"""Service for managing and analyzing links between notes."""

import logging
import os
import re
import json
import hashlib
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LinkService:
    """Manages link relationships and suggestions between notes."""

    def __init__(self, vector_store, chunking_service=None, embedding_service=None):
        """
        Initialize the link service.

        Args:
            vector_store: Instance of VectorStoreService for content queries
        """
        self.vector_store = vector_store
        self.chunking_service = chunking_service or vector_store.chunking_service
        self.embedding_service = embedding_service or vector_store.embedding_service

        # Get the notes base directory from config and ensure it's absolute
        self.base_path = os.path.expanduser(
            self.vector_store.config.get("notes_base_dir", "~/Documents/notes")
        )
        # Ensure base_path is absolute
        self.base_path = os.path.abspath(self.base_path)

        # Get the base directory name for path extraction
        self.base_dir_name = os.path.basename(self.base_path)

        # Get the parent directory of base_path
        self.parent_dir = os.path.dirname(self.base_path)

        logger.info(f"Using notes base directory: {self.base_path}")
        logger.info(f"Base directory name: {self.base_dir_name}")
        logger.info(f"Parent directory: {self.parent_dir}")

        # Store analysis state alongside vector store data
        vector_store_dir = os.path.join(self.base_path, ".vector_store")
        self.analysis_state_file = os.path.join(vector_store_dir, ".note_analysis_state")

        # Ensure vector store directory exists
        os.makedirs(vector_store_dir, exist_ok=True)

        self._load_analysis_state()

    def _load_analysis_state(self) -> None:
        """Load the analysis state from disk."""
        try:
            with open(self.analysis_state_file, "r") as f:
                self.analysis_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.analysis_state = {}

    def _save_analysis_state(self) -> None:
        """Save the analysis state to disk."""
        with open(self.analysis_state_file, "w") as f:
            json.dump(self.analysis_state, f, indent=2)

    def _get_content_hash(self, content: str) -> str:
        """Generate a hash of the note content, ignoring the auto-generated references section."""
        # Split at the auto-generated references section if it exists
        if "## Auto generated references" in content:
            content = content.split("## Auto generated references")[0].rstrip()
            # Remove trailing horizontal rule if present
            if content.endswith("\n---"):
                content = content[:-4].rstrip()
            elif content.endswith("\n---\n"):
                content = content[:-5].rstrip()
        return hashlib.md5(content.encode()).hexdigest()

    def _needs_analysis(self, note_id: str, content: str) -> bool:
        """
        Check if a note needs to be analyzed based on its content hash and last analysis time.

        Args:
            note_id: ID of the note
            content: Current content of the note

        Returns:
            bool: True if the note needs analysis, False otherwise
        """
        current_hash = self._get_content_hash(content)
        state = self.analysis_state.get(note_id, {})

        # If no previous analysis or content has changed
        if not state or state.get("content_hash") != current_hash:
            return True

        return False

    def _update_analysis_state(self, note_id: str, content: str) -> None:
        """
        Update the analysis state for a note.

        Args:
            note_id: ID of the note
            content: Current content of the note
        """
        self.analysis_state[note_id] = {
            "content_hash": self._get_content_hash(content),
            "last_analyzed": time.time()
        }
        self._save_analysis_state()

    def analyze_relationships(self, note_id: str, auto_index: bool = False) -> Dict[str, Any]:
        """
        Analyze relationships for a specific note.

        Args:
            note_id: ID of the note to analyze
            auto_index: Whether to automatically index the note if not found in vector store

        Returns:
            Dictionary containing relationship analysis
        """
        try:
            # Convert to absolute path if it's not already
            if not os.path.isabs(note_id):
                note_id = os.path.join(self.base_path, note_id)

            # Normalize the path
            note_id = os.path.normpath(note_id)
            logger.debug(f"Analyzing note with absolute path: {note_id}")

            # Make the path relative to base_path for vector store operations
            relative_note_id = self._get_relative_note_id(note_id)
            logger.debug(f"Final relative note ID: {relative_note_id}")

            # Check if file exists
            if not os.path.exists(note_id):
                logger.warning(f"Note file not found: {note_id}")
                return {
                    "direct_links": [],
                    "semantic_links": [],
                    "backlinks": [],
                    "suggested_links": []
                }

            # Get note content using relative path
            note_content = self.vector_store.get_note_content(relative_note_id)

            # If note is not in vector store, read it from file and index it if auto_index is True
            if not note_content:
                if auto_index:
                    logger.info(f"Note {relative_note_id} not found in vector store, indexing it now")
                    try:
                        with open(note_id, "r") as f:
                            file_content = f.read()

                        # Get file metadata
                        stat = os.stat(note_id)
                        metadata = {
                            "id": relative_note_id,
                            "path": note_id,
                            "modified_time": stat.st_mtime,
                            "created_time": stat.st_ctime,
                            "type": "note"
                        }

                        # Generate chunks and embeddings
                        chunks = self.chunking_service.chunk_document(file_content, doc_type="note")
                        if not chunks:
                            logger.warning(f"No chunks generated for note: {relative_note_id}")
                            return {
                                "direct_links": [],
                                "semantic_links": [],
                                "backlinks": [],
                                "suggested_links": []
                            }

                        chunk_texts = [chunk["content"] for chunk in chunks]
                        embeddings = self.embedding_service.embed_chunks(chunk_texts)

                        # Add document to vector store
                        self.vector_store.add_document(
                            doc_id=relative_note_id,
                            chunks=chunk_texts,
                            embeddings=embeddings,
                            metadata=metadata
                        )

                        # Get note content again after indexing
                        note_content = self.vector_store.get_note_content(relative_note_id)
                        if not note_content:
                            logger.error(f"Failed to index note: {relative_note_id}")
                            return {
                                "direct_links": [],
                                "semantic_links": [],
                                "backlinks": [],
                                "suggested_links": []
                            }
                    except Exception as e:
                        logger.error(f"Error indexing note {relative_note_id}: {str(e)}")
                        return {
                            "direct_links": [],
                            "semantic_links": [],
                            "backlinks": [],
                            "suggested_links": []
                        }
                else:
                    logger.warning(f"Could not get content for note: {relative_note_id} (auto-indexing disabled)")
                    return {
                        "direct_links": [],
                        "semantic_links": [],
                        "backlinks": [],
                        "suggested_links": []
                    }

            # Check if we need to analyze this note
            if not self._needs_analysis(relative_note_id, note_content["content"]):
                logger.info(f"Skipping analysis for {relative_note_id} - content unchanged since last analysis")
                return {
                    "direct_links": self.vector_store.find_connected_notes(relative_note_id),
                    "semantic_links": [],  # Skip semantic analysis for unchanged content
                    "backlinks": self._find_backlinks(relative_note_id),
                    "suggested_links": []  # Skip suggestions since content hasn't changed
                }

            logger.info(f"Starting relationship analysis for note: {relative_note_id}")

            # Get existing connections
            direct_links = self.vector_store.find_connected_notes(relative_note_id)
            logger.info(f"Found {len(direct_links)} direct links")

            # Find semantic relationships
            semantic_links = self.vector_store.find_similar(
                query_embedding=note_content["embedding"],
                limit=5,  # Keep original limit for semantic links
                threshold=0.6,  # Keep original threshold for semantic links
            )

            # Get backlinks
            backlinks = self._find_backlinks(relative_note_id)

            # Get suggested connections
            suggested_links = self._suggest_connections(relative_note_id, direct_links, backlinks)

            # Update analysis state and vector store
            self._update_analysis_state(relative_note_id, note_content["content"])
            current_time = time.time()
            self.vector_store.set_last_update_time(current_time)
            logger.info(f"Updated last_update timestamp to {current_time}")

            return {
                "direct_links": direct_links,
                "semantic_links": semantic_links,
                "backlinks": backlinks,
                "suggested_links": suggested_links,
            }

        except Exception as e:
            logger.error(f"Error analyzing relationships for {note_id}: {str(e)}")
            return {
                "direct_links": [],
                "semantic_links": [],
                "backlinks": [],
                "suggested_links": []
            }

    def _find_backlinks(self, note_id: str) -> List[Dict[str, Any]]:
        """
        Find all notes that link to this note.

        Args:
            note_id: ID of the note to find backlinks for

        Returns:
            List of notes linking to this note
        """
        return self.vector_store.find_backlinks(note_id)

    def _suggest_connections(
        self, note_id: str, existing_links: List[Dict[str, Any]] = None, backlinks: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest potential connections based on content similarity.

        Args:
            note_id: ID of the note to find suggestions for
            existing_links: Optional list of existing direct links to avoid duplicate calls
            backlinks: Optional list of backlinks to avoid duplicate calls

        Returns:
            List of suggested connections
        """
        # Get note content
        note_content = self.vector_store.get_note_content(note_id)
        if not note_content:
            logger.warning(f"Could not get content for note: {note_id}")
            return []

        # Find similar content with lower threshold for suggestions
        similar = self.vector_store.find_similar(
            query_embedding=note_content["embedding"],
            limit=10,  # Increased limit to find more potential matches
            threshold=0.5,  # Lower threshold for suggestions
        )

        logger.info(f"Found {len(similar)} similar documents for {note_id}")

        # Get existing connections including backlinks
        existing = set()
        direct_links = existing_links or self.vector_store.find_connected_notes(note_id)
        backlinks = backlinks or self.vector_store.find_backlinks(note_id)

        # Add both direct links and backlinks to existing set
        existing.update(link["target_id"] for link in direct_links)
        existing.update(link["source_id"] for link in backlinks)

        logger.info(f"Found {len(existing)} existing connections")

        # Group results by note_id and calculate aggregate scores
        note_matches = {}
        for result in similar:
            result_id = result["metadata"].get("doc_id")
            # Skip self-links and already existing links
            if (
                result_id
                and result_id not in existing
                and result_id != note_id
                and os.path.normpath(result_id) != os.path.normpath(note_id)
            ):
                if result_id not in note_matches:
                    note_matches[result_id] = {
                        "note_id": result_id,
                        "max_similarity": result["similarity"],
                        "chunk_similarities": [result["similarity"]],
                        "best_preview": result["content"][:200].replace("\n", " ").strip(),
                        "chunk_count": 1
                    }
                else:
                    match = note_matches[result_id]
                    match["chunk_similarities"].append(result["similarity"])
                    match["chunk_count"] += 1
                    if result["similarity"] > match["max_similarity"]:
                        match["max_similarity"] = result["similarity"]
                        match["best_preview"] = result["content"][:200].replace("\n", " ").strip()

        # Calculate aggregate scores and create final suggestions
        suggestions = []
        for note_id, match in note_matches.items():
            # Calculate aggregate score that rewards multiple chunks
            # We use a combination of:
            # 1. The maximum similarity of any chunk
            # 2. A bonus based on number of chunks and their similarities
            chunk_bonus = sum(match["chunk_similarities"]) / len(match["chunk_similarities"]) * min(1, match["chunk_count"] / 3)
            aggregate_score = match["max_similarity"] + (chunk_bonus * 0.2)  # Adjust bonus weight as needed

            suggestions.append({
                "note_id": note_id,
                "similarity": round(aggregate_score, 3),
                "reason": f"Content similarity ({match['max_similarity']:.2f}) with {match['chunk_count']} matching chunks",
                "preview": match["best_preview"]
            })
            logger.debug(
                f"Added suggestion: {note_id} with aggregate score {aggregate_score:.3f} from {match['chunk_count']} chunks"
            )

        # Sort suggestions by aggregate score
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(f"Generated {len(suggestions)} suggestions for {note_id}")
        return suggestions

    def update_obsidian_links(
        self, note_path: str, links: List[Dict[str, Any]], update_backlinks: bool = True, skip_vector_update: bool = False
    ) -> None:
        """
        Update Obsidian-style wiki links in a note file.

        Args:
            note_path: Full path to the note file
            links: List of links to add/update
            update_backlinks: Whether to update backlinks in target notes
            skip_vector_update: Whether to skip updating the vector store (useful for batch operations)

        Raises:
            FileNotFoundError: If the note file does not exist
            IOError: If there is an error reading or writing the file
        """
        # Ensure we have the full path
        note_path = os.path.expanduser(note_path)

        # Get relative ID for vector store
        note_id = self._get_relative_note_id(note_path)

        logger.debug(f"Using note ID: {note_id} for path: {note_path}")

        try:
            # Read the entire file first
            with open(note_path, "r") as f:
                original_content = f.read()

            # Split content at the auto-generated references section if it exists
            if "## Auto generated references" in original_content:
                main_content = original_content.split("## Auto generated references")[0].rstrip()
                # Remove any trailing horizontal rule if it's right before the auto-generated section
                if main_content.endswith("\n---\n"):
                    main_content = main_content[:-5].rstrip()
            else:
                # Split at horizontal rule if it exists
                if "\n---\n" in original_content:
                    parts = original_content.split("\n---\n", 1)
                    main_content = parts[0]
                    # Keep any non-auto-generated content after the horizontal rule
                    if len(parts) > 1 and parts[1].strip():
                        main_content += "\n---\n" + parts[1]
                else:
                    main_content = original_content

            # Process links to add
            new_links = []
            for link in links:
                if link.get("add_wiki_link"):
                    target = link["target_id"]
                    # Use provided alias if available, otherwise generate one
                    alias = link.get("alias") or self._generate_alias(target)
                    new_link = f"[[{target}|{alias}]]"

                    # Check if link already exists in main content
                    if not self._has_wiki_link(main_content, target):
                        new_links.append(new_link)

                        # Update backlinks in target note if requested
                        if update_backlinks:
                            # Pass the target ID directly, not a path constructed from the current note path
                            self._update_target_backlinks(target, note_id, note_path)

            # Only add the auto-generated section if we have links to add
            if new_links:
                # Start with the main content
                updated_content = main_content.rstrip()

                # Add horizontal rule if not present
                if not updated_content.endswith("\n---\n"):
                    updated_content += "\n\n---"

                # Add the auto-generated references section
                updated_content += "\n\n## Auto generated references"
                for link in new_links:
                    updated_content += f"\n{link}"
                updated_content += "\n"

                # Write changes to the file
                with open(note_path, "w") as f:
                    logger.info(f"Writing changes to file: {note_path}")
                    f.write(updated_content)
                logger.info(f"Updated links in file: {note_path}")

                # Update analysis state
                self._update_analysis_state(note_id, updated_content)

                # Update the vector store if not skipped
                if not skip_vector_update:
                    self._update_vector_store(note_id, updated_content)
                    current_time = time.time()
                    self.vector_store.set_last_update_time(current_time)
                    logger.info(f"Updated last_update timestamp to {current_time}")
            else:
                logger.info(f"No new links to add for: {note_path}")

        except FileNotFoundError as e:
            logger.error(f"Error reading file {note_path}: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"Error accessing file {note_path}: {str(e)}")
            raise

    def _generate_alias(self, target: str) -> str:
        """Generate a readable alias from the target ID."""
        # Remove file extension if present
        base_name = os.path.basename(target).replace(".md", "")

        # Replace hyphens and underscores with spaces
        words = base_name.replace("-", " ").replace("_", " ").split()

        # Capitalize each word
        return " ".join(word.capitalize() for word in words)

    def _has_wiki_link(self, content: str, target: str) -> bool:
        """
        Check if content already contains a wiki link to target.

        Args:
            content: The note content to check
            target: The target note ID to look for

        Returns:
            bool: True if the link already exists
        """
        # Check for exact wiki link match
        exact_pattern = rf"\[\[{re.escape(target)}(?:\|[^\]]+)?\]\]"
        if re.search(exact_pattern, content):
            logger.debug(f"Found exact wiki link match for {target}")
            return True

        # Check for link in any section
        sections = ["## Related", "## Links", "## References", "## Backlinks"]
        for section in sections:
            if section in content:
                section_content = content.split(section, 1)[1].split("\n\n")[0]
                if target in section_content:
                    logger.debug(f"Found {target} in {section} section")
                    return True

        # Check for any mention of the target file name
        base_name = os.path.basename(target).replace(".md", "")
        if f"[[{base_name}" in content:
            logger.debug(f"Found mention of {base_name} in content")
            return True

        return False

    def _insert_wiki_link(self, content: str, new_link: str) -> str:
        """Insert a wiki link at the bottom of the document."""
        # Check if we already have a horizontal line separator
        if "\n---\n" not in content:
            # Add horizontal line and links section
            return f"{content.rstrip()}\n\n---\n{new_link}"
        else:
            # Add to existing links section after the horizontal line
            parts = content.rsplit("\n---\n", 1)
            return f"{parts[0]}\n---\n{parts[1].rstrip()}\n{new_link}"

    def _remove_wiki_link(self, content: str, target: str) -> str:
        """Remove a wiki link from the content."""
        pattern = rf"\[\[{re.escape(target)}(?:\|[^\]]+)?\]\]\n?"
        return re.sub(pattern, "", content)

    def _update_target_backlinks(
        self, target_id: str, source_id: str, source_path: str
    ) -> None:
        """
        Update backlinks in a target note.

        Args:
            target_id: ID of the target note
            source_id: ID of the source note
            source_path: Path to the source note
        """
        # Get target note content and path
        note_content = self.vector_store.get_note_content(target_id)
        if not note_content:
            logger.warning(f"Could not get content for target note: {target_id}")
            return

        # Construct target note path
        # First convert to absolute path if it's not already
        if not os.path.isabs(target_id):
            # Use the base_path directly to construct the target path
            target_path = os.path.join(self.base_path, target_id)
            logger.debug(f"Constructed target path from base_path: {target_path}")
        else:
            target_path = target_id
            logger.debug(f"Using absolute target path: {target_path}")

        logger.debug(f"Looking for target note at: {target_path}")

        if not os.path.exists(target_path):
            logger.warning(f"Target note file not found: {target_path}")
            return

        try:
            # Read the target note content
            with open(target_path, "r") as f:
                content = f.read()

            # Check if backlink already exists
            if f"[[{source_id}]]" not in content:
                # Add backlink at the end of the file
                if not content.endswith("\n"):
                    content += "\n"

                # Add auto-generated references section if it doesn't exist
                if "## Auto generated references" not in content:
                    content += "\n---\n\n## Auto generated references\n"

                # Add the backlink
                content += f"[[{source_id}]]\n"

                # Write updated content
                with open(target_path, "w") as f:
                    f.write(content)
                logger.info(f"Added backlink to {source_id} in {target_path}")

        except IOError as e:
            logger.error(f"Error updating backlinks in {target_path}: {str(e)}")
            return

    def _update_vector_store(self, note_id: str, content: str) -> None:
        """
        Update the vector store for a note.

        Args:
            note_id: ID of the note to update
            content: New content of the note
        """
        try:
            chunks = self.vector_store.chunking_service.chunk_document(
                content, doc_type="note"
            )
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = self.vector_store.embedding_service.embed_chunks(
                chunk_texts
            )
            self.vector_store.update_document(
                doc_id=note_id, new_chunks=chunk_texts, new_embeddings=embeddings
            )
            logger.info(f"Updated vector store for: {note_id}")
        except Exception as e:
            logger.error(f"Failed to update vector store for {note_id}: {str(e)}")

    def _get_relative_note_id(self, path: str) -> str:
        """
        Convert an absolute path to a relative note ID for vector store operations.

        Args:
            path: Absolute or relative path to a note

        Returns:
            Relative note ID for vector store operations
        """
        # If it's already a relative path, return it as is
        if not os.path.isabs(path):
            return path

        # Normalize the path
        path = os.path.normpath(path)
        
        # Log the path we're trying to convert
        logger.debug(f"Converting path to relative ID: {path}")
        logger.debug(f"Base path: {self.base_path}")

        # Try to make it relative to the base path
        try:
            if path.startswith(self.base_path):
                relative_path = os.path.relpath(path, self.base_path)
                logger.debug(f"Path is relative to base_path: {relative_path}")
                return relative_path

            # If it's not under base_path, check if it contains the base directory name
            base_dir_pattern = f"/{self.base_dir_name}/"
            if base_dir_pattern in path:
                parts = path.split(base_dir_pattern)
                if len(parts) > 1:
                    relative_path = parts[1]
                    logger.debug(f"Path extracted after base dir: {relative_path}")
                    return relative_path
                
            # Try another approach - find the common part with base_path
            common_prefix = os.path.commonpath([path, self.base_path])
            if common_prefix and common_prefix != "/":
                relative_path = path[len(common_prefix):].lstrip('/')
                logger.debug(f"Path relative to common prefix: {relative_path}")
                return relative_path
                
        except Exception as e:
            logger.warning(f"Error converting path to relative ID: {str(e)}")

        # If all else fails, just return the basename
        basename = os.path.basename(path)
        logger.debug(f"Using basename as fallback: {basename}")
        return basename
