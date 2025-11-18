"""Service for chunking documents using LangChain's RecursiveCharacterTextSplitter."""

import logging
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingService:
    """Manages the chunking of documents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chunking service.

        Args:
            config: Configuration dictionary containing chunking settings
        """
        self.config = config
        chunk_config = config.get("chunking_config", {}).get("recursive", {})
        self.chunk_size = chunk_config.get("chunk_size", 300)
        self.chunk_overlap = chunk_config.get("chunk_overlap", 100)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=chunk_config.get(
                "separators",
                ["\n\n", "\n### ", "\n## ", "\n# ", "\n", ". ", "? ", "! ", "; "]
            ),
            is_separator_regex=False,
        )

    def chunk_document(
        self,
        content: str,
        doc_type: str = "note",
        title: str = "",
        tags: List[str] = None,
        frontmatter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split a document into chunks using LangChain's RecursiveCharacterTextSplitter.

        Args:
            content: Content to chunk
            doc_type: Type of document
            title: Document title
            tags: Document tags
            frontmatter: Document frontmatter

        Returns:
            List of chunks with metadata
        """
        try:
            # Split into initial chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Created {len(chunks)} initial chunks")

            # Further split any chunks that exceed the size limit
            result = []
            for i, chunk in enumerate(chunks, 1):
                # If chunk exceeds size limit, split it further
                if len(chunk) > self.chunk_size:
                    subchunks = [chunk[i:i + self.chunk_size] 
                               for i in range(0, len(chunk), self.chunk_size)]
                    for j, subchunk in enumerate(subchunks, 1):
                        chunk_num = f"{i}.{j}"
                        self._add_chunk_to_result(result, subchunk, doc_type, title, 
                                                tags, frontmatter, chunk_num)
                else:
                    self._add_chunk_to_result(result, chunk, doc_type, title, 
                                            tags, frontmatter, str(i))

            return result

        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            # Fallback: return entire content as single chunk
            return [
                {
                    "content": content,
                    "metadata": {
                        "title": title or "Single chunk",
                        "char_count": len(content),
                        "doc_type": doc_type,
                        "tags": tags or [],
                    },
                }
            ]
    def _add_chunk_to_result(
        self, 
        result: List[Dict[str, Any]], 
        chunk: str,
        doc_type: str,
        title: str,
        tags: List[str],
        frontmatter: Dict[str, Any],
        chunk_num: str
    ) -> None:
        """Add a chunk with its metadata to the result list."""
        chunk_metadata = {
            "title": f"{title or 'Chunk'} (Part {chunk_num})",
            "char_count": len(chunk),
            "doc_type": doc_type,
            "doc_title": title,
            "tags": tags or [],
        }

        # Add relevant frontmatter fields
        if frontmatter:
            for key, value in frontmatter.items():
                if key not in ["content", "title", "tags"]:
                    chunk_metadata[f"frontmatter_{key}"] = value

        result.append({"content": chunk, "metadata": chunk_metadata})
