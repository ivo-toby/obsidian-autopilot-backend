"""
Note processing and summarization tool.

This module provides functionality for processing daily notes, weekly summaries,
meeting notes and learning entries. It interfaces with various services to
generate summaries, extract tasks, and manage reminders.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any
from services.learning_service import LearningService
from services.summary_service import SummaryService
from services.meeting_service import MeetingService
from services.openai_service import OpenAIService
from services.vector_store import VectorStoreService, EmbeddingService, ChunkingService
from services.knowledge.link_service import LinkService
from utils.config_loader import load_config
from utils.cli import setup_argparser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_daily_notes(cfg, cli_args):
    """Process daily notes to generate summaries and extract tasks."""
    summary_service = SummaryService(cfg)
    summary_service.process_daily_notes(
        date_str=cli_args.date,
        dry_run=cli_args.dry_run,
        skip_reminders=cli_args.skip_reminders,
        replace_summary=cli_args.replace_summary
    )

def process_weekly_notes(cfg, cli_args):
    """Process and generate weekly note summaries."""
    summary_service = SummaryService(cfg)
    summary_service.process_weekly_notes(
        date_str=cli_args.date,
        dry_run=cli_args.dry_run,
        replace_summary=cli_args.replace_summary
    )

def process_meeting_notes(cfg, cli_args):
    """Process daily notes to generate structured meeting notes."""
    meeting_service = MeetingService(cfg)
    meeting_service.process_meeting_notes(
        date_str=cli_args.date,
        dry_run=cli_args.dry_run
    )

def process_new_learnings(cfg, cli_args):
    """Process and extract new learnings from notes."""
    learning_service = LearningService(
        cfg["learnings_file"], cfg["learnings_output_dir"]
    )
    learning_service.process_new_learnings(
        OpenAIService(api_key=cfg["api_key"], model=cfg["model"])
    )

def process_knowledge_base(cfg, cli_args):
    """Handle knowledge base operations."""
    try:
        vector_store = VectorStoreService(cfg)
        embedding_service = EmbeddingService(cfg)
        chunking_service = ChunkingService(cfg)
        summary_service = SummaryService(cfg)
        link_service = LinkService(vector_store)

        if cli_args.reindex or cli_args.update:
            if cli_args.reindex:
                logger.info("Reindexing all notes...")
                notes = summary_service.get_all_notes()
            else:
                last_update = vector_store.get_last_update_time()
                logger.info(f"Checking for notes modified since {datetime.fromtimestamp(last_update)}")
                notes = [note for note in summary_service.get_all_notes() 
                        if note.get('modified_time', 0) > last_update]
                if not notes:
                    logger.info("No notes need updating")
                    return
                logger.info(f"Found {len(notes)} notes to update")

            if not cli_args.dry_run:
                current_time = time.time()
                for note in notes:
                    logger.info(f"Processing note: {note['id']}")
                    chunks = chunking_service.chunk_document(
                        note['content'],
                        doc_type=note.get('type', 'note')
                    )
                    if not chunks:
                        logger.warning(f"No chunks generated for note: {note['id']}")
                        continue
                        
                    chunk_texts = [chunk['content'] for chunk in chunks]
                    logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
                    embeddings = embedding_service.embed_chunks(chunk_texts)
                    logger.info("Embeddings generated successfully")
                    vector_store.add_document(
                        doc_id=note['id'],
                        chunks=chunk_texts,
                        embeddings=embeddings,
                        metadata=note
                    )
                
                if cli_args.update:
                    vector_store.set_last_update_time(current_time)
                    logger.info(f"Updated last_update timestamp to {datetime.fromtimestamp(current_time)}")
            else:
                logger.info("Dry run - no changes made")

        elif cli_args.query:
            # Search for similar content
            logger.info(f"Searching for: {cli_args.query}")
            query_embedding = embedding_service.embed_text(cli_args.query)
            results = vector_store.find_similar(
                query_embedding=query_embedding,
                limit=cli_args.limit,
                doc_type=cli_args.note_type,
                threshold=cfg.get('vector_store', {}).get('similarity_threshold', 0.3)  # Lower default threshold
            )
            if not results:
                logger.info("No matching results found")
            else:
                logger.info(f"Found {len(results)} matching results")
                _display_search_results(results)

        elif cli_args.show_connections:
            note_path = os.path.expanduser(cli_args.show_connections)
            connections = vector_store.find_connected_notes(note_path)
            if cli_args.graph:
                _display_connections_graph(note_path, connections)
            else:
                _display_connections(note_path, connections)

        elif cli_args.find_by_tag:
            query = f"tag:{cli_args.find_by_tag}"
            query_embedding = embedding_service.embed_text(query)
            results = vector_store.find_similar(
                query_embedding=query_embedding,
                limit=cli_args.limit,
                threshold=0.5  # Lower threshold for tag search
            )
            _display_search_results(results, focus='tags')

        elif cli_args.find_by_date:
            try:
                date = datetime.strptime(cli_args.find_by_date, "%Y-%m-%d")
                query = f"date:{date.strftime('%Y-%m-%d')}"
                query_embedding = embedding_service.embed_text(query)
                results = vector_store.find_similar(
                    query_embedding=query_embedding,
                    limit=cli_args.limit,
                    threshold=0.5  # Lower threshold for date search
                )
                _display_search_results(results, focus='dates')
            except ValueError:
                logger.error("Invalid date format. Please use YYYY-MM-DD")

        elif cli_args.analyze_links:
            note_path = os.path.expanduser(cli_args.analyze_links)
            logger.info(f"Analyzing links for: {note_path}")
            analysis = link_service.analyze_relationships(note_path)
            _display_link_analysis(note_path, analysis)

            # Update links based on suggestions
            if not cli_args.dry_run and analysis['suggested_links']:
                if cli_args.auto_link or input("\nAdd suggested links to note? (y/N): ").lower() == 'y':
                    links_to_add = [
                        {
                            'add_wiki_link': True,
                            'target_id': suggestion['note_id'],
                            'alias': None  # Let the service generate an alias
                        }
                        for suggestion in analysis['suggested_links']
                    ]
                    link_service.update_obsidian_links(note_path, links_to_add)
                    logger.info("Links updated successfully")

        elif cli_args.note_structure:
            note_path = os.path.expanduser(cli_args.note_structure)
            try:
                with open(note_path, 'r') as f:
                    content = f.read()
                chunks = chunking_service.chunk_document(
                    content,
                    doc_type=cli_args.note_type or 'note'
                )
                _display_note_structure(chunks)
            except FileNotFoundError:
                logger.error(f"Note not found: {note_path}")

    except Exception as e:
        logger.error(f"Error processing knowledge base: {str(e)}")
        raise

def _display_search_results(results, focus=None):
    """Display search results with optional focus on specific metadata."""
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.2f}")
        print(f"Source: {result['metadata'].get('source', 'unknown')}")
        if focus == 'tags' and 'tags' in result['metadata']:
            print(f"Tags: {', '.join(result['metadata']['tags'])}")
        if focus == 'dates' and 'dates' in result['metadata']:
            print(f"Dates: {', '.join(result['metadata']['dates'])}")
        print(f"Content: {result['content'][:200]}...")

def _display_connections(note_path, connections):
    """Display note connections in text format."""
    print(f"\nConnections for {note_path}:")
    for conn in connections:
        print(f"\n- {conn['target_id']}")
        print(f"  Relationship: {conn['relationship']}")
        print(f"  Type: {conn['link_type']}")
        if conn.get('context'):
            print(f"  Context: {conn['context'][:100]}...")

def _display_connections_graph(note_path, connections):
    """Display note connections in Mermaid graph format."""
    print("\n```mermaid")
    print("graph TD")
    source_id = os.path.basename(note_path).replace('.md', '')
    print(f"    {source_id}[{os.path.basename(note_path)}]")
    
    for conn in connections:
        target_id = os.path.basename(conn['target_id']).replace('.md', '')
        print(f"    {source_id} -->|{conn['relationship']}| {target_id}[{os.path.basename(conn['target_id'])}]")
    
    print("```")

def _display_link_analysis(note_path: str, analysis: Dict[str, Any]) -> None:
    """Display link analysis results."""
    print(f"\nLink Analysis for {note_path}:")
    
    print("\nDirect Links:")
    for link in analysis['direct_links']:
        print(f"- {link['target_id']} ({link['relationship']})")
        if link.get('context'):
            print(f"  Context: {link['context'][:100]}...")

    print("\nBacklinks:")
    for link in analysis['backlinks']:
        print(f"- {link['source_id']} ({link['relationship']})")
        if link.get('context'):
            print(f"  Context: {link['context'][:100]}...")

    print("\nSemantic Relationships:")
    for link in analysis['semantic_links']:
        print(f"- {link['metadata'].get('doc_id', 'Unknown')} "
              f"(similarity: {link['similarity']:.2f})")
        print(f"  Preview: {link['content'][:100]}...")

    print("\nSuggested Connections:")
    for suggestion in analysis['suggested_links']:
        print(f"- {suggestion['note_id']} "
              f"(similarity: {suggestion['similarity']:.2f})")
        print(f"  Reason: {suggestion['reason']}")
        print(f"  Preview: {suggestion['preview']}")

def _display_note_structure(chunks):
    """Display semantic structure of a note."""
    print("\nNote Structure:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{i}. {chunk['metadata']['title']}")
        print(f"   Type: {chunk['metadata'].get('doc_type', 'note')}")
        if chunk['metadata'].get('tags'):
            print(f"   Tags: {', '.join(chunk['metadata']['tags'])}")
        if chunk['metadata'].get('dates'):
            print(f"   Dates: {', '.join(chunk['metadata']['dates'])}")
        print(f"   Size: {chunk['metadata']['char_count']} characters")
        print(f"   Preview: {chunk['content'][:100]}...")

if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == 'notes':
        if args.process_learnings:
            process_new_learnings(cfg, args)
        elif args.meetingnotes:
            process_meeting_notes(cfg, args)
        elif args.weekly:
            process_weekly_notes(cfg, args)
        else:
            process_daily_notes(cfg, args)
            process_meeting_notes(cfg, args)
            process_new_learnings(cfg, args)
    elif args.command == 'kb':
        process_knowledge_base(cfg, args)
    else:
        # Default behavior for backward compatibility
        if hasattr(args, 'process_learnings') and args.process_learnings:
            process_new_learnings(cfg, args)
        elif hasattr(args, 'meetingnotes') and args.meetingnotes:
            process_meeting_notes(cfg, args)
        elif hasattr(args, 'weekly') and args.weekly:
            process_weekly_notes(cfg, args)
        else:
            process_daily_notes(cfg, args)
            process_meeting_notes(cfg, args)
            process_new_learnings(cfg, args)
