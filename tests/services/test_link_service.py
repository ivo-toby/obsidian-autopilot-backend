"""
Tests for the LinkService class.

This module contains comprehensive tests for the LinkService functionality,
including relationship analysis, link management, and content suggestions.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
# Import pathlib
import pathlib
from services.knowledge.link_service import LinkService

# Test data
SAMPLE_NOTE_CONTENT = """# Test Note

This is a test note with some [[wiki-link]] and a [[another-link|Custom Title]].
It also has [external links](https://example.com).

## Related
- [[related-note]]

---
[[backlink-test]]
"""

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = Mock()

    # Mock config and path
    mock.config = {
        "path": "~/Documents/notes/.vector_store"
    }

    # Set default return values for core methods accessed by LinkService
    # These will be used unless overridden in specific tests
    mock.find_connected_notes.return_value = []
    mock.find_backlinks.return_value = []
    # Ensure get_note_content returns something valid by default for hashing etc.
    mock.get_note_content.return_value = {
        "content": SAMPLE_NOTE_CONTENT, # Default content
        "embedding": [0.1] * 1536,     # Default embedding
        "metadata": {"type": "note"}
    }
    mock.find_similar.return_value = []
    mock.update_note = MagicMock()

    # Mock find_connected_notes - specific setup moved to test_analyze_relationships if needed
    # mock.find_connected_notes.return_value = [
    #     {
    #         "target_id": "wiki-link",
    #         "relationship": "references",
    #         "link_type": "wiki",
    #         "context": "Context for wiki-link"
    #     }
    # ]

    # Mock find_backlinks - specific setup moved to test if needed
    # mock.find_backlinks.return_value = [
    #     {
    #         "source_id": "source-note",
    #         "relationship": "references",
    #         "link_type": "wiki",
    #         "context": "Context from source note"
    #     }
    # ]

    # Mock get_note_content - default is now set above
    # mock.get_note_content.return_value = {
    #     "content": SAMPLE_NOTE_CONTENT,
    #     "embedding": [0.1] * 1536,
    #     "metadata": {"type": "note"}
    # }

    # Mock find_similar - specific setup moved to test if needed
    # mock.find_similar.return_value = [
    #     {
    #         "chunk_id": "similar-note_chunk_1",
    #         "content": "Similar content",
    #         "metadata": {"doc_id": "similar-note"},
    #         "similarity": 0.85
    #     }
    # ]

    # Set chunking_service and embedding_service
    mock.chunking_service = Mock()
    mock.chunking_service.chunk_document.return_value = [
        {
            "content": SAMPLE_NOTE_CONTENT,
            "metadata": {"type": "note"}
        }
    ]

    mock.embedding_service = Mock()
    mock.embedding_service.embed_chunks.return_value = [[0.1] * 1536]

    return mock

@pytest.fixture
def mock_chunking_service():
    """Create a mock chunking service."""
    mock = Mock()
    mock.chunk_document.return_value = [
        {
            "content": SAMPLE_NOTE_CONTENT,
            "metadata": {"type": "note"}
        }
    ]
    return mock

@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = Mock()
    mock.embed_chunks.return_value = [[0.1] * 1536]
    return mock

@pytest.fixture
def link_service(mock_vector_store, mock_chunking_service, mock_embedding_service, tmp_path):
    """Create a LinkService instance with mocked dependencies."""
    # Mock os.path.exists INSTEAD of pathlib
    with patch('services.knowledge.link_service.os.path.exists', return_value=True, autospec=True) as mock_os_exists:
        # Mock the analysis state file path resolution to use tmp_path
        with patch('services.knowledge.link_service.os.path.expanduser', return_value=str(tmp_path)):
             service = LinkService(mock_vector_store, mock_chunking_service, mock_embedding_service)
             service.base_path = str(tmp_path)
             service.analysis_state_file = os.path.join(str(tmp_path), "analysis_state.json")
             # Initialize state - use os.path.exists which should be mocked
             if not os.path.exists(service.analysis_state_file):
                 service.analysis_state = {}
             return service

def test_analyze_relationships(link_service, mock_vector_store):
    """Test relationship analysis for a note."""
    note_id = "test-note"
    # Create the dummy note file WITHOUT .md extension
    # Convert base_path string to Path object for joining
    note_path = pathlib.Path(link_service.base_path) / note_id
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_NOTE_CONTENT)

    # --- Mocks Setup ---
    # Set up SPECIFIC mock returns for THIS test
    mock_vector_store.find_connected_notes.return_value = [{"target_id": "wiki-link", "relationship": "references", "link_type": "wiki", "context": "ctx"}]
    mock_vector_store.find_backlinks.return_value = [{"source_id": "source-note", "relationship": "references", "link_type": "wiki", "context": "ctx"}]
    mock_vector_store.get_note_content.return_value = {"content": SAMPLE_NOTE_CONTENT, "embedding": [0.1] * 1536}
    mock_vector_store.find_similar.side_effect = [
        [{"metadata": {"doc_id": "similar-note"}, "content": "similar content", "similarity": 0.8}],
        [{"metadata": {"doc_id": "suggested-note"}, "content": "suggested content", "similarity": 0.55}]
    ]

    # Reset mocks before the call
    mock_vector_store.reset_mock()
    # Re-apply mocks
    mock_vector_store.find_connected_notes.return_value = [{"target_id": "wiki-link", "relationship": "references", "link_type": "wiki", "context": "ctx"}]
    mock_vector_store.find_backlinks.return_value = [{"source_id": "source-note", "relationship": "references", "link_type": "wiki", "context": "ctx"}]
    mock_vector_store.get_note_content.return_value = {"content": SAMPLE_NOTE_CONTENT, "embedding": [0.1] * 1536}
    mock_vector_store.find_similar.side_effect = [
        [{"metadata": {"doc_id": "similar-note"}, "content": "similar content", "similarity": 0.8}],
        [{"metadata": {"doc_id": "suggested-note"}, "content": "suggested content", "similarity": 0.55}]
    ]

    # --- Run Analysis ---
    link_service.analysis_state = {} # Clear state
    analysis = link_service.analyze_relationships(note_id)

    # --- Assertions ---
    # Verify vector store calls
    mock_vector_store.get_note_content.assert_called_with(note_id) # Relative ID should be used here
    mock_vector_store.find_connected_notes.assert_called_with(note_id)
    mock_vector_store.find_backlinks.assert_called_with(note_id)
    assert mock_vector_store.find_similar.call_count == 2

    # Verify results structure (content already verified via mock return values)
    assert "direct_links" in analysis and len(analysis["direct_links"]) == 1
    assert "backlinks" in analysis and len(analysis["backlinks"]) == 1
    assert "semantic_links" in analysis and len(analysis["semantic_links"]) == 1
    assert "suggested_links" in analysis and len(analysis["suggested_links"]) == 1

def test_update_obsidian_links(link_service, tmp_path):
    """Test updating Obsidian-style wiki links in a note."""
    # Create a test note file
    note_path = tmp_path / "test-note.md"
    with open(note_path, "w") as f:
        f.write(SAMPLE_NOTE_CONTENT)

    # Define links to add
    links_to_add = [
        {
            "add_wiki_link": True,
            "target_id": "new-link",
            "alias": "New Link"
        }
    ]

    # Mock chunking_service to return proper chunks
    link_service.chunking_service.chunk_document.return_value = [
        {
            "content": SAMPLE_NOTE_CONTENT + "\n[[new-link|New Link]]",
            "metadata": {"type": "note"}
        }
    ]

    # Update links
    link_service.update_obsidian_links(str(note_path), links_to_add)

    # Verify the file was updated
    with open(note_path) as f:
        content = f.read()
        assert "[[new-link|New Link]]" in content

def test_generate_alias(link_service):
    """Test alias generation for wiki links."""
    test_cases = [
        ("test-note", "Test Note"),
        ("my_test_note", "My Test Note"),
        ("complex-test-note.md", "Complex Test Note"),
    ]

    for input_id, expected_alias in test_cases:
        alias = link_service._generate_alias(input_id)
        assert alias == expected_alias

def test_has_wiki_link(link_service):
    """Test detection of existing wiki links."""
    content = SAMPLE_NOTE_CONTENT

    # Test existing links
    assert link_service._has_wiki_link(content, "wiki-link")
    assert link_service._has_wiki_link(content, "another-link")
    assert link_service._has_wiki_link(content, "backlink-test")

    # Test non-existent links
    assert not link_service._has_wiki_link(content, "nonexistent-link")

def test_insert_wiki_link(link_service):
    """Test insertion of new wiki links."""
    # Test insertion with existing separator
    content = "Test content\n\n---\nExisting links"
    new_link = "[[new-link]]"
    result = link_service._insert_wiki_link(content, new_link)
    assert "---\nExisting links\n[[new-link]]" in result

    # Test insertion without existing separator
    content = "Test content"
    result = link_service._insert_wiki_link(content, new_link)
    assert "Test content\n\n---\n[[new-link]]" in result

def test_update_target_backlinks(link_service, tmp_path):
    """Test updating backlinks in target notes."""
    source_note_id = "source-note"
    target_note_id = "target-note"
    # Create file WITHOUT .md extension
    target_path = tmp_path / target_note_id
    expected_source_path = tmp_path / source_note_id # Path only, file doesn't need to exist
    initial_content = "# Target Note\\n\\nSome content" # NO references section initially

    # Create the target note file
    with open(target_path, "w") as f:
        f.write(initial_content)

    # Patch os.path.exists specifically for this test scope
    with patch('services.knowledge.link_service.os.path.exists', autospec=True) as mock_exists:
        # Ensure it returns True specifically for the target path the service checks
        expected_target_path_in_service = os.path.join(link_service.base_path, target_note_id)
        mock_exists.side_effect = lambda p: str(p) == str(expected_target_path_in_service)

        # Mock the vector store update method
        link_service.vector_store.update_note = MagicMock()
        # Mock dependencies needed if vector_store.update_note is called by the service
        link_service.chunking_service.chunk_document.return_value = [{'content': 'updated', 'metadata': {}}]
        link_service.embedding_service.embed_chunks.return_value = [[0.5] * 1536]

        # Call the method under test
        link_service._update_target_backlinks(
            target_note_id,
            source_note_id,
            str(expected_source_path)
        )

    # Verify the actual file content WAS modified correctly
    with open(target_path, 'r', encoding='utf-8') as f:
        updated_content = f.read()

    print(f"--- Updated Content in {target_path} ---") # Debug print
    print(updated_content)
    print("----------------------------------------")

    assert "# Target Note" in updated_content
    assert "Some content" in updated_content
    # Check specifically if the references section was added correctly
    assert "## Auto generated references" in updated_content
    assert f"[[{source_note_id}]]" in updated_content

    # Verify vector store update was called AFTER the file modification
    # _update_target_backlinks doesn't call update_note, so this should NOT be called here
    link_service.vector_store.update_note.assert_not_called()

def test_error_handling(link_service):
    """Test error handling in various scenarios."""
    # Mock os.path.exists to return False for the nonexistent file
    with patch('services.knowledge.link_service.os.path.exists', return_value=False):
        # Test with non-existent note by trying to update links
        with pytest.raises(FileNotFoundError):
            link_service.update_obsidian_links(
                "nonexistent.md",
                [{"target_id": "test", "add_wiki_link": True}]
            )

def test_content_hash_tracking(link_service, tmp_path):
    """Test that content hash tracking correctly identifies changes."""
    # Create a test note
    note_path = tmp_path / "test-note.md"
    initial_content = "# Test Note\n\nInitial content"
    with open(note_path, "w") as f:
        f.write(initial_content)

    # First analysis should indicate need for analysis
    assert link_service._needs_analysis("test-note", initial_content) == True

    # Update analysis state
    link_service._update_analysis_state("test-note", initial_content)

    # Same content should not need analysis
    assert link_service._needs_analysis("test-note", initial_content) == False

    # Modified content should need analysis
    modified_content = initial_content + "\nNew content"
    assert link_service._needs_analysis("test-note", modified_content) == True

    # Auto-generated references should not trigger analysis
    content_with_refs = initial_content + "\n\n---\n## Auto generated references\n[[link1]]\n[[link2]]"
    assert link_service._needs_analysis("test-note", content_with_refs) == False

def test_auto_generated_references_handling(link_service, tmp_path):
    """Test handling of auto-generated references section."""
    # Create a test note with existing content and references
    note_path = tmp_path / "test-note.md"
    initial_content = """# Test Note

Some content here.

---
## Manual References
[[manual-link]]

## Auto generated references
[[old-link|Old Link]]
"""
    with open(note_path, "w") as f:
        f.write(initial_content)

    # Add new links
    new_links = [
        {
            "add_wiki_link": True,
            "target_id": "new-link-1",
            "alias": "New Link 1"
        },
        {
            "add_wiki_link": True,
            "target_id": "new-link-2",
            "alias": "New Link 2"
        }
    ]

    # Update links
    link_service.update_obsidian_links(str(note_path), new_links, update_backlinks=False)

    # Read updated content
    with open(note_path) as f:
        updated_content = f.read()

    # Verify:
    # 1. Manual content is preserved
    assert "## Manual References" in updated_content
    assert "[[manual-link]]" in updated_content

    # 2. Old auto-generated links are removed
    assert "[[old-link|Old Link]]" not in updated_content

    # 3. New links are added
    assert "[[new-link-1|New Link 1]]" in updated_content
    assert "[[new-link-2|New Link 2]]" in updated_content

    # 4. Auto-generated section is properly formatted
    assert "## Auto generated references" in updated_content

def test_analysis_state_persistence(link_service, tmp_path):
    """Test that analysis state is properly saved and loaded."""
    # Create a test note
    note_path = tmp_path / "test-note.md"
    content = "# Test Note\n\nTest content"
    with open(note_path, "w") as f:
        f.write(content)

    # Update analysis state
    link_service._update_analysis_state("test-note", content)

    # Create new instance of LinkService to test persistence
    new_link_service = LinkService(link_service.vector_store)

    # Verify state was loaded
    assert "test-note" in new_link_service.analysis_state
    assert "content_hash" in new_link_service.analysis_state["test-note"]
    assert "last_analyzed" in new_link_service.analysis_state["test-note"]

    # Verify content hash matches
    assert new_link_service.analysis_state["test-note"]["content_hash"] == link_service._get_content_hash(content)

def test_skip_unchanged_notes(link_service, mock_vector_store):
    """Test that unchanged notes skip full analysis."""
    note_id = "test-note"
    content = "# Test Note\\n\\nTest content"
    # Create the dummy note file WITHOUT .md extension
    # Convert base_path string to Path object for joining
    note_path = pathlib.Path(link_service.base_path) / note_id
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Ensure analysis state is clean initially
    link_service.analysis_state = {}

    # --- Mock Setup First Run ---
    mock_vector_store.reset_mock() # Ensure clean state
    mock_vector_store.get_note_content.return_value = {"content": content, "embedding": [0.1] * 1536 }
    mock_vector_store.find_connected_notes.return_value = [{"target_id": "existing-link"}]
    mock_vector_store.find_backlinks.return_value = [{"source_id": "backlink"}]
    mock_vector_store.find_similar.side_effect = [
        [{"metadata": {"doc_id": "semantic"}, "content": "sem", "similarity": 0.8}],
        [{"metadata": {"doc_id": "suggested"}, "content": "sug", "similarity": 0.7}]
    ]

    # --- First analysis ---
    analysis1 = link_service.analyze_relationships(note_id)

    # Verify analysis ran fully the first time
    mock_vector_store.get_note_content.assert_called_with(note_id)
    mock_vector_store.find_connected_notes.assert_called_with(note_id)
    mock_vector_store.find_backlinks.assert_called_with(note_id)
    assert mock_vector_store.find_similar.call_count == 2
    assert len(analysis1.get("semantic_links", [])) == 1
    assert len(analysis1.get("suggested_links", [])) == 1

    # --- Mock Setup Second Run ---
    # Note content hash should now be in analysis_state
    mock_vector_store.reset_mock()
    # Re-apply mocks for methods expected to be called even when skipped
    mock_vector_store.find_connected_notes.return_value = [{"target_id": "existing-link"}]
    mock_vector_store.find_backlinks.return_value = [{"source_id": "backlink"}]
    # get_note_content IS called again for the hash check
    mock_vector_store.get_note_content.return_value = {"content": content, "embedding": [0.1] * 1536 }

    # --- Second analysis ---
    analysis2 = link_service.analyze_relationships(note_id)

    # Verify:
    # 1. Methods called
    mock_vector_store.get_note_content.assert_called_once_with(note_id) # Called for hash check
    mock_vector_store.find_connected_notes.assert_called_once_with(note_id)
    mock_vector_store.find_backlinks.assert_called_once_with(note_id)
    mock_vector_store.find_similar.assert_not_called() # Should be skipped

    # 2. Results structure (semantic/suggested should be empty)
    assert "direct_links" in analysis2 and len(analysis2["direct_links"]) == 1
    assert "backlinks" in analysis2 and len(analysis2["backlinks"]) == 1
    assert analysis2.get("semantic_links", []) == []
    assert analysis2.get("suggested_links", []) == []

if __name__ == "__main__":
    pytest.main([__file__])