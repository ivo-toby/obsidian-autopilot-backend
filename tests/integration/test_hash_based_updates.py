"""
Integration tests for hash-based change detection in kb updates.

These tests verify that the content hash-based change detection works
correctly end-to-end, preventing unnecessary re-indexing when only
timestamps change (e.g., from cloud sync).
"""

import os
import tempfile
import time
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest
from services.vector_store.store_service import VectorStoreService


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()

    # Mock embedding service
    mock_embedding = Mock()
    mock_embedding.embed_text.return_value = [0.1] * 1024

    config = {
        "vector_store": {
            "path": temp_dir,
            "hnsw_config": {"m": 128, "ef_construction": 400, "ef_search": 200}
        }
    }

    # Create the vector store with mocked embedding dimensions
    with patch.object(VectorStoreService, '_get_embedding_dimensions', return_value=1024):
        store = VectorStoreService(config, embedding_service=mock_embedding)

    yield store

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_hash_stored_when_adding_document(temp_vector_store):
    """Test that content hash is stored when adding a document."""
    doc_id = "test-note.md"
    content = "# Test Note\n\nThis is test content"
    chunks = ["Test chunk 1", "Test chunk 2"]
    embeddings = [[0.1] * 1024, [0.2] * 1024]

    metadata = {
        "content": content,
        "modified_time": time.time(),
        "type": "note"
    }

    # Add document
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Verify hash was stored
    stored_metadata = temp_vector_store.get_document_metadata(doc_id)
    assert stored_metadata is not None
    assert "content_hash" in stored_metadata
    assert "hash_algorithm" in stored_metadata
    assert stored_metadata["hash_algorithm"] == "sha256"
    assert "indexed_at" in stored_metadata

    # Verify hash is correct
    expected_hash = VectorStoreService.compute_content_hash(content)
    assert stored_metadata["content_hash"] == expected_hash


def test_unchanged_content_detected_correctly(temp_vector_store):
    """Test that unchanged content is detected even with updated timestamp."""
    doc_id = "test-note.md"
    content = "# Test Note\n\nThis is test content"
    chunks = ["Test chunk"]
    embeddings = [[0.1] * 1024]

    # Add document with initial timestamp
    metadata = {
        "content": content,
        "modified_time": time.time(),
        "type": "note"
    }
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Simulate timestamp update (e.g., from cloud sync) but same content
    time.sleep(0.1)  # Ensure different timestamp

    # Check if content changed (should return False - content is same)
    has_changed = temp_vector_store.has_content_changed(doc_id, content)
    assert has_changed is False, "Unchanged content should not be detected as changed"


def test_changed_content_detected_correctly(temp_vector_store):
    """Test that actual content changes are detected."""
    doc_id = "test-note.md"
    original_content = "# Test Note\n\nOriginal content"
    modified_content = "# Test Note\n\nModified content"
    chunks = ["Test chunk"]
    embeddings = [[0.1] * 1024]

    # Add document with original content
    metadata = {
        "content": original_content,
        "modified_time": time.time(),
        "type": "note"
    }
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Check with modified content
    has_changed = temp_vector_store.has_content_changed(doc_id, modified_content)
    assert has_changed is True, "Changed content should be detected"


def test_new_document_detected_as_changed(temp_vector_store):
    """Test that new documents (not yet indexed) are always marked as changed."""
    doc_id = "new-note.md"
    content = "# New Note\n\nNew content"

    has_changed = temp_vector_store.has_content_changed(doc_id, content)
    assert has_changed is True, "New documents should always be marked as changed"


def test_legacy_document_without_hash_detected_as_changed(temp_vector_store):
    """Test that legacy documents without hash are marked as changed."""
    doc_id = "legacy-note.md"
    content = "# Legacy Note\n\nLegacy content"
    chunks = ["Legacy chunk"]
    embeddings = [[0.1] * 1024]

    # Add document without content in metadata (simulating legacy entry)
    metadata = {
        "modified_time": time.time(),
        "type": "note"
        # No "content" field - hash won't be computed
    }
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Verify no hash was stored
    stored_metadata = temp_vector_store.get_document_metadata(doc_id)
    assert "content_hash" not in stored_metadata or stored_metadata.get("content_hash") is None

    # Check if content changed (should return True for legacy entries)
    has_changed = temp_vector_store.has_content_changed(doc_id, content)
    assert has_changed is True, "Legacy documents without hash should be marked as changed"


def test_rehash_document_adds_hash_to_legacy_entry(temp_vector_store):
    """Test that rehash_document can add hash to legacy entries."""
    doc_id = "legacy-note.md"
    content = "# Legacy Note\n\nLegacy content"
    chunks = ["Legacy chunk"]
    embeddings = [[0.1] * 1024]

    # Add document without hash (legacy entry)
    metadata = {
        "modified_time": time.time(),
        "type": "note"
    }
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Rehash the document
    result = temp_vector_store.rehash_document(doc_id, content)
    assert result is True, "Rehashing should succeed for legacy entry"

    # Verify hash was added
    stored_metadata = temp_vector_store.get_document_metadata(doc_id)
    assert "content_hash" in stored_metadata
    assert stored_metadata["content_hash"] == VectorStoreService.compute_content_hash(content)


def test_rehash_document_skips_entries_with_hash(temp_vector_store):
    """Test that rehash_document skips documents that already have hash."""
    doc_id = "test-note.md"
    content = "# Test Note\n\nTest content"
    chunks = ["Test chunk"]
    embeddings = [[0.1] * 1024]

    # Add document with hash
    metadata = {
        "content": content,
        "modified_time": time.time(),
        "type": "note"
    }
    temp_vector_store.add_document(doc_id, chunks, embeddings, metadata)

    # Try to rehash (should skip)
    result = temp_vector_store.rehash_document(doc_id, content)
    assert result is False, "Rehashing should skip documents that already have hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
