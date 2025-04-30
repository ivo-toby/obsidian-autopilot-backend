"""
Tests for the VectorStoreService class.

This module contains comprehensive tests for the VectorStoreService functionality,
including document storage, retrieval, and relationship management.
"""

import json
import os
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
from chromadb.api.models.Collection import Collection
from services.vector_store.store_service import VectorStoreService
import chromadb
from chromadb.config import Settings
from sqlite3 import OperationalError

# Test data
SAMPLE_DOCUMENT = {
    "id": "test-note",
    "content": "This is a test note with [[wiki-link]] and [external link](https://example.com)",
    "type": "note",
    "modified_time": time.time(),
}

SAMPLE_CHUNKS = [
    "This is a test note with [[wiki-link]]",
    "and [external link](https://example.com)"
]

SAMPLE_EMBEDDINGS = [
    [0.1] * 1024,  # mxbai-embed-large dimension
    [0.2] * 1024
]

# Constants
DB_PATH = "/tmp/test_vector_store"

@pytest.fixture(autouse=True)
def cleanup_db():
    # Clean up before test if needed
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
    yield
    # Clean up after test
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)

@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client."""
    client = Mock()

    # Create mock collections
    notes_collection = Mock(spec=Collection)
    links_collection = Mock(spec=Collection)
    metadata_collection = Mock(spec=Collection)
    system_collection = Mock(spec=Collection)
    references_collection = Mock(spec=Collection)

    # Setup collection returns
    def get_or_create_collection(name, **kwargs):
        collections = {
            "notes": notes_collection,
            "links": links_collection,
            "metadata": metadata_collection,
            "system": system_collection,
            "references": references_collection
        }
        return collections[name]

    client.get_or_create_collection = Mock(side_effect=get_or_create_collection)

    return client

@pytest.fixture
def mock_chunking_service():
    """Create a mock chunking service."""
    mock = Mock()
    mock.chunk_document.return_value = SAMPLE_CHUNKS
    return mock

@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = Mock()
    # Mock embed_text to return a fixed-size embedding
    mock.embed_text.return_value = [0.1] * 1024 # Example dimension
    # Mock get_embedding_dimensions if called by init
    mock.get_embedding_dimensions.return_value = 1024
    return mock

@pytest.fixture
def store_config():
    """Create test configuration."""
    return {
        "vector_store": {
            "path": DB_PATH,
            "distance_func": "cosine",
            "hnsw_space": "cosine",
            "hnsw_config": {"m": 128, "ef_construction": 400, "ef_search": 200},
        }
    }

@pytest.fixture
def mock_settings():
    """Mock ChromaDB settings."""
    settings = Mock(spec=Settings)
    settings.persist_directory = "/tmp/test_vector_store"
    settings.allow_reset = True
    settings.anonymized_telemetry = False
    return settings

@pytest.fixture
def vector_store(store_config, mock_embedding_service):
    """Fixture to create a VectorStoreService instance with mocks."""
    # Patch the ChromaDB client initialization and collection creation
    with patch('chromadb.PersistentClient') as MockPersistentClient:
        mock_client_instance = MockPersistentClient.return_value

        # Mock the get_or_create_collection method
        mock_collections = {}
        def mock_get_or_create(name, metadata):
            if name not in mock_collections:
                mock_coll = MagicMock()
                mock_coll.name = name
                mock_coll.metadata = metadata
                # Add necessary methods used in tests if not already MagicMocked
                mock_coll.get = MagicMock(return_value={'ids': [], 'metadatas': [], 'documents': [], 'embeddings': []})
                mock_coll.query = MagicMock(return_value={'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'embeddings': [[]], 'distances': [[]]})
                mock_coll.upsert = MagicMock()
                mock_coll.delete = MagicMock()
                mock_collections[name] = mock_coll
            return mock_collections[name]

        mock_client_instance.get_or_create_collection.side_effect = mock_get_or_create

        # Mock the embedding service call within init if needed
        with patch.object(mock_embedding_service, 'embed_text', return_value=[0.1] * 1024):
            # Need to also patch _get_embedding_dimensions or ensure mock has it
            with patch('services.vector_store.store_service.VectorStoreService._get_embedding_dimensions', return_value=1024):
                service = VectorStoreService(store_config, embedding_service=mock_embedding_service)
                # Manually assign the mocked collections dict, as the real client is bypassed
                service.collections = mock_collections
                # Ensure the client mock is stored if service uses it directly later
                service.client = mock_client_instance
                return service

def test_initialization(vector_store, store_config):
    """Test initialization of collections and parameters."""
    # Verify collections were created in the dictionary
    assert "notes" in vector_store.collections
    assert "links" in vector_store.collections
    # Check the keys expected by the code
    assert "metadata" in vector_store.collections
    assert "system" in vector_store.collections
    assert "references" in vector_store.collections
    # Check one collection object exists
    assert vector_store.collections["notes"] is not None

def test_add_document(vector_store):
    """Test adding a new document."""
    # Patch the extract methods to handle content directly
    with patch.object(vector_store, '_extract_wiki_links') as mock_wiki_links, \
         patch.object(vector_store, '_extract_external_refs') as mock_external_refs, \
         patch.object(vector_store.embedding_service, '__class__') as mock_class:
        # Set up mock for Ollama detection
        mock_class.__name__ = 'OllamaEmbedding'
        mock_wiki_links.return_value = []
        mock_external_refs.return_value = []

        # Add document
        vector_store.add_document(
            doc_id="test-note",
            chunks=SAMPLE_CHUNKS,
            embeddings=SAMPLE_EMBEDDINGS
        )

        # Verify notes collection was updated
        notes_collection = vector_store.collections["notes"]
        notes_collection.upsert.assert_called_once_with(
            ids=["test-note_chunk_0", "test-note_chunk_1"],
            documents=SAMPLE_CHUNKS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=[{
                "doc_id": "test-note",
                "chunk_index": idx,
                "doc_type": "note",
                "source_path": "",
                "date": "",
                "filename": ""
            } for idx in range(len(SAMPLE_CHUNKS))]
        )

def test_find_connected_notes(vector_store):
    """Test finding connected notes."""
    # Mock links collection to return some connections
    links_collection = vector_store.collections["links"]
    links_collection.query.return_value = {
        "ids": ["connection1"],
        "metadatas": [[{  # Note the double list
            "target_id": "connected-note",
            "relationship": "references",
            "link_type": "wiki"
        }]]
    }

    results = vector_store.find_connected_notes("test-note")

    # Verify query was executed
    links_collection.query.assert_called_once()

    # Verify results
    assert len(results) == 1
    assert results[0]["target_id"] == "connected-note"
    assert results[0]["relationship"] == "references"
    assert results[0]["link_type"] == "wiki"

def test_find_backlinks(vector_store):
    """Test finding backlinks."""
    # Mock links collection to return some backlinks
    links_collection = vector_store.collections["links"]
    links_collection.query.return_value = {
        "ids": ["backlink1"],
        "metadatas": [[{  # Note the double list
            "source_id": "source-note",
            "relationship": "references",
            "link_type": "wiki"
        }]]
    }

    results = vector_store.find_backlinks("test-note")

    # Verify query was executed
    links_collection.query.assert_called_once()

    # Verify results
    assert len(results) == 1
    assert results[0]["source_id"] == "source-note"
    assert results[0]["relationship"] == "references"
    assert results[0]["link_type"] == "wiki"

def test_get_note_content(vector_store):
    """Test retrieving note content."""
    # Set up mock return value
    notes_collection = vector_store.collections["notes"]
    notes_collection.query.return_value = {
        "ids": [["test-note_chunk_0"]],
        "documents": [["Test content"]],
        "metadatas": [[{"doc_id": "test-note", "chunk_index": 0, "doc_type": "note"}]],
        "embeddings": [[[0.1] * 1024]]
    }

    result = vector_store.get_note_content("test-note")

    # Verify query was executed with correct parameters
    notes_collection.query.assert_called_once_with(
        query_embeddings=[[1.0] * 1024],
        where={"doc_id": "test-note"},
        include=["documents", "metadatas", "embeddings"]
    )

    # Verify result format
    assert result == {
        "content": "Test content",
        "metadata": {"doc_id": "test-note", "chunk_index": 0, "doc_type": "note"},
        "embedding": [0.1] * 1024
    }

def test_update_document(vector_store):
    """Test updating an existing document."""
    new_chunks = ["Updated content"]
    new_embeddings = [[0.3] * 1024]

    vector_store.update_document(
        doc_id="test-note",
        new_chunks=new_chunks,
        new_embeddings=new_embeddings
    )

    # Verify old content was deleted
    notes_collection = vector_store.collections["notes"]
    notes_collection.delete.assert_called_once_with(where={"doc_id": "test-note"})

    # Verify links were deleted
    links_collection = vector_store.collections["links"]
    links_collection.delete.assert_called_once_with(where={"source_id": "test-note"})

    # Verify new content was added
    notes_collection.upsert.assert_called_once_with(
        ids=["test-note_chunk_0"],
        documents=new_chunks,
        embeddings=new_embeddings,
        metadatas=[{
            "doc_id": "test-note",
            "chunk_index": 0,
            "doc_type": "note",
            "source_path": "",
            "date": "",
            "filename": ""
        }]
    )

def test_needs_update(vector_store):
    """Test checking if a document needs updating."""
    doc_id = "test-note"
    current_time = time.time()

    # Mock the response from the 'metadata' collection
    mock_meta_collection = vector_store.collections["metadata"]

    # Case 1: Document exists, needs update (file newer)
    mock_meta_collection.get.return_value = {
        "ids": [doc_id],
        "metadatas": [{"modified_time": current_time - 100}]
    }
    assert vector_store.needs_update(doc_id, current_time) is True
    mock_meta_collection.get.assert_called_once_with(ids=[doc_id], include=["metadatas"])

    # Case 2: Document exists, doesn't need update (file older or same)
    mock_meta_collection.reset_mock()
    mock_meta_collection.get.return_value = {
        "ids": [doc_id],
        "metadatas": [{"modified_time": current_time + 100}]
    }
    assert vector_store.needs_update(doc_id, current_time) is False
    mock_meta_collection.get.assert_called_once_with(ids=[doc_id], include=["metadatas"])

    # Case 3: Document doesn't exist in metadata
    mock_meta_collection.reset_mock()
    mock_meta_collection.get.return_value = {"ids": [], "metadatas": []}
    assert vector_store.needs_update(doc_id, current_time) is True
    mock_meta_collection.get.assert_called_once_with(ids=[doc_id], include=["metadatas"])

    # Case 4: Error during get - should default to True
    mock_meta_collection.reset_mock()
    mock_meta_collection.get.side_effect = Exception("DB error")
    assert vector_store.needs_update(doc_id, current_time) is True
    mock_meta_collection.get.assert_called_once_with(ids=[doc_id], include=["metadatas"])

def test_last_update_time(vector_store):
    """Test getting and setting last update time using metadata.json."""
    metadata_path = os.path.join(vector_store.db_path, "metadata.json")
    timestamp = time.time()

    # --- Test set_last_update_time ---
    # Case 1: File does not exist
    m_open = mock_open()
    with patch('builtins.open', m_open):
        with patch('os.path.exists', return_value=False):
            # Patch json.dump for this specific call scope
            with patch('json.dump') as mock_json_dump:
                vector_store.set_last_update_time(timestamp)

                # Verify open was called correctly for write
                m_open.assert_called_once_with(metadata_path, "w")
                # Get the mock file handle that open() returned
                mock_file_handle = m_open()
                # Verify json.dump was called with the correct data and handle
                expected_data = {"last_update": timestamp}
                mock_json_dump.assert_called_once_with(expected_data, mock_file_handle)

    # Case 2: File exists
    existing_data_dict = {"other_key": "value"}
    existing_content = json.dumps(existing_data_dict)
    m_open = mock_open(read_data=existing_content)
    with patch('builtins.open', m_open):
        with patch('os.path.exists', return_value=True):
            # Patch json.dump again for this scope
            with patch('json.dump') as mock_json_dump:
                new_timestamp = timestamp + 100
                vector_store.set_last_update_time(new_timestamp)

                # Verify open called for read then write
                m_open.assert_any_call(metadata_path, "r")
                m_open.assert_any_call(metadata_path, "w")

                # Get the mock file handle (mock_open resets, need the handle from the write call context)
                # We assume the handle passed to json.dump is the one from the 'w' call
                # Find the handle used in the json.dump call
                assert mock_json_dump.call_count == 1
                dump_args, _ = mock_json_dump.call_args
                written_dict = dump_args[0]
                # Verify the dictionary passed to json.dump is correct
                expected_data = {"other_key": "value", "last_update": new_timestamp}
                assert written_dict == expected_data

    # --- Test get_last_update_time ---
    # Simulate file existing
    existing_data = {"last_update": timestamp}
    m_open = mock_open(read_data=json.dumps(existing_data))
    with patch('builtins.open', m_open):
        with patch('os.path.exists', return_value=True):
            retrieved_time = vector_store.get_last_update_time()
            m_open.assert_called_once_with(metadata_path, "r")
            assert retrieved_time == timestamp

    # Simulate file not existing
    with patch('os.path.exists', return_value=False):
        with patch('builtins.open', mock_open()):
            retrieved_time = vector_store.get_last_update_time()
            assert retrieved_time == 0
            mock_open().assert_not_called()

    # Simulate error during read
    m_open = mock_open()
    # Configure the mock handle returned by open() to raise IOError on read
    mock_handle = m_open.return_value
    mock_handle.read.side_effect = IOError("Read error")
    with patch('builtins.open', m_open):
         with patch('os.path.exists', return_value=True):
             retrieved_time = vector_store.get_last_update_time()
             assert retrieved_time == 0 # Should default to 0 on error
             # Check that open was called for read (multiple times due to retry)
             assert m_open.call_count == vector_store.max_retries # Check retry count
             # Verify the arguments of any of the calls
             m_open.assert_any_call(metadata_path, "r")
             # Check that read was called (also multiple times)
             assert mock_handle.read.call_count == vector_store.max_retries

def test_retry_operation(vector_store):
    """Test retry mechanism for database operations."""
    operation = Mock(side_effect=[OperationalError("DB locked"), OperationalError("DB locked"), "success"])
    result = vector_store._retry_operation(operation)
    assert result == "success"
    assert operation.call_count == 3

def test_error_handling(vector_store):
    """Test error handling for various operations."""
    # Test error handling for get_note_content
    vector_store.collections["notes"].query.side_effect = Exception("Database error")
    result = vector_store.get_note_content("test-note")
    assert result is None

    # Test error handling for update_document
    vector_store.collections["notes"].delete.side_effect = Exception("Database error")
    with pytest.raises(Exception):
        vector_store.update_document(
            doc_id="test-note",
            new_chunks=["content"],
            new_embeddings=[[0.1] * 1024]
        )

if __name__ == "__main__":
    pytest.main([__file__])
