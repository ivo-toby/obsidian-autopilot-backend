# Content Hash-Based Change Detection

## Problem Statement

The current knowledge base update mechanism relies on file modification time (`mtime`) to detect changes. This approach fails when:

1. **Cloud sync services** (Google Drive, iCloud, Dropbox) update `mtime` during sync operations without actual content changes
2. **File system operations** (backups, indexing, git operations) touch files and update timestamps
3. **Cross-platform sync** can result in timestamp drift between systems

**Current Impact:**
- Running `kb --update` after Google Drive sync triggers full re-indexing of 800+ notes
- Each note requires:
  - Content reading
  - Semantic chunking
  - Embedding generation (expensive LLM API calls)
  - Vector store update
- Result: 10-30 minutes of processing for changes that might only affect 1-2 files

## Solution Overview

Implement content-based change detection using cryptographic hashes (SHA-256):

1. Store content hash for each indexed document in vector store metadata
2. On update, compute hash of current content and compare to stored hash
3. Only re-index documents where content hash has changed
4. Ignore `mtime` changes that don't reflect actual content modifications

**Benefits:**
- Immune to cloud sync timestamp updates
- Only processes genuinely modified content
- Significantly faster incremental updates
- Maintains backward compatibility with existing vector store

## Implementation Plan

### Phase 1: Metadata Schema Extension

**File:** `services/vector_store/store_service.py`

**Changes:**

1. Extend document metadata structure:
```python
metadata = {
    "doc_id": str,           # Existing: relative path
    "type": str,             # Existing: note type
    "tags": List[str],       # Existing: extracted tags
    "dates": List[str],      # Existing: extracted dates
    "modified_time": float,  # Existing: file mtime
    "content_hash": str,     # NEW: SHA-256 of content
    "hash_algorithm": str,   # NEW: "sha256" (for future flexibility)
    "indexed_at": float,     # NEW: when this version was indexed
}
```

2. Add hash computation utility:
```python
def compute_content_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of content.

    Args:
        content: Document content to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex-encoded hash string
    """
    import hashlib
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()
```

3. Add content change detection method:
```python
def has_content_changed(self, doc_id: str, content: str) -> bool:
    """
    Check if document content has changed since last indexing.

    Args:
        doc_id: Document identifier
        content: Current document content

    Returns:
        True if content has changed or document not indexed
    """
    stored_metadata = self.get_document_metadata(doc_id)
    if not stored_metadata:
        return True  # Not indexed yet

    stored_hash = stored_metadata.get("content_hash")
    if not stored_hash:
        return True  # Legacy entry without hash

    current_hash = compute_content_hash(content)
    return current_hash != stored_hash
```

4. Add metadata retrieval method:
```python
def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve stored metadata for a document.

    Args:
        doc_id: Document identifier

    Returns:
        Metadata dict if found, None otherwise
    """
    # Query ChromaDB for document by ID
    # Return metadata if exists
```

### Phase 2: Update Logic Integration

**File:** `main.py`

**Modify:** `process_knowledge_base()` function, `--update` branch (lines 143-201)

**Current Logic:**
```python
notes = [
    note
    for note in summary_service.get_all_notes()
    if note.get("modified_time", 0) > last_update
]
```

**New Logic:**
```python
all_notes = summary_service.get_all_notes()
notes = []

for note in all_notes:
    # Check both mtime and content hash
    mtime = note.get("modified_time", 0)

    # Quick mtime check first (optimization)
    if mtime <= last_update:
        continue  # File hasn't been touched

    # File was touched - check if content actually changed
    if vector_store.has_content_changed(note["id"], note["content"]):
        notes.append(note)
    else:
        logger.debug(
            f"Skipping {note['id']} - touched but content unchanged "
            f"(mtime: {datetime.fromtimestamp(mtime)})"
        )

if not notes:
    logger.info("No notes with content changes need updating")
    return

logger.info(
    f"Found {len(notes)} notes with actual content changes "
    f"(filtered from {len([n for n in all_notes if n.get('modified_time', 0) > last_update])} touched files)"
)
```

**Enhancement:** Add `--force-hash-check` flag to bypass mtime optimization:
```python
kb_parser.add_argument(
    "--force-hash-check",
    action="store_true",
    help="Check content hash for all notes, ignoring mtime (useful after migration)"
)
```

### Phase 3: Hash Storage and Migration

**File:** `services/vector_store/store_service.py`

**Modify:** `add_document()` method to include hash in metadata

**Before:**
```python
def add_document(
    self, doc_id: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]
) -> None:
    # Store chunks with metadata
```

**After:**
```python
def add_document(
    self, doc_id: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]
) -> None:
    # Compute content hash from original content
    content = metadata.get("content", "")
    if content:
        metadata["content_hash"] = compute_content_hash(content)
        metadata["hash_algorithm"] = "sha256"
        metadata["indexed_at"] = time.time()

    # Store chunks with enhanced metadata
```

**Migration Strategy:**

1. **Backward Compatibility:** Existing entries without `content_hash` are treated as "needs update"
2. **Lazy Migration:** Hashes are added as documents are naturally updated
3. **Optional Full Migration:** Add `--rehash-all` command:
```python
kb_parser.add_argument(
    "--rehash-all",
    action="store_true",
    help="Recompute and store content hashes for all indexed documents without re-embedding"
)
```

This command:
- Reads all documents from vector store
- Computes hashes for entries missing `content_hash`
- Updates metadata without regenerating embeddings
- Fast operation (no LLM calls)

### Phase 4: Observability and Debugging

**Add logging and statistics:**

1. **Update summary statistics:**
```python
logger.info(f"Update completed:")
logger.info(f"  - Total notes scanned: {len(all_notes)}")
logger.info(f"  - Files touched since last update: {len(touched_files)}")
logger.info(f"  - Files with content changes: {len(notes)}")
logger.info(f"  - Files skipped (timestamp-only changes): {len(touched_files) - len(notes)}")
```

2. **Add `--explain-changes` flag:**
```python
kb_parser.add_argument(
    "--explain-changes",
    action="store_true",
    help="Show detailed explanation of which files changed and why"
)
```

Output:
```
File: daily/2025-10-22.md
  Status: CHANGED
  Reason: Content hash mismatch
  Old hash: a1b2c3d4...
  New hash: e5f6g7h8...
  Modified: 2025-10-22 15:31:23

File: meeting/standup-2025-10-21.md
  Status: SKIPPED
  Reason: Content unchanged (sync timestamp update)
  Hash: x9y8z7w6... (unchanged)
  Modified: 2025-10-22 15:31:23
```

3. **Add `--debug-hash` command:**
```python
kb_parser.add_argument(
    "--debug-hash",
    type=str,
    metavar="NOTE_PATH",
    help="Compare stored hash vs current hash for specific note"
)
```

### Phase 5: Configuration Options

**File:** `config.yaml`

Add configuration section:
```yaml
vector_store:
  path: "~/.vector_store"

  # Change detection strategy
  change_detection:
    method: "content_hash"  # Options: "mtime", "content_hash", "both"
    hash_algorithm: "sha256"  # Future: could support other algorithms

    # Performance tuning
    skip_mtime_check: false  # If true, always compute hashes (slower but most accurate)
    cache_hashes: true       # Cache computed hashes in memory during single run
```

**Behavior by method:**
- `"mtime"`: Current behavior (backward compatible)
- `"content_hash"`: New behavior (recommended for cloud-synced vaults)
- `"both"`: Paranoid mode - file must pass both mtime AND hash check to skip

## Testing Strategy

### Unit Tests

**File:** `tests/services/vector_store/test_content_hash.py`

```python
def test_compute_content_hash():
    """Test hash computation is deterministic."""
    content = "# Test Note\n\nSome content"
    hash1 = compute_content_hash(content)
    hash2 = compute_content_hash(content)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex length

def test_content_hash_sensitivity():
    """Test hash changes with content."""
    content1 = "# Test Note"
    content2 = "# Test Note\n"  # Added newline
    assert compute_content_hash(content1) != compute_content_hash(content2)

def test_has_content_changed_new_document():
    """Test new documents always marked as changed."""
    vector_store = setup_vector_store()
    assert vector_store.has_content_changed("new-note.md", "content")

def test_has_content_changed_unchanged():
    """Test unchanged content detected correctly."""
    # Index document
    # Check same content returns False

def test_has_content_changed_modified():
    """Test modified content detected correctly."""
    # Index document
    # Check different content returns True

def test_legacy_entries_without_hash():
    """Test backward compatibility with entries lacking content_hash."""
    # Simulate legacy entry
    # Verify it's treated as "needs update"
```

### Integration Tests

**File:** `tests/integration/test_kb_update_with_hash.py`

```python
def test_update_skips_unchanged_content(temp_vault):
    """Test that touched files with unchanged content are skipped."""
    # 1. Create note and index it
    # 2. Update timestamp but not content
    # 3. Run kb --update
    # 4. Verify note was not re-indexed

def test_update_processes_changed_content(temp_vault):
    """Test that content changes trigger re-indexing."""
    # 1. Create note and index it
    # 2. Modify content
    # 3. Run kb --update
    # 4. Verify note was re-indexed with new hash

def test_google_drive_sync_scenario(temp_vault):
    """Simulate Google Drive updating all mtimes."""
    # 1. Index 100 notes
    # 2. Update all mtimes to same timestamp (simulating sync)
    # 3. Run kb --update
    # 4. Verify 0 notes re-indexed
```

### Manual Testing Checklist

- [ ] Full reindex (`kb --reindex`) adds hashes to all new documents
- [ ] Update with no changes (`kb --update`) after Google Drive sync processes 0 notes
- [ ] Update with 1 changed file (`kb --update`) processes exactly 1 note
- [ ] Legacy migration (`kb --rehash-all`) adds hashes without re-embedding
- [ ] Performance: Update after sync completes in <10 seconds (vs 10-30 minutes before)
- [ ] `kb --explain-changes` shows accurate change detection
- [ ] `kb --debug-hash <note>` displays hash comparison correctly

## Performance Considerations

### Hash Computation Cost

SHA-256 hashing is very fast:
- ~500 MB/s on modern CPUs
- Average note size: 5-10 KB
- Hash computation: <1ms per note
- 1000 notes: ~1 second total

**Optimization:** Can compute hashes in parallel if needed:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    hashes = list(executor.map(compute_content_hash, note_contents))
```

### Memory Usage

For hash caching (if enabled):
- SHA-256 hash: 32 bytes (64 hex chars)
- 1000 notes: 64 KB memory
- Negligible impact

### Storage Overhead

Per document in ChromaDB:
- Hash string: 64 bytes
- Metadata fields: ~100 bytes total
- For 1000 documents: ~160 KB
- Negligible vs embedding storage (768-dim vectors)

## Migration Path

### For Existing Users

**Step 1: Update codebase**
```bash
git pull
# No config changes required - uses content_hash by default
```

**Step 2: Next update auto-migrates**
```bash
python main.py kb --update
# Adds hashes to documents as they're naturally updated
```

**Step 3 (Optional): Force migration**
```bash
python main.py kb --rehash-all
# Immediately adds hashes to all existing documents
```

### Rollback Strategy

If issues arise, disable via config:
```yaml
vector_store:
  change_detection:
    method: "mtime"  # Fall back to old behavior
```

No data loss - hashes stored as extra metadata, ignored when using `mtime` method.

## Future Enhancements

### 1. Content-Addressable Storage
Store embeddings by content hash (like Git):
- Identical content across different notes shares embeddings
- Saves storage and computation for duplicated content

### 2. Incremental Chunk Detection
Instead of re-embedding entire document on any change:
- Hash individual chunks
- Only re-embed chunks that changed
- Useful for large documents with small edits

### 3. Change History
Track hash history for documents:
```python
{
    "doc_id": "daily/2025-10-22.md",
    "hash_history": [
        {"hash": "abc123...", "indexed_at": 1700000000},
        {"hash": "def456...", "indexed_at": 1700010000},
    ]
}
```
Enables:
- Rollback to previous versions
- Change frequency analysis
- Document evolution tracking

### 4. Semantic Change Detection
Combine with semantic similarity:
- Hash detects ANY change
- Semantic similarity detects MEANINGFUL changes
- Skip re-indexing for trivial edits (typo fixes, formatting)

```python
if has_content_changed(doc_id, content):
    if is_semantically_similar(old_embedding, new_embedding, threshold=0.98):
        logger.info(f"Skipping {doc_id} - trivial change")
        return
```

## References

- **ChromaDB Documentation:** https://docs.trychroma.com/
- **Python hashlib:** https://docs.python.org/3/library/hashlib.html
- **Content-Addressable Storage (Git model):** https://git-scm.com/book/en/v2/Git-Internals-Git-Objects

## Implementation Checklist

- [ ] Phase 1: Metadata schema extension
  - [ ] Add `compute_content_hash()` utility
  - [ ] Add `has_content_changed()` method
  - [ ] Add `get_document_metadata()` method
  - [ ] Write unit tests for hash functions
- [ ] Phase 2: Update logic integration
  - [ ] Modify `kb --update` to use hash check
  - [ ] Add `--force-hash-check` flag
  - [ ] Add detailed logging
  - [ ] Write integration tests
- [ ] Phase 3: Hash storage and migration
  - [ ] Modify `add_document()` to store hashes
  - [ ] Implement `--rehash-all` command
  - [ ] Test backward compatibility
- [ ] Phase 4: Observability
  - [ ] Add update statistics logging
  - [ ] Implement `--explain-changes` flag
  - [ ] Implement `--debug-hash` command
- [ ] Phase 5: Configuration
  - [ ] Add config section
  - [ ] Implement strategy selection
  - [ ] Document configuration options
- [ ] Documentation
  - [ ] Update CLAUDE.md with new behavior
  - [ ] Add usage examples to README
  - [ ] Document migration steps
- [ ] Testing
  - [ ] Run full test suite
  - [ ] Manual testing with Google Drive sync
  - [ ] Performance benchmarking
