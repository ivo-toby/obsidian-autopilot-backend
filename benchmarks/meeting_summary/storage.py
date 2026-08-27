"""Atomic, content-addressed storage for meeting benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(content: str) -> str:
    return sha256_bytes(content.encode("utf-8"))


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


class RunStore:
    """Store one benchmark run below an immutable run directory."""

    def __init__(
        self, run_dir: Path, manifest: Optional[Dict[str, object]] = None
    ):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest = manifest or {}

    @classmethod
    def create(
        cls,
        output_dir: Path,
        config_path: Path,
        scope: Optional[Dict[str, object]] = None,
        config_snapshot: Optional[Dict[str, object]] = None,
    ) -> "RunStore":
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        config_value = config_path
        if hasattr(config_value, "source"):
            config_value = getattr(config_value, "source")
        config = Path(config_value).expanduser().resolve()
        config_bytes = (
            config.read_bytes()
            if config.is_file()
            else str(config).encode("utf-8")
        )
        config_hash = sha256_bytes(config_bytes)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output / (run_id + "-" + config_hash[:8])
        suffix = 1
        while run_dir.exists():
            run_dir = output / (
                "%s-%s-%d"
                % (
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                    config_hash[:8],
                    suffix,
                )
            )
            suffix += 1
        run_dir.mkdir()
        store = cls(run_dir)
        store.manifest = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "config_sha256": config_hash,
            "config_path": str(config),
        }
        if scope is not None:
            store.manifest["scope"] = scope
        if config_snapshot is not None:
            store.manifest["config_snapshot"] = config_snapshot
        store.write_json(Path("manifest.json"), store.manifest)
        return store

    @classmethod
    def open(cls, run_dir: Path) -> "RunStore":
        directory = Path(run_dir).expanduser().resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Run manifest does not exist: %s" % manifest_path
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("Run manifest must be a JSON object")
        return cls(directory, manifest)

    def find_completed(self, operation: str, cache_key: str) -> Optional[Path]:
        for path in self.run_dir.rglob("*.json"):
            if path == self.manifest_path:
                continue
            try:
                payload = self.read_json(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if (
                payload.get("operation") == operation
                and payload.get("cache_key") == cache_key
                and payload.get("status") == "complete"
            ):
                return path
        return None

    def write_json(
        self, relative_path: Path, payload: Dict[str, object]
    ) -> Path:
        destination = self._destination(relative_path)
        content = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        return self._atomic_write(destination, content)

    def write_text(self, relative_path: Path, content: str) -> Path:
        return self._atomic_write(self._destination(relative_path), content)

    def read_json(self, path: Path) -> Dict[str, object]:
        target = Path(path)
        if not target.is_absolute():
            target = self.run_dir / target
        with target.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("Stored JSON must be an object: %s" % target)
        return value

    def store_input(
        self, category: str, identifier: str, content: str
    ) -> Path:
        digest = sha256_text(content)
        return self.write_text(
            Path("inputs") / category / (identifier + "-" + digest + ".md"),
            content,
        )

    def _destination(self, relative_path: Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "Storage paths must be relative to the run directory"
            )
        return self.run_dir / relative

    @staticmethod
    def _atomic_write(destination: Path, content: str) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            ".%s.tmp-%s" % (destination.name, os.getpid())
        )
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination
