"""Build-time manifest and runtime verification for the baked Chroma index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict

import chromadb


DEFAULT_COLLECTION = "pkd_knowledge_base_openai"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "index_manifest.json"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[1] / "openai_chroma_data"


class BakedIndexError(RuntimeError):
    """Raised when the production image does not contain a usable index."""


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise BakedIndexError(f"Baked index directory is missing: {root}")
    digest = hashlib.sha256()
    # Chroma opens SQLite in writable mode even for read-only retrieval and may
    # update housekeeping pages. The logical count/metadata/probe below verify
    # SQLite; hash the immutable HNSW artifacts so ordinary startup cannot
    # invalidate its own manifest.
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("chroma.sqlite3")
        and not path.name.endswith(("-wal", "-shm"))
    )
    if not files:
        raise BakedIndexError(f"Baked index directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def inspect_index(
    index_path: Path = DEFAULT_INDEX_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=str(index_path))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as error:
        raise BakedIndexError(
            f"Baked collection {collection_name!r} is unavailable"
        ) from error

    count = collection.count()
    metadata = dict(collection.metadata or {})
    if count <= 0:
        raise BakedIndexError("Baked collection contains no vectors")
    if metadata.get("index_build_state", "ready") != "ready":
        raise BakedIndexError(
            f"Baked collection state is {metadata.get('index_build_state')!r}"
        )
    try:
        probe = collection.peek(limit=1)
    except Exception as error:
        raise BakedIndexError("Baked collection failed its local probe") from error
    if not probe.get("ids"):
        raise BakedIndexError("Baked collection probe returned no records")
    return {
        "client": client,
        "collection": collection,
        "vector_count": count,
        "collection_metadata": metadata,
    }


def build_manifest(
    index_path: Path = DEFAULT_INDEX_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, Any]:
    inspected = inspect_index(index_path, collection_name)
    metadata = inspected["collection_metadata"]
    return {
        "manifest_version": 1,
        "collection_name": collection_name,
        "index_schema_version": str(metadata.get("index_schema_version", "")),
        "index_build_state": metadata.get("index_build_state", "ready"),
        "vector_count": inspected["vector_count"],
        "embedding_model": embedding_model,
        "tree_sha256": _tree_sha256(index_path),
    }


def write_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, Any]:
    manifest = build_manifest(index_path, collection_name, embedding_model)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_index_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    *,
    verify_checksum: bool = True,
) -> Dict[str, Any]:
    """Validate the immutable image files without opening Chroma's SQLite DB."""
    if not manifest_path.is_file():
        raise BakedIndexError(f"Index manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BakedIndexError("Index manifest is invalid JSON") from error
    required = {
        "collection_name",
        "embedding_model",
        "index_schema_version",
        "index_build_state",
        "tree_sha256",
        "vector_count",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise BakedIndexError(f"Index manifest is missing: {', '.join(missing)}")
    if manifest["index_build_state"] != "ready":
        raise BakedIndexError("Baked index manifest is not marked ready")
    if int(manifest["vector_count"]) <= 0:
        raise BakedIndexError("Baked index manifest has no vectors")
    if verify_checksum and manifest["tree_sha256"] != _tree_sha256(index_path):
        raise BakedIndexError("Baked index checksum does not match its manifest")
    return manifest


def materialize_runtime_index(
    index_path: Path = DEFAULT_INDEX_PATH,
) -> Path:
    """Create a writable runtime view while preserving the baked image bytes.

    Chroma opens SQLite in writable mode and updates housekeeping pages even
    for retrieval-only use. Copying SQLite to ephemeral storage ensures startup
    and requests never mutate the database embedded in the container image.
    Large immutable HNSW directories are symlinked to avoid a second 1.1 GB
    copy; production marks those source artifacts read-only.
    """
    runtime_root = Path(tempfile.mkdtemp(prefix="cysticcare-chroma-"))
    runtime_index = runtime_root / "index"
    runtime_index.mkdir()
    for source in index_path.iterdir():
        target = runtime_index / source.name
        if source.is_dir():
            target.symlink_to(source.resolve(), target_is_directory=True)
        elif source.name.startswith("chroma.sqlite3"):
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())
    return runtime_index


def verify_baked_index(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    *,
    verify_checksum: bool = True,
) -> Dict[str, Any]:
    manifest = load_index_manifest(
        manifest_path,
        index_path,
        verify_checksum=verify_checksum,
    )
    collection_name = str(manifest.get("collection_name") or "")
    if not collection_name:
        raise BakedIndexError("Index manifest has no collection name")
    inspected = inspect_index(index_path, collection_name)
    metadata = inspected["collection_metadata"]
    comparisons = {
        "vector_count": inspected["vector_count"],
        "index_schema_version": str(metadata.get("index_schema_version", "")),
        "index_build_state": metadata.get("index_build_state", "ready"),
    }
    for key, actual in comparisons.items():
        if manifest.get(key) != actual:
            raise BakedIndexError(
                f"Index manifest mismatch for {key}: "
                f"expected {manifest.get(key)!r}, found {actual!r}"
            )
    return {**inspected, "manifest": manifest}


def runtime_checksum_enabled() -> bool:
    return os.getenv("VERIFY_INDEX_CHECKSUM", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
