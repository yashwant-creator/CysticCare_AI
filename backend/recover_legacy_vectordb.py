"""Stage and safely promote a recovered Chroma knowledge base.

This command recovers the pre-rebuild Chroma snapshot without pretending that
the currently available PDF directory is the full historical corpus.  It reads
the legacy snapshot locally, re-embeds every recovered passage with the current
OpenAI embedding model, and writes the result to a *separate* Chroma directory.
The running index is never deleted by this command.

Run from ``backend/``::

    # Validate the backup, recovered inventory, current API credentials, and
    # the optional current-PDF additions.  No stage directory is created.
    python recover_legacy_vectordb.py --dry-run

    # Create and validate a stage.  Inspect it before switching production.
    python recover_legacy_vectordb.py --stage-dir app/openai_chroma_data.recovery_stage

    # Promote an already-ready stage using a reversible rename.  Or include
    # --promote-stage on the initial command to build and promote in one run.
    python recover_legacy_vectordb.py \
        --stage-dir app/openai_chroma_data.recovery_stage --promote-stage

The default stage path deliberately begins with ``openai_chroma_data.recovery``
so it remains excluded from source control and deployment contexts.  Promotion
renames the active target to a timestamped sibling first, then renames the
validated stage into place.  If the second rename fails, the original target is
restored.

Current-PDF-only additions are text chunks made with the existing
``process_pdf_file_with_metadata`` structure-aware chunker.  Figure descriptions
are intentionally not regenerated here: doing so would require additional
vision-model calls and would make a recovery run no longer a deterministic
re-embedding of the source corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import chromadb
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR / "app"
DEFAULT_BACKUP_DIR = APP_DIR / "openai_chroma_data.prebuild_backup.20260629_205005"
DEFAULT_STAGE_DIR = APP_DIR / "openai_chroma_data.recovery_stage"
DEFAULT_TARGET_DIR = APP_DIR / "openai_chroma_data"
DEFAULT_COLLECTION_NAME = "pkd_knowledge_base_openai"
INDEX_SCHEMA_VERSION = "2"
RECOVERY_ARTIFACT = "legacy_vector_recovery"

# Load the same backend-local environment file used by the service before any
# OpenAI client is created.  It is intentionally harmless when the file does
# not exist, so a deployment can provide credentials through its environment.
load_dotenv(BACKEND_DIR / ".env")
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RecoveryError(RuntimeError):
    """A recovery input, staging, or promotion safety contract failed."""


@dataclass(frozen=True)
class ChromaRecord:
    """One normalized record ready for embedding and insertion into Chroma."""

    id: str
    document: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RecoveryPayload:
    """Preflight output, kept independent of Chroma/OpenAI client lifetimes."""

    records: list[ChromaRecord]
    backup_dir: Path
    source_collection: str
    legacy_vector_count: int
    legacy_logical_document_count: int
    current_pdf_only_document_count: int
    current_pdf_only_chunk_count: int
    current_pdf_only_files: list[str]
    current_pdf_reconciliation: dict[str, Any]
    inventory: dict[str, Any]

    @property
    def expected_vector_count(self) -> int:
        return len(self.records)

    @property
    def expected_logical_document_count(self) -> int:
        return self.legacy_logical_document_count + self.current_pdf_only_document_count


def _resolve_path(value: str | Path, *, base: Path = BACKEND_DIR) -> Path:
    """Resolve a user path deterministically without creating it."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _mapping_value(value: Any) -> dict[str, Any]:
    """Turn a helper inventory/dataclass into JSON-safe diagnostic metadata."""
    if value is None:
        return {}
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return {"value": _json_safe(value)}


def _json_safe(value: Any) -> Any:
    """Avoid logging a non-serializable third-party metadata object."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Normalize recovered metadata to Chroma's scalar-only metadata contract."""
    normalized: dict[str, str | int | float | bool] = {}
    for raw_key, value in metadata.items():
        if value is None:
            continue
        key = str(raw_key)
        if not key:
            continue
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int):
            normalized[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                normalized[key] = value
        elif isinstance(value, str):
            normalized[key] = value
        else:
            # Legacy Chroma metadata is scalar in practice, but make recovery
            # robust to a future nested provenance field without losing it.
            normalized[key] = json.dumps(_json_safe(value), sort_keys=True)
    return normalized


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _inventory_count(inventory: Any, *names: str) -> Optional[int]:
    """Read a count from a recovery helper without coupling to its dataclass."""
    for name in names:
        value = _record_value(inventory, name)
        if isinstance(value, bool):
            continue
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _legacy_record_to_chroma(record: Any) -> ChromaRecord:
    """Add recovery provenance while retaining the helper's canonical IDs/text."""
    record_id = str(_record_value(record, "id", "")).strip()
    document = _record_value(record, "document", "")
    metadata = _record_value(record, "metadata", {})
    if not record_id:
        raise RecoveryError("The legacy recovery helper returned a record without an ID")
    if not isinstance(document, str) or not document.strip():
        raise RecoveryError(f"The legacy recovery helper returned an empty document for {record_id}")
    if not isinstance(metadata, Mapping):
        raise RecoveryError(f"The legacy recovery helper returned non-mapping metadata for {record_id}")

    result = _chroma_metadata(metadata)
    result.setdefault("chunk_id", record_id)
    result.setdefault("content_type", "text")
    result["recovery_source"] = "legacy_backup"
    return ChromaRecord(id=record_id, document=document, metadata=result)


def _load_metadata_cache() -> dict[str, Any]:
    """Read cache data without calling MetadataManager (which writes it back)."""
    cache_path = APP_DIR / "metadata_cache.json"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read metadata cache %s: %s", cache_path, error)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _is_known_legacy_pdf(
    pdf_path: Path,
    metadata: Mapping[str, Any],
    *,
    legacy_paper_ids: set[str],
    legacy_titles: set[str],
    legacy_file_names: set[str],
    legacy_match_keys: set[str],
) -> bool:
    """Compare current PDFs to recovered papers using stable and legacy keys."""
    from app.utils.paper_identity import normalize_paper_identity, stable_paper_id

    title = metadata.get("title") or ""
    title_identity = normalize_paper_identity(title)
    paper_id = stable_paper_id(title, pdf_path.stem)
    file_stem_identity = normalize_paper_identity(pdf_path.stem)
    return (
        paper_id in legacy_paper_ids
        or bool(title_identity and title_identity in legacy_titles)
        or pdf_path.name.casefold() in legacy_file_names
        or bool(title_identity and title_identity in legacy_match_keys)
        or bool(file_stem_identity and file_stem_identity in legacy_match_keys)
    )


def _current_pdf_metadata(pdf_path: Path, cache: Mapping[str, Any]) -> dict[str, Any]:
    """Return cache metadata or inexpensive PDF metadata for duplicate matching."""
    cached = cache.get(pdf_path.name)
    if isinstance(cached, Mapping):
        metadata = dict(cached)
    else:
        from app.utils.openai_utils import extract_metadata

        metadata = extract_metadata(str(pdf_path))
    metadata["file_name"] = pdf_path.name
    metadata["file_path"] = str(pdf_path)
    return metadata


def _display_metadata(pdf_path: Path, basic_metadata: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    """Build scalar source metadata without mutating the repository cache."""
    cached = cache.get(pdf_path.name)
    merged: dict[str, Any] = {}
    if isinstance(basic_metadata, Mapping):
        merged.update(basic_metadata)
    if isinstance(cached, Mapping):
        merged.update(cached)

    title = str(merged.get("title") or pdf_path.stem).strip() or pdf_path.stem
    author = str(merged.get("author") or "Unknown Author").strip() or "Unknown Author"
    year = str(merged.get("year") or "Unknown").strip() or "Unknown"
    merged.update(
        {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "title": title,
            "author": author,
            "year": year,
            "source_type": str(merged.get("source_type") or "scientific_paper"),
            "citation": str(merged.get("citation") or f"{author} ({year}). {title}"),
            "display_name": str(merged.get("display_name") or f"{author} {year}"),
        }
    )
    return _chroma_metadata(merged)


def _format_current_chunk_for_embedding(
    chunk: str,
    metadata: Mapping[str, Any],
) -> str:
    """Match the current indexer's title/section embedding context convention."""
    section_heading = str(metadata.get("section_heading") or "").strip()
    page_number = metadata.get("page_number") or metadata.get("page_start")
    title = str(metadata.get("title") or "Unknown").strip() or "Unknown"
    prefix = f"[Paper: {title}]"
    if section_heading:
        prefix += f" [Section: {section_heading}]"
    if page_number:
        prefix += f" [Page: {page_number}]"

    # ``process_pdf_file_with_metadata`` already adds a preliminary context
    # prefix.  Replace it with the cache-enriched paper title instead of
    # embedding two competing titles.
    first_line, separator, remainder = chunk.partition("\n")
    body = remainder if separator and first_line.startswith("[") and first_line.endswith("]") else chunk
    return f"{prefix}\n{body}".strip()


def _build_current_pdf_only_records(
    legacy_records: Sequence[ChromaRecord],
    *,
    pdf_directory: Path,
    limit: int,
    legacy_match_keys: Optional[Iterable[Any]] = None,
) -> tuple[list[ChromaRecord], list[str], dict[str, Any]]:
    """Add up to ``limit`` physical PDFs absent from the recovered corpus.

    The historical corpus is authoritative for the 355 logical papers.  This
    optional append only handles PDFs that really are absent from it; it never
    re-ingests the 81-PDF local subset wholesale.
    """
    if limit <= 0:
        return [], [], {
            "status": "disabled",
            "filename_nonoverlap_files": [],
            "canonical_aliases": {},
            "current_only_candidates": [],
            "included_current_only_files": [],
        }
    if not pdf_directory.is_dir():
        logger.warning("Current PDF directory is unavailable; no PDF-only records added: %s", pdf_directory)
        return [], [], {
            "status": "pdf_directory_unavailable",
            "pdf_directory": str(pdf_directory),
            "filename_nonoverlap_files": [],
            "canonical_aliases": {},
            "current_only_candidates": [],
            "included_current_only_files": [],
        }

    from app.utils.openai_utils import get_pdf_files, process_pdf_file_with_metadata
    from app.utils.paper_identity import normalize_paper_identity, stable_paper_id

    cache = _load_metadata_cache()
    legacy_paper_ids = {str(record.metadata.get("paper_id") or "") for record in legacy_records}
    legacy_paper_ids.discard("")
    legacy_titles = {
        normalize_paper_identity(record.metadata.get("title"))
        for record in legacy_records
        if normalize_paper_identity(record.metadata.get("title"))
    }
    legacy_file_names = {
        str(record.metadata.get("file_name") or "").casefold()
        for record in legacy_records
        if record.metadata.get("file_name")
    }
    helper_match_keys = {str(value) for value in (legacy_match_keys or []) if value}

    candidates: list[tuple[Path, dict[str, Any]]] = []
    filename_nonoverlap_files: list[str] = []
    canonical_aliases: dict[str, list[str]] = {}
    for raw_path in get_pdf_files(str(pdf_directory)):
        pdf_path = Path(raw_path)
        candidate_metadata = _current_pdf_metadata(pdf_path, cache)
        title_identity = normalize_paper_identity(candidate_metadata.get("title"))
        if pdf_path.name.casefold() not in legacy_file_names:
            filename_nonoverlap_files.append(pdf_path.name)
            aliases = sorted(
                {
                    str(record.metadata.get("file_name"))
                    for record in legacy_records
                    if title_identity
                    and normalize_paper_identity(record.metadata.get("title")) == title_identity
                    and record.metadata.get("file_name")
                }
            )
            if aliases:
                canonical_aliases[pdf_path.name] = aliases
        if not _is_known_legacy_pdf(
            pdf_path,
            candidate_metadata,
            legacy_paper_ids=legacy_paper_ids,
            legacy_titles=legacy_titles,
            legacy_file_names=legacy_file_names,
            legacy_match_keys=helper_match_keys,
        ):
            candidates.append((pdf_path, candidate_metadata))

    candidate_names = [pdf_path.name for pdf_path, _metadata in candidates]
    selected_candidates = candidates[:limit]
    deferred_candidates = candidate_names[limit:]
    reconciliation: dict[str, Any] = {
        "status": "reconciled",
        "pdf_directory": str(pdf_directory),
        "filename_nonoverlap_files": filename_nonoverlap_files,
        "canonical_aliases": canonical_aliases,
        "current_only_candidates": candidate_names,
        "included_current_only_files": [],
        "deferred_current_only_files": deferred_candidates,
    }
    candidates = selected_candidates
    if not candidates:
        logger.info(
            "No true current-PDF-only papers were found; %s filename non-overlaps resolve to legacy aliases",
            len(canonical_aliases),
        )
        return [], [], reconciliation

    existing_ids = {record.id for record in legacy_records}
    current_records: list[ChromaRecord] = []
    added_files: list[str] = []
    for pdf_path, _candidate_metadata in candidates:
        chunks, basic_metadata, chunk_metadatas = process_pdf_file_with_metadata(str(pdf_path))
        enhanced_metadata = _display_metadata(pdf_path, basic_metadata, cache)
        paper_id = stable_paper_id(enhanced_metadata.get("title"), pdf_path.stem)
        chunks_added = 0
        for chunk_index, chunk in enumerate(chunks):
            raw_chunk_metadata = (
                chunk_metadatas[chunk_index]
                if chunk_index < len(chunk_metadatas)
                else {"chunk_index": chunk_index, "content_type": "text"}
            )
            chunk_metadata = dict(raw_chunk_metadata)
            previous_index = chunk_metadata.get("previous_chunk_index", -1)
            next_index = chunk_metadata.get("next_chunk_index", -1)
            record_id = f"{paper_id}_current_{chunk_index}"
            if record_id in existing_ids:
                raise RecoveryError(
                    f"Current-PDF-only chunk ID collision for {pdf_path.name}: {record_id}"
                )
            existing_ids.add(record_id)
            chunk_metadata.update(
                {
                    "paper_id": paper_id,
                    "chunk_id": record_id,
                    "content_type": chunk_metadata.get("content_type", "text"),
                    "previous_chunk_id": (
                        f"{paper_id}_current_{previous_index}"
                        if isinstance(previous_index, int) and previous_index >= 0
                        else ""
                    ),
                    "next_chunk_id": (
                        f"{paper_id}_current_{next_index}"
                        if isinstance(next_index, int) and next_index >= 0
                        else ""
                    ),
                    "recovery_source": "current_pdf_only",
                    "recovery_source_pdf": pdf_path.name,
                }
            )
            metadata = _chroma_metadata({**enhanced_metadata, **chunk_metadata})
            current_records.append(
                ChromaRecord(
                    id=record_id,
                    document=_format_current_chunk_for_embedding(chunk, metadata),
                    metadata=metadata,
                )
            )
            chunks_added += 1
        if chunks_added:
            added_files.append(pdf_path.name)
            logger.info("Added %s current-PDF-only chunks from %s", chunks_added, pdf_path.name)
        else:
            logger.warning("Current PDF-only candidate had no extractable text and was skipped: %s", pdf_path)

    reconciliation["included_current_only_files"] = added_files
    return current_records, added_files, reconciliation


def _load_legacy_recovery(backup_dir: Path, source_collection: str) -> Any:
    """Use the dedicated local-only recovery helper, with a clear setup error."""
    try:
        from app.services.legacy_corpus_recovery import recover_legacy_backup
    except ImportError as error:
        raise RecoveryError(
            "Legacy corpus recovery helpers are unavailable. Ensure "
            "app.services.legacy_corpus_recovery is deployed with this command."
        ) from error
    try:
        return recover_legacy_backup(
            str(backup_dir),
            collection_name=source_collection,
            page_size=500,
        )
    except Exception as error:
        raise RecoveryError(
            f"Could not recover legacy collection {source_collection!r} from {backup_dir}: {error}"
        ) from error


def prepare_recovery_payload(
    *,
    backup_dir: Path,
    source_collection: str,
    include_current_pdf_only: bool,
    current_pdf_directory: Path,
    current_pdf_only_limit: int,
) -> RecoveryPayload:
    """Read/normalize the full local recovery corpus before any stage exists."""
    if not backup_dir.is_dir():
        raise RecoveryError(f"Legacy backup directory does not exist: {backup_dir}")
    if not (backup_dir / "chroma.sqlite3").is_file():
        raise RecoveryError(f"Legacy backup has no chroma.sqlite3 database: {backup_dir}")
    if current_pdf_only_limit < 0:
        raise RecoveryError("--current-pdf-only-limit must be zero or greater")

    recovered = _load_legacy_recovery(backup_dir, source_collection)
    helper_records = list(getattr(recovered, "records", []) or [])
    if not helper_records:
        raise RecoveryError("Legacy recovery helper returned no records")
    legacy_records = [_legacy_record_to_chroma(record) for record in helper_records]
    legacy_ids = [record.id for record in legacy_records]
    if len(legacy_ids) != len(set(legacy_ids)):
        raise RecoveryError("Legacy recovery helper returned duplicate Chroma IDs")

    inventory_value = getattr(recovered, "inventory", None)
    inventory = _mapping_value(inventory_value)
    legacy_vector_count = _inventory_count(
        inventory_value,
        "source_vector_count",
        "legacy_vector_count",
        "vector_count",
        "record_count",
    ) or len(legacy_records)
    if legacy_vector_count != len(legacy_records):
        raise RecoveryError(
            "Legacy recovery record count does not match its source-vector inventory: "
            f"{len(legacy_records)} recovered vs {legacy_vector_count} expected"
        )
    legacy_logical_document_count = _inventory_count(
        inventory_value,
        "logical_document_count",
        "logical_documents",
        "unique_paper_count",
        "paper_count",
    )
    if legacy_logical_document_count is None:
        legacy_logical_document_count = len(
            {str(record.metadata.get("paper_id") or record.metadata.get("title") or record.id) for record in legacy_records}
        )

    current_records: list[ChromaRecord] = []
    current_files: list[str] = []
    current_pdf_reconciliation: dict[str, Any] = {
        "status": "disabled",
        "filename_nonoverlap_files": [],
        "canonical_aliases": {},
        "current_only_candidates": [],
        "included_current_only_files": [],
    }
    if include_current_pdf_only and current_pdf_only_limit:
        current_records, current_files, current_pdf_reconciliation = _build_current_pdf_only_records(
            legacy_records,
            pdf_directory=current_pdf_directory,
            limit=current_pdf_only_limit,
            legacy_match_keys=getattr(recovered, "legacy_match_keys", None),
        )

    all_records = legacy_records + current_records
    all_ids = [record.id for record in all_records]
    if len(all_ids) != len(set(all_ids)):
        raise RecoveryError("Recovered stage would contain duplicate Chroma IDs")

    return RecoveryPayload(
        records=all_records,
        backup_dir=backup_dir,
        source_collection=source_collection,
        legacy_vector_count=legacy_vector_count,
        legacy_logical_document_count=legacy_logical_document_count,
        current_pdf_only_document_count=len(current_files),
        current_pdf_only_chunk_count=len(current_records),
        current_pdf_only_files=current_files,
        current_pdf_reconciliation=current_pdf_reconciliation,
        inventory=inventory,
    )


def validate_openai_connection() -> Any:
    """Create the embedding client and make the mandatory API preflight call."""
    from app.services.openai_service import OpenAIService
    from app.utils.openai_utils import load_session_config

    try:
        config = load_session_config()
        service = OpenAIService(
            embedding_model=config["embedding_model"],
            chat_model=os.getenv("BASE_CHAT_MODEL", "").strip() or config["chat_model"],
            vision_model=config["vision_model"],
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"],
        )
    except Exception as error:
        raise RecoveryError(f"Could not configure the OpenAI embedding client: {error}") from error
    try:
        available = service.validate_connection()
    except Exception as error:
        raise RecoveryError(f"OpenAI embedding preflight failed: {error}") from error
    if not available:
        raise RecoveryError("OpenAI embedding preflight failed: service reported unavailable")
    logger.info("OpenAI embedding preflight passed")
    return service


def _build_metadata(payload: RecoveryPayload) -> dict[str, str | int | float | bool]:
    """Metadata written at collection creation; state remains building initially."""
    return {
        "hnsw:space": "cosine",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "index_build_state": "building",
        "recovery_artifact": RECOVERY_ARTIFACT,
        "recovery_source_backup": str(payload.backup_dir),
        "recovery_source_collection": payload.source_collection,
        "recovery_source_vectors": payload.legacy_vector_count,
        "recovery_source_logical_documents": payload.legacy_logical_document_count,
        "recovery_current_pdf_only_documents": payload.current_pdf_only_document_count,
        "recovery_current_pdf_only_chunks": payload.current_pdf_only_chunk_count,
        "recovery_expected_vectors": payload.expected_vector_count,
        "recovery_expected_logical_documents": payload.expected_logical_document_count,
        "recovery_created_at": datetime.now(timezone.utc).isoformat(),
    }


def _close_client(client: Any) -> None:
    """Release Chroma's SQLite/HNSW handles before a directory rename."""
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception as error:  # pragma: no cover - best effort across Chroma versions
        logger.warning("Could not fully close Chroma client before filesystem operation: %s", error)


def _validate_stage_collection(
    stage_dir: Path,
    collection_name: str,
    *,
    require_ready: bool,
) -> dict[str, Any]:
    """Validate count, provenance, state, and sample retrieval from a stage."""
    if not stage_dir.is_dir() or not (stage_dir / "chroma.sqlite3").is_file():
        raise RecoveryError(f"Recovery stage does not contain a Chroma database: {stage_dir}")
    client = chromadb.PersistentClient(path=str(stage_dir))
    try:
        collection = client.get_collection(name=collection_name)
        metadata = dict(collection.metadata or {})
        expected = metadata.get("recovery_expected_vectors")
        try:
            expected_count = int(expected)
        except (TypeError, ValueError):
            raise RecoveryError("Recovery stage is missing a valid recovery_expected_vectors marker")
        actual_count = collection.count()
        if actual_count != expected_count:
            raise RecoveryError(
                f"Recovery stage vector count mismatch: {actual_count} stored vs {expected_count} expected"
            )
        if metadata.get("recovery_artifact") != RECOVERY_ARTIFACT:
            raise RecoveryError("Recovery stage is missing its legacy-recovery provenance marker")
        if str(metadata.get("index_schema_version")) != INDEX_SCHEMA_VERSION:
            raise RecoveryError("Recovery stage has an incompatible index schema version")
        state = metadata.get("index_build_state")
        allowed_states = {"ready"} if require_ready else {"building", "ready"}
        if state not in allowed_states:
            raise RecoveryError(f"Recovery stage is marked {state!r}, not one of {sorted(allowed_states)}")
        if actual_count:
            sample = collection.get(limit=1, include=["documents", "metadatas"])
            if not sample.get("ids") or not sample.get("documents"):
                raise RecoveryError("Recovery stage count is nonzero but a sample record cannot be read")
        return metadata
    finally:
        _close_client(client)


def build_stage(
    *,
    stage_dir: Path,
    collection_name: str,
    payload: RecoveryPayload,
    service: Any,
    embedding_batch_size: int,
    chroma_batch_size: int,
) -> dict[str, Any]:
    """Write a new immutable recovery stage, leaving it marked building on error."""
    if stage_dir.exists():
        raise RecoveryError(
            f"Refusing to overwrite an existing stage directory: {stage_dir}. "
            "Choose a new --stage-dir or inspect/promote the existing stage."
        )
    if embedding_batch_size < 1 or chroma_batch_size < 1:
        raise RecoveryError("Embedding and Chroma batch sizes must be positive")

    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    client: Any = None
    try:
        client = chromadb.PersistentClient(path=str(stage_dir))
        collection = client.create_collection(
            name=collection_name,
            metadata=_build_metadata(payload),
        )
        logger.info(
            "Created recovery stage %s with %s expected vectors (%s logical documents)",
            stage_dir,
            payload.expected_vector_count,
            payload.expected_logical_document_count,
        )

        for start in range(0, len(payload.records), chroma_batch_size):
            batch = payload.records[start : start + chroma_batch_size]
            documents = [record.document for record in batch]
            embeddings = service.get_embeddings_batch(documents, batch_size=embedding_batch_size)
            if len(embeddings) != len(batch):
                raise RecoveryError(
                    f"Embedding count mismatch for records {start}:{start + len(batch)}: "
                    f"got {len(embeddings)}, expected {len(batch)}"
                )
            collection.add(
                ids=[record.id for record in batch],
                documents=documents,
                metadatas=[record.metadata for record in batch],
                embeddings=embeddings,
            )
            logger.info(
                "Staged %s/%s vectors",
                min(start + len(batch), payload.expected_vector_count),
                payload.expected_vector_count,
            )

        actual_count = collection.count()
        if actual_count != payload.expected_vector_count:
            raise RecoveryError(
                f"Recovery stage write count mismatch: {actual_count} stored vs "
                f"{payload.expected_vector_count} expected"
            )

        # Validate the vector rows before making the collection serviceable.
        if actual_count:
            sample = collection.get(limit=1, include=["documents", "metadatas"])
            if not sample.get("ids") or not sample.get("documents"):
                raise RecoveryError("Recovery stage cannot read a sample vector after write")

        # Chroma does not allow immutable HNSW creation settings to be passed
        # to ``modify`` again, even if their values are unchanged.
        ready_metadata = {
            key: value
            for key, value in dict(collection.metadata or {}).items()
            if not str(key).startswith("hnsw:")
        }
        ready_metadata["index_build_state"] = "ready"
        ready_metadata["recovery_validated_at"] = datetime.now(timezone.utc).isoformat()
        collection.modify(metadata=ready_metadata)
        logger.info("Recovery stage passed count/sample validation and is marked ready")
    except Exception:
        # The initial metadata intentionally stays ``building`` when an error
        # occurs before ``modify``.  Keep the failed stage intact for diagnosis;
        # it cannot be served or promoted by this command.
        raise
    finally:
        if client is not None:
            _close_client(client)

    return _validate_stage_collection(stage_dir, collection_name, require_ready=True)


def _renamed_backup_path(target_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = target_dir.with_name(f"{target_dir.name}.pre_recovery_backup.{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = target_dir.with_name(
            f"{target_dir.name}.pre_recovery_backup.{timestamp}_{suffix}"
        )
        suffix += 1
    return candidate


def safe_promote_paths(stage_dir: Path, target_dir: Path) -> Optional[Path]:
    """Promote a ready stage by reversible same-filesystem renames.

    Returns the preserved target directory, or ``None`` when the target was
    absent.  This low-level function deliberately does not inspect Chroma; the
    caller must validate the stage first.
    """
    stage_dir = stage_dir.resolve()
    target_dir = target_dir.resolve()
    if stage_dir == target_dir:
        raise RecoveryError("The recovery stage and target directory must be different")
    if not stage_dir.is_dir():
        raise RecoveryError(f"Recovery stage directory does not exist: {stage_dir}")
    if not target_dir.parent.exists():
        raise RecoveryError(f"Target directory parent does not exist: {target_dir.parent}")
    try:
        if stage_dir.stat().st_dev != target_dir.parent.stat().st_dev:
            raise RecoveryError(
                "Stage and target are on different filesystems; safe rename promotion is unavailable"
            )
    except OSError as error:
        raise RecoveryError(f"Could not inspect stage/target filesystem: {error}") from error

    preserved_target: Optional[Path] = None
    if target_dir.exists():
        preserved_target = _renamed_backup_path(target_dir)
        logger.info("Preserving active target as %s", preserved_target)
        target_dir.rename(preserved_target)

    try:
        logger.info("Promoting validated recovery stage %s to %s", stage_dir, target_dir)
        stage_dir.rename(target_dir)
    except Exception as error:
        if preserved_target is not None and not target_dir.exists():
            try:
                preserved_target.rename(target_dir)
                logger.error("Promotion failed; restored the prior target directory")
            except Exception as rollback_error:
                raise RecoveryError(
                    f"Promotion failed ({error}) and rollback also failed ({rollback_error}). "
                    f"Prior target remains at {preserved_target}"
                ) from rollback_error
        raise RecoveryError(f"Could not promote recovery stage: {error}") from error
    return preserved_target


def promote_stage(*, stage_dir: Path, target_dir: Path, collection_name: str) -> Optional[Path]:
    """Validate a ready stage immediately before filesystem promotion."""
    _validate_stage_collection(stage_dir, collection_name, require_ready=True)
    return safe_promote_paths(stage_dir, target_dir)


def _summary(payload: RecoveryPayload, *, stage_dir: Path, target_dir: Path) -> dict[str, Any]:
    return {
        "legacy_backup": str(payload.backup_dir),
        "source_collection": payload.source_collection,
        "legacy_vectors": payload.legacy_vector_count,
        "legacy_logical_documents": payload.legacy_logical_document_count,
        "current_pdf_only_documents": payload.current_pdf_only_document_count,
        "current_pdf_only_chunks": payload.current_pdf_only_chunk_count,
        "current_pdf_only_files": payload.current_pdf_only_files,
        "current_pdf_reconciliation": payload.current_pdf_reconciliation,
        "expected_vectors": payload.expected_vector_count,
        "expected_logical_documents": payload.expected_logical_document_count,
        "stage_dir": str(stage_dir),
        "target_dir": str(target_dir),
        "inventory": payload.inventory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover the legacy Chroma corpus into a validated, separately staged index."
    )
    parser.add_argument(
        "--backup-dir",
        default=str(DEFAULT_BACKUP_DIR),
        help="Legacy Chroma persist directory (default: pre-rebuild local snapshot).",
    )
    parser.add_argument(
        "--source-collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name in the legacy backup.",
    )
    parser.add_argument(
        "--stage-dir",
        default=str(DEFAULT_STAGE_DIR),
        help="New, initially absent Chroma persist directory used for staging.",
    )
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help="Active Chroma directory to replace only during a validated promotion.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name to create in the recovery stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run local and OpenAI preflight but do not create, write, or promote a stage.",
    )
    parser.add_argument(
        "--promote-stage",
        action="store_true",
        help=(
            "Promote a ready existing --stage-dir, or promote the stage immediately after "
            "this command builds and validates it."
        ),
    )
    parser.add_argument(
        "--include-current-pdf-only",
        dest="include_current_pdf_only",
        action="store_true",
        help="Append current physical PDFs that are absent from the legacy corpus (default).",
    )
    parser.add_argument(
        "--no-current-pdf-only",
        dest="include_current_pdf_only",
        action="store_false",
        help="Recover only legacy vectors; do not inspect the local PDF subset.",
    )
    parser.set_defaults(include_current_pdf_only=True)
    parser.add_argument(
        "--current-pdf-dir",
        default=str(APP_DIR / "papers"),
        help="Directory searched only for current-PDF-only additions.",
    )
    parser.add_argument(
        "--current-pdf-only-limit",
        type=int,
        default=5,
        help="Maximum number of absent current PDFs to append (default: 5; zero disables).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=100,
        help="Texts per OpenAI embedding batch (default: 100).",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=500,
        help="Vectors per Chroma insertion batch (default: 500).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute recovery, preserving the active target until promotion succeeds."""
    backup_dir = _resolve_path(args.backup_dir)
    stage_dir = _resolve_path(args.stage_dir)
    target_dir = _resolve_path(args.target_dir)
    current_pdf_dir = _resolve_path(args.current_pdf_dir)

    # A previous successful stage can be promoted without re-reading the legacy
    # database or making an unnecessary OpenAI call.  Its persisted provenance
    # and exact vector count are revalidated immediately before the rename.
    if args.promote_stage and stage_dir.exists():
        if args.dry_run:
            stage_metadata = _validate_stage_collection(
                stage_dir, args.collection_name, require_ready=True
            )
            logger.info("Dry run: ready stage validated; it would be promoted to %s", target_dir)
            print(
                json.dumps(
                    {
                        "status": "dry_run_ready_stage",
                        "stage_dir": str(stage_dir),
                        "target_dir": str(target_dir),
                        "expected_vectors": stage_metadata.get("recovery_expected_vectors"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        preserved = promote_stage(
            stage_dir=stage_dir,
            target_dir=target_dir,
            collection_name=args.collection_name,
        )
        print(
            json.dumps(
                {
                    "status": "promoted",
                    "stage_dir": str(stage_dir),
                    "target_dir": str(target_dir),
                    "preserved_previous_target": str(preserved) if preserved else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if stage_dir.exists():
        raise RecoveryError(
            f"Stage directory already exists: {stage_dir}. Use --promote-stage to promote a "
            "validated ready stage, or choose a new --stage-dir."
        )

    # Strict preflight ordering: reconstruct the local source inventory and
    # verify OpenAI *before* Chroma creates the stage directory.
    payload = prepare_recovery_payload(
        backup_dir=backup_dir,
        source_collection=args.source_collection,
        include_current_pdf_only=args.include_current_pdf_only,
        current_pdf_directory=current_pdf_dir,
        current_pdf_only_limit=args.current_pdf_only_limit,
    )
    service = validate_openai_connection()
    summary = _summary(payload, stage_dir=stage_dir, target_dir=target_dir)

    if args.dry_run:
        logger.info("Dry run passed; no recovery stage was created")
        print(json.dumps({"status": "dry_run_passed", **summary}, indent=2, sort_keys=True))
        return 0

    build_stage(
        stage_dir=stage_dir,
        collection_name=args.collection_name,
        payload=payload,
        service=service,
        embedding_batch_size=args.embedding_batch_size,
        chroma_batch_size=args.chroma_batch_size,
    )
    if args.promote_stage:
        preserved = promote_stage(
            stage_dir=stage_dir,
            target_dir=target_dir,
            collection_name=args.collection_name,
        )
        summary["preserved_previous_target"] = str(preserved) if preserved else None
        summary["status"] = "staged_and_promoted"
    else:
        summary["status"] = "staged_ready"
        summary["next_step"] = (
            f"Inspect the ready stage, then run: python recover_legacy_vectordb.py "
            f"--stage-dir {stage_dir} --promote-stage"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except RecoveryError as error:
        logger.error("Legacy vector recovery aborted safely: %s", error)
        return 1
    except KeyboardInterrupt:
        logger.error("Legacy vector recovery interrupted; any created stage remains marked building")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
