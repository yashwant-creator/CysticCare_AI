"""Offline recovery of documents stored only in a legacy Chroma collection.

The old index contains useful extracted text for papers whose PDFs are no
longer present in the workspace.  This module turns those stored documents
into *new* records that can be embedded into a schema-v2 collection.  It does
not open PDFs, call an embedding provider, or make any network requests.

Only complete duplicate source-file sets are collapsed.  In particular, two
filenames are treated as aliases only when the complete multiset of their
stored chunk documents is identical.  Merely sharing a title is never enough
to merge papers.

The reader intentionally relies on Chroma's public ``count`` and paginated
``get`` APIs.  Keeping the migration at this boundary makes it work across
Chroma storage implementations without reading SQLite or HNSW internals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Protocol, Sequence, Tuple

from ..utils.paper_identity import normalize_paper_identity, stable_paper_id


DEFAULT_LEGACY_COLLECTION_NAME = "pkd_knowledge_base_openai"
LEGACY_CHUNK_ORIGIN = "legacy_chroma"
_METADATA_INCLUDE = ["documents", "metadatas"]
_LEGACY_CHUNK_SUFFIX = re.compile(
    r"(?i)(?:[_-](?:pdf[_-]?)?chunk[_-]?\d+)$"
)
_TRAILING_NUMBER = re.compile(r"(?i)(?:[_-](?:pdf[_-]?)?chunk[_-]?)(\d+)$")


class LegacyCorpusRecoveryError(RuntimeError):
    """Raised when a legacy collection cannot be read without losing rows."""


class ChromaCollectionLike(Protocol):
    """The small public Chroma collection surface used by this module."""

    name: str

    def count(self) -> int:
        ...

    def get(self, *, limit: int, offset: int, include: Sequence[str]) -> Mapping[str, Any]:
        ...


PaperIdFactory = Callable[[Any, Any], str]


@dataclass(frozen=True)
class LegacyChromaRow:
    """One text chunk read from the old collection through Chroma's public API."""

    legacy_embedding_id: str
    document: str
    metadata: Mapping[str, Any]
    source_position: int


@dataclass(frozen=True)
class RecoveryRecord:
    """A Chroma-ready, schema-v2-compatible document to re-embed."""

    id: str
    document: str
    metadata: Mapping[str, Any]

    @property
    def chunk_id(self) -> str:
        return self.id

    def as_chroma_record(self) -> Dict[str, Any]:
        """Return the conventional ``id/document/metadata`` record shape."""
        return {
            "id": self.id,
            "document": self.document,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LegacyPaperProvenance:
    """File/title provenance for one logical recovered paper."""

    paper_id: str
    title: str
    canonical_source_file_name: str
    source_file_names: Tuple[str, ...]
    source_file_paths: Tuple[str, ...]
    title_variants: Tuple[str, ...]
    legacy_source_fingerprint: str
    source_row_count: int
    emitted_chunk_count: int
    match_keys: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "canonical_source_file_name": self.canonical_source_file_name,
            "source_file_names": list(self.source_file_names),
            "source_file_paths": list(self.source_file_paths),
            "title_variants": list(self.title_variants),
            "legacy_source_fingerprint": self.legacy_source_fingerprint,
            "source_row_count": self.source_row_count,
            "emitted_chunk_count": self.emitted_chunk_count,
            "match_keys": list(self.match_keys),
        }


@dataclass(frozen=True)
class LegacyCorpusInventory:
    """Counts needed to audit a legacy-only recovery before adding PDFs."""

    source_row_count: int
    source_file_count: int
    source_title_count: int
    logical_paper_count: int
    recovered_chunk_count: int
    skipped_empty_document_count: int
    duplicate_alias_source_set_count: int
    aliased_source_file_count: int
    duplicate_alias_source_file_count: int
    duplicate_alias_row_count: int

    @property
    def legacy_vector_count(self) -> int:
        """Compatibility name for the number of source Chroma rows."""
        return self.source_row_count

    @property
    def legacy_document_count(self) -> int:
        """Compatibility name for the number of legacy filenames."""
        return self.source_file_count

    @property
    def recovered_vector_count(self) -> int:
        return self.recovered_chunk_count

    def to_dict(self) -> Dict[str, int]:
        return {
            "source_row_count": self.source_row_count,
            "source_file_count": self.source_file_count,
            "source_title_count": self.source_title_count,
            "logical_paper_count": self.logical_paper_count,
            "recovered_chunk_count": self.recovered_chunk_count,
            "skipped_empty_document_count": self.skipped_empty_document_count,
            "duplicate_alias_source_set_count": self.duplicate_alias_source_set_count,
            "aliased_source_file_count": self.aliased_source_file_count,
            "duplicate_alias_source_file_count": self.duplicate_alias_source_file_count,
            "duplicate_alias_row_count": self.duplicate_alias_row_count,
        }


@dataclass(frozen=True)
class LegacyCorpusRecovery:
    """Legacy-only records and audit information.

    ``records`` deliberately contains no current-PDF records.  A rebuild
    caller can compare ``legacy_match_keys`` with its PDF/cache inventory,
    then append only the current-PDF documents it intends to keep.
    """

    records: Tuple[RecoveryRecord, ...]
    inventory: LegacyCorpusInventory
    paper_provenance: Tuple[LegacyPaperProvenance, ...]
    collection_name: str = ""

    @property
    def legacy_vector_count(self) -> int:
        return self.inventory.legacy_vector_count

    @property
    def legacy_document_count(self) -> int:
        return self.inventory.legacy_document_count

    @property
    def provenance(self) -> Tuple[LegacyPaperProvenance, ...]:
        """Short alias used by command-line callers."""
        return self.paper_provenance

    @property
    def legacy_match_keys(self) -> frozenset[str]:
        """Normalized title/file keys for a caller's current-PDF comparison."""
        return frozenset(
            key
            for paper in self.paper_provenance
            for key in paper.match_keys
            if key
        )

    @property
    def source_backup(self) -> Dict[str, Any]:
        """Small serializable description of the input collection."""
        return {
            "kind": LEGACY_CHUNK_ORIGIN,
            "collection_name": self.collection_name,
            **self.inventory.to_dict(),
        }

    def as_chroma_payload(self) -> Dict[str, List[Any]]:
        """Return lists directly suitable for ``Collection.add`` after embedding."""
        return {
            "ids": [record.id for record in self.records],
            "documents": [record.document for record in self.records],
            "metadatas": [dict(record.metadata) for record in self.records],
        }


@dataclass
class _SourceBucket:
    source_file_name: str
    rows: List[LegacyChromaRow]

    @property
    def fingerprint(self) -> str:
        # Sort digest values rather than relying on legacy insertion order;
        # duplicate aliases sometimes use different Chroma IDs.
        digests = sorted(_document_digest(row.document) for row in self.rows)
        digest = hashlib.sha256()
        for item in digests:
            digest.update(item.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    @property
    def source_paths(self) -> Tuple[str, ...]:
        return _sorted_unique(
            _metadata_text(row.metadata, "file_path") for row in self.rows
        )


def iter_legacy_chroma_rows(
    collection: ChromaCollectionLike,
    *,
    page_size: int = 500,
) -> Iterator[LegacyChromaRow]:
    """Yield every old Chroma row using public, offset-based pagination.

    The function rejects malformed or changing pages rather than quietly
    zipping unequal arrays and producing an unrecoverable partial corpus.
    """
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")

    try:
        expected_count = int(collection.count())
    except Exception as error:  # pragma: no cover - delegated Chroma errors
        raise LegacyCorpusRecoveryError(f"Could not count legacy collection: {error}") from error
    if expected_count < 0:
        raise LegacyCorpusRecoveryError("Legacy collection returned a negative row count")

    offset = 0
    while offset < expected_count:
        limit = min(page_size, expected_count - offset)
        try:
            page = collection.get(
                limit=limit,
                offset=offset,
                include=_METADATA_INCLUDE,
            )
        except Exception as error:  # pragma: no cover - delegated Chroma errors
            raise LegacyCorpusRecoveryError(
                f"Could not read legacy collection at offset {offset}: {error}"
            ) from error

        if not isinstance(page, Mapping):
            raise LegacyCorpusRecoveryError(
                f"Legacy collection returned a non-mapping page at offset {offset}"
            )
        ids = page.get("ids")
        documents = page.get("documents")
        metadatas = page.get("metadatas")
        if not isinstance(ids, list) or not isinstance(documents, list) or not isinstance(metadatas, list):
            raise LegacyCorpusRecoveryError(
                "Legacy Chroma get() must return list-valued ids, documents, and metadatas"
            )
        if not ids:
            raise LegacyCorpusRecoveryError(
                f"Legacy collection ended early at offset {offset}; expected {expected_count} rows"
            )
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise LegacyCorpusRecoveryError(
                f"Legacy Chroma page arrays are not aligned at offset {offset}: "
                f"ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)}"
            )
        if len(ids) > limit or offset + len(ids) > expected_count:
            raise LegacyCorpusRecoveryError(
                f"Legacy collection returned an invalid page length ({len(ids)}) at offset {offset}"
            )

        for index, (raw_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
            legacy_embedding_id = _as_text(raw_id)
            if not legacy_embedding_id:
                raise LegacyCorpusRecoveryError(
                    f"Legacy collection has an empty embedding ID at offset {offset + index}"
                )
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, Mapping):
                raise LegacyCorpusRecoveryError(
                    f"Legacy collection has non-mapping metadata for {legacy_embedding_id!r}"
                )
            yield LegacyChromaRow(
                legacy_embedding_id=legacy_embedding_id,
                document=_as_text(document),
                metadata=dict(metadata),
                source_position=offset + index,
            )
        offset += len(ids)

    if offset != expected_count:  # defensive guard if a collection changes during reads
        raise LegacyCorpusRecoveryError(
            f"Legacy collection changed while reading: recovered {offset} of {expected_count} rows"
        )


def read_legacy_collection_rows(
    collection: ChromaCollectionLike,
    *,
    page_size: int = 500,
) -> Tuple[LegacyChromaRow, ...]:
    """Materialize the public-paginated legacy rows for auditing or recovery."""
    return tuple(iter_legacy_chroma_rows(collection, page_size=page_size))


def recover_legacy_collection(
    collection: ChromaCollectionLike,
    *,
    page_size: int = 500,
    paper_id_factory: PaperIdFactory = stable_paper_id,
) -> LegacyCorpusRecovery:
    """Build deterministic, re-embedding-ready records from one old collection.

    No embedding is created here.  The caller owns vector generation and can
    append independently selected current-PDF records after this result.
    """
    rows = read_legacy_collection_rows(collection, page_size=page_size)
    return recover_legacy_rows(
        rows,
        paper_id_factory=paper_id_factory,
        collection_name=_as_text(getattr(collection, "name", "")),
    )


def build_recovery_records(
    legacy_collection: ChromaCollectionLike,
    *,
    page_size: int = 500,
    paper_id_factory: PaperIdFactory = stable_paper_id,
) -> LegacyCorpusRecovery:
    """Command-facing alias for :func:`recover_legacy_collection`."""
    return recover_legacy_collection(
        legacy_collection,
        page_size=page_size,
        paper_id_factory=paper_id_factory,
    )


def recover_legacy_rows(
    rows: Iterable[LegacyChromaRow],
    *,
    paper_id_factory: PaperIdFactory = stable_paper_id,
    collection_name: str = "",
) -> LegacyCorpusRecovery:
    """Recover records from already-read legacy rows (useful for offline tests)."""
    materialized_rows = tuple(rows)
    source_buckets = _bucket_rows_by_source_file(materialized_rows)
    fingerprint_buckets: Dict[str, List[_SourceBucket]] = defaultdict(list)
    for source in source_buckets.values():
        fingerprint_buckets[source.fingerprint].append(source)

    records: List[RecoveryRecord] = []
    provenance: List[LegacyPaperProvenance] = []
    emitted_paper_ids: Dict[str, str] = {}
    duplicate_source_set_count = 0
    aliased_source_file_count = 0
    duplicate_alias_source_file_count = 0
    duplicate_alias_row_count = 0
    skipped_empty_document_count = 0

    # Sort by canonical filename and fingerprint to make every generated ID
    # independent of public-get pagination order.
    logical_groups = []
    for fingerprint, aliases in fingerprint_buckets.items():
        ordered_aliases = sorted(aliases, key=lambda source: _source_sort_key(source.source_file_name))
        canonical = ordered_aliases[0]
        logical_groups.append((canonical.source_file_name, fingerprint, ordered_aliases))
        if len(ordered_aliases) > 1:
            duplicate_source_set_count += 1
            aliased_source_file_count += len(ordered_aliases)
            duplicate_alias_source_file_count += len(ordered_aliases) - 1
            duplicate_alias_row_count += sum(len(source.rows) for source in ordered_aliases[1:])

    for _, fingerprint, alias_sources in sorted(
        logical_groups,
        key=lambda item: (_source_sort_key(item[0]), item[1]),
    ):
        canonical_source = alias_sources[0]
        source_file_names = tuple(source.source_file_name for source in alias_sources)
        source_file_paths = _sorted_unique(
            path for source in alias_sources for path in source.source_paths
        )
        title_variants = _title_variants(alias_sources)
        title = _choose_preferred_text(title_variants, fallback=_file_stem(canonical_source.source_file_name))
        requested_paper_id = _as_text(
            paper_id_factory(title, _file_stem(canonical_source.source_file_name))
        )
        if not requested_paper_id:
            requested_paper_id = _fallback_paper_id(title, canonical_source.source_file_name)
        paper_id = _unique_paper_id(requested_paper_id, fingerprint, emitted_paper_ids)

        # A document's legacy ID generally contains a chunk ordinal.  It is
        # used only to produce a stable sequence; the v2 chunk IDs below are
        # entirely fresh and cannot collide with the legacy index.
        canonical_rows = sorted(canonical_source.rows, key=_legacy_row_sort_key)
        nonempty_rows = [row for row in canonical_rows if row.document.strip()]
        skipped_empty_document_count += len(canonical_rows) - len(nonempty_rows)
        records_for_paper = _records_for_logical_paper(
            paper_id=paper_id,
            title=title,
            canonical_source=canonical_source,
            alias_sources=alias_sources,
            source_file_names=source_file_names,
            source_file_paths=source_file_paths,
            title_variants=title_variants,
            legacy_source_fingerprint=fingerprint,
            rows=nonempty_rows,
        )
        records.extend(records_for_paper)
        provenance.append(
            LegacyPaperProvenance(
                paper_id=paper_id,
                title=title,
                canonical_source_file_name=canonical_source.source_file_name,
                source_file_names=source_file_names,
                source_file_paths=source_file_paths,
                title_variants=title_variants,
                legacy_source_fingerprint=fingerprint,
                source_row_count=sum(len(source.rows) for source in alias_sources),
                emitted_chunk_count=len(records_for_paper),
                match_keys=_paper_match_keys(title_variants, source_file_names),
            )
        )

    source_titles = {
        normalize_paper_identity(_metadata_text(row.metadata, "title"))
        for row in materialized_rows
        if normalize_paper_identity(_metadata_text(row.metadata, "title"))
    }
    inventory = LegacyCorpusInventory(
        source_row_count=len(materialized_rows),
        source_file_count=len(source_buckets),
        source_title_count=len(source_titles),
        logical_paper_count=len(logical_groups),
        recovered_chunk_count=len(records),
        skipped_empty_document_count=skipped_empty_document_count,
        duplicate_alias_source_set_count=duplicate_source_set_count,
        aliased_source_file_count=aliased_source_file_count,
        duplicate_alias_source_file_count=duplicate_alias_source_file_count,
        duplicate_alias_row_count=duplicate_alias_row_count,
    )
    return LegacyCorpusRecovery(
        records=tuple(records),
        inventory=inventory,
        paper_provenance=tuple(provenance),
        collection_name=collection_name,
    )


def open_legacy_collection(
    persist_directory: str | Path,
    *,
    collection_name: str = DEFAULT_LEGACY_COLLECTION_NAME,
) -> ChromaCollectionLike:
    """Open a local legacy Chroma collection without touching PDFs or network I/O."""
    try:
        import chromadb
    except ImportError as error:  # pragma: no cover - only relevant in bare environments
        raise LegacyCorpusRecoveryError("chromadb is required to open a persisted backup") from error

    client = chromadb.PersistentClient(path=str(persist_directory))
    return client.get_collection(name=collection_name)


def recover_legacy_backup(
    persist_directory: str | Path,
    *,
    collection_name: str = DEFAULT_LEGACY_COLLECTION_NAME,
    page_size: int = 500,
    paper_id_factory: PaperIdFactory = stable_paper_id,
) -> LegacyCorpusRecovery:
    """Open and recover one local persisted Chroma backup."""
    collection = open_legacy_collection(
        persist_directory,
        collection_name=collection_name,
    )
    return recover_legacy_collection(
        collection,
        page_size=page_size,
        paper_id_factory=paper_id_factory,
    )


def _bucket_rows_by_source_file(
    rows: Iterable[LegacyChromaRow],
) -> Dict[str, _SourceBucket]:
    buckets: Dict[str, _SourceBucket] = {}
    for row in rows:
        source_file_name = _source_file_name(row)
        bucket = buckets.get(source_file_name)
        if bucket is None:
            bucket = _SourceBucket(source_file_name=source_file_name, rows=[])
            buckets[source_file_name] = bucket
        bucket.rows.append(row)
    return buckets


def _records_for_logical_paper(
    *,
    paper_id: str,
    title: str,
    canonical_source: _SourceBucket,
    alias_sources: Sequence[_SourceBucket],
    source_file_names: Tuple[str, ...],
    source_file_paths: Tuple[str, ...],
    title_variants: Tuple[str, ...],
    legacy_source_fingerprint: str,
    rows: Sequence[LegacyChromaRow],
) -> List[RecoveryRecord]:
    alias_names_json = _json_list(source_file_names)
    alias_paths_json = _json_list(source_file_paths)
    titles_json = _json_list(title_variants)
    canonical_metadata = _preferred_metadata(alias_sources)
    source_file_path = _choose_preferred_text(
        canonical_source.source_paths,
        fallback=_metadata_text(canonical_metadata, "file_path"),
    )
    records: List[RecoveryRecord] = []
    for chunk_index, row in enumerate(rows):
        chunk_id = f"{paper_id}_legacy_{chunk_index}"
        previous_chunk_id = f"{paper_id}_legacy_{chunk_index - 1}" if chunk_index else ""
        next_chunk_id = (
            f"{paper_id}_legacy_{chunk_index + 1}"
            if chunk_index + 1 < len(rows)
            else ""
        )
        metadata = _normalized_metadata(
            canonical_metadata=canonical_metadata,
            row_metadata=row.metadata,
            paper_id=paper_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            title=title,
            source_file_name=canonical_source.source_file_name,
            source_file_path=source_file_path,
            legacy_embedding_id=row.legacy_embedding_id,
            alias_names_json=alias_names_json,
            alias_paths_json=alias_paths_json,
            titles_json=titles_json,
            alias_source_count=len(alias_sources),
            legacy_source_fingerprint=legacy_source_fingerprint,
        )
        records.append(RecoveryRecord(id=chunk_id, document=row.document, metadata=metadata))
    return records


def _normalized_metadata(
    *,
    canonical_metadata: Mapping[str, Any],
    row_metadata: Mapping[str, Any],
    paper_id: str,
    chunk_id: str,
    chunk_index: int,
    previous_chunk_id: str,
    next_chunk_id: str,
    title: str,
    source_file_name: str,
    source_file_path: str,
    legacy_embedding_id: str,
    alias_names_json: str,
    alias_paths_json: str,
    titles_json: str,
    alias_source_count: int,
    legacy_source_fingerprint: str,
) -> Dict[str, Any]:
    """Flatten legacy fields and make unavailable PDF structure explicit."""
    # Keep useful existing metadata under the familiar keys.  Per-row values
    # win only for fields that actually vary in the old collection.
    metadata: Dict[str, Any] = {}
    for field in (
        "author",
        "citation",
        "creation_date",
        "display_name",
        "indexed_date",
        "source_type",
        "subject",
        "year",
    ):
        value = _metadata_text(row_metadata, field) or _metadata_text(canonical_metadata, field)
        if value:
            metadata[field] = value

    metadata.update(
        {
            # Current downstream code has historically read title/file_name.
            "title": title,
            "paper_title": title,
            "file_name": source_file_name,
            "file_path": source_file_path,
            # Explicit provenance for the migration, including the particular
            # old vector that supplied this text.
            "source_file_name": source_file_name,
            "source_file_path": source_file_path,
            "legacy_embedding_id": legacy_embedding_id,
            "chunk_origin": LEGACY_CHUNK_ORIGIN,
            "legacy_source_fingerprint": legacy_source_fingerprint,
            "legacy_source_file_aliases_json": alias_names_json,
            "legacy_source_file_paths_json": alias_paths_json,
            "legacy_title_variants_json": titles_json,
            "legacy_alias_source_count": alias_source_count,
            # Schema-v2 identity and stable logical adjacency.
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "previous_chunk_id": previous_chunk_id,
            "next_chunk_id": next_chunk_id,
            "parent_document_id": paper_id,
            "parent_chunk_id": "",
            "content_type": "text",
            # No page/section information survives in this pre-structure
            # legacy index.  Use false/zero/empty values rather than inventing
            # a page or section for a retrieved medical statement.
            "structure_available": False,
            "structure_unavailable": True,
            "structure_status": "unavailable_from_legacy_chroma",
            "page_number": 0,
            "page_start": 0,
            "page_end": 0,
            "section_heading": "",
        }
    )
    return metadata


def _preferred_metadata(alias_sources: Sequence[_SourceBucket]) -> Mapping[str, Any]:
    """Choose deterministic field values across canonical rows and aliases."""
    all_rows = [row for source in alias_sources for row in source.rows]
    result: Dict[str, Any] = {}
    for field in (
        "author",
        "citation",
        "creation_date",
        "display_name",
        "file_path",
        "indexed_date",
        "source_type",
        "subject",
        "year",
    ):
        values = [_metadata_text(row.metadata, field) for row in all_rows]
        selected = _choose_preferred_text(values, fallback="")
        if selected:
            result[field] = selected
    return result


def _title_variants(alias_sources: Sequence[_SourceBucket]) -> Tuple[str, ...]:
    values = [
        _metadata_text(row.metadata, "title")
        or _metadata_text(row.metadata, "display_name")
        for source in alias_sources
        for row in source.rows
    ]
    return _sorted_unique(values)


def _paper_match_keys(
    title_variants: Sequence[str], source_file_names: Sequence[str]
) -> Tuple[str, ...]:
    keys = []
    for title in title_variants:
        normalized = normalize_paper_identity(title)
        if normalized:
            keys.append(f"title:{normalized}")
    for file_name in source_file_names:
        normalized = normalize_paper_identity(_file_stem(file_name))
        if normalized:
            keys.append(f"file:{normalized}")
    return _sorted_unique(keys)


def _source_file_name(row: LegacyChromaRow) -> str:
    file_name = _metadata_text(row.metadata, "file_name")
    if file_name:
        return _basename(file_name)
    file_path = _metadata_text(row.metadata, "file_path")
    if file_path:
        return _basename(file_path)

    # A filename is absent in some hand-made Chroma records.  Derive a stable
    # synthetic one from the old ID, which still preserves a traceable origin.
    prefix = _LEGACY_CHUNK_SUFFIX.sub("", row.legacy_embedding_id).strip("_-")
    if not prefix:
        prefix = hashlib.sha256(row.legacy_embedding_id.encode("utf-8")).hexdigest()[:16]
    if not prefix.casefold().endswith(".pdf"):
        prefix = f"{prefix}.pdf"
    return prefix


def _legacy_row_sort_key(row: LegacyChromaRow) -> Tuple[int, int, str, str, int]:
    match = _TRAILING_NUMBER.search(row.legacy_embedding_id)
    if match:
        return (
            0,
            int(match.group(1)),
            row.legacy_embedding_id.casefold(),
            _document_digest(row.document),
            row.source_position,
        )
    return (
        1,
        0,
        row.legacy_embedding_id.casefold(),
        _document_digest(row.document),
        row.source_position,
    )


def _unique_paper_id(
    requested_paper_id: str,
    source_fingerprint: str,
    emitted_paper_ids: Dict[str, str],
) -> str:
    existing_fingerprint = emitted_paper_ids.get(requested_paper_id)
    if existing_fingerprint is None:
        emitted_paper_ids[requested_paper_id] = source_fingerprint
        return requested_paper_id
    if existing_fingerprint == source_fingerprint:
        return requested_paper_id

    # Same legacy title can legitimately describe different documents.  Keep
    # the normal title-stable ID for the first deterministic group and suffix
    # later collisions with a content-derived identifier rather than merging.
    candidate = f"{requested_paper_id}_{source_fingerprint[:12]}"
    while candidate in emitted_paper_ids and emitted_paper_ids[candidate] != source_fingerprint:
        candidate = f"{candidate}_{source_fingerprint[len(candidate) % 48:][:4]}"
    emitted_paper_ids[candidate] = source_fingerprint
    return candidate


def _fallback_paper_id(title: str, source_file_name: str) -> str:
    identity = normalize_paper_identity(title) or normalize_paper_identity(source_file_name)
    digest = hashlib.sha256((identity or "legacy-paper").encode("utf-8")).hexdigest()[:12]
    return f"paper_{digest}"


def _choose_preferred_text(values: Iterable[Any], *, fallback: str) -> str:
    """Choose most frequent text, resolving ties lexically for stability."""
    nonempty = [_as_text(value) for value in values]
    nonempty = [value for value in nonempty if value]
    if not nonempty:
        return fallback
    counts = Counter(nonempty)
    return min(
        counts,
        key=lambda value: (-counts[value], normalize_paper_identity(value), value.casefold()),
    )


def _metadata_text(metadata: Mapping[str, Any], field: str) -> str:
    return _as_text(metadata.get(field)) if metadata else ""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _basename(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def _file_stem(value: str) -> str:
    name = _basename(value)
    return name.rsplit(".", 1)[0] if "." in name else name


def _source_sort_key(value: str) -> Tuple[str, str]:
    return (normalize_paper_identity(value), value.casefold())


def _sorted_unique(values: Iterable[Any]) -> Tuple[str, ...]:
    unique = {_as_text(value) for value in values if _as_text(value)}
    return tuple(sorted(unique, key=lambda value: (normalize_paper_identity(value), value.casefold())))


def _document_digest(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _json_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
