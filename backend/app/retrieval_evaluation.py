"""Dependency-free retrieval evaluation for the CysticCare benchmark.

This module evaluates *retrieval*, rather than the language model's answer.  It
accepts the benchmark already present in this repository and a ranked list of
retrieved sources for each question.  The benchmark contains paper-level labels
in ``retrieval_targets`` and passage-level labels in ``supporting_passages``.

The evaluator deliberately keeps paper and chunk evaluation separate:

* ``label_level="paper"`` assesses whether the right research papers were found.
* ``label_level="chunk"`` assesses whether the exact labelled supporting chunks
  were found.

Paper-level evaluation is the default because it works with the existing
``cysticcare_metadata.sources`` output.  Exact ``paper_id``/``chunk_id`` values
are preferred, but legacy ``file``, ``display_name``, ``title``, and citation
fields are used as a conservative normalized-title fallback.

Example
-------

.. code-block:: bash

    cd backend
    python evaluate_retrieval.py \
      --benchmark app/pipeline_results.json \
      --predictions app/pipeline_results_with_cysticcare.json \
      --k 1,3,5,10 --label-level paper \
      --output retrieval_metrics.json

Prediction records are joined to benchmark records by ``id``, ``question_id``,
``query_id``, or finally the question text.  A prediction can contain one of
the following ranked lists:

* ``retrieved`` / ``retrieval_results`` / ``retrieved_chunks``
* ``sources``
* ``cysticcare_metadata.retrieved_chunks`` / ``cysticcare_metadata.sources``

Each item should ideally include ``paper_id`` and, for chunk evaluation,
``chunk_id``.  ``metadata`` objects are also inspected, which makes the module
compatible with raw retriever output.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_CUTOFFS: Tuple[int, ...] = (1, 3, 5, 10)
SUPPORTED_LABEL_LEVELS = frozenset({"paper", "chunk", "auto"})

_RECORD_KEY_FIELDS = ("id", "question_id", "query_id")
_QUESTION_FIELDS = ("question", "query")
_PREDICTION_LIST_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("retrieved",),
    ("retrieval_results",),
    ("retrieved_chunks",),
    ("retrieval",),
    ("candidates",),
    ("cysticcare_metadata", "retrieved_chunks"),
    ("cysticcare_metadata", "sources"),
    ("metadata", "sources"),
    ("sources",),
    ("cysticcare_sources",),
)
_PAPER_ID_KEYS = ("paper_id", "paperId", "source_paper_id", "document_paper_id")
_CHUNK_ID_KEYS = ("chunk_id", "chunkId", "passage_id", "passageId")
_TITLE_KEYS = (
    "paper_title",
    "title",
    "display_name",
    "file",
    "citation",
    "source_title",
)


def _as_text(value: Any) -> str:
    """Return a stripped string or an empty string for non-useful values."""
    if value is None:
        return ""
    return str(value).strip()


def normalise_label(value: Any) -> str:
    """Normalize IDs/titles for exact, case-insensitive fallback matching.

    This intentionally is not fuzzy matching.  Fuzzy title matching can turn a
    clinically unrelated paper with a similar title into a false hit.  IDs are
    always the preferred evaluation key.
    """
    text = _as_text(value).casefold()
    text = re.sub(r"\s+", " ", text)
    # Punctuation differences in PDF filenames/citations are not meaningful.
    return re.sub(r"[^\w]+", "", text)


def _append_unique(values: List[str], value: Any) -> None:
    text = _as_text(value)
    if text and text not in values:
        values.append(text)


def _iter_values(value: Any) -> Iterable[Any]:
    """Yield values from a scalar/list/dict label field without splitting text."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _get_nested(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def record_key(record: Mapping[str, Any]) -> str:
    """Return a stable join key, preferring explicit IDs over question text."""
    for field in _RECORD_KEY_FIELDS + _QUESTION_FIELDS:
        value = _as_text(record.get(field))
        if value:
            return value
    return ""


def _parse_json_cell(value: str) -> Any:
    """Decode JSON-shaped CSV cells while leaving ordinary strings untouched."""
    value = value.strip()
    if not value or value[0] not in "[{":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON, JSONL/NDJSON, or CSV benchmark/predictions file.

    JSON files may contain a top-level list or a wrapper object with one of
    ``records``, ``data``, ``items``, or ``results``.  CSV columns containing
    JSON arrays/objects are decoded automatically, which is useful for exports
    where a ``retrieved`` column contains a serialized ranked list.

    Raises:
        ValueError: if the input cannot be interpreted as a list of records.
    """
    input_path = Path(path)
    suffix = input_path.suffix.casefold()
    try:
        if suffix in {".jsonl", ".ndjson"}:
            records: List[Any] = []
            with input_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        records.append(json.loads(text))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_number} of {input_path}"
                        ) from exc
        elif suffix == ".csv":
            with input_path.open("r", encoding="utf-8", newline="") as handle:
                records = [
                    {key: _parse_json_cell(value or "") for key, value in row.items()}
                    for row in csv.DictReader(handle)
                ]
        else:
            with input_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, Mapping):
                records = None  # type: ignore[assignment]
                for wrapper_key in ("records", "data", "items", "results"):
                    candidate = payload.get(wrapper_key)
                    if isinstance(candidate, list):
                        records = candidate
                        break
                if records is None:
                    raise ValueError(
                        f"Expected a list or a records/data/items/results wrapper in {input_path}"
                    )
            else:
                raise ValueError(f"Expected a JSON list of records in {input_path}")
    except OSError as exc:
        raise ValueError(f"Could not read {input_path}: {exc}") from exc

    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise ValueError(f"Every record in {input_path} must be a JSON object")
    return [dict(row) for row in records]


def _add_grade(labels: Dict[str, float], raw_id: Any, grade: Any = 1.0) -> None:
    label = _as_text(raw_id)
    if not label:
        return
    try:
        numeric_grade = float(grade)
    except (TypeError, ValueError):
        numeric_grade = 1.0
    # Zero/negative relevance labels are not useful as a gold target.
    if numeric_grade > 0:
        labels[label] = max(labels.get(label, 0.0), numeric_grade)


def _positive_grade(value: Any) -> float:
    """Return a positive numeric relevance grade, defaulting to binary relevance."""
    try:
        grade = float(value)
    except (TypeError, ValueError):
        return 1.0
    return grade if grade > 0 else 1.0


def _extract_id_grades(value: Any, labels: Dict[str, float], id_keys: Sequence[str]) -> None:
    """Add labels from scalar IDs, lists, maps, or objects with relevance grades."""
    for item in _iter_values(value):
        if isinstance(item, Mapping):
            item_id = ""
            for key in tuple(id_keys) + ("id", "key"):
                if _as_text(item.get(key)):
                    item_id = _as_text(item.get(key))
                    break
            if item_id:
                _add_grade(
                    labels,
                    item_id,
                    item.get("grade", item.get("relevance", item.get("score", 1.0))),
                )
            else:
                # Supports {"paper-a": 3, "paper-b": 1} style graded labels.
                # A nested object is ignored unless it exposes one of the ID
                # fields above, avoiding accidental use of metadata keys as IDs.
                for only_id, only_grade in item.items():
                    if not isinstance(only_grade, Mapping):
                        _add_grade(labels, only_id, only_grade)
        else:
            _add_grade(labels, item)


def _parse_legacy_supporting_passages(value: str) -> List[Dict[str, str]]:
    """Best-effort parser for older human-readable ``[P1] title (chunk_id)`` text."""
    passages: List[Dict[str, str]] = []
    # Titles may contain parentheses, so capture only the final id-like suffix.
    pattern = re.compile(
        r"^\s*\[P[^\]]+\]\s*(?P<title>.*?)\s*\((?P<chunk>[A-Za-z0-9-]+_\d+)\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(value):
        chunk_id = match.group("chunk")
        paper_id = _paper_id_from_chunk_id(chunk_id)
        passages.append(
            {
                "chunk_id": chunk_id,
                "paper_id": paper_id or "",
                "paper_title": match.group("title").strip(),
            }
        )
    return passages


def _paper_id_from_chunk_id(chunk_id: str) -> str:
    """Infer paper ID from the conventional ``paper-id_ordinal`` chunk ID."""
    match = re.match(r"^(?P<paper>.+)_\d+$", _as_text(chunk_id))
    return match.group("paper") if match else ""


@dataclass
class GoldLabels:
    """Gold labels extracted from one benchmark row."""

    paper_grades: Dict[str, float] = field(default_factory=dict)
    chunk_grades: Dict[str, float] = field(default_factory=dict)
    paper_title_grades: Dict[str, float] = field(default_factory=dict)
    # Stable canonical targets prevent duplicate historical IDs for the same
    # paper from being counted as separate retrieval requirements.
    paper_id_target_keys: Dict[str, str] = field(default_factory=dict)
    paper_title_target_keys: Dict[str, str] = field(default_factory=dict)
    paper_target_grades: Dict[str, float] = field(default_factory=dict)

    def finalize_paper_targets(self) -> None:
        """Group ID/title aliases that refer to the same labelled paper."""
        for paper_id, grade in self.paper_grades.items():
            target = self.paper_id_target_keys.get(paper_id, f"id:{paper_id}")
            self.paper_target_grades[target] = max(
                self.paper_target_grades.get(target, 0.0), grade
            )
        for title, grade in self.paper_title_grades.items():
            target = self.paper_title_target_keys.get(title, f"title:{title}")
            self.paper_target_grades[target] = max(
                self.paper_target_grades.get(target, 0.0), grade
            )

    @property
    def has_papers(self) -> bool:
        return bool(self.paper_grades or self.paper_title_grades)

    @property
    def has_chunks(self) -> bool:
        return bool(self.chunk_grades)


def extract_gold_labels(record: Mapping[str, Any]) -> GoldLabels:
    """Extract paper/chunk labels from current and legacy benchmark formats.

    The current benchmark's authoritative paper labels are ``retrieval_targets``.
    ``supporting_passages`` provides exact chunk labels.  Additional explicit
    fields are accepted so a hand-curated label sidecar can add graded relevance
    without changing this evaluator.
    """
    labels = GoldLabels()

    for field in ("retrieval_targets", "gold_paper_ids", "source_paper_ids", "paper_ids"):
        _extract_id_grades(record.get(field), labels.paper_grades, _PAPER_ID_KEYS)

    # Support common sidecar forms, e.g. {"gold": {"paper_ids": [...], ...}}.
    gold_container = record.get("gold")
    if isinstance(gold_container, Mapping):
        for field in ("paper_ids", "papers", "retrieval_targets"):
            _extract_id_grades(gold_container.get(field), labels.paper_grades, _PAPER_ID_KEYS)
        for field in ("chunk_ids", "chunks", "supporting_chunks"):
            _extract_id_grades(gold_container.get(field), labels.chunk_grades, _CHUNK_ID_KEYS)

    for field in ("gold_chunk_ids", "supporting_chunk_ids", "chunk_ids"):
        _extract_id_grades(record.get(field), labels.chunk_grades, _CHUNK_ID_KEYS)

    passages = record.get("supporting_passages")
    if isinstance(passages, str):
        passages = _parse_legacy_supporting_passages(passages)
    for passage in _iter_values(passages):
        if not isinstance(passage, Mapping):
            continue
        grade = passage.get("grade", passage.get("relevance", 1.0))
        chunk_id = _as_text(passage.get("chunk_id") or passage.get("id"))
        paper_id = _as_text(passage.get("paper_id"))
        title = _as_text(passage.get("paper_title") or passage.get("title"))
        _add_grade(labels.chunk_grades, chunk_id, grade)
        _add_grade(labels.paper_grades, paper_id, grade)
        if not paper_id:
            _add_grade(labels.paper_grades, _paper_id_from_chunk_id(chunk_id), grade)
        normalized_title = normalise_label(title)
        if normalized_title:
            labels.paper_title_grades[normalized_title] = max(
                labels.paper_title_grades.get(normalized_title, 0.0), _positive_grade(grade)
            )
            target = f"title:{normalized_title}"
            labels.paper_title_target_keys[normalized_title] = target
            if paper_id:
                labels.paper_id_target_keys[paper_id] = target

    # ``source_papers`` is a useful title alias for legacy prediction exports.
    for source in _iter_values(record.get("source_papers")):
        if isinstance(source, Mapping):
            grade = source.get("grade", source.get("relevance", 1.0))
            paper_id = _as_text(source.get("paper_id") or source.get("id"))
            title = _as_text(source.get("paper_title") or source.get("title"))
            _add_grade(labels.paper_grades, paper_id, grade)
            normalized_title = normalise_label(title)
            if normalized_title:
                labels.paper_title_grades[normalized_title] = max(
                    labels.paper_title_grades.get(normalized_title, 0.0), _positive_grade(grade)
                )
                target = f"title:{normalized_title}"
                labels.paper_title_target_keys[normalized_title] = target
                if paper_id:
                    labels.paper_id_target_keys[paper_id] = target
        elif isinstance(source, str):
            # Old exports use one paper per line as "title — authors".
            for line in source.splitlines() or [source]:
                title = line.split(" — ", 1)[0].strip()
                normalized_title = normalise_label(title)
                if normalized_title:
                    labels.paper_title_grades[normalized_title] = max(
                        labels.paper_title_grades.get(normalized_title, 0.0), 1.0
                    )

    labels.finalize_paper_targets()
    return labels


def _parse_legacy_source_lines(value: str) -> List[Dict[str, str]]:
    """Parse a simple legacy source list without treating relevance as identity."""
    results: List[Dict[str, str]] = []
    for line in value.splitlines():
        text = line.strip()
        if not text:
            continue
        # ``1. Foo (rel=0.9)`` from the historical benchmark export.
        text = re.sub(r"^\d+\.\s*", "", text)
        text = re.sub(r"\s*\(rel(?:evance)?(?:_score)?\s*=\s*[-+]?\d*\.?\d+\)\s*$", "", text, flags=re.I)
        chunk_match = re.search(r"\(([A-Za-z0-9-]+_\d+)\)\s*$", text)
        if chunk_match:
            chunk_id = chunk_match.group(1)
            title = text[: chunk_match.start()].strip()
            results.append({"chunk_id": chunk_id, "paper_title": title})
        else:
            results.append({"title": text})
    return results


@dataclass(frozen=True)
class RetrievedItem:
    """A normalized ranked prediction item."""

    paper_id: str = ""
    chunk_id: str = ""
    title_aliases: Tuple[str, ...] = ()
    raw_rank: int = 0

    def identity(self, label_level: str) -> str:
        """Stable key used to remove duplicates before calculating ranking metrics."""
        if label_level == "chunk":
            return f"chunk:{normalise_label(self.chunk_id)}" if self.chunk_id else f"raw:{self.raw_rank}"
        if self.paper_id:
            return f"paper:{normalise_label(self.paper_id)}"
        inferred = _paper_id_from_chunk_id(self.chunk_id)
        if inferred:
            return f"paper:{normalise_label(inferred)}"
        if self.title_aliases:
            return f"title:{self.title_aliases[0]}"
        return f"raw:{self.raw_rank}"


def _first_text(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _as_text(mapping.get(key))
        if value:
            return value
    return ""


def _normalise_retrieved_item(raw: Any, raw_rank: int) -> RetrievedItem:
    if isinstance(raw, str):
        raw = {"title": raw}
    if not isinstance(raw, Mapping):
        return RetrievedItem(raw_rank=raw_rank)

    # Raw retriever output may put all identity fields under metadata.
    combined: Dict[str, Any] = {}
    metadata = raw.get("metadata")
    if isinstance(metadata, Mapping):
        combined.update(metadata)
    combined.update(raw)

    chunk_id = _first_text(combined, _CHUNK_ID_KEYS)
    paper_id = _first_text(combined, _PAPER_ID_KEYS)
    generic_id = _as_text(combined.get("id"))
    if not chunk_id and generic_id and _paper_id_from_chunk_id(generic_id):
        chunk_id = generic_id
    elif not paper_id and generic_id:
        # Chroma's id is normally a chunk ID; only treat a non-chunk generic ID
        # as a paper ID if an explicit chunk ID is absent.
        paper_id = generic_id
    if not paper_id:
        paper_id = _paper_id_from_chunk_id(chunk_id)

    aliases: List[str] = []
    for key in _TITLE_KEYS:
        _append_unique(aliases, combined.get(key))
    # Citation may include author/year before the title; retaining it as an alias
    # does no harm, while exact title/file aliases still take precedence.
    normalized_aliases = tuple(
        normalized
        for normalized in (normalise_label(alias) for alias in aliases)
        if normalized
    )
    return RetrievedItem(
        paper_id=paper_id,
        chunk_id=chunk_id,
        title_aliases=normalized_aliases,
        raw_rank=raw_rank,
    )


def extract_retrieved_items(record: Mapping[str, Any]) -> List[RetrievedItem]:
    """Find and normalize a ranked retrieval list from one prediction record.

    The first populated recognized field wins, so callers should provide one
    authoritative ranked list rather than a mix of candidates and final context.
    ``cysticcare_metadata.sources`` is supported for the existing generated
    benchmark output.
    """
    raw_items: Any = None
    for path in _PREDICTION_LIST_PATHS:
        candidate = _get_nested(record, path)
        if candidate not in (None, "", []):
            raw_items = candidate
            break
    if isinstance(raw_items, str):
        raw_items = _parse_legacy_source_lines(raw_items)
    if isinstance(raw_items, Mapping):
        # A source map keyed by rank or ID is accepted as a convenience.
        raw_items = list(raw_items.values())
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    return [_normalise_retrieved_item(item, rank) for rank, item in enumerate(raw_items, start=1)]


def _grade_for_item(item: RetrievedItem, labels: GoldLabels, label_level: str) -> float:
    if label_level == "chunk":
        return labels.chunk_grades.get(item.chunk_id, 0.0)

    target = _relevant_target_key(item, labels, label_level)
    return labels.paper_target_grades.get(target, 0.0) if target else 0.0


def _gold_target_count(labels: GoldLabels, label_level: str) -> int:
    if label_level == "chunk":
        return len(labels.chunk_grades)
    return len(labels.paper_target_grades)


def _deduplicate(
    items: Sequence[RetrievedItem], label_level: str, labels: Optional[GoldLabels] = None
) -> List[RetrievedItem]:
    seen: Set[str] = set()
    unique: List[RetrievedItem] = []
    for item in items:
        # If the benchmark can prove two representations resolve to the same
        # target paper, collapse them even when one result has only a legacy
        # title and another has a stable paper ID.
        matched_target = (
            _relevant_target_key(item, labels, label_level) if labels is not None else ""
        )
        identity = f"target:{matched_target}" if matched_target else item.identity(label_level)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _relevant_target_key(item: RetrievedItem, labels: GoldLabels, label_level: str) -> str:
    """Return the canonical gold target key matched by a relevant item."""
    if label_level == "chunk":
        return item.chunk_id if item.chunk_id in labels.chunk_grades else ""
    for paper_id in (item.paper_id, _paper_id_from_chunk_id(item.chunk_id)):
        if paper_id in labels.paper_grades:
            return labels.paper_id_target_keys.get(paper_id, f"id:{paper_id}")
    for alias in item.title_aliases:
        if alias in labels.paper_title_grades:
            return labels.paper_title_target_keys.get(alias, f"title:{alias}")
    return ""


def evaluate_ranked_items(
    items: Sequence[RetrievedItem],
    labels: GoldLabels,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    label_level: str = "paper",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Calculate Recall@k, MRR@k, nDCG@k, and context precision@k.

    ``Recall@k`` is the fraction of unique labelled targets recovered.  ``MRR``
    uses the first relevant rank, ``nDCG`` uses explicit grades when supplied,
    and context precision is relevant selected context divided by the number of
    returned (not requested) items.  Duplicate papers/chunks are removed before
    evaluation so repeated context cannot inflate scores.
    """
    if label_level not in {"paper", "chunk"}:
        raise ValueError("label_level must be 'paper' or 'chunk' for metric calculation")
    clean_cutoffs = _validate_cutoffs(cutoffs)
    target_count = _gold_target_count(labels, label_level)
    if target_count == 0:
        return {
            str(k): {
                "recall": None,
                "mrr": None,
                "ndcg": None,
                "context_precision": None,
            }
            for k in clean_cutoffs
        }

    ranked = _deduplicate(items, label_level, labels)
    grades = [_grade_for_item(item, labels, label_level) for item in ranked]
    target_keys = [_relevant_target_key(item, labels, label_level) for item in ranked]
    if label_level == "chunk":
        ideal_grades = sorted(labels.chunk_grades.values(), reverse=True)
    else:
        # Use the same canonical targets as recall.  Historical exports can
        # expose two paper IDs for one title; counting both in the ideal DCG
        # would unfairly penalize a retriever that correctly returns that
        # paper once.
        ideal_grades = sorted(labels.paper_target_grades.values(), reverse=True)

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for k in clean_cutoffs:
        top_grades = grades[:k]
        top_keys = {key for key in target_keys[:k] if key}
        retrieved_count = len(top_grades)
        relevant_count = sum(1 for grade in top_grades if grade > 0)
        first_relevant_rank = next(
            (rank for rank, grade in enumerate(top_grades, start=1) if grade > 0), None
        )
        dcg = sum(
            (math.pow(2.0, grade) - 1.0) / math.log2(rank + 1)
            for rank, grade in enumerate(top_grades, start=1)
        )
        ideal_dcg = sum(
            (math.pow(2.0, grade) - 1.0) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades[:k], start=1)
        )
        results[str(k)] = {
            # A legacy title alias can occasionally coexist with a stable ID
            # without a mapping between them. Metrics are bounded by definition
            # even in that imperfect historical data shape.
            "recall": min(1.0, len(top_keys) / target_count),
            "mrr": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
            "ndcg": min(1.0, dcg / ideal_dcg) if ideal_dcg else 0.0,
            "context_precision": (relevant_count / retrieved_count) if retrieved_count else 0.0,
        }
    return results


def _validate_cutoffs(cutoffs: Sequence[int]) -> Tuple[int, ...]:
    parsed: List[int] = []
    for cutoff in cutoffs:
        try:
            value = int(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cutoff: {cutoff!r}") from exc
        if value <= 0:
            raise ValueError(f"Cutoffs must be positive integers, got {value}")
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one cutoff is required")
    return tuple(sorted(parsed))


def _select_label_level(
    requested_level: str, labels: GoldLabels, items: Sequence[RetrievedItem]
) -> Optional[str]:
    if requested_level not in SUPPORTED_LABEL_LEVELS:
        raise ValueError(
            f"label_level must be one of {', '.join(sorted(SUPPORTED_LABEL_LEVELS))}"
        )
    if requested_level == "paper":
        return "paper" if labels.has_papers else None
    if requested_level == "chunk":
        return "chunk" if labels.has_chunks else None
    # Auto chooses the stricter passage metric only when predictions actually
    # contain stable chunk IDs; otherwise paper-level output is more meaningful.
    if labels.has_chunks and any(item.chunk_id for item in items):
        return "chunk"
    return "paper" if labels.has_papers else None


def _mean_metric(rows: Sequence[Dict[str, Dict[str, Optional[float]]]], cutoff: str, name: str) -> Optional[float]:
    values = [row[cutoff][name] for row in rows if row[cutoff][name] is not None]
    return (sum(values) / len(values)) if values else None


def evaluate_records(
    benchmark_records: Sequence[Mapping[str, Any]],
    prediction_records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    label_level: str = "paper",
    include_per_query: bool = False,
) -> Dict[str, Any]:
    """Evaluate ranked predictions against a benchmark and return JSON-safe data.

    If ``prediction_records`` is omitted, retrieval fields are read directly
    from the benchmark rows.  This supports a single output file containing both
    benchmark labels and CysticCare responses/metadata.
    """
    clean_cutoffs = _validate_cutoffs(cutoffs)
    if label_level not in SUPPORTED_LABEL_LEVELS:
        raise ValueError(
            f"label_level must be one of {', '.join(sorted(SUPPORTED_LABEL_LEVELS))}"
        )

    predictions_by_key: Dict[str, Mapping[str, Any]] = {}
    duplicate_prediction_keys: List[str] = []
    if prediction_records is not None:
        for prediction in prediction_records:
            key = record_key(prediction)
            if not key:
                continue
            if key in predictions_by_key:
                duplicate_prediction_keys.append(key)
                continue
            predictions_by_key[key] = prediction

    metric_rows: List[Dict[str, Dict[str, Optional[float]]]] = []
    per_query: List[Dict[str, Any]] = []
    skipped = Counter()
    diagnostics = Counter()
    label_level_counts = Counter()
    seen_benchmark_keys: Set[str] = set()

    for index, benchmark in enumerate(benchmark_records):
        key = record_key(benchmark)
        if key and key in seen_benchmark_keys:
            skipped["duplicate_benchmark_key"] += 1
            continue
        if key:
            seen_benchmark_keys.add(key)
        prediction = predictions_by_key.get(key) if prediction_records is not None else benchmark
        prediction_is_missing = prediction is None
        if prediction_is_missing:
            diagnostics["missing_prediction"] += 1
            # Missing output is a retrieval failure, not an excuse to remove a
            # labelled benchmark query from the aggregate.
            prediction = {}
        labels = extract_gold_labels(benchmark)
        items = extract_retrieved_items(prediction)
        active_level = _select_label_level(label_level, labels, items)
        if active_level is None:
            skipped["missing_gold_labels"] += 1
            continue
        if not items and not prediction_is_missing:
            diagnostics["empty_prediction"] += 1
            # Empty retrieval is still a valid failed query and must lower scores.
        metrics = evaluate_ranked_items(items, labels, clean_cutoffs, active_level)
        metric_rows.append(metrics)
        label_level_counts[active_level] += 1
        if include_per_query:
            per_query.append(
                {
                    "record_key": key or f"row:{index}",
                    "question": _as_text(benchmark.get("question")),
                    "label_level": active_level,
                    "gold_target_count": _gold_target_count(labels, active_level),
                    "retrieved_count": len(_deduplicate(items, active_level, labels)),
                    "metrics": metrics,
                }
            )

    aggregate = {
        str(k): {
            metric: _mean_metric(metric_rows, str(k), metric)
            for metric in ("recall", "mrr", "ndcg", "context_precision")
        }
        for k in clean_cutoffs
    }
    report: Dict[str, Any] = {
        "label_level_requested": label_level,
        "cutoffs": list(clean_cutoffs),
        "queries_evaluated": len(metric_rows),
        "benchmark_records": len(benchmark_records),
        "prediction_records": len(prediction_records) if prediction_records is not None else len(benchmark_records),
        "label_level_counts": dict(sorted(label_level_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "prediction_diagnostics": dict(sorted(diagnostics.items())),
        "duplicate_prediction_keys": sorted(set(duplicate_prediction_keys)),
        "metrics": aggregate,
    }
    if include_per_query:
        report["per_query"] = per_query
    return report


def parse_cutoffs(value: str) -> Tuple[int, ...]:
    """Parse a CLI-friendly comma-separated cutoff string such as ``1,5,10``."""
    return _validate_cutoffs([part.strip() for part in value.split(",") if part.strip()])


def write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a deterministic, human-readable JSON report."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
