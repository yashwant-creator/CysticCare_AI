"""Stable research-paper identities for retrieval metadata and evaluation.

Paper filenames are convenient operational identifiers but change when a PDF is
renamed. This module uses a normalized title as the stable source of truth and
optionally preserves IDs from the repository's existing retrieval benchmark.
The optional mapping affects identifiers only; it never supplies retrieval
content or relevance labels to the chatbot.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_benchmark_title_ids: Optional[Dict[str, str]] = None
_GENERIC_TITLE_IDENTITIES = frozenset({"unknown", "untitled", "na", "none", "null"})


def normalize_paper_identity(value: Any) -> str:
    """Normalize a title for deterministic identity matching, not fuzzy search."""
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^\w]+", "", text)


def _benchmark_path() -> Path:
    return Path(__file__).resolve().parent.parent / "pipeline_results.json"


def _load_benchmark_title_ids(path: Optional[Path] = None) -> Dict[str, str]:
    """Load a title -> historic paper ID map when the evaluation corpus exists."""
    source = path or _benchmark_path()
    try:
        with source.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.debug("No benchmark paper identity map at %s: %s", source, error)
        return {}

    if not isinstance(records, list):
        return {}

    title_ids: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        entries = []
        source_papers = record.get("source_papers", [])
        supporting_passages = record.get("supporting_passages", [])
        if isinstance(source_papers, list):
            entries.extend(source_papers)
        if isinstance(supporting_passages, list):
            entries.extend(supporting_passages)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = normalize_paper_identity(entry.get("paper_title") or entry.get("title"))
            paper_id = str(entry.get("paper_id") or "").strip()
            if title and paper_id:
                # A title should resolve consistently. Preserve the first map
                # rather than silently changing an established benchmark ID.
                title_ids.setdefault(title, paper_id)
    return title_ids


def benchmark_title_ids() -> Dict[str, str]:
    """Return the cached optional compatibility map for existing benchmarks."""
    global _benchmark_title_ids
    if _benchmark_title_ids is None:
        _benchmark_title_ids = _load_benchmark_title_ids()
    return _benchmark_title_ids


def stable_paper_id(title: Any, fallback_name: Any) -> str:
    """Return a title-stable paper ID, preserving a known benchmark ID if any."""
    normalized_title = normalize_paper_identity(title)
    # PDF metadata frequently reports a placeholder title. Treat it as absent
    # so unrelated files do not collapse into one ``Unknown`` paper during a
    # rebuild; the filename is the deterministic fallback in that situation.
    if normalized_title in _GENERIC_TITLE_IDENTITIES:
        normalized_title = ""
    known_id = benchmark_title_ids().get(normalized_title)
    if known_id:
        return known_id

    normalized_fallback = normalize_paper_identity(fallback_name)
    identity = normalized_title or normalized_fallback or "unknown-paper"
    # A readable prefix avoids an opaque standalone hash while keeping the ID
    # safe for Chroma and stable across runs and filename changes.
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"paper_{digest}"
