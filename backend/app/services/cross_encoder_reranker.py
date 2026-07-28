"""Self-hosted cross-encoder reranking for research-paper evidence.

The MedCPT model is a biomedical query--passage cross-encoder.  It is kept
behind a lazy import so the ordinary LLM reranker remains usable in lightweight
development and test environments.  MedCPT emits *raw logits*, not calibrated
probabilities; ranking always uses those logits and any rejection threshold is
an explicit, separately calibrated raw-logit value.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .semantic_reranker import RerankerConfig

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


def _optional_env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; disabling its threshold", name)
        return None


def _env_nonnegative_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


@dataclass(frozen=True)
class CrossEncoderConfig:
    """Runtime controls for the local MedCPT cross-encoder.

    ``min_logit`` deliberately defaults to ``None``.  MedCPT scores are raw
    logits whose scale is not a portable confidence probability, so reusing
    the LLM reranker's 0.50 cutoff would create a false safety signal.  Set a
    value only after calibrating it on held-out PKD retrieval labels.
    """

    enabled: bool = True
    candidate_limit: int = 24
    max_chunks_per_document: int = 2
    max_chars_per_candidate: int = 1_200
    model_name: str = "ncbi/MedCPT-Cross-Encoder"
    revision: Optional[str] = None
    max_length: int = 512
    batch_size: int = 8
    device: str = "auto"
    local_files_only: bool = False
    min_logit: Optional[float] = None
    min_logit_margin: float = 0.0

    @classmethod
    def from_environment(
        cls,
        base_config: Optional[RerankerConfig] = None,
    ) -> "CrossEncoderConfig":
        base = base_config or RerankerConfig.from_environment()
        model_name = os.getenv(
            "RAG_CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder"
        ).strip()
        return cls(
            enabled=base.enabled,
            candidate_limit=base.candidate_limit,
            max_chunks_per_document=base.max_chunks_per_document,
            max_chars_per_candidate=base.max_chars_per_candidate,
            model_name=model_name or "ncbi/MedCPT-Cross-Encoder",
            revision=os.getenv("RAG_CROSS_ENCODER_REVISION", "").strip() or None,
            max_length=min(512, _env_int("RAG_CROSS_ENCODER_MAX_LENGTH", 512, 64)),
            batch_size=min(64, _env_int("RAG_CROSS_ENCODER_BATCH_SIZE", 8, 1)),
            device=os.getenv("RAG_CROSS_ENCODER_DEVICE", "auto").strip() or "auto",
            local_files_only=_env_bool("RAG_CROSS_ENCODER_LOCAL_ONLY", False),
            min_logit=_optional_env_float("RAG_CROSS_ENCODER_MIN_LOGIT"),
            min_logit_margin=_env_nonnegative_float(
                "RAG_CROSS_ENCODER_MIN_LOGIT_MARGIN", 0.0
            ),
        )


class MedCPTCrossEncoder:
    """Lazy, process-cached wrapper around Hugging Face MedCPT inference."""

    _cache: Dict[Tuple[str, Optional[str], str, bool], Tuple[Any, Any, Any, str]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: CrossEncoderConfig):
        self.config = config

    @staticmethod
    def _resolve_device(torch_module: Any, requested: str) -> str:
        choice = (requested or "auto").strip().lower()
        if choice != "auto":
            return choice
        if getattr(torch_module.cuda, "is_available", lambda: False)():
            return "cuda"
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps and mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def _load_model(
        cls,
        config: CrossEncoderConfig,
    ) -> Tuple[Any, Any, Any, str]:
        """Load one immutable model/tokenizer pair per process and device."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Cross-encoder dependencies are unavailable; install torch and transformers"
            ) from error

        device = cls._resolve_device(torch, config.device)
        key = (config.model_name, config.revision, device, config.local_files_only)
        with cls._cache_lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
                revision=config.revision,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
                revision=config.revision,
            )
            model.to(device)
            model.eval()
            loaded = (torch, tokenizer, model, device)
            cls._cache[key] = loaded
            logger.info(
                "Loaded MedCPT cross-encoder model=%s revision=%s device=%s",
                config.model_name,
                config.revision or "default",
                device,
            )
            return loaded

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        """Return one MedCPT raw relevance logit for every query/passage pair."""
        if not passages:
            return []
        torch, tokenizer, model, device = self._load_model(self.config)
        logits: List[float] = []
        for start in range(0, len(passages), self.config.batch_size):
            batch = list(passages[start : start + self.config.batch_size])
            pairs = [[query, passage] for passage in batch]
            encoded = tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.config.max_length,
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                batch_logits = model(**encoded).logits
            values = batch_logits.detach().to("cpu").reshape(-1).tolist()
            logits.extend(float(value) for value in values)

        if len(logits) != len(passages):
            raise ValueError(
                f"Cross-encoder returned {len(logits)} scores for {len(passages)} passages"
            )
        if not all(math.isfinite(score) for score in logits):
            raise ValueError("Cross-encoder returned non-finite relevance scores")
        return logits


class CrossEncoderReranker:
    """Rerank evidence with MedCPT while preserving RAG diversity safeguards."""

    backend_name = "medcpt"

    def __init__(
        self,
        config: Optional[CrossEncoderConfig] = None,
        scorer: Optional[Any] = None,
    ):
        self.config = config or CrossEncoderConfig.from_environment()
        self.scorer = scorer or MedCPTCrossEncoder(self.config)

    @staticmethod
    def _document_key(candidate: Dict[str, Any]) -> str:
        metadata = candidate.get("metadata") or {}
        return str(
            metadata.get("paper_id")
            or metadata.get("parent_document_id")
            or metadata.get("document_key")
            or metadata.get("file_name")
            or candidate.get("id")
            or "unknown"
        )

    def _format_candidate(self, candidate: Dict[str, Any]) -> str:
        metadata = candidate.get("metadata") or {}
        text = " ".join(str(candidate.get("document", "")).split())
        if len(text) > self.config.max_chars_per_candidate:
            text = text[: self.config.max_chars_per_candidate].rsplit(" ", 1)[0] + "…"
        title = metadata.get("title") or metadata.get("display_name") or "Unknown paper"
        section = metadata.get("section_heading") or "Unspecified section"
        page = metadata.get("page_number") or metadata.get("page_start") or "unknown"
        return f"Paper: {title}\nSection: {section}\nPage: {page}\nPassage: {text}"

    def _apply_document_diversity(
        self,
        candidates: Iterable[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            result_limit = int(top_k)
        except (TypeError, ValueError):
            return []
        if result_limit <= 0:
            return []

        selected: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for candidate in candidates:
            document_key = self._document_key(candidate)
            if counts.get(document_key, 0) >= self.config.max_chunks_per_document:
                continue
            counts[document_key] = counts.get(document_key, 0) + 1
            selected.append(candidate)
            if len(selected) >= result_limit:
                break
        return selected

    @staticmethod
    def _display_score(logit: float) -> float:
        """Stable display value only; it is not a calibrated probability."""
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
        exp_value = math.exp(max(logit, -60.0))
        return exp_value / (1.0 + exp_value)

    def _confidence(
        self,
        top_logit: float,
        margin: Optional[float],
    ) -> Tuple[str, bool]:
        threshold = self.config.min_logit
        if threshold is None:
            return "ranked_unthresholded", True
        if top_logit < threshold:
            return "insufficient", False
        if margin is not None and margin < self.config.min_logit_margin:
            return "ambiguous", True
        return "ranked", True

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidate_subset = list(candidates[: min(self.config.candidate_limit, 60)])
        base_metadata: Dict[str, Any] = {
            "reranker_enabled": self.config.enabled,
            "reranker_backend": self.backend_name,
            "candidate_count": len(candidate_subset),
            "threshold": self.config.min_logit,
            "threshold_type": "raw_logit" if self.config.min_logit is not None else "disabled",
            "min_score_margin": self.config.min_logit_margin,
            "reranker_used": False,
            "score_type": "fused_rank",
            "confidence": "unavailable",
            "accepted": bool(candidate_subset),
        }
        if not candidate_subset:
            base_metadata["selected_count"] = 0
            return [], base_metadata
        if not self.config.enabled:
            selected = self._apply_document_diversity(candidate_subset, top_k)
            base_metadata["selected_count"] = len(selected)
            return selected, base_metadata

        try:
            started = time.perf_counter()
            logits = self.scorer.score_pairs(
                query,
                [self._format_candidate(candidate) for candidate in candidate_subset],
            )
            latency_ms = round((time.perf_counter() - started) * 1_000, 2)
            if len(logits) != len(candidate_subset):
                raise ValueError(
                    f"Cross-encoder returned {len(logits)} scores for {len(candidate_subset)} candidates"
                )
            if not all(math.isfinite(float(logit)) for logit in logits):
                raise ValueError("Cross-encoder returned non-finite relevance scores")
        except Exception as error:
            logger.warning("Cross-encoder reranking failed; retaining fused candidates: %s", error)
            base_metadata.update(
                {
                    "reranker_error": "cross_encoder_unavailable",
                    "confidence": "fallback",
                }
            )
            selected = self._apply_document_diversity(candidate_subset, top_k)
            base_metadata["selected_count"] = len(selected)
            return selected, base_metadata

        ranked: List[Dict[str, Any]] = []
        for candidate, logit in zip(candidate_subset, logits):
            result = dict(candidate)
            result["metadata"] = dict(candidate.get("metadata") or {})
            raw_score = float(logit)
            display_score = self._display_score(raw_score)
            result["fused_score"] = candidate.get("relevance_score")
            result["reranker_raw_score"] = round(raw_score, 4)
            result["reranker_score"] = round(display_score, 4)
            # Keep the UI/source ordering field bounded, but never use this
            # sigmoid value as the evidence threshold.
            result["relevance_score"] = round(display_score, 4)
            result["_cross_encoder_sort_score"] = raw_score
            ranked.append(result)
        ranked.sort(key=lambda item: item["_cross_encoder_sort_score"], reverse=True)
        for rank, candidate in enumerate(ranked, start=1):
            candidate["reranker_rank"] = rank

        top_logit = ranked[0]["_cross_encoder_sort_score"]
        margin = (
            round(top_logit - ranked[1]["_cross_encoder_sort_score"], 4)
            if len(ranked) > 1
            else None
        )
        confidence, accepted = self._confidence(top_logit, margin)
        thresholded = (
            [
                candidate
                for candidate in ranked
                if candidate["_cross_encoder_sort_score"] >= self.config.min_logit
            ]
            if self.config.min_logit is not None
            else ranked
        )
        selected = self._apply_document_diversity(thresholded, top_k) if accepted else []
        for candidate in ranked:
            candidate.pop("_cross_encoder_sort_score", None)

        base_metadata.update(
            {
                "reranker_used": True,
                "score_type": "cross_encoder_logit",
                "score_normalization": "sigmoid_for_display_only",
                "top_raw_score": round(top_logit, 4),
                "top_score": ranked[0]["reranker_score"],
                "score_margin": margin,
                "confidence": confidence,
                "accepted": accepted,
                "selected_count": len(selected),
                "latency_ms": latency_ms,
            }
        )
        return selected, base_metadata
