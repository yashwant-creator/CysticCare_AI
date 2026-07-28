#!/usr/bin/env python3
"""Benchmark retrieval directly against a local, ready Chroma collection.

Unlike ``generate_cysticcare_responses.py``, this command never calls the
chatbot's HTTP endpoint and never generates an answer.  It runs the same
query-embedding -> hybrid (dense + BM25/RRF) -> reranking path against the
specified local vector database, then evaluates the ranked chunks with
``app.retrieval_evaluation``.

The output trace is a JSON list with a ``retrieved`` field on each row, so it
can be evaluated again with ``evaluate_retrieval.py``.  The metrics output is
a JSON evaluator report augmented with reproducibility metadata.

Examples
--------

  # Benchmark MedCPT locally.  This does not call /chat or Cloud Run.
  cd backend
  python run_retrieval_benchmark.py \
      --chroma-path app/openai_chroma_data \
      --reranker medcpt \
      --trace-output app/retrieval_trace_medcpt.json \
      --metrics-output app/retrieval_metrics_medcpt.json

  # Compare the hybrid retriever with no second-stage reranker.
  python run_retrieval_benchmark.py --reranker off \
      --trace-output app/retrieval_trace_hybrid.json \
      --metrics-output app/retrieval_metrics_hybrid.json

Query embeddings are still produced by the configured OpenAI embedding model;
``--reranker medcpt`` performs no LLM reranking and explicitly disables its
LLM fallback.  ``--reranker llm`` makes one direct helper-model judgement per
query.  No mode uses ``CYSTICCARE_BACKEND_URL`` or a deployed chatbot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    # Make ``python backend/run_retrieval_benchmark.py`` work from the project
    # root as well as ``python run_retrieval_benchmark.py`` from backend/.
    sys.path.insert(0, str(HERE))

DEFAULT_BENCHMARK = HERE / "app" / "pipeline_results.json"
DEFAULT_CHROMA_PATH = HERE / "app" / "openai_chroma_data"
DEFAULT_COLLECTION = "pkd_knowledge_base_openai"
DEFAULT_TRACE_OUTPUT = HERE / "app" / "retrieval_trace_medcpt.json"
DEFAULT_METRICS_OUTPUT = HERE / "app" / "retrieval_metrics_medcpt.json"
DEFAULT_CUTOFFS = "1,3,5,10"


class BenchmarkError(RuntimeError):
    """An actionable validation or retrieval failure for this benchmark."""


@dataclass(frozen=True)
class ReadyCollection:
    """The validated local collection and its non-sensitive run metadata."""

    client: Any
    collection: Any
    vector_count: int
    metadata: Dict[str, Any]
    path: Path
    name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK),
        help="Labelled benchmark JSON/JSONL/CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Persistent local Chroma directory; never a service URL (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Existing, ready Chroma collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--reranker",
        choices=("off", "medcpt", "llm"),
        default="medcpt",
        help=(
            "Second-stage reranker: off = hybrid only; medcpt = local cross-encoder "
            "with no LLM fallback; llm = direct helper-model judge (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Final chunks passed to the evaluator, from 1 to 20 (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        default=DEFAULT_CUTOFFS,
        help="Evaluation cutoffs, comma-separated; each must be <= --top-k (default: %(default)s)",
    )
    parser.add_argument(
        "--label-level",
        choices=("paper", "chunk", "auto"),
        default="paper",
        help="Evaluate paper labels, exact chunks, or auto-select the strictest usable level",
    )
    parser.add_argument(
        "--embedding-model",
        help=(
            "Override OPENAI_EMBEDDING_MODEL for query embeddings. It must match the "
            "model/dimension used to build the collection."
        ),
    )
    parser.add_argument(
        "--query-suffix",
        default="",
        help=(
            "Optional text appended to every benchmark question. Default is empty so the "
            "benchmark measures its original queries."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N benchmark rows (0 = all rows; default: %(default)s)",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Include per-query metric rows in the metrics JSON.",
    )
    parser.add_argument(
        "--trace-output",
        default=str(DEFAULT_TRACE_OUTPUT),
        help="Write evaluator-compatible ranked retrieval trace here (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(DEFAULT_METRICS_OUTPUT),
        help="Write JSON metric report here (default: %(default)s)",
    )
    return parser


def _normalise_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _as_nonempty_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _display_path(path: Path) -> str:
    """Keep paths readable and deterministic in JSON reports."""
    return str(path)


def load_ready_collection(
    chroma_path: str | Path,
    collection_name: str,
    *,
    client_factory: Optional[Callable[..., Any]] = None,
) -> ReadyCollection:
    """Open an existing local collection only after strict readiness checks.

    A runner must never benchmark an empty or partially rebuilt index: both
    would create misleading accuracy metrics.  This deliberately uses
    ``get_collection`` rather than a get-or-create operation, so a typo cannot
    silently create a blank collection.
    """
    path = _normalise_path(chroma_path)
    name = _as_nonempty_text(collection_name)
    if not name:
        raise BenchmarkError("--collection must not be empty")
    if not path.exists():
        raise BenchmarkError(f"Chroma path does not exist: {path}")
    if not path.is_dir():
        raise BenchmarkError(f"Chroma path is not a directory: {path}")

    if client_factory is None:
        try:
            import chromadb
        except ImportError as error:  # pragma: no cover - depends on install
            raise BenchmarkError(
                "ChromaDB is not installed; install backend/app/requirements_openai.txt"
            ) from error
        client_factory = chromadb.PersistentClient

    try:
        client = client_factory(path=str(path))
        collection = client.get_collection(name=name)
    except Exception as error:  # Chroma error classes vary by installed version.
        raise BenchmarkError(
            f"Could not open collection {name!r} at {path}: {type(error).__name__}: {error}"
        ) from error

    try:
        vector_count = int(collection.count())
    except Exception as error:
        raise BenchmarkError(
            f"Could not count collection {name!r}: {type(error).__name__}: {error}"
        ) from error
    if vector_count <= 0:
        raise BenchmarkError(
            f"Collection {name!r} is empty ({vector_count} vectors); rebuild it before benchmarking"
        )

    raw_metadata = getattr(collection, "metadata", None) or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    build_state = _as_nonempty_text(metadata.get("index_build_state")).lower()
    if build_state != "ready":
        reported_state = repr(metadata.get("index_build_state", None))
        raise BenchmarkError(
            f"Collection {name!r} is not ready (index_build_state={reported_state}). "
            "It may be incomplete; finish a clean rebuild before benchmarking."
        )

    return ReadyCollection(
        client=client,
        collection=collection,
        vector_count=vector_count,
        metadata=metadata,
        path=path,
        name=name,
    )


def build_hybrid_retriever(
    collection: Any,
    *,
    retriever_factory: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Build BM25 state from the same local Chroma collection used for dense search."""
    if retriever_factory is None:
        from app.services.hybrid_retriever import HybridRetriever

        retriever_factory = HybridRetriever

    try:
        retriever = retriever_factory(collection)
    except Exception as error:
        raise BenchmarkError(
            f"Could not initialize the hybrid retriever: {type(error).__name__}: {error}"
        ) from error

    if not getattr(retriever, "initialized", False) or int(
        getattr(retriever, "corpus_size", 0) or 0
    ) <= 0:
        raise BenchmarkError(
            "Hybrid BM25 index did not initialize from the local Chroma collection; "
            "do not fall back to a different retrieval path for this benchmark."
        )
    return retriever


def create_openai_services(
    embedding_model: Optional[str] = None,
    *,
    include_llm_helper: bool = False,
    service_factory: Optional[Callable[..., Any]] = None,
    session_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Optional[Any]]:
    """Create direct provider clients without starting or calling an HTTP chatbot.

    The first return value is always used for embeddings.  The second exists
    only for explicit ``--reranker llm`` runs and honours the same
    ``HELPER_CHAT_MODEL`` setting as the application.
    """
    if session_config is None:
        from app.utils.openai_utils import load_session_config

        session_config = load_session_config()
    if service_factory is None:
        from app.services.openai_service import OpenAIService

        service_factory = OpenAIService

    configured_embedding = _as_nonempty_text(embedding_model) or _as_nonempty_text(
        session_config.get("embedding_model")
    )
    if not configured_embedding:
        raise BenchmarkError("No embedding model is configured")
    base_chat_model = _as_nonempty_text(os.getenv("BASE_CHAT_MODEL")) or _as_nonempty_text(
        session_config.get("chat_model")
    )

    common = {
        "embedding_model": configured_embedding,
        "vision_model": _as_nonempty_text(session_config.get("vision_model")),
        "max_retries": session_config.get("max_retries", 3),
        "retry_delay": session_config.get("retry_delay", 2),
    }
    try:
        embedding_service = service_factory(chat_model=base_chat_model, **common)
        if not include_llm_helper:
            return embedding_service, None

        helper_model = _as_nonempty_text(os.getenv("HELPER_CHAT_MODEL"))
        if not helper_model or helper_model == base_chat_model:
            return embedding_service, embedding_service
        helper_service = service_factory(chat_model=helper_model, **common)
        return embedding_service, helper_service
    except Exception as error:
        raise BenchmarkError(
            f"Could not configure direct embedding/reranker service: {type(error).__name__}: {error}"
        ) from error


def build_reranker(
    mode: str,
    llm_service: Optional[Any],
    *,
    reranker_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Create a deliberately explicit reranker configuration for a comparison run."""
    selected_mode = _as_nonempty_text(mode).lower()
    if selected_mode not in {"off", "medcpt", "llm"}:
        raise BenchmarkError(f"Unsupported reranker mode: {mode!r}")

    from app.services.semantic_reranker import RerankerConfig

    if reranker_factory is None:
        from app.services.semantic_reranker import create_evidence_reranker

        reranker_factory = create_evidence_reranker

    base_config = RerankerConfig.from_environment()
    if selected_mode == "off":
        config = replace(
            base_config,
            backend="off",
            enabled=False,
            fallback_to_llm=False,
        )
        service = None
    elif selected_mode == "medcpt":
        # This override is important: benchmark results for MedCPT must not
        # silently become LLM-reranker results if the local model cannot load.
        config = replace(
            base_config,
            backend="medcpt",
            enabled=True,
            fallback_to_llm=False,
        )
        service = None
    else:
        if llm_service is None:
            raise BenchmarkError("--reranker llm requires a configured helper OpenAI service")
        config = replace(
            base_config,
            backend="llm",
            enabled=True,
            fallback_to_llm=False,
        )
        service = llm_service

    try:
        return reranker_factory(service, config=config)
    except Exception as error:
        raise BenchmarkError(
            f"Could not configure the {selected_mode} reranker: {type(error).__name__}: {error}"
        ) from error


def _question_for_record(record: Mapping[str, Any], index: int) -> str:
    for field in ("question", "query"):
        value = _as_nonempty_text(record.get(field))
        if value:
            return value
    raise BenchmarkError(f"Benchmark row {index} has no non-empty question/query field")


def _prediction_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy evaluator join keys without inventing IDs that would not match gold rows."""
    identity: Dict[str, Any] = {}
    for field in ("id", "question_id", "query_id"):
        value = record.get(field)
        if _as_nonempty_text(value):
            identity[field] = value
            break
    return identity


def _preview(value: Any, limit: int = 400) -> str:
    text = " ".join(_as_nonempty_text(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _trace_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain stable identity and scoring fields while keeping trace files manageable."""
    raw_metadata = result.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    candidate_id = _as_nonempty_text(result.get("id"))
    chunk_id = _as_nonempty_text(metadata.get("chunk_id")) or candidate_id
    paper_id = _as_nonempty_text(metadata.get("paper_id"))

    traced: Dict[str, Any] = {
        "id": candidate_id,
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "metadata": metadata,
        "document_preview": _preview(result.get("document")),
    }
    for score_field in (
        "relevance_score",
        "dense_score",
        "sparse_score",
        "rrf_score",
        "fused_score",
        "reranker_score",
        "reranker_raw_score",
        "reranker_rank",
    ):
        if score_field in result:
            traced[score_field] = result[score_field]
    return traced


def run_retrieval_queries(
    benchmark_records: Sequence[Mapping[str, Any]],
    *,
    retriever: Any,
    embedding_service: Any,
    reranker: Any,
    top_k: int,
    query_suffix: str = "",
    expected_reranker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run only retrieval for each benchmark question and return ranked trace rows.

    Any failed embedding or retrieval aborts the run instead of becoming an
    unlabelled empty prediction.  When a specific LLM or MedCPT reranker is
    requested, a fail-open reranker error also aborts: otherwise a hybrid-only
    result could be incorrectly reported as that reranker's score.
    """
    if not 1 <= int(top_k) <= 20:
        raise BenchmarkError("--top-k must be an integer between 1 and 20")

    try:
        configured_candidates = int(getattr(reranker.config, "candidate_limit", top_k))
    except (AttributeError, TypeError, ValueError):
        configured_candidates = top_k
    candidate_limit = max(int(top_k), configured_candidates)
    # Match ``search_knowledge_base``: pull a broad candidate set, then let
    # the selected reranker decide the final context-safe ranking.
    candidate_pool_limit = max(candidate_limit, 40)

    traces: List[Dict[str, Any]] = []
    for index, record in enumerate(benchmark_records):
        question = _question_for_record(record, index)
        retrieval_query = question + query_suffix
        started = time.perf_counter()
        try:
            query_embedding = embedding_service.get_embedding(retrieval_query)
            if not isinstance(query_embedding, Sequence) or isinstance(
                query_embedding, (str, bytes)
            ) or not query_embedding:
                raise ValueError("embedding service returned an empty or invalid vector")
            candidates = retriever.hybrid_search(
                query=retrieval_query,
                query_embedding=list(query_embedding),
                top_k=int(top_k),
                candidate_pool_limit=candidate_pool_limit,
                result_limit=candidate_limit,
            )
            if not candidates:
                raise ValueError(
                    "hybrid retrieval returned no candidates from a non-empty ready collection"
                )
            selected, retrieval_metadata = reranker.rerank(
                query=retrieval_query,
                candidates=candidates,
                top_k=int(top_k),
            )
            if (
                expected_reranker in {"medcpt", "llm"}
                and isinstance(retrieval_metadata, Mapping)
                and retrieval_metadata.get("reranker_error")
            ):
                raise ValueError(
                    f"{expected_reranker} reranker was unavailable "
                    f"({retrieval_metadata['reranker_error']})"
                )
        except Exception as error:
            snippet = _preview(question, 100)
            raise BenchmarkError(
                f"Retrieval failed at benchmark row {index} ({snippet!r}): "
                f"{type(error).__name__}: {str(error)[:300]}"
            ) from error

        trace: Dict[str, Any] = {
            **_prediction_identity(record),
            # Preserve the original question for evaluator joins. The executed
            # query is separate, so an optional suffix cannot break matching.
            "question": question,
            "retrieval_query": retrieval_query,
            "benchmark_index": index,
            "candidate_count": len(candidates),
            "retrieved": [_trace_result(item) for item in selected],
            "retrieval_metadata": dict(retrieval_metadata or {}),
            "elapsed_ms": round((time.perf_counter() - started) * 1_000, 2),
        }
        traces.append(trace)
    return traces


def _write_json_atomic(payload: Any, output_path: str | Path) -> Path:
    path = _normalise_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BenchmarkError(f"Could not write {path}: {error}") from error
    return path


def _validate_cli_arguments(args: argparse.Namespace) -> Tuple[int, ...]:
    from app.retrieval_evaluation import parse_cutoffs

    if not 1 <= args.top_k <= 20:
        raise BenchmarkError("--top-k must be an integer between 1 and 20")
    if args.limit < 0:
        raise BenchmarkError("--limit must be zero or a positive integer")
    try:
        cutoffs = parse_cutoffs(args.k)
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    if max(cutoffs) > args.top_k:
        raise BenchmarkError(
            f"--top-k ({args.top_k}) must be at least the largest requested cutoff ({max(cutoffs)})"
        )

    benchmark_path = _normalise_path(args.benchmark)
    trace_path = _normalise_path(args.trace_output)
    metrics_path = _normalise_path(args.metrics_output)
    if trace_path == metrics_path:
        raise BenchmarkError("--trace-output and --metrics-output must be different paths")
    if trace_path == benchmark_path or metrics_path == benchmark_path:
        raise BenchmarkError("Benchmark input must not be overwritten by an output path")
    return cutoffs


def _load_dotenv_if_available() -> None:
    """Load backend/.env for local runs without making python-dotenv mandatory for --help/tests."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(HERE / ".env")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cutoffs = _validate_cli_arguments(args)
        _load_dotenv_if_available()

        from app.retrieval_evaluation import evaluate_records, load_records

        benchmark_records = load_records(args.benchmark)
        if args.limit:
            benchmark_records = benchmark_records[: args.limit]
        if not benchmark_records:
            raise BenchmarkError("No benchmark records were selected")

        ready = load_ready_collection(args.chroma_path, args.collection)
        retriever = build_hybrid_retriever(ready.collection)

        # Construct direct provider services only after local benchmark/index
        # validation. A partial collection therefore never triggers an API call.
        embedding_service, llm_service = create_openai_services(
            args.embedding_model,
            include_llm_helper=args.reranker == "llm",
        )
        reranker = build_reranker(args.reranker, llm_service)
        trace_records = run_retrieval_queries(
            benchmark_records,
            retriever=retriever,
            embedding_service=embedding_service,
            reranker=reranker,
            top_k=args.top_k,
            query_suffix=args.query_suffix,
            expected_reranker=args.reranker,
        )
        report = evaluate_records(
            benchmark_records,
            trace_records,
            cutoffs=cutoffs,
            label_level=args.label_level,
            include_per_query=args.per_query,
        )
        if not report.get("queries_evaluated"):
            raise BenchmarkError(
                "The selected benchmark contains no usable retrieval labels; no accuracy metric was produced"
            )

        run_metadata = {
            "runner": "direct_local_chroma_retrieval",
            "benchmark_path": _display_path(_normalise_path(args.benchmark)),
            "benchmark_records_selected": len(benchmark_records),
            "chroma_path": _display_path(ready.path),
            "collection": ready.name,
            "collection_vector_count": ready.vector_count,
            "collection_metadata": ready.metadata,
            "top_k": args.top_k,
            "cutoffs": list(cutoffs),
            "reranker": args.reranker,
            "query_suffix": args.query_suffix,
            "embedding_model": getattr(embedding_service, "embedding_model", args.embedding_model),
            "http_chatbot_endpoint_used": False,
        }
        metrics_payload = dict(report)
        metrics_payload["run_metadata"] = run_metadata

        trace_path = _write_json_atomic(trace_records, args.trace_output)
        metrics_path = _write_json_atomic(metrics_payload, args.metrics_output)

        print(f"Wrote retrieval trace: {trace_path}")
        print(f"Wrote retrieval metrics: {metrics_path}")
        print(
            f"Evaluated {report['queries_evaluated']} labelled queries from "
            f"{len(benchmark_records)} benchmark rows; reranker={args.reranker}, top_k={args.top_k}."
        )
        for cutoff in cutoffs:
            metric = report["metrics"][str(cutoff)]
            print(
                f"@{cutoff}: recall={metric['recall']:.4f}, "
                f"context_precision={metric['context_precision']:.4f}, "
                f"mrr={metric['mrr']:.4f}, ndcg={metric['ndcg']:.4f}"
            )
        return 0
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: benchmark interrupted; no completed metrics report was written", file=sys.stderr)
        return 130
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        print(f"error: unexpected {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
