# Retrieval tuning

The retrieval pipeline now uses three stages:

1. Dense embedding search and BM25, fused with reciprocal-rank fusion (RRF),
   produce a broad candidate set.
2. A selected evidence reranker scores whether each candidate contains evidence
   for the exact question. The default is an LLM judge; the optional MedCPT
   biomedical cross-encoder is available for comparison.
3. Only passages that meet a calibrated evidence policy are sent to the answer
   model. The selected context is diversified across papers.

The RRF score is a relative rank and is **not** a confidence probability. Do
not use it as a cutoff. `RAG_MIN_RELEVANCE_SCORE` applies only when the semantic
reranker succeeded.

## Defaults and environment controls

| Variable | Default | Purpose |
| --- | ---: | --- |
| `RAG_RERANKER_BACKEND` | `llm` | `llm` (default), `medcpt`, or `off`. Keep `llm` until MedCPT is benchmark-calibrated. |
| `RAG_RERANK_ENABLED` | `true` | Enables semantic evidence reranking. |
| `RAG_RERANK_CANDIDATES` | `24` | Number of fused candidates judged by the reranker. |
| `RAG_RERANK_MAX_CHARS` | `1200` | Maximum passage characters shown to the reranker. |
| `RAG_RERANK_MAX_TOKENS` | `2000` | Completion-token budget for the batched semantic judge. |
| `RAG_MIN_RELEVANCE_SCORE` | `0.50` | LLM-only evidence cutoff. Tune from labelled evaluation, not intuition. |
| `RAG_MIN_SCORE_MARGIN` | `0.05` | Marks close top results as `ambiguous`; it does not discard usable evidence. |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | `2` | Prevents one paper from filling all answer context. |
| `RAG_CROSS_ENCODER_MODEL` | `ncbi/MedCPT-Cross-Encoder` | Biomedical query--passage model used when backend is `medcpt`. |
| `RAG_CROSS_ENCODER_REVISION` | unset | Pin a Hugging Face commit revision before production deployment. |
| `RAG_CROSS_ENCODER_MAX_LENGTH` | `512` | MedCPT token limit for each query/passage pair. |
| `RAG_CROSS_ENCODER_BATCH_SIZE` | `8` | Number of pairs scored together; start conservatively on CPU. |
| `RAG_CROSS_ENCODER_DEVICE` | `auto` | `cpu`, `cuda`, `mps`, or automatic selection. |
| `RAG_CROSS_ENCODER_LOCAL_ONLY` | `false` | Require a locally cached model instead of downloading it at runtime. |
| `RAG_CROSS_ENCODER_MIN_LOGIT` | unset | Optional **raw MedCPT-logit** cutoff; do not set until benchmark-calibrated. |
| `RAG_CROSS_ENCODER_MIN_LOGIT_MARGIN` | `0` | Optional raw-logit margin for marking close results as ambiguous. |
| `RAG_CROSS_ENCODER_FALLBACK_TO_LLM` | `true` | If MedCPT cannot load or score, use the existing LLM judge before falling back to RRF. |
| `RAG_HYBRID_CANDIDATE_POOL` | `60` | Dense/BM25 pool before fusion and reranking. |
| `RAG_DENSE_RRF_WEIGHT` | `1.0` | Dense-search contribution to RRF. |
| `RAG_SPARSE_RRF_WEIGHT` | `1.0` | BM25 contribution to RRF. |

Set `HELPER_CHAT_MODEL` to a lower-latency model that supports structured JSON
output if the answer model is expensive. The LLM reranker makes one batched
helper call per retrieval request. If that call fails, retrieval deliberately
falls back to the fused candidate order and records `confidence: "fallback"`;
it does not treat RRF as a threshold score.

## Try the MedCPT cross-encoder

MedCPT is a self-hosted biomedical cross-encoder. It scores each query and
passage directly, so it avoids the LLM rerank call and returns raw relevance
logits. `torch` and `transformers` are included in the backend requirements;
the model itself is loaded lazily and cached once per process.

For a local trial:

```bash
export RAG_RERANKER_BACKEND=medcpt
export RAG_CROSS_ENCODER_DEVICE=cpu
```

For production, bake/cache the exact model revision in the image or mounted
model volume and set `RAG_CROSS_ENCODER_LOCAL_ONLY=true`. Do not rely on a
first user request to download model weights. Start a CPU deployment with at
least 2 GiB memory and low concurrency, then measure p95 latency.

If MedCPT fails to load, the default `RAG_CROSS_ENCODER_FALLBACK_TO_LLM=true`
uses the prior LLM evidence judge. A successful MedCPT ranking is never
overridden by that fallback.

## Rebuild the index

The improved chunk structure is index schema version 2. Existing vectors are
not modified automatically. Rebuild deliberately, from `backend/`:

```bash
python rebuild_vectordb.py
```

This command deletes and recreates the local Chroma collection, so make a copy
of any collection that must be retained first. It verifies that source PDFs and
OpenAI embeddings are available before deletion and exits nonzero on an
incomplete rebuild. A running application exposes `requires_rebuild` through
its initialization response and collection stats when it detects an older or
interrupted collection.

The new index stores section/page metadata, stable `paper_id` values, and fresh
`chunk_id` values. Where a paper title exists in the existing benchmark, a
canonical historic paper ID is preserved for paper-level metric continuity.
Chunk-level labels must be regenerated because the new structure-aware chunks
intentionally have new boundaries.

## Calibrate the threshold

For the LLM backend, start with `0.50`, then run a held-out evaluation over a
range such as `0.35, 0.45, 0.50, 0.60, 0.70`. Choose the lowest threshold that
meets your medical-context precision target while retaining acceptable
Recall@k.

For MedCPT, leave `RAG_CROSS_ENCODER_MIN_LOGIT` unset for the first ranking
comparison. Its raw logits and their sigmoid display values are **not**
calibrated probabilities, so `0.50` is not a valid inherited cutoff. On a
held-out set, sweep observed raw logits and choose a cutoff only if it meets
the precision/recall target. A useful initial goal is high context precision
(for example, 90%+) without a material drop in Recall@20.

Use the evaluator after generating results with
`include_retrieval_debug: true`. After an index rebuild or tuning change, force
fresh benchmark calls rather than resuming stale outputs:

```bash
python generate_cysticcare_responses.py \
  --force \
  --output /tmp/pipeline_results_with_cysticcare_fresh.json

python evaluate_retrieval.py \
  --benchmark app/pipeline_results.json \
  --predictions /tmp/pipeline_results_with_cysticcare_fresh.json \
  --k 1,3,5,10 \
  --label-level paper \
  --per-query \
  --output /tmp/retrieval_metrics.json
```

`generate_cysticcare_responses.py` pins standard RAG (rather than adaptive,
CoT, or Stepback routing) so every evaluation row contains one meaningful
ranked list. Its default URL is the deployed service; set
`CYSTICCARE_BACKEND_URL` to the environment that actually contains the rebuilt
index before generating a fresh export.

Use `--label-level chunk` only after rebuilding the index and regenerating
supporting-passage labels for the new chunk boundaries.
