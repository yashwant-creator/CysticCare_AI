import logging
import math
import os
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _positive_env_number(name: str, default: float) -> float:
    """Read a positive retrieval tuning value without breaking a request."""
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Vector Search (ChromaDB) and
    Sparse Keyword Search (BM25) with Reciprocal Rank Fusion (RRF).
    """
    _instance: Optional['HybridRetriever'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HybridRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self, collection=None):
        # Prevent re-initialization if already done
        if hasattr(self, 'initialized') and self.initialized:
            # If a new collection is passed, re-build the index
            if collection is not None and collection != self.collection:
                self.collection = collection
                self.build_sparse_index()
            return

        self.collection = collection
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # BM25 corpus statistics
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # list of {term -> term frequency} per doc

        # BM25 Hyperparameters (standard defaults)
        self.k1 = 1.5
        self.b = 0.75

        self.initialized = False
        if collection is not None:
            self.build_sparse_index()

    def build_sparse_index(self):
        """Fetch all documents from ChromaDB and build the BM25 sparse index."""
        if self.collection is None:
            logger.error("Cannot build sparse index: ChromaDB collection is None")
            return

        try:
            logger.info("Fetching all stored chunks from ChromaDB to build sparse BM25 index...")
            # Fetch all stored documents, metadatas and ids
            data = self.collection.get(include=["documents", "metadatas"])

            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", [])
            self.ids = data.get("ids", [])

            self.corpus_size = len(self.documents)
            logger.info(f"Loaded {self.corpus_size} chunks from ChromaDB for sparse index construction")

            if self.corpus_size == 0:
                logger.warning("No documents found in ChromaDB collection. Sparse search index remains uninitialized.")
                return

            self.doc_lengths = []
            self.term_freqs = []
            self.doc_freqs = {}

            total_len = 0
            for doc in self.documents:
                tokens = self._tokenize(doc)
                doc_len = len(tokens)
                self.doc_lengths.append(doc_len)
                total_len += doc_len

                # Compute term frequencies (TF) for this document
                tf = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                self.term_freqs.append(tf)

                # Accumulate document frequency (DF) for each distinct term in the document
                for t in tf.keys():
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

            self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
            self.initialized = True
            logger.info(f"✓ HybridRetriever sparse index built successfully. Corpus: {self.corpus_size} chunks. Avg chunk size: {self.avg_doc_len:.1f} tokens.")

        except Exception as e:
            logger.error(f"Error while building sparse BM25 index: {e}", exc_info=True)

    def _tokenize(self, text: str) -> List[str]:
        """Simple, consistent English tokenization (lowercased, alphanumeric words)."""
        if not text:
            return []
        text = text.lower()
        # Find all alphanumeric sequences representing words/numbers
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency (IDF) with standard BM25 smoothing."""
        df = self.doc_freqs.get(term, 0)
        # BM25 standard IDF formula with floor/smoothing to avoid negative values
        val = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
        return max(val, 0.0001)

    def sparse_search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Search the corpus using BM25 sparse keyword matching."""
        if not self.initialized or self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx in range(self.corpus_size):
            doc_score = 0.0
            tf_dict = self.term_freqs[idx]
            doc_len = self.doc_lengths[idx]

            for token in query_tokens:
                if token in tf_dict:
                    tf = tf_dict[token]
                    idf = self._idf(token)

                    # BM25 scoring formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    doc_score += idf * (numerator / denominator)

            if doc_score > 0.0:
                scores.append((idx, doc_score))

        # Sort docs by BM25 score descending
        scores.sort(key=lambda x: -x[1])

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "id": self.ids[idx],
                "sparse_score": round(score, 4)
            })

        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_pool_limit: Optional[int] = None,
        result_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining Dense and Sparse search results
        using Reciprocal Rank Fusion (RRF).
        """
        # Fetch a broad candidate pool before any semantic reranking. `top_k`
        # remains the backwards-compatible output size unless `result_limit` is
        # supplied by the caller.
        configured_pool = min(
            200, int(_positive_env_number("RAG_HYBRID_CANDIDATE_POOL", 60))
        )
        minimum_pool = max(top_k * 4, 30)
        candidate_pool_limit = max(
            candidate_pool_limit or minimum_pool,
            configured_pool,
            minimum_pool,
        )
        result_limit = result_limit or top_k
        dense_weight = _positive_env_number("RAG_DENSE_RRF_WEIGHT", 1.0)
        sparse_weight = _positive_env_number("RAG_SPARSE_RRF_WEIGHT", 1.0)

        # 1. Fetch dense candidates from ChromaDB
        dense_results = []
        try:
            dense_raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_pool_limit,
                include=["documents", "metadatas", "distances"]
            )

            dense_docs = dense_raw["documents"][0] if dense_raw["documents"] else []
            dense_metas = dense_raw["metadatas"][0] if dense_raw["metadatas"] else []
            dense_ids = dense_raw["ids"][0] if dense_raw["ids"] else []
            dense_distances = dense_raw["distances"][0] if dense_raw["distances"] else []

            for doc, meta, doc_id, dist in zip(dense_docs, dense_metas, dense_ids, dense_distances):
                # Map distance (which is often cosine distance e.g. 0..2) to standard relevance score (0..1)
                dense_results.append({
                    "document": doc,
                    "metadata": meta,
                    "id": doc_id,
                    "relevance_score": round(min(1.0, max(0.0, 1.0 - dist)), 4)
                })
        except Exception as e:
            logger.error(f"Error fetching dense candidates from ChromaDB during hybrid search: {e}")

        # 2. Fetch sparse candidates from local BM25 index
        sparse_results = []
        try:
            sparse_results = self.sparse_search(query, top_k=candidate_pool_limit)
        except Exception as e:
            logger.error(f"Error fetching sparse candidates during hybrid search: {e}")

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}  # id -> combined rrf score
        doc_details = {}  # id -> {document, metadata}
        dense_scores = {}  # id -> relevance score (0..1)
        sparse_scores = {}  # id -> raw bm25 score

        # Populate ranks from dense retrieval
        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                dense_weight / (rrf_k + rank + 1)
            )
            doc_details[doc_id] = {"document": res["document"], "metadata": res["metadata"]}
            dense_scores[doc_id] = res["relevance_score"]

        # Populate ranks from sparse retrieval
        for rank, res in enumerate(sparse_results):
            doc_id = res["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                sparse_weight / (rrf_k + rank + 1)
            )
            if doc_id not in doc_details:
                doc_details[doc_id] = {"document": res["document"], "metadata": res["metadata"]}
            sparse_scores[doc_id] = res["sparse_score"]

        # 4. Sort fused candidates by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])

        # This is a normalized rank-fusion score, not a calibrated relevance
        # probability. The semantic reranker supplies the confidence score used
        # for threshold decisions later in the pipeline.
        max_possible_rrf = (dense_weight + sparse_weight) / (rrf_k + 1)

        final_results = []
        for doc_id in sorted_ids[:result_limit]:
            details = doc_details[doc_id]
            rrf_score = rrf_scores[doc_id]

            # This preserves the existing UI-facing fused score. It must not be
            # used as clinical/retrieval confidence; the semantic reranker adds
            # a calibrated threshold score when enabled.
            normalized_score = round(rrf_score / max_possible_rrf, 4)

            # Fallback value if found only in sparse stream: map BM25 score to reasonable range
            fallback_score = 0.5
            relevance_score = dense_scores.get(doc_id, fallback_score)

            final_results.append({
                "document": details["document"],
                "metadata": details["metadata"],
                "id": doc_id,
                "relevance_score": normalized_score,  # Swap standard distance score with normalized RRF score
                "dense_score": relevance_score,
                "sparse_score": sparse_scores.get(doc_id, 0.0),
                "rrf_score": round(rrf_score, 6)
            })

        logger.info(
            f"Hybrid search returned {len(final_results)} RRF-fused chunks (merged "
            f"{len(dense_results)} dense and {len(sparse_results)} sparse candidates)."
        )
        return final_results
