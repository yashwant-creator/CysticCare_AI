"""
OpenAI RAG Initialization Module
Handles PDF processing and ChromaDB integration for the OpenAI pipeline
"""

import asyncio
import os
import logging
from typing import List, Dict, Any, Mapping, Optional
import chromadb
from .openai_service import OpenAIService
from ..utils.openai_utils import (
    process_pdf_file_with_metadata,
    process_pdf_images,
    get_pdf_files,
    load_session_config
)
from ..utils.metadata_manager import get_metadata_manager
from ..utils.paper_identity import stable_paper_id
from ..utils.refusal_utils import REFUSAL_MESSAGE, get_guardrail_response, is_refusal_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STANDARD_RAG_SYSTEM_PROMPT = f"""You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD), ADPKD, and kidney-related disease topics.

If the question is asking anything other than PKD, ADPKD, or any kidney related disease, respond EXACTLY with:
"{REFUSAL_MESSAGE}"

If the provided context does not contain enough relevant information to answer the question, respond EXACTLY with:
"{REFUSAL_MESSAGE}"

Provide accurate, evidence-based information based on the provided medical literature.

When answering questions:
- Be clear, concise, and compassionate
- Cite evidence from the provided sources when making claims
- Acknowledge uncertainty when information is limited
- Remind users to consult their healthcare provider for medical decisions
- Use professional medical terminology but explain complex concepts"""

# Global ChromaDB client and collection
openai_chroma_client: Optional[chromadb.Client] = None
openai_collection: Optional[Any] = None
openai_service: Optional[OpenAIService] = None
# Optional cheaper model for non-answer "helper" calls (query rewriting,
# validation, follow-up generation). Defaults to the main service (so prod
# behaviour is unchanged) unless HELPER_CHAT_MODEL is set, e.g. gpt-4o-mini.
openai_helper_service: Optional[OpenAIService] = None

EMPTY_KNOWLEDGE_BASE_MESSAGE = (
    "I don't have any indexed PKD or ADPKD source documents loaded right now, "
    "so I can't provide a sourced answer yet."
)

# Increment this whenever stored chunk structure or metadata changes in a way
# that requires regenerating embeddings. Existing collections are never deleted
# implicitly; the rebuild command is the explicit migration path.
INDEX_SCHEMA_VERSION = "2"

INSUFFICIENT_RETRIEVAL_MESSAGE = (
    "I found source material, but not enough directly relevant evidence in the "
    "indexed research papers to answer that reliably."
)


def _mutable_collection_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return collection metadata safe to pass to Chroma's ``modify`` API.

    HNSW settings are fixed when a collection is created.  Chroma rejects a
    later ``modify`` request that includes them, even when the value did not
    change.  Runtime/build-state metadata remains mutable.
    """
    return {
        key: value
        for key, value in dict(metadata or {}).items()
        if not str(key).startswith("hnsw:")
    }


def _resolve_pdf_directory(pdf_directory: str) -> str:
    """Resolve the configured PDF directory with a fallback to the repo data folder."""
    app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [pdf_directory]

    if pdf_directory == "papers":
        candidates.append(os.path.join("data", "papers"))

    existing_directories = []
    for candidate in candidates:
        resolved = os.path.join(app_root, candidate)
        if os.path.exists(resolved):
            existing_directories.append(resolved)
            if get_pdf_files(resolved):
                return resolved

    if existing_directories:
        return existing_directories[0]

    return os.path.join(app_root, pdf_directory)


async def build_openai_rag_system(
    pdf_directory: str = "papers",
    collection_name: str = "pkd_knowledge_base_openai"
) -> Dict[str, Any]:
    """
    Offline-only vector database build:
    1. Load PDFs
    2. Extract and chunk text
    3. Generate OpenAI embeddings
    4. Store in ChromaDB

    Args:
        pdf_directory: Directory containing PDFs
        collection_name: Name for ChromaDB collection

    Returns:
        Initialization status dictionary
    """
    global openai_chroma_client, openai_collection, openai_service

    try:
        logger.info("Starting offline OpenAI vector database build...")

        config = load_session_config()

        global openai_service, openai_helper_service
        # Base answer-generation model. Defaults to the configured model
        # (gpt-5.5, production) unless BASE_CHAT_MODEL overrides it, e.g. gpt-4o.
        base_model = os.getenv("BASE_CHAT_MODEL", "").strip() or config["chat_model"]
        openai_service = OpenAIService(
            embedding_model=config["embedding_model"],
            chat_model=base_model,
            vision_model=config["vision_model"],
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"]
        )
        if base_model != config["chat_model"]:
            logger.info(f"Base answer-generation model overridden to '{base_model}'")

        # Route helper calls (query-rewrite, validation, follow-ups) to a cheaper
        # model when HELPER_CHAT_MODEL is set; otherwise reuse the main service so
        # production behaviour is identical.
        helper_model = os.getenv("HELPER_CHAT_MODEL", "").strip()
        if helper_model and helper_model != base_model:
            openai_helper_service = OpenAIService(
                embedding_model=config["embedding_model"],
                chat_model=helper_model,
                vision_model=config["vision_model"],
                max_retries=config["max_retries"],
                retry_delay=config["retry_delay"]
            )
            logger.info(
                f"Helper LLM calls (query-rewrite, validation, follow-ups) routed to '{helper_model}'"
            )
        else:
            openai_helper_service = openai_service

        if not openai_service.validate_connection():
            logger.error("Failed to validate OpenAI connection")
            return {
                "status": "error",
                "message": "Failed to connect to OpenAI API",
                "documents_processed": 0,
                "chunks_created": 0
            }

        logger.info("OpenAI connection validated")

        chroma_data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "openai_chroma_data"
        )
        os.makedirs(chroma_data_path, exist_ok=True)

        logger.info(f"Initializing ChromaDB with path: {chroma_data_path}")
        global openai_chroma_client, openai_collection
        openai_chroma_client = chromadb.PersistentClient(path=chroma_data_path)

        def create_v2_collection() -> Any:
            """Create a clean collection with creation-time index settings."""
            return openai_chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "index_schema_version": INDEX_SCHEMA_VERSION,
                    "index_build_state": "building",
                },
            )

        resolved_pdf_directory = _resolve_pdf_directory(pdf_directory)

        try:
            openai_collection = openai_chroma_client.get_collection(name=collection_name)
        except Exception as collection_error:
            logger.info(
                "Creating ChromaDB collection %s (existing collection unavailable: %s)",
                collection_name,
                collection_error,
            )
            openai_collection = create_v2_collection()
        else:
            total_vectors = openai_collection.count()
            logger.info(f"Using existing ChromaDB collection: {collection_name}")
            logger.info(f"Collection contains {total_vectors} vectors")
            collection_metadata = openai_collection.metadata or {}
            build_state = collection_metadata.get("index_build_state", "ready")
            if total_vectors > 0:
                existing_schema = collection_metadata.get("index_schema_version")
                if build_state != "ready":
                    logger.error(
                        "Collection %s is marked %r and may be incomplete; refusing to serve it. "
                        "Run `python rebuild_vectordb.py` from backend/.",
                        collection_name,
                        build_state,
                    )
                    return {
                        "status": "error",
                        "message": "Knowledge-base rebuild is incomplete; run rebuild_vectordb.py",
                        "documents_processed": 0,
                        "chunks_created": 0,
                        "total_vectors": total_vectors,
                        "requires_rebuild": True,
                    }
                if existing_schema != INDEX_SCHEMA_VERSION:
                    logger.warning(
                        "Collection schema is %r; retrieval improvements require a rebuild "
                        "to schema %s. Run `python rebuild_vectordb.py` from backend/.",
                        existing_schema,
                        INDEX_SCHEMA_VERSION,
                    )
                # Build Hybrid Retriever BM25 sparse index
                try:
                    from .hybrid_retriever import HybridRetriever
                    import asyncio
                    await asyncio.to_thread(HybridRetriever, openai_collection)
                except Exception as hr_err:
                    logger.error(f"Failed to build Hybrid Retriever on startup: {hr_err}")

                return {
                    "status": "success",
                    "message": f"ChromaDB collection '{collection_name}' already initialized",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "total_vectors": total_vectors,
                    "requires_rebuild": existing_schema != INDEX_SCHEMA_VERSION,
                }
            logger.warning(
                "Existing ChromaDB collection '%s' is empty; recreating it with schema %s before ingestion from %s",
                collection_name,
                INDEX_SCHEMA_VERSION,
                resolved_pdf_directory,
            )
            # HNSW distance space is a collection creation-time setting, so an
            # empty legacy collection is recreated instead of metadata-modified.
            openai_chroma_client.delete_collection(collection_name)
            openai_collection = create_v2_collection()

        pdf_directory = resolved_pdf_directory

        if not os.path.exists(pdf_directory):
            logger.warning(f"PDF directory not found: {pdf_directory}")
            return {
                "status": "warning",
                "message": f"PDF directory not found: {pdf_directory}",
                "documents_processed": 0,
                "chunks_created": 0
            }

        pdf_files = get_pdf_files(pdf_directory)

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return {
                "status": "warning",
                "message": f"No PDF files found in {pdf_directory}",
                "documents_processed": 0,
                "chunks_created": 0,
                "total_vectors": openai_collection.count() if openai_collection is not None else 0,
            }

        logger.info(f"Processing {len(pdf_files)} PDF files...")

        metadata_manager = get_metadata_manager()

        all_chunks = []
        all_metadatas = []
        all_ids = []
        total_chunks = 0
        indexed_documents = 0
        seen_paper_ids: Dict[str, str] = {}

        for pdf_file in pdf_files:
            try:
                # Preserve document structure alongside each chunk. The legacy
                # helper remains available for callers outside the indexer, but
                # the retrieval index needs page/section/neighbor metadata.
                chunks, basic_metadata, chunk_metadatas = process_pdf_file_with_metadata(pdf_file)
                enhanced_metadata = metadata_manager.extract_enhanced_metadata(
                    pdf_file, basic_metadata
                )
                paper_id = stable_paper_id(
                    enhanced_metadata.get("title"),
                    os.path.splitext(os.path.basename(pdf_file))[0],
                )
                if paper_id in seen_paper_ids:
                    logger.info(
                        "Skipping duplicate paper identity %s (%s; already indexed from %s)",
                        paper_id,
                        pdf_file,
                        seen_paper_ids[paper_id],
                    )
                    continue
                seen_paper_ids[paper_id] = pdf_file
                indexed_documents += 1

                paper_chunks = []
                paper_metadatas = []
                paper_ids = []

                for chunk_idx, chunk in enumerate(chunks):
                    # Stable title IDs preserve paper-level benchmark joins even
                    # when a source PDF filename changes. The new chunking
                    # scheme deliberately receives fresh chunk ordinals.
                    chunk_id = f"{paper_id}_{chunk_idx}"
                    raw_chunk_metadata = (
                        chunk_metadatas[chunk_idx]
                        if chunk_idx < len(chunk_metadatas)
                        else {"chunk_index": chunk_idx, "content_type": "text"}
                    )
                    chunk_metadata = dict(raw_chunk_metadata)
                    section_heading = chunk_metadata.get("section_heading") or ""
                    page_number = chunk_metadata.get("page_number") or chunk_metadata.get("page_start")

                    # The chunker exposes adjacency by its own stable IDs;
                    # translate them to the IDs actually stored in Chroma.
                    # This keeps future context-window expansion safe after
                    # filename sanitization.
                    previous_index = chunk_metadata.get("previous_chunk_index", -1)
                    next_index = chunk_metadata.get("next_chunk_index", -1)
                    chunk_metadata["previous_chunk_id"] = (
                        f"{paper_id}_{previous_index}"
                        if isinstance(previous_index, int) and previous_index >= 0
                        else ""
                    )
                    chunk_metadata["next_chunk_id"] = (
                        f"{paper_id}_{next_index}"
                        if isinstance(next_index, int) and next_index >= 0
                        else ""
                    )

                    # Embedding the paper title and local section context makes
                    # otherwise generic clinical passages distinguishable.
                    embedding_prefix = f"[Paper: {enhanced_metadata.get('title', 'Unknown')}]"
                    if section_heading:
                        embedding_prefix += f" [Section: {section_heading}]"
                    if page_number:
                        embedding_prefix += f" [Page: {page_number}]"
                    # The structure-aware chunker may carry a preliminary
                    # PDF-metadata prefix. Replace it with the enriched title
                    # from MetadataManager instead of embedding it twice.
                    chunk_body = chunk
                    first_line, separator, remainder = chunk.partition("\n")
                    if separator and first_line.startswith("[") and first_line.endswith("]"):
                        chunk_body = remainder

                    final_chunk_text = f"{embedding_prefix}\n{chunk_body}"
                    paper_chunks.append(final_chunk_text)
                    paper_metadatas.append({
                        **enhanced_metadata,
                        **chunk_metadata,
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                        "content_type": chunk_metadata.get("content_type", "text"),
                    })
                    paper_ids.append(chunk_id)

                # Post-process paper chunks to populate parent_text in their metadata
                for index, metadata in enumerate(paper_metadatas):
                    previous_metadata = paper_metadatas[index - 1] if index > 0 else None
                    next_metadata = paper_metadatas[index + 1] if index + 1 < len(paper_metadatas) else None

                    parts = []
                    # Add previous chunk if it exists and belongs to the same section
                    if previous_metadata and previous_metadata.get("parent_chunk_id") == metadata.get("parent_chunk_id"):
                        parts.append(paper_chunks[index - 1])
                    parts.append(paper_chunks[index])
                    # Add next chunk if it exists and belongs to the same section
                    if next_metadata and next_metadata.get("parent_chunk_id") == metadata.get("parent_chunk_id"):
                        parts.append(paper_chunks[index + 1])

                    metadata["parent_text"] = "\n\n".join(parts)

                all_chunks.extend(paper_chunks)
                all_metadatas.extend(paper_metadatas)
                all_ids.extend(paper_ids)

                total_chunks += len(chunks)
                logger.info(f"✓ Processed {pdf_file}: {len(chunks)} chunks - {enhanced_metadata['display_name']}")

                # Describe embedded figures with the vision model and embed them too
                if config.get("enable_image_descriptions", True):
                    image_chunks = process_pdf_images(pdf_file, openai_service)
                    for img_idx, (img_text, img_info) in enumerate(image_chunks):
                        all_chunks.append(img_text)
                        all_metadatas.append({
                            **enhanced_metadata,
                            "content_type": "figure",
                            "page_number": img_info["page_number"],
                            "paper_id": paper_id,
                            "chunk_id": f"{paper_id}_fig_{img_idx}",
                        })
                        all_ids.append(
                            f"{paper_id}_fig_{img_idx}"
                        )
                    total_chunks += len(image_chunks)
                    if image_chunks:
                        logger.info(
                            f"✓ Added {len(image_chunks)} figure descriptions from {pdf_file}"
                        )

            except Exception as e:
                logger.error(f"✗ Error processing {pdf_file}: {e}")
                continue

        if not all_chunks:
            logger.warning("No chunks created from PDF files")
            return {
                "status": "warning",
                "message": "No chunks created from PDF files",
                "documents_processed": len(pdf_files),
                "chunks_created": 0
            }

        logger.info(f"Generating OpenAI embeddings for {len(all_chunks)} chunks...")
        embeddings = openai_service.get_embeddings_batch(all_chunks, batch_size=100)

        if len(embeddings) != len(all_chunks):
            logger.error(f"Embedding count mismatch: got {len(embeddings)}, expected {len(all_chunks)}")
            return {
                "status": "error",
                "message": "Embedding generation failed",
                "documents_processed": len(pdf_files),
                "chunks_created": 0
            }

        logger.info("Storing embeddings and documents in ChromaDB...")
        chroma_batch_size = 5000
        for i in range(0, len(all_chunks), chroma_batch_size):
            openai_collection.add(
                embeddings=embeddings[i:i + chroma_batch_size],
                documents=all_chunks[i:i + chroma_batch_size],
                metadatas=all_metadatas[i:i + chroma_batch_size],
                ids=all_ids[i:i + chroma_batch_size]
            )
            logger.info(f"Stored batch {i // chroma_batch_size + 1}: {min(i + chroma_batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

        total_vectors = openai_collection.count()
        if total_vectors != len(all_chunks):
            logger.error(
                "Index write count mismatch: stored %s vectors for %s chunks",
                total_vectors,
                len(all_chunks),
            )
            return {
                "status": "error",
                "message": "Index write was incomplete; rerun rebuild_vectordb.py",
                "documents_processed": indexed_documents,
                "chunks_created": total_chunks,
                "total_vectors": total_vectors,
                "requires_rebuild": True,
            }
        try:
            # Chroma collection creation settings (for example
            # ``hnsw:space``) are immutable.  Passing them back to
            # ``modify`` makes Chroma treat even an unchanged value as an
            # unsupported distance-function change, leaving an otherwise
            # complete rebuild stuck in ``building`` state.
            current_collection_metadata = _mutable_collection_metadata(
                openai_collection.metadata
            )
            current_collection_metadata["index_schema_version"] = INDEX_SCHEMA_VERSION
            current_collection_metadata["index_build_state"] = "ready"
            openai_collection.modify(metadata=current_collection_metadata)
        except Exception as metadata_error:
            # Leave the collection in ``building`` state so it cannot be
            # mistaken for a ready corpus on the next application startup.
            logger.error("Could not mark the rebuilt collection ready: %s", metadata_error)
            return {
                "status": "error",
                "message": "Index build could not be marked ready; rerun rebuild_vectordb.py",
                "documents_processed": indexed_documents,
                "chunks_created": total_chunks,
                "total_vectors": total_vectors,
                "requires_rebuild": True,
            }
        logger.info(f"✓ Successfully stored {len(all_chunks)} chunks in ChromaDB")

        # Build Hybrid Retriever BM25 sparse index for fresh build
        try:
            from .hybrid_retriever import HybridRetriever
            HybridRetriever(openai_collection)
        except Exception as hr_err:
            logger.error(f"Failed to build Hybrid Retriever on fresh build: {hr_err}")

        return {
            "status": "success",
            "message": "OpenAI RAG system initialized successfully",
            "documents_processed": indexed_documents,
            "chunks_created": total_chunks,
            "total_vectors": total_vectors
        }

    except Exception as e:
        logger.error(f"Error building OpenAI vector database: {e}")
        return {
            "status": "error",
            "message": f"Build failed: {str(e)}",
            "documents_processed": 0,
            "chunks_created": 0
        }


async def initialize_openai_rag_system(
    pdf_directory: str = "papers",
    collection_name: str = "pkd_knowledge_base_openai",
) -> Dict[str, Any]:
    """Load the baked production index without network calls or mutations.

    ``pdf_directory`` remains in the signature for compatibility with older
    callers, but production startup deliberately ignores it. Index creation is
    owned exclusively by :func:`build_openai_rag_system`.
    """
    del pdf_directory
    global openai_chroma_client, openai_collection

    try:
        from .index_manifest import (
            DEFAULT_INDEX_PATH,
            DEFAULT_MANIFEST_PATH,
            load_index_manifest,
            materialize_runtime_index,
            runtime_checksum_enabled,
            verify_baked_index,
        )

        logger.info("Verifying baked vector database before accepting traffic")
        manifest = await asyncio.to_thread(
            load_index_manifest,
            DEFAULT_MANIFEST_PATH,
            DEFAULT_INDEX_PATH,
            verify_checksum=runtime_checksum_enabled(),
        )
        runtime_index_path = await asyncio.to_thread(
            materialize_runtime_index,
            DEFAULT_INDEX_PATH,
        )
        verified = await asyncio.to_thread(
            verify_baked_index,
            DEFAULT_MANIFEST_PATH,
            runtime_index_path,
            verify_checksum=False,
        )
        if manifest["collection_name"] != collection_name:
            raise RuntimeError(
                f"Baked collection is {manifest['collection_name']!r}, "
                f"not {collection_name!r}"
            )
        openai_chroma_client = verified["client"]
        openai_collection = verified["collection"]

        from .hybrid_retriever import HybridRetriever

        await asyncio.to_thread(HybridRetriever, openai_collection)

        if os.getenv("PRELOAD_MEDCPT", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .cross_encoder_reranker import (
                CrossEncoderConfig,
                MedCPTCrossEncoder,
            )

            config = CrossEncoderConfig.from_environment()
            await asyncio.to_thread(
                MedCPTCrossEncoder._load_model,
                config,
            )

        return {
            "status": "success",
            "message": "Baked vector database loaded and verified",
            "documents_processed": 0,
            "chunks_created": 0,
            "total_vectors": manifest["vector_count"],
            "requires_rebuild": False,
            "index_manifest": manifest,
        }
    except Exception as error:
        logger.error("Baked vector database startup failed: %s", error)
        openai_chroma_client = None
        openai_collection = None
        return {
            "status": "error",
            "message": f"Baked index unavailable: {error}",
            "documents_processed": 0,
            "chunks_created": 0,
            "total_vectors": 0,
            "requires_rebuild": True,
        }


async def search_knowledge_base(
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search the ChromaDB knowledge base for relevant documents

    Args:
        query: Search query text
        top_k: Number of top results to return

    Returns:
        Dictionary with search results
    """
    global openai_service, openai_helper_service, openai_collection

    if openai_collection is None:
        logger.error("ChromaDB collection not initialized")
        return {
            "status": "error",
            "message": "Knowledge base not initialized",
            "results": []
        }

    collection_metadata = getattr(openai_collection, "metadata", None) or {}
    if collection_metadata.get("index_build_state") == "building":
        logger.error("Knowledge base is still marked as building; refusing partial retrieval")
        return {
            "status": "error",
            "message": "Knowledge-base rebuild is incomplete; rerun rebuild_vectordb.py",
            "results": [],
        }

    if openai_service is None:
        logger.error("OpenAI service not initialized")
        return {
            "status": "error",
            "message": "Embedding service not initialized",
            "results": []
        }

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 0
    if not 1 <= top_k <= 20:
        return {
            "status": "error",
            "message": "top_k must be an integer between 1 and 20",
            "results": [],
        }
    try:
        # Check if SOTA HyDE (Hypothetical Document Embeddings) is enabled
        use_hyde = os.getenv("RAG_HYDE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        dense_query_text = query

        if use_hyde:
            try:
                from .query_rewriter import get_query_rewriter
                query_rewriter = get_query_rewriter(openai_helper_service or openai_service)
                hyde_passage = await query_rewriter.generate_hyde_passage(query)
                if hyde_passage and hyde_passage != query:
                    dense_query_text = hyde_passage
                    logger.info("SOTA HyDE: Generated hypothetical passage for dense retrieval.")
            except Exception as hyde_err:
                logger.warning(f"Failed to generate HyDE passage, falling back to original query: {hyde_err}")
                dense_query_text = query

        query_embedding = openai_service.get_embedding(dense_query_text)

        from .semantic_reranker import create_evidence_reranker
        reranker = create_evidence_reranker(openai_helper_service or openai_service)
        candidate_limit = max(top_k, reranker.config.candidate_limit)

        from .hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()

        if retriever.initialized:
            candidates = retriever.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                candidate_pool_limit=max(candidate_limit, 40),
                result_limit=candidate_limit,
            )
        else:
            logger.warning("HybridRetriever not initialized; falling back to pure dense vector search")
            results = openai_collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_limit,
                include=["documents", "metadatas", "distances"]
            )

            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            ids = results.get("ids", [[]])[0] if results.get("ids") else []

            candidates = []
            for index, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
                candidates.append({
                    "document": doc,
                    "metadata": meta,
                    "id": ids[index] if index < len(ids) else f"dense_candidate_{index}",
                    "relevance_score": round(min(1.0, max(0.0, 1 - distance)), 4),
                    "dense_score": round(min(1.0, max(0.0, 1 - distance)), 4),
                    "sparse_score": 0.0,
                })

        formatted_results, retrieval_metadata = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        # Check if parent-child context resolution is enabled
        use_parent_child = os.getenv("RAG_PARENT_CHILD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        if use_parent_child:
            parent_swaps = 0
            for item in formatted_results:
                metadata = item.get("metadata") or {}
                parent_text = metadata.get("parent_text")
                if parent_text:
                    item["document"] = parent_text
                    parent_swaps += 1
            if parent_swaps > 0:
                logger.info(f"Hierarchical Parent-Child Context: Expanded parent text for {parent_swaps} chunks.")

        logger.info(
            "Knowledge base search returned %s context-safe results from %s candidates "
            "(confidence=%s)",
            len(formatted_results),
            len(candidates),
            retrieval_metadata.get("confidence"),
        )

        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "retrieval_metadata": retrieval_metadata,
        }

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "results": []
        }


async def get_rag_response(
    query: str,
    top_k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    use_query_rewriting: bool = True
) -> Dict[str, Any]:
    """
    Generate a response using the RAG pattern:
    1. Rewrite query (optional, improves retrieval)
    2. Search knowledge base
    3. Format context
    4. Call OpenAI GPT API
    5. Return response with sources

    Args:
        query: User query
        top_k: Number of documents to retrieve
        temperature: Model temperature
        max_tokens: Maximum response tokens
        use_query_rewriting: Whether to use query rewriting agent

    Returns:
        Dictionary with response and sources
    """
    global openai_service

    if openai_service is None or openai_collection is None:
        logger.error("RAG system not initialized")
        return {
            "status": "error",
            "message": "RAG system not initialized",
            "response": None,
            "sources": []
        }

    try:
        guardrail_response = get_guardrail_response(query)
        if guardrail_response is not None:
            logger.info("Standard RAG answered query via guardrail before retrieval")
            return {
                "status": "success",
                "response": guardrail_response,
                "sources": [],
                "query": query,
                "retrieved_chunks": [],
                "refused": guardrail_response == REFUSAL_MESSAGE,
            }

        search_query = query
        if use_query_rewriting:
            from .query_rewriter import get_query_rewriter
            query_rewriter = get_query_rewriter(openai_helper_service or openai_service)

            search_query = await query_rewriter.rewrite_query_simple(query)
            logger.info(f"Query rewritten: '{query}' → '{search_query}'")

        search_results = await search_knowledge_base(search_query, top_k)

        if search_results["status"] != "success":
            logger.error(f"Knowledge base search failed: {search_results['message']}")
            return {
                "status": "error",
                "message": f"Search failed: {search_results['message']}",
                "response": None,
                "sources": []
            }

        results = search_results["results"]
        if not results:
            retrieval_metadata = search_results.get("retrieval_metadata", {})
            insufficient_evidence = retrieval_metadata.get("confidence") == "insufficient"
            logger.warning(
                "No context-safe chunks available for query; confidence=%s",
                retrieval_metadata.get("confidence", "unknown"),
            )
            return {
                "status": "success",
                "response": (
                    INSUFFICIENT_RETRIEVAL_MESSAGE
                    if insufficient_evidence
                    else EMPTY_KNOWLEDGE_BASE_MESSAGE
                ),
                "sources": [],
                "query": query,
                "retrieved_chunks": [],
                "refused": True,
                "retrieval_metadata": retrieval_metadata,
            }

        context_parts = []
        unique_sources = {}

        for i, result in enumerate(results):
            meta = result.get("metadata", {})
            display_name = meta.get("display_name", f"Source {i+1}")
            citation = meta.get("citation", "Unknown Source")
            file_key = display_name or meta.get("file_name") or citation or f"chunk_{i}"

            context_parts.append(f"[Source {i+1}: {display_name}]\n{result['document']}")

            score = result.get("relevance_score", 0)
            if file_key not in unique_sources or score > unique_sources[file_key]["relevance_score"]:
                unique_sources[file_key] = {
                    "title": meta.get("title", "Unknown"),
                    "author": meta.get("author", "Unknown"),
                    "year": meta.get("year", "Unknown"),
                    "file": file_key,
                    "citation": citation,
                    "display_name": display_name,
                    "relevance_score": score,
                }

        sources = [
            {"index": i + 1, **s}
            for i, s in enumerate(
                sorted(unique_sources.values(), key=lambda x: -x["relevance_score"])
            )
        ]

        context = "\n\n".join(context_parts)

        response = openai_service.get_chat_completion_with_context(
            context=context,
            user_query=query,
            system_instruction=STANDARD_RAG_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if is_refusal_response(response):
            logger.info("Standard RAG returned refusal response")
            return {
                "status": "success",
                "response": REFUSAL_MESSAGE,
                "sources": [],
                "query": query,
                "retrieved_chunks": [],
                # Keep retrieval evaluation evidence separate from public
                # sources. The API exposes it only for an explicit debug
                # request, so a model refusal does not bias benchmark data.
                "retrieval_debug_chunks": results,
                "refused": True,
                "retrieval_metadata": search_results.get("retrieval_metadata", {}),
            }

        logger.info(f"Generated RAG response ({len(response)} chars, {len(sources)} sources)")

        return {
            "status": "success",
            "response": response,
            "sources": sources,
            "query": query,
            "retrieved_chunks": results,
            "refused": False,
            "retrieval_metadata": search_results.get("retrieval_metadata", {}),
        }

    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return {
            "status": "error",
            "message": f"Response generation failed: {str(e)}",
            "response": None,
            "sources": []
        }


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the ChromaDB collection

    Returns:
        Dictionary with collection statistics
    """
    global openai_collection

    if openai_collection is None:
        return {
            "status": "error",
            "message": "Collection not initialized"
        }

    try:
        count = openai_collection.count()
        collection_metadata = openai_collection.metadata or {}
        build_state = collection_metadata.get("index_build_state", "ready")
        return {
            "status": "success",
            "collection_name": openai_collection.name,
            "total_vectors": count,
            "embedding_dimension": 1536,  # OpenAI embeddings are 1536-dimensional
            "model": "text-embedding-3-small",
            "index_schema_version": collection_metadata.get("index_schema_version"),
            "index_build_state": build_state,
            "requires_rebuild": (
                collection_metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION
                or build_state != "ready"
            ),
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to get stats: {str(e)}"
        }
