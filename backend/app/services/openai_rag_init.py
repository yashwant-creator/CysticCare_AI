"""
OpenAI RAG Initialization Module
Handles PDF processing and ChromaDB integration for the OpenAI pipeline
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from .openai_service import OpenAIService
from ..utils.openai_utils import (
    process_pdf_file,
    get_pdf_files,
    create_chunk_id,
    load_session_config
)
from ..utils.metadata_manager import get_metadata_manager
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

EMPTY_KNOWLEDGE_BASE_MESSAGE = (
    "I don't have any indexed PKD or ADPKD source documents loaded right now, "
    "so I can't provide a sourced answer yet."
)


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


async def initialize_openai_rag_system(
    pdf_directory: str = "papers",
    collection_name: str = "pkd_knowledge_base_openai"
) -> Dict[str, Any]:
    """
    Initialize the OpenAI RAG system:
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
        logger.info("Starting OpenAI RAG system initialization...")

        config = load_session_config()

        global openai_service
        openai_service = OpenAIService(
            embedding_model=config["embedding_model"],
            chat_model=config["chat_model"],
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"]
        )

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

        resolved_pdf_directory = _resolve_pdf_directory(pdf_directory)

        try:
            openai_collection = openai_chroma_client.get_collection(name=collection_name)
            total_vectors = openai_collection.count()
            logger.info(f"Using existing ChromaDB collection: {collection_name}")
            logger.info(f"Collection contains {total_vectors} vectors")
            if total_vectors > 0:
                return {
                    "status": "success",
                    "message": f"ChromaDB collection '{collection_name}' already initialized",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "total_vectors": total_vectors
                }
            logger.warning(
                "Existing ChromaDB collection '%s' is empty; attempting to rebuild from PDFs in %s",
                collection_name,
                resolved_pdf_directory,
            )
        except Exception:
            logger.info(f"Creating new ChromaDB collection: {collection_name}")
            openai_collection = openai_chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

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
        
        for pdf_file in pdf_files:
            try:
                chunks, basic_metadata = process_pdf_file(pdf_file)
                enhanced_metadata = metadata_manager.extract_enhanced_metadata(
                    pdf_file, basic_metadata
                )
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append(enhanced_metadata)
                    all_ids.append(create_chunk_id(os.path.basename(pdf_file), chunk_idx))
                
                total_chunks += len(chunks)
                logger.info(f"✓ Processed {pdf_file}: {len(chunks)} chunks - {enhanced_metadata['display_name']}")
                
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
        logger.info(f"✓ Successfully stored {len(all_chunks)} chunks in ChromaDB")
        
        return {
            "status": "success",
            "message": "OpenAI RAG system initialized successfully",
            "documents_processed": len(pdf_files),
            "chunks_created": total_chunks,
            "total_vectors": total_vectors
        }
        
    except Exception as e:
        logger.error(f"Error initializing OpenAI RAG system: {e}")
        return {
            "status": "error",
            "message": f"Initialization failed: {str(e)}",
            "documents_processed": 0,
            "chunks_created": 0
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
    global openai_service, openai_collection
    
    if openai_collection is None:
        logger.error("ChromaDB collection not initialized")
        return {
            "status": "error",
            "message": "Knowledge base not initialized",
            "results": []
        }
    
    try:
        query_embedding = openai_service.get_embedding(query)
        
        results = openai_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        formatted_results = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            formatted_results.append({
                "document": doc,
                "metadata": meta,
                "relevance_score": round(1 - distance, 4)
            })
        
        logger.info(f"Knowledge base search returned {len(formatted_results)} results")
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
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
            query_rewriter = get_query_rewriter(openai_service)
            
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
            logger.warning("No retrieved chunks available for query; returning empty-knowledge-base response")
            return {
                "status": "success",
                "response": EMPTY_KNOWLEDGE_BASE_MESSAGE,
                "sources": [],
                "query": query,
                "retrieved_chunks": [],
                "refused": True,
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
                "refused": True,
            }
        
        logger.info(f"Generated RAG response ({len(response)} chars, {len(sources)} sources)")
        
        return {
            "status": "success",
            "response": response,
            "sources": sources,
            "query": query,
            "retrieved_chunks": results,
            "refused": False,
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
        return {
            "status": "success",
            "collection_name": openai_collection.name,
            "total_vectors": count,
            "embedding_dimension": 1536,  # OpenAI embeddings are 1536-dimensional
            "model": "text-embedding-3-small"
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to get stats: {str(e)}"
        }
