"""
OpenAI Pipeline Backend - Main Application
FastAPI application for the OpenAI-based RAG pipeline
Runs on port 8001 (parallel to original pipeline on 8000)
"""

import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from services.openai_rag_init import (
    initialize_openai_rag_system,
    get_rag_response,
    get_collection_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class InitializeRequest(BaseModel):
    """Request model for RAG system initialization"""
    pdf_directory: str = "papers"
    collection_name: str = "pkd_knowledge_base_openai"


class InitializeResponse(BaseModel):
    """Response model for initialization"""
    status: str
    message: str
    documents_processed: int
    chunks_created: int
    total_vectors: int = 0


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str
    session_id: str = "default"
    top_k: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000


class SourceInfo(BaseModel):
    """Information about a source document"""
    index: int
    title: str
    author: str
    file: str
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    status: str
    response: str
    sources: List[SourceInfo]
    query: str
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    timestamp: str
    collection_stats: Dict[str, Any]


class QuickQuestionsResponse(BaseModel):
    """Response model for quick questions"""
    questions: List[str]


# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan - startup and shutdown
    """
    # Startup
    logger.info("=" * 80)
    logger.info("OpenAI Pipeline Backend Starting")
    logger.info("=" * 80)
    
    try:
        # Initialize RAG system on startup
        logger.info("Initializing OpenAI RAG system...")
        result = await initialize_openai_rag_system(
            pdf_directory="papers",
            collection_name="pkd_knowledge_base_openai"
        )
        logger.info(f"RAG System Initialization: {result['status']}")
        logger.info(f"  - Documents processed: {result['documents_processed']}")
        logger.info(f"  - Chunks created: {result['chunks_created']}")
        if 'total_vectors' in result:
            logger.info(f"  - Total vectors: {result['total_vectors']}")
        logger.info(result['message'])
        
    except Exception as e:
        logger.error(f"Error during RAG initialization: {e}")
    
    logger.info("Application startup complete")
    logger.info("Listening on http://0.0.0.0:8001")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("OpenAI Pipeline Backend Shutting Down")
    logger.info("=" * 80)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CysticCare AI - OpenAI Pipeline",
    description="OpenAI-based RAG pipeline for PKD knowledge base",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/initialize", response_model=InitializeResponse)
async def initialize_endpoint(request: InitializeRequest) -> InitializeResponse:
    """
    Initialize the RAG system with PDFs from specified directory
    
    Args:
        request: InitializeRequest with pdf_directory and collection_name
        
    Returns:
        InitializeResponse with status and statistics
    """
    try:
        logger.info(f"Initialize endpoint called with directory: {request.pdf_directory}")
        
        result = await initialize_openai_rag_system(
            pdf_directory=request.pdf_directory,
            collection_name=request.collection_name
        )
        
        return InitializeResponse(
            status=result["status"],
            message=result["message"],
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            total_vectors=result.get("total_vectors", 0)
        )
        
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint - process user query and return RAG response
    
    Args:
        request: ChatRequest with query and parameters
        
    Returns:
        ChatResponse with answer and sources
    """
    try:
        logger.info(f"Chat endpoint called - Query: {request.query[:50]}...")
        logger.info(f"  Session: {request.session_id}, Top-K: {request.top_k}")
        
        # Get RAG response
        result = await get_rag_response(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if result["status"] != "success":
            logger.error(f"RAG response failed: {result['message']}")
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to generate response")
            )
        
        # Format sources
        sources = [
            SourceInfo(
                index=s["index"],
                title=s["title"],
                author=s["author"],
                file=s["file"],
                relevance_score=s["relevance_score"]
            )
            for s in result.get("sources", [])
        ]
        
        response = ChatResponse(
            status="success",
            response=result["response"],
            sources=sources,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Chat response generated successfully ({len(sources)} sources)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """
    Health check endpoint - verify system is running and initialized
    
    Returns:
        HealthResponse with status and collection statistics
    """
    try:
        stats = get_collection_stats()
        
        health = HealthResponse(
            status="healthy" if stats.get("status") == "success" else "degraded",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            collection_stats=stats
        )
        
        logger.info(f"Health check - Status: {health.status}")
        return health
        
    except Exception as e:
        logger.error(f"Error in health endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quick-questions", response_model=QuickQuestionsResponse)
async def quick_questions_endpoint() -> QuickQuestionsResponse:
    """
    Get suggested questions for quick testing
    
    Returns:
        QuickQuestionsResponse with list of suggested questions
    """
    suggested_questions = [
        "What is Polycystic Kidney Disease (PKD)?",
        "What are the symptoms of PKD?",
        "How is PKD diagnosed?",
        "What treatment options are available for PKD?",
        "What is the progression rate of PKD?",
        "Are there genetic factors in PKD?",
        "How does PKD affect kidney function?",
        "What lifestyle changes help manage PKD?"
    ]
    
    logger.info("Quick questions endpoint called")
    return QuickQuestionsResponse(questions=suggested_questions)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "CysticCare AI - OpenAI Pipeline",
        "version": "1.0.0",
        "description": "OpenAI-based RAG pipeline for PKD knowledge base",
        "endpoints": {
            "POST /initialize": "Initialize RAG system with PDFs",
            "POST /chat": "Chat with the AI (RAG response)",
            "GET /health": "Health check",
            "GET /quick-questions": "Get suggested questions",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation"
        },
        "info": {
            "embedding_model": "text-embedding-3-small (1536 dimensions)",
            "llm_model": "gpt-4o",
            "vector_store": "ChromaDB (persistent disk-based)",
            "pdf_processing": "pdfplumber + PyPDF2",
            "chunking": "400-word chunks"
        }
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "status": "error",
        "detail": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled Exception: {str(exc)}")
    return {
        "status": "error",
        "detail": "Internal server error",
        "message": str(exc)
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run on port 8001 (original pipeline runs on 8000)
    logger.info("Starting OpenAI Pipeline on port 8001")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
