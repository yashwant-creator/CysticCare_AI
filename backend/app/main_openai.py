
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from services.openai_rag_init import (
    initialize_openai_rag_system,
    get_rag_response,
    get_collection_stats
)

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))  # legacy, but also try backend/.env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))  # fallback for backend/.env
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    use_query_rewriting: bool = True  # Enable query rewriting by default
    use_cot: bool = True  # Enable Chain of Thought reasoning
    use_stepback: bool = False  # Enable Stepback Query Decomposition
    use_adaptive_agent: bool = True  # Auto-select best agent


class SourceInfo(BaseModel):
    """Information about a source document"""
    index: int
    title: str
    author: str
    year: str = "Unknown"
    file: str
    citation: str = ""
    display_name: str = ""
    relevance_score: float


class ReasoningStep(BaseModel):
    """Model for a single reasoning step in CoT"""
    step: int
    sub_question: str
    reasoning: str
    sources_used: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    status: str
    response: str
    sources: List[SourceInfo]
    query: str
    timestamp: str
    reasoning_chain: List[ReasoningStep] = []  # Empty if CoT not used
    cot_enabled: bool = False
    stepback_query: str = ""  # Stepback query if stepback mode used


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

# Global flag for initialization status
_rag_initialized = False
_initialization_task = None

async def initialize_rag_background():
    """Initialize RAG system in background after server starts"""
    global _rag_initialized
    try:
        logger.info("Initializing OpenAI RAG system in background...")
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
        _rag_initialized = True
    except Exception as e:
        logger.error(f"Error during RAG initialization: {e}")
        _rag_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan - startup and shutdown
    """
    global _initialization_task
    # Startup
    logger.info("=" * 80)
    logger.info("OpenAI Pipeline Backend Starting")
    logger.info("=" * 80)
    
    # Start RAG initialization in background (non-blocking)
    import asyncio
    _initialization_task = asyncio.create_task(initialize_rag_background())
    
    logger.info("Application startup complete")
    logger.info("Listening on http://0.0.0.0:8080")
    logger.info("RAG system will initialize in background")
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

# Add CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # Auto-select agent if adaptive mode enabled
        if request.use_adaptive_agent:
            from services.adaptive_agent_selector import AdaptiveAgentSelector
            agent_config = await AdaptiveAgentSelector.select_agent(request.query)
            logger.info(f"Adaptive Agent Selected: {agent_config['recommendation']} (reason: {agent_config['reason']})")
            request.use_stepback = agent_config['use_stepback']
            request.use_cot = agent_config['use_cot']
        
        logger.info(f"Chat endpoint called - Query: {request.query[:50]}...")
        logger.info(f"  Session: {request.session_id}, Top-K: {request.top_k}, CoT: {request.use_cot}, Stepback: {request.use_stepback}")
        
        # Use Stepback Query Decomposition if requested
        if request.use_stepback:
            from services.stepback_agent import StepbackAgent
            from services.openai_rag_init import openai_service
            
            if openai_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG system not initialized yet"
                )
            
            stepback_agent = StepbackAgent(openai_service)
            result = await stepback_agent.answer_with_stepback(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        # Use Chain of Thought RAG if requested
        elif request.use_cot:
            from services.cot_rag_service import get_cot_rag_service
            from services.openai_rag_init import openai_service
            
            if openai_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG system not initialized yet"
                )
            
            cot_service = get_cot_rag_service(openai_service)
            result = await cot_service.get_cot_rag_response(
                query=request.query,
                top_k_per_step=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            # Get standard RAG response (with query rewriting if enabled)
            result = await get_rag_response(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_query_rewriting=request.use_query_rewriting
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
                year=s.get("year", "Unknown"),
                file=s["file"],
                citation=s.get("citation", ""),
                display_name=s.get("display_name", ""),
                relevance_score=s["relevance_score"]
            )
            for s in result.get("sources", [])
        ]
        
        # Format reasoning chain if CoT was used
        reasoning_chain = []
        if request.use_cot and "reasoning_chain" in result:
            reasoning_chain = [
                ReasoningStep(
                    step=step["step"],
                    sub_question=step["sub_question"],
                    reasoning=step["reasoning"],
                    sources_used=step["sources_used"]
                )
                for step in result.get("reasoning_chain", [])
            ]
        
        response = ChatResponse(
            status="success",
            response=result["response"],
            sources=sources,
            query=request.query,
            timestamp=datetime.now().isoformat(),
            reasoning_chain=reasoning_chain,
            cot_enabled=request.use_cot,
            stepback_query=result.get("stepback_query", "")
        )
        
        logger.info(f"Chat response generated successfully ({len(sources)} sources)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-query")
async def analyze_query_endpoint(request: ChatRequest):
    """
    Analyze a query and show what agent would be selected
    
    Args:
        request: ChatRequest with query
        
    Returns:
        Agent selection recommendation and reasoning
    """
    try:
        from services.adaptive_agent_selector import AdaptiveAgentSelector
        
        logger.info(f"Analyze query endpoint called - Query: {request.query[:50]}...")
        
        # Get agent selection
        selection = await AdaptiveAgentSelector.select_agent(request.query)
        
        return {
            "status": "success",
            "query": request.query,
            "recommendation": selection['recommendation'],
            "reason": selection['reason'],
            "agent_config": {
                "use_stepback": selection['use_stepback'],
                "use_cot": selection['use_cot']
            },
            "explanation": get_agent_explanation(selection['recommendation'])
        }
        
    except Exception as e:
        logger.error(f"Error in analyze query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_agent_explanation(recommendation: str) -> str:
    """Get explanation for why this agent was selected"""
    explanations = {
        "stepback": "Using Stepback Query Decomposition for better context - generates broader queries to retrieve both specific and foundational knowledge.",
        "cot": "Using Chain of Thought reasoning - breaks down complex questions into sub-questions for structured multi-step analysis.",
        "standard_rag": "Using Standard RAG - direct retrieval and answer generation, optimal for straightforward questions."
    }
    return explanations.get(recommendation, "")


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """
    Health check endpoint - verify system is running and initialized
    
    Returns:
        HealthResponse with status and collection statistics
    """
    try:
        # Check if RAG is initialized
        if not _rag_initialized:
            stats = {
                "status": "initializing",
                "message": "RAG system is initializing in background",
                "total_documents": 0,
                "total_chunks": 0
            }
            health = HealthResponse(
                status="initializing",
                version="1.0.0",
                timestamp=datetime.now().isoformat(),
                collection_stats=stats
            )
        else:
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


@app.post("/stepback-demo")
async def stepback_demo_endpoint(request: ChatRequest):
    """
    Demonstration endpoint showing stepback query generation
    Returns both original and stepback queries with their retrieval results
    
    Args:
        request: ChatRequest with query
        
    Returns:
        Detailed stepback analysis
    """
    try:
        from services.stepback_agent import StepbackAgent
        from services.openai_rag_init import openai_service
        
        if openai_service is None:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized yet"
            )
        
        stepback_agent = StepbackAgent(openai_service)
        
        # Generate stepback query
        stepback_query = await stepback_agent.generate_stepback_query(request.query)
        
        # Get retrieval results with stepback
        retrieval_results = await stepback_agent.retrieve_with_stepback(
            request.query,
            request.top_k
        )
        
        return {
            "status": "success",
            "original_query": request.query,
            "stepback_query": stepback_query,
            "retrieval_stats": {
                "original_results": retrieval_results.get("original_count", 0),
                "stepback_results": retrieval_results.get("stepback_count", 0),
                "combined_unique": retrieval_results.get("combined_count", 0)
            },
            "explanation": "The stepback query captures broader concepts to provide better context"
        }
        
    except Exception as e:
        logger.error(f"Error in stepback demo: {e}")
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
