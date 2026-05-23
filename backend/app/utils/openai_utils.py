"""
OpenAI Pipeline Utilities
Shared utilities for the OpenAI-based RAG pipeline
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any
import pdfplumber
from PyPDF2 import PdfReader
from pathlib import Path
import re
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If OPENAI_API_KEY not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key


_tokenizer = None
_MAX_TOKENS = 8000  # safe margin below OpenAI's 8192 hard limit


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _enforce_token_limit(chunk: str) -> List[str]:
    """Split a chunk that exceeds _MAX_TOKENS into token-safe pieces."""
    enc = _get_tokenizer()
    tokens = enc.encode(chunk)
    if len(tokens) <= _MAX_TOKENS:
        return [chunk]
    mid = len(tokens) // 2
    left = enc.decode(tokens[:mid])
    right = enc.decode(tokens[mid:])
    return _enforce_token_limit(left) + _enforce_token_limit(right)


def chunk_up_context(text: str, chunk_length: int = 400) -> List[str]:
    """
    Split text into chunks based on word count, then enforce a token limit.

    Args:
        text: Text to chunk
        chunk_length: Target words per chunk (default 400)

    Returns:
        List of text chunks, each guaranteed to be under _MAX_TOKENS tokens
    """
    words = text.split()
    raw_chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1

        if current_length >= chunk_length:
            raw_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        raw_chunks.append(" ".join(current_chunk))

    # Guarantee every chunk fits within OpenAI's embedding token limit
    safe_chunks = []
    for chunk in raw_chunks:
        safe_chunks.extend(_enforce_token_limit(chunk))

    return safe_chunks


def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with metadata: title, author, subject, creation_date
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            metadata = reader.metadata
            
            return {
                "title": metadata.get("/Title", "Unknown") if metadata else "Unknown",
                "author": metadata.get("/Author", "Unknown") if metadata else "Unknown",
                "subject": metadata.get("/Subject", "Unknown") if metadata else "Unknown",
                "creation_date": metadata.get("/CreationDate", "Unknown") if metadata else "Unknown",
                "file_name": os.path.basename(pdf_path),
                "file_path": pdf_path
            }
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        return {
            "title": "Unknown",
            "author": "Unknown",
            "subject": "Unknown",
            "creation_date": "Unknown",
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path
        }


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def process_pdf_file(pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Process a single PDF file: extract text and create chunks
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (chunks, metadata)
    """
    text = extract_text_from_pdf(pdf_path)
    metadata = extract_metadata(pdf_path)
    chunks = chunk_up_context(text, chunk_length=400)
    
    logger.info(f"Processed {pdf_path}: {len(chunks)} chunks created")
    return chunks, metadata


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files from a directory
    
    Args:
        directory: Directory path
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    path = Path(directory)
    
    for pdf_file in path.glob("**/*.pdf"):
        pdf_files.append(str(pdf_file))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files


def sanitize_id(text: str) -> str:
    """
    Sanitize text to create valid ChromaDB ID
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized ID (alphanumeric + underscore/dash)
    """
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^\w\-]', '_', text)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Truncate to reasonable length
    return sanitized[:255]


def create_chunk_id(file_name: str, chunk_index: int) -> str:
    """
    Create unique ID for a chunk
    
    Args:
        file_name: Name of source file
        chunk_index: Index of chunk within file
        
    Returns:
        Unique chunk ID
    """
    base = sanitize_id(file_name)
    return f"{base}_chunk_{chunk_index}"


def format_context_for_prompt(documents: List[str], metadatas: List[Dict], distances: List[float]) -> Tuple[str, List[Dict]]:
    """
    Format retrieved documents into a context string for the LLM prompt
    
    Args:
        documents: List of document chunks
        metadatas: List of metadata for each chunk
        distances: List of distance scores
        
    Returns:
        Tuple of (formatted_context_string, sources_list)
    """
    sources = []
    context_parts = []
    
    for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
        # Add to context
        context_parts.append(f"[Source {i+1}]\n{doc}\n")
        
        # Track source
        source = {
            "index": i + 1,
            "title": meta.get("title", "Unknown") if isinstance(meta, dict) else "Unknown",
            "author": meta.get("author", "Unknown") if isinstance(meta, dict) else "Unknown",
            "file": meta.get("file_name", "Unknown") if isinstance(meta, dict) else "Unknown",
            "relevance_score": round(1 - distance, 4)  # Convert distance to similarity
        }
        sources.append(source)
    
    context_string = "\n".join(context_parts)
    return context_string, sources


def create_system_prompt(context: str) -> str:
    """
    Create system prompt for OpenAI API with context
    
    Args:
        context: Retrieved context from ChromaDB
        
    Returns:
        Formatted system prompt
    """
    return f"""You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD). 
You have access to medical literature and knowledge about PKD.

Use the provided context to answer questions accurately and cite your sources.
If the context doesn't contain relevant information, say so explicitly.
Always prioritize accuracy and cite the source documents.

CONTEXT FROM MEDICAL LITERATURE:
{context}

Instructions:
1. Answer the user's question based on the provided context
2. Cite which source(s) you used
3. If information is not in the context, say "This information is not in my available knowledge base"
4. Provide clear, medically accurate information
5. Format your response clearly with sections if needed
"""


def load_session_config() -> Dict[str, Any]:
    """
    Load session configuration from environment or defaults
    
    Returns:
        Configuration dictionary
    """
    return {
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        "retry_delay": int(os.getenv("OPENAI_RETRY_DELAY", "2")),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        "top_k_results": int(os.getenv("OPENAI_TOP_K_RESULTS", "5"))
    }
