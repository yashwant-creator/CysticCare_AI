"""
Standalone script to rebuild the ChromaDB vector database from PDFs.
Run from the backend/ directory: python rebuild_vectordb.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    import chromadb

    chroma_path = os.path.join(os.path.dirname(__file__), "app", "openai_chroma_data")
    collection_name = "pkd_knowledge_base_openai"

    logger.info("Deleting existing collection if present...")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        logger.info("No existing collection to delete")

    from app.services.openai_rag_init import initialize_openai_rag_system

    logger.info("Starting vector DB rebuild from PDFs...")
    result = await initialize_openai_rag_system(
        pdf_directory="papers",
        collection_name=collection_name,
    )

    logger.info("=" * 60)
    logger.info("Result: %s", result["status"])
    logger.info("Message: %s", result["message"])
    logger.info("Documents processed: %s", result.get("documents_processed", 0))
    logger.info("Chunks created: %s", result.get("chunks_created", 0))
    logger.info("Total vectors: %s", result.get("total_vectors", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
