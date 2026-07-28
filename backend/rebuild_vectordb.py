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


async def preflight() -> bool:
    """Validate rebuild inputs before deleting the currently usable index."""
    from app.services.openai_rag_init import _resolve_pdf_directory
    from app.services.openai_service import OpenAIService
    from app.utils.openai_utils import get_pdf_files, load_session_config

    pdf_directory = _resolve_pdf_directory("papers")
    pdf_files = get_pdf_files(pdf_directory)
    if not pdf_files:
        logger.error("Rebuild preflight failed: no PDFs found in %s", pdf_directory)
        return False

    try:
        config = load_session_config()
        service = OpenAIService(
            embedding_model=config["embedding_model"],
            chat_model=os.getenv("BASE_CHAT_MODEL", "").strip() or config["chat_model"],
            vision_model=config["vision_model"],
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"],
        )
    except Exception as error:
        logger.error("Rebuild preflight failed while configuring OpenAI: %s", error)
        return False

    if not service.validate_connection():
        logger.error("Rebuild preflight failed: OpenAI embeddings are unavailable")
        return False

    logger.info("Rebuild preflight passed: %s PDFs and OpenAI embeddings available", len(pdf_files))
    return True


async def main() -> int:
    import chromadb

    chroma_path = os.path.join(os.path.dirname(__file__), "app", "openai_chroma_data")
    collection_name = "pkd_knowledge_base_openai"

    if not await preflight():
        logger.error("Aborting rebuild before deleting the existing collection")
        return 1

    logger.info("Deleting existing collection if present...")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collections = client.list_collections()
        collection_names = {getattr(collection, "name", str(collection)) for collection in collections}
    except Exception as error:
        logger.error("Could not inspect existing ChromaDB collections: %s", error)
        return 1

    if collection_name in collection_names:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception as error:
            logger.error("Could not delete existing collection '%s': %s", collection_name, error)
            return 1
    else:
        logger.info("No existing collection to delete")

    from app.integrations.traceai import setup_traceai
    from app.services import openai_rag_init

    setup_traceai()

    logger.info("Starting vector DB rebuild from PDFs...")
    result = await openai_rag_init.build_openai_rag_system(
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
    if result.get("status") != "success":
        logger.error("Rebuild failed; the collection remains marked incomplete and will not be served")
        return 1
    from app.services.index_manifest import write_manifest

    manifest = write_manifest()
    logger.info(
        "Wrote baked-index manifest: schema=%s vectors=%s",
        manifest["index_schema_version"],
        manifest["vector_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
