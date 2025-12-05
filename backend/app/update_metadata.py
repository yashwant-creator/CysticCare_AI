"""
Update ChromaDB Metadata Script
Updates existing ChromaDB collection with enhanced metadata from PDFs
Run this to fix missing metadata in your deployed vector store
"""

import os
import sys
import logging
from pathlib import Path
import chromadb
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.openai_utils import extract_metadata, get_pdf_files
from utils.metadata_manager import MetadataManager

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_chromadb_metadata(
    chroma_path: str = None,
    collection_name: str = "pkd_knowledge_base_openai",
    pdf_directory: str = "papers"
):
    """
    Update metadata in existing ChromaDB collection
    
    Args:
        chroma_path: Path to ChromaDB storage
        collection_name: Name of the collection to update
        pdf_directory: Directory containing PDF files
    """
    # Initialize paths
    if chroma_path is None:
        chroma_path = os.path.join(os.path.dirname(__file__), "openai_chroma_data")
    
    pdf_directory = os.path.join(os.path.dirname(__file__), pdf_directory)
    
    logger.info("=" * 80)
    logger.info("CHROMADB METADATA UPDATE UTILITY")
    logger.info("=" * 80)
    logger.info(f"ChromaDB path: {chroma_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"PDF directory: {pdf_directory}")
    logger.info("")
    
    # Check if ChromaDB exists
    if not os.path.exists(chroma_path):
        logger.error(f"ChromaDB path not found: {chroma_path}")
        return
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_directory):
        logger.error(f"PDF directory not found: {pdf_directory}")
        return
    
    try:
        # Initialize ChromaDB client
        logger.info("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=chroma_path)
        
        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✓ Found collection: {collection_name}")
            logger.info(f"  Total vectors: {collection.count()}")
        except Exception as e:
            logger.error(f"Collection not found: {e}")
            return
        
        # Initialize metadata manager
        metadata_manager = MetadataManager()
        
        # Get all PDF files
        pdf_files = get_pdf_files(pdf_directory)
        logger.info(f"\nFound {len(pdf_files)} PDF files")
        
        # Create metadata mapping
        metadata_map = {}
        logger.info("\nExtracting metadata from PDFs...")
        
        for pdf_path in pdf_files:
            try:
                # Extract basic metadata
                basic_metadata = extract_metadata(pdf_path)
                
                # Enhance with metadata manager
                enhanced_metadata = metadata_manager.extract_enhanced_metadata(
                    pdf_path, basic_metadata
                )
                
                file_name = os.path.basename(pdf_path)
                metadata_map[file_name] = enhanced_metadata
                
                logger.info(f"  ✓ {enhanced_metadata['display_name']}: {enhanced_metadata['citation']}")
                
            except Exception as e:
                logger.error(f"  ✗ Error processing {pdf_path}: {e}")
        
        # Get all items from collection
        logger.info("\nRetrieving collection data...")
        all_data = collection.get(
            include=["metadatas", "documents", "embeddings"]
        )
        
        ids = all_data["ids"]
        metadatas = all_data["metadatas"]
        documents = all_data["documents"]
        embeddings = all_data["embeddings"]
        
        logger.info(f"Retrieved {len(ids)} items from collection")
        
        # Update metadata
        logger.info("\nUpdating metadata...")
        updated_count = 0
        
        updated_metadatas = []
        for i, (item_id, metadata) in enumerate(zip(ids, metadatas)):
            # Extract file name from ID (format: filename_chunk_N)
            id_parts = item_id.split("_chunk_")
            if len(id_parts) >= 2:
                # Reconstruct filename (everything before last _chunk_N)
                file_name_base = "_chunk_".join(id_parts[:-1])
                # Add .pdf extension if not present
                if not file_name_base.endswith('.pdf'):
                    file_name_base += '.pdf'
            else:
                file_name_base = metadata.get("file_name", "unknown.pdf")
            
            # Find matching metadata
            enhanced_metadata = None
            for pdf_file, pdf_meta in metadata_map.items():
                if pdf_file in file_name_base or file_name_base in pdf_file:
                    enhanced_metadata = pdf_meta
                    break
            
            if enhanced_metadata:
                # Update metadata with enhanced version
                updated_metadata = {
                    **metadata,  # Keep existing metadata
                    "title": enhanced_metadata["title"],
                    "author": enhanced_metadata["author"],
                    "year": enhanced_metadata["year"],
                    "citation": enhanced_metadata["citation"],
                    "display_name": enhanced_metadata["display_name"],
                    "source_type": enhanced_metadata["source_type"],
                    "file_name": enhanced_metadata["file_name"]
                }
                updated_metadatas.append(updated_metadata)
                updated_count += 1
            else:
                # Keep existing metadata
                updated_metadatas.append(metadata)
                logger.warning(f"  No enhanced metadata found for: {file_name_base}")
        
        # Update collection with new metadata
        if updated_count > 0:
            logger.info(f"\nUpdating {updated_count} items in ChromaDB...")
            
            # ChromaDB doesn't have a direct update method, so we need to upsert
            collection.upsert(
                ids=ids,
                metadatas=updated_metadatas,
                documents=documents,
                embeddings=embeddings
            )
            
            logger.info(f"✓ Successfully updated {updated_count} items")
        else:
            logger.warning("No items were updated")
        
        # Export metadata summary
        logger.info("\n" + "=" * 80)
        summary = metadata_manager.export_metadata_summary(
            output_path=os.path.join(os.path.dirname(__file__), "metadata_summary.txt")
        )
        logger.info("\nMetadata Summary:")
        logger.info(summary)
        
        logger.info("\n" + "=" * 80)
        logger.info("METADATA UPDATE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total items in collection: {collection.count()}")
        logger.info(f"Items updated: {updated_count}")
        logger.info(f"Metadata cache saved to: {metadata_manager.metadata_cache_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update ChromaDB metadata with enhanced PDF information"
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Path to ChromaDB storage directory"
    )
    parser.add_argument(
        "--collection",
        default="pkd_knowledge_base_openai",
        help="Name of ChromaDB collection"
    )
    parser.add_argument(
        "--pdf-dir",
        default="papers",
        help="Directory containing PDF files"
    )
    
    args = parser.parse_args()
    
    update_chromadb_metadata(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        pdf_directory=args.pdf_dir
    )
