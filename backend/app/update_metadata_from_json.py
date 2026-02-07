"""
Update ChromaDB Metadata from JSON
Updates existing ChromaDB collection with metadata from a JSON file
Use this when PDF files are not available (e.g., in deployed environments)
"""

import os
import sys
import logging
import json
import re
import chromadb
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_filename_for_metadata(filename: str) -> dict:
    """
    Extract author and year from filename patterns
    Patterns: Author_Year.pdf, Author_et_al_Year.pdf
    """
    result = {"author": "Unknown", "year": "Unknown"}
    
    # Remove .pdf extension
    base_name = filename.replace(".pdf", "")
    
    # Pattern 1: Author_Year (e.g., Bergmann_2018.pdf)
    pattern1 = r'^([A-Za-z]+)_(\d{4})'
    match = re.match(pattern1, base_name)
    if match:
        result["author"] = match.group(1)
        result["year"] = match.group(2)
        return result
    
    # Pattern 2: Author_et_al_Year
    pattern2 = r'^([A-Za-z]+)_et_al_(\d{4})'
    match = re.match(pattern2, base_name)
    if match:
        result["author"] = f"{match.group(1)} et al."
        result["year"] = match.group(2)
        return result
    
    # Pattern 3: Extract any year (4 digits)
    year_match = re.search(r'(\d{4})', base_name)
    if year_match:
        result["year"] = year_match.group(1)
    
    # Pattern 4: Extract potential author name (first word before underscore)
    words = base_name.split('_')
    if words and len(words[0]) > 2:
        result["author"] = words[0]
    
    return result


def generate_metadata_template():
    """
    Generate a metadata JSON template from existing ChromaDB data
    This merges new papers with existing metadata instead of overwriting
    """
    chroma_path = os.path.join(os.path.dirname(__file__), "openai_chroma_data")
    collection_name = "pkd_knowledge_base_openai"
    output_file = os.path.join(os.path.dirname(__file__), "papers_metadata.json")
    
    logger.info("Generating metadata template from ChromaDB...")
    
    try:
        # Load existing metadata if it exists
        existing_metadata = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_metadata = json.load(f)
            logger.info(f"✓ Loaded {len(existing_metadata)} existing entries")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)
        
        # Get all unique filenames
        all_data = collection.get(include=["metadatas"])
        filenames = set()
        
        for metadata in all_data["metadatas"]:
            filename = metadata.get("file_name", "")
            if filename:
                filenames.add(filename)
        
        logger.info(f"✓ Found {len(filenames)} papers in ChromaDB")
        
        # Start with existing metadata
        metadata_template = existing_metadata.copy()
        
        # Add new entries for papers not in existing metadata
        new_count = 0
        for filename in sorted(filenames):
            if filename not in metadata_template:
                # Try to infer from filename
                inferred = parse_filename_for_metadata(filename)
                
                # Get title from first chunk if available
                title = "Unknown"
                for metadata in all_data["metadatas"]:
                    if metadata.get("file_name") == filename and metadata.get("title"):
                        title = metadata.get("title")
                        break
                
                metadata_template[filename] = {
                    "author": inferred["author"],
                    "year": inferred["year"],
                    "title": title,
                    "journal": "Unknown",
                    "notes": "Please update this entry with correct information"
                }
                new_count += 1
                logger.info(f"  + Added new entry: {filename}")
        
        # Save merged template
        with open(output_file, 'w') as f:
            json.dump(metadata_template, f, indent=2)
        
        logger.info(f"\n✓ Template updated: {output_file}")
        logger.info(f"✓ Total papers: {len(metadata_template)}")
        logger.info(f"✓ New papers added: {new_count}")
        logger.info(f"✓ Existing papers preserved: {len(existing_metadata)}")
        
        if new_count > 0:
            logger.info("\nNext steps:")
            logger.info("1. Edit papers_metadata.json to update the new entries")
            logger.info("2. Run this script again (without --generate-template) to apply metadata to ChromaDB")
        else:
            logger.info("\n✓ No new papers found - all papers already have metadata entries")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error generating template: {e}", exc_info=True)
        return None


def update_chromadb_from_json(
    metadata_file: str = None,
    chroma_path: str = None,
    collection_name: str = "pkd_knowledge_base_openai"
):
    """
    Update ChromaDB metadata from JSON file
    
    Args:
        metadata_file: Path to JSON file with metadata
        chroma_path: Path to ChromaDB storage
        collection_name: Name of the collection to update
    """
    # Initialize paths
    if chroma_path is None:
        chroma_path = os.path.join(os.path.dirname(__file__), "openai_chroma_data")
    
    if metadata_file is None:
        metadata_file = os.path.join(os.path.dirname(__file__), "papers_metadata.json")
    
    logger.info("=" * 80)
    logger.info("CHROMADB METADATA UPDATE FROM JSON")
    logger.info("=" * 80)
    logger.info(f"ChromaDB path: {chroma_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Metadata file: {metadata_file}")
    logger.info("")
    
    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file not found: {metadata_file}")
        logger.info("\nGenerating template file...")
        generate_metadata_template()
        return
    
    try:
        # Load metadata from JSON
        logger.info("Loading metadata from JSON...")
        with open(metadata_file, 'r') as f:
            metadata_map = json.load(f)
        
        logger.info(f"✓ Loaded metadata for {len(metadata_map)} papers")
        
        # Initialize ChromaDB client
        logger.info("\nConnecting to ChromaDB...")
        client = chromadb.PersistentClient(path=chroma_path)
        
        # Get collection
        collection = client.get_collection(name=collection_name)
        logger.info(f"✓ Found collection: {collection_name}")
        logger.info(f"  Total vectors: {collection.count()}")
        
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
        missing_count = 0
        
        updated_metadatas = []
        for i, (item_id, metadata) in enumerate(zip(ids, metadatas)):
            file_name = metadata.get("file_name", "unknown.pdf")
            
            # Find matching metadata in JSON
            enhanced_metadata = metadata_map.get(file_name)
            
            if enhanced_metadata:
                # Create citation
                author = enhanced_metadata.get("author", "Unknown")
                year = enhanced_metadata.get("year", "Unknown")
                title = enhanced_metadata.get("title", metadata.get("title", "Unknown"))
                
                # Format display name
                display_name = f"{author} {year}" if year != "Unknown" else author
                
                # Format citation
                citation = f"{author} ({year}). {title}"
                
                # Update metadata with enhanced version
                updated_metadata = {
                    **metadata,  # Keep existing metadata
                    "title": title,
                    "author": author,
                    "year": year,
                    "citation": citation,
                    "display_name": display_name,
                    "source_type": "scientific_paper",
                    "journal": enhanced_metadata.get("journal", "Unknown")
                }
                updated_metadatas.append(updated_metadata)
                updated_count += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{len(ids)} items...")
            else:
                # Keep existing metadata but add defaults for missing fields
                updated_metadata = {
                    **metadata,
                    "author": metadata.get("author", "Unknown"),
                    "year": "Unknown",
                    "citation": metadata.get("title", "Unknown Source"),
                    "display_name": file_name.replace(".pdf", ""),
                    "source_type": "scientific_paper"
                }
                updated_metadatas.append(updated_metadata)
                missing_count += 1
        
        # Update collection with new metadata
        logger.info(f"\nUpdating ChromaDB collection...")
        
        # ChromaDB upsert to update metadata
        collection.upsert(
            ids=ids,
            metadatas=updated_metadatas,
            documents=documents,
            embeddings=embeddings
        )
        
        logger.info(f"✓ Successfully updated metadata in ChromaDB")
        
        logger.info("\n" + "=" * 80)
        logger.info("METADATA UPDATE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total items in collection: {collection.count()}")
        logger.info(f"Items with updated metadata: {updated_count}")
        logger.info(f"Items with default metadata: {missing_count}")
        
        if missing_count > 0:
            logger.warning(f"\n⚠️  {missing_count} items don't have entries in metadata JSON")
            logger.info("Add their entries to papers_metadata.json for complete sourcing")
        
        logger.info("=" * 80)
        logger.info("\n✓ Sourcing feature is now fixed!")
        logger.info("✓ Citations will appear in chatbot responses")
        
    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update ChromaDB metadata from JSON file")
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate a metadata template JSON file"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        help="Path to metadata JSON file"
    )
    
    args = parser.parse_args()
    
    if args.generate_template:
        generate_metadata_template()
    else:
        update_chromadb_from_json(metadata_file=args.metadata_file)
