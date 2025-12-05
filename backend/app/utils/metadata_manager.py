"""
Metadata Manager for RAG System
Handles extraction, storage, and retrieval of PDF metadata for better source attribution
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages metadata for PDF documents in the RAG system
    Provides consistent metadata extraction and formatting
    """
    
    def __init__(self, metadata_cache_path: Optional[str] = None):
        """
        Initialize metadata manager
        
        Args:
            metadata_cache_path: Path to cache metadata JSON file
        """
        self.metadata_cache_path = metadata_cache_path or os.path.join(
            os.path.dirname(__file__), "..", "metadata_cache.json"
        )
        self.metadata_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load metadata cache from disk"""
        if os.path.exists(self.metadata_cache_path):
            try:
                with open(self.metadata_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save metadata cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.metadata_cache_path), exist_ok=True)
            with open(self.metadata_cache_path, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
            logger.info(f"Saved metadata cache to {self.metadata_cache_path}")
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
    
    def extract_enhanced_metadata(self, pdf_path: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and enhance metadata from PDF file
        
        Args:
            pdf_path: Path to PDF file
            pdf_metadata: Basic metadata from PyPDF2
            
        Returns:
            Enhanced metadata dictionary
        """
        file_name = os.path.basename(pdf_path)
        
        # Try to extract author and year from filename (common pattern: Author_Year.pdf)
        filename_parts = self._parse_filename(file_name)
        
        # Get basic metadata
        title = pdf_metadata.get("title", "Unknown")
        author = pdf_metadata.get("author", "Unknown")
        
        # If metadata is missing or generic, try to infer from filename
        if title == "Unknown" or title == "" or "untitled" in title.lower():
            title = filename_parts.get("title", file_name.replace(".pdf", ""))
        
        if author == "Unknown" or author == "":
            author = filename_parts.get("author", "Unknown Author")
        
        # Create enhanced metadata
        enhanced_metadata = {
            "file_name": file_name,
            "file_path": pdf_path,
            "title": self._clean_metadata_string(title),
            "author": self._clean_metadata_string(author),
            "year": filename_parts.get("year", "Unknown"),
            "subject": pdf_metadata.get("subject", "Polycystic Kidney Disease"),
            "creation_date": pdf_metadata.get("creation_date", "Unknown"),
            "source_type": "scientific_paper",
            "indexed_date": datetime.now().isoformat(),
            # Additional fields for better attribution
            "citation": self._create_citation(
                author=self._clean_metadata_string(author),
                year=filename_parts.get("year", "Unknown"),
                title=self._clean_metadata_string(title)
            ),
            "display_name": self._create_display_name(
                author=self._clean_metadata_string(author),
                year=filename_parts.get("year", "Unknown")
            )
        }
        
        # Cache the metadata
        self.metadata_cache[file_name] = enhanced_metadata
        self._save_cache()
        
        return enhanced_metadata
    
    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse author and year from filename patterns like:
        - Author_Year.pdf
        - Author_et_al_Year.pdf
        - LastName_YYYY.pdf
        
        Args:
            filename: PDF filename
            
        Returns:
            Dictionary with parsed information
        """
        result = {}
        
        # Remove .pdf extension
        base_name = filename.replace(".pdf", "")
        
        # Pattern 1: Author_Year (e.g., Bergmann_2018.pdf)
        pattern1 = r'^([A-Za-z]+)_(\d{4})'
        match = re.match(pattern1, base_name)
        if match:
            result["author"] = match.group(1)
            result["year"] = match.group(2)
            result["title"] = f"{match.group(1)} et al. ({match.group(2)})"
            return result
        
        # Pattern 2: Author_et_al_Year
        pattern2 = r'^([A-Za-z]+)_et_al_(\d{4})'
        match = re.match(pattern2, base_name)
        if match:
            result["author"] = f"{match.group(1)} et al."
            result["year"] = match.group(2)
            result["title"] = f"{match.group(1)} et al. ({match.group(2)})"
            return result
        
        # Pattern 3: Extract any year (4 digits)
        year_match = re.search(r'(\d{4})', base_name)
        if year_match:
            result["year"] = year_match.group(1)
        
        # Pattern 4: Extract potential author name (first word)
        words = base_name.split('_')
        if words:
            result["author"] = words[0]
            result["title"] = base_name.replace('_', ' ')
        
        return result
    
    def _clean_metadata_string(self, value: str) -> str:
        """Clean and normalize metadata strings"""
        if not value or value == "Unknown":
            return value
        
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'\s+', ' ', value)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _create_citation(self, author: str, year: str, title: str) -> str:
        """
        Create a formatted citation string
        
        Args:
            author: Author name
            year: Publication year
            title: Paper title
            
        Returns:
            Formatted citation string
        """
        if author == "Unknown" or author == "":
            author = "Unknown Author"
        if year == "Unknown" or year == "":
            year = "n.d."
        if title == "Unknown" or title == "":
            title = "Untitled"
        
        return f"{author} ({year}). {title}"
    
    def _create_display_name(self, author: str, year: str) -> str:
        """
        Create a short display name for source attribution
        
        Args:
            author: Author name
            year: Publication year
            
        Returns:
            Display name like "Bergmann 2018"
        """
        if author == "Unknown" or author == "":
            author = "Unknown"
        if year == "Unknown" or year == "":
            year = "n.d."
        
        return f"{author} {year}"
    
    def get_metadata_for_file(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached metadata for a file
        
        Args:
            file_name: Name of the PDF file
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata_cache.get(file_name)
    
    def format_source_citation(self, metadata: Dict[str, Any], chunk_index: int = None) -> str:
        """
        Format metadata into a human-readable source citation
        
        Args:
            metadata: Metadata dictionary
            chunk_index: Optional chunk index for specific reference
            
        Returns:
            Formatted citation string
        """
        citation = metadata.get("citation", "")
        if not citation:
            display_name = metadata.get("display_name", "Unknown Source")
            citation = display_name
        
        if chunk_index is not None:
            citation += f" [Chunk {chunk_index}]"
        
        return citation
    
    def export_metadata_summary(self, output_path: str = None) -> str:
        """
        Export a summary of all cached metadata
        
        Args:
            output_path: Path to save summary (optional)
            
        Returns:
            Summary as formatted string
        """
        if not self.metadata_cache:
            return "No metadata cached"
        
        summary_lines = [
            "=" * 80,
            "PDF METADATA SUMMARY",
            "=" * 80,
            f"Total documents: {len(self.metadata_cache)}",
            "",
            "Documents:"
        ]
        
        for file_name, metadata in sorted(self.metadata_cache.items()):
            summary_lines.append(f"\n  File: {file_name}")
            summary_lines.append(f"    Citation: {metadata.get('citation', 'N/A')}")
            summary_lines.append(f"    Display: {metadata.get('display_name', 'N/A')}")
            summary_lines.append(f"    Author: {metadata.get('author', 'N/A')}")
            summary_lines.append(f"    Year: {metadata.get('year', 'N/A')}")
        
        summary = "\n".join(summary_lines)
        
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(summary)
                logger.info(f"Metadata summary saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving metadata summary: {e}")
        
        return summary


# Global metadata manager instance
_metadata_manager = None


def get_metadata_manager() -> MetadataManager:
    """Get or create global metadata manager instance"""
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MetadataManager()
    return _metadata_manager
