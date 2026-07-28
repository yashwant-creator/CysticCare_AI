"""
OpenAI Pipeline Utilities
Shared utilities for the OpenAI-based RAG pipeline
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
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

# The old ingestion path used 400-word, non-overlapping chunks.  Keep that
# path intact for callers of ``process_pdf_file``, but use this smaller,
# token-aware target for the metadata-rich PDF ingestion path below.  It is a
# useful size for paper passages: large enough to retain a result or
# recommendation, small enough for a reranker to distinguish relevant text.
DEFAULT_STRUCTURED_CHUNK_SIZE_TOKENS = 320
DEFAULT_STRUCTURED_CHUNK_OVERLAP_TOKENS = 50
_UNSECTIONED_HEADING = "Unsectioned text"


class _WhitespaceTokenizer:
    """Offline-safe, conservative fallback when tiktoken's model file is absent.

    tiktoken occasionally downloads its encoding definition on first use. PDF
    ingestion should still work in an offline deployment, so this fallback uses
    whitespace-delimited tokens. It is only a size estimate; OpenAI's encoding
    remains the preferred implementation whenever available.
    """

    name = "whitespace-fallback"

    def encode(self, text: str) -> List[str]:
        return re.findall(r"\S+", text or "")

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


def _get_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # Do not make document ingestion unavailable just because the
            # encoding cache is cold and the deployment has no outbound access.
            logger.warning(
                "Unable to load tiktoken's cl100k_base encoding; using a "
                "whitespace token estimate instead: %s",
                e,
            )
            _tokenizer = _WhitespaceTokenizer()
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


def _normalise_inline_whitespace(value: str) -> str:
    """Collapse inline whitespace while preserving the caller's line boundaries."""
    return re.sub(r"\s+", " ", value or "").strip()


def _normalise_heading(value: str) -> str:
    """Remove common heading numbering while preserving the human-readable label."""
    heading = _normalise_inline_whitespace(value)
    heading = re.sub(
        r"^(?:\d+(?:\.\d+){0,4}|[IVXLCM]{1,8})(?:\s*[.)-]\s*|\s+)",
        "",
        heading,
    )
    return heading.rstrip(":").strip() or _UNSECTIONED_HEADING


_KNOWN_SECTION_HEADINGS = {
    "abstract",
    "background",
    "introduction",
    "methods",
    "materials and methods",
    "methodology",
    "patients and methods",
    "results",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
    "clinical implications",
    "limitations",
    "strengths and limitations",
    "acknowledgments",
    "acknowledgements",
    "funding",
    "conflicts of interest",
    "declarations",
    "references",
    "supplementary material",
    "supplemental material",
}


def _is_likely_section_heading(line: str) -> bool:
    """Best-effort heading detection for plain text extracted from a PDF.

    PDF text extraction generally loses font information, so this deliberately
    favours conservative, recognisable research-paper headings and short
    numbered/all-caps labels. False negatives simply stay in the surrounding
    chunk; false positives are more harmful because they fragment a passage.
    """
    candidate = _normalise_inline_whitespace(line)
    if not candidate or len(candidate) > 140:
        return False
    if candidate.endswith((".", "!", "?", ";", ",")):
        return False

    canonical = _normalise_heading(candidate).casefold()
    if canonical in _KNOWN_SECTION_HEADINGS:
        return True

    words = candidate.split()
    if len(words) > 14:
        return False

    # Numbered headings such as "2.3 Kidney volume measurement".
    if re.match(
        r"^(?:\d+(?:\.\d+){0,4}|[IVXLCM]{1,8})\s*[.)-]?\s+[A-Z][A-Za-z0-9/&(), -]+$",
        candidate,
        re.IGNORECASE,
    ):
        return True

    # Contact details, page counters, and table labels are often short and
    # title-cased/all-caps after PDF extraction. Once an explicit numbered
    # heading has been handled above, digits are a strong signal that this is
    # not a useful semantic section boundary.
    if any(char.isdigit() for char in candidate):
        return False

    letters = [char for char in candidate if char.isalpha()]
    if not letters:
        return False

    # All-capital labels are common in journal PDFs. Require at least three
    # letters so a gene symbol such as PKD1 is not routinely treated as a
    # heading.
    upper_fraction = sum(char.isupper() for char in letters) / len(letters)
    if len(letters) >= 3 and upper_fraction >= 0.85:
        return True

    # A short title-cased label is also likely a heading, but only when nearly
    # every substantive word is title cased. This avoids treating a normal
    # sentence that merely starts with a capital as a section boundary.
    alpha_words = re.findall(r"[A-Za-z]+", candidate)
    if 1 < len(alpha_words) <= 10:
        title_cased = sum(word[0].isupper() for word in alpha_words if word)
        if title_cased / len(alpha_words) >= 0.8:
            return True

    return False


def _token_count(text: str) -> int:
    return len(_get_tokenizer().encode(text or ""))


def _split_long_unit(unit: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
    """Split an unusually long extracted line into token-safe source units.

    Most PDF lines are one sentence or paragraph, which lets the main chunker
    preserve natural boundaries. A few PDFs emit an entire page as one line;
    only those fall back to token-level slicing.
    """
    text = unit["text"]
    if _token_count(text) <= max_tokens:
        return [unit]

    # First divide on sentence endings. This keeps ordinary scientific prose
    # coherent before using a token-level fallback for genuinely long sentences.
    sentence_boundaries = [0]
    sentence_boundaries.extend(match.end() for match in re.finditer(r"[.!?](?=\s|$)", text))
    if sentence_boundaries[-1] != len(text):
        sentence_boundaries.append(len(text))

    sentence_units: List[Dict[str, Any]] = []
    for start, end in zip(sentence_boundaries, sentence_boundaries[1:]):
        raw_piece = text[start:end]
        stripped_piece = raw_piece.strip()
        if not stripped_piece:
            continue
        leading_whitespace = len(raw_piece) - len(raw_piece.lstrip())
        piece_start = start + leading_whitespace
        sentence_units.append({
            **unit,
            "text": stripped_piece,
            "start": unit["start"] + piece_start,
            "end": unit["start"] + piece_start + len(stripped_piece),
        })

    # If sentence splitting did not help (e.g. a table row or a very long
    # sentence), split the remaining unit by token count.
    safe_units: List[Dict[str, Any]] = []
    for sentence_unit in sentence_units or [unit]:
        if _token_count(sentence_unit["text"]) <= max_tokens:
            safe_units.append(sentence_unit)
            continue

        token_values = _get_tokenizer().encode(sentence_unit["text"])
        search_from = 0
        for token_start in range(0, len(token_values), max_tokens):
            raw_piece = _get_tokenizer().decode(token_values[token_start:token_start + max_tokens])
            piece = raw_piece.strip()
            if not piece:
                continue
            position = sentence_unit["text"].find(piece, search_from)
            if position < 0:
                # Decoding token ranges may normalize a boundary in a rare
                # encoding edge case. Preserve monotonic offsets in that case.
                position = search_from
            search_from = min(len(sentence_unit["text"]), position + len(piece))
            safe_units.append({
                **sentence_unit,
                "text": piece,
                "start": sentence_unit["start"] + position,
                "end": sentence_unit["start"] + position + len(piece),
            })

    return safe_units


def _join_units(units: List[Dict[str, Any]]) -> str:
    return " ".join(unit["text"] for unit in units if unit.get("text")).strip()


def _tail_overlap_units(units: List[Dict[str, Any]], overlap_tokens: int) -> List[Dict[str, Any]]:
    """Return the final source units needed to create a bounded chunk overlap."""
    if overlap_tokens <= 0 or not units:
        return []

    selected_reversed: List[Dict[str, Any]] = []
    consumed = 0
    for unit in reversed(units):
        unit_tokens = _token_count(unit["text"])
        remaining = overlap_tokens - consumed
        if remaining <= 0:
            break
        if unit_tokens <= remaining:
            selected_reversed.append(unit)
            consumed += unit_tokens
            continue

        token_values = _get_tokenizer().encode(unit["text"])
        raw_tail = _get_tokenizer().decode(token_values[-remaining:])
        tail = raw_tail.strip()
        if tail:
            position = unit["text"].rfind(tail)
            if position < 0:
                position = max(0, len(unit["text"]) - len(tail))
            selected_reversed.append({
                **unit,
                "text": tail,
                "start": unit["start"] + position,
                "end": unit["start"] + position + len(tail),
            })
        break

    return list(reversed(selected_reversed))


def _context_prefix(document_title: Optional[str], section_heading: str) -> str:
    """Give embeddings compact paper/section context without hiding source text."""
    labels = []
    title = _normalise_inline_whitespace(str(document_title or ""))
    if title and title.casefold() not in {"unknown", "untitled"}:
        labels.append(f"Document: {title[:240]}")
    if section_heading:
        labels.append(f"Section: {section_heading[:160]}")
    return f"[{' | '.join(labels)}]\n" if labels else ""


def chunk_pdf_pages(
    pages: List[Dict[str, Any]],
    *,
    document_key: str,
    document_title: Optional[str] = None,
    chunk_size_tokens: int = DEFAULT_STRUCTURED_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_STRUCTURED_CHUNK_OVERLAP_TOKENS,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Create page- and section-aware chunks from extracted PDF page text.

    ``pages`` contains dictionaries with ``page_number`` and ``text`` keys, as
    returned by :func:`extract_text_pages_from_pdf`. The returned metadata is
    deliberately flat (strings and integers only), so it can be stored directly
    in Chroma metadata alongside the document-level metadata.

    ``chunk_start`` and ``chunk_end`` are offsets in the normalized, joined PDF
    text. ``page_number`` is the first source page for a chunk; ``page_start``
    and ``page_end`` make cross-page passages explicit.
    """
    if chunk_size_tokens < 64:
        raise ValueError("chunk_size_tokens must be at least 64")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")

    chunk_size_tokens = min(chunk_size_tokens, _MAX_TOKENS)
    chunk_overlap_tokens = min(chunk_overlap_tokens, chunk_size_tokens - 1)
    normalized_document_key = sanitize_id(document_key) or "document"

    sections: List[Dict[str, Any]] = [{
        "heading": _UNSECTIONED_HEADING,
        "section_index": 0,
        "units": [],
    }]
    current_section = sections[0]
    document_offset = 0

    for page_position, page in enumerate(pages):
        try:
            page_number = int(page.get("page_number", page_position + 1))
        except (TypeError, ValueError):
            page_number = page_position + 1

        raw_text = str(page.get("text") or "")
        normalized_lines = [
            _normalise_inline_whitespace(line)
            for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        normalized_page_text = "\n".join(normalized_lines)
        line_offset = 0

        for line in normalized_lines:
            line_start = document_offset + line_offset
            line_end = line_start + len(line)
            line_offset += len(line) + 1

            if not line:
                continue
            if _is_likely_section_heading(line):
                current_section = {
                    "heading": _normalise_heading(line),
                    "section_index": len(sections),
                    "units": [],
                }
                sections.append(current_section)
                continue

            current_section["units"].append({
                "text": line,
                "page_number": page_number,
                "start": line_start,
                "end": line_end,
            })

        # One normalized newline separates source pages. This makes offsets
        # monotonic even when an extracted page is blank.
        document_offset += len(normalized_page_text) + 1

    raw_chunk_records: List[Dict[str, Any]] = []
    for section in sections:
        section_heading = section["heading"]
        prefix = _context_prefix(document_title, section_heading)
        # Reserve room for the title/section prefix within the requested chunk
        # size. Long titles are truncated in _context_prefix, so 64 tokens is a
        # safe lower bound for meaningful source content.
        body_token_budget = max(64, chunk_size_tokens - _token_count(prefix))
        source_units: List[Dict[str, Any]] = []
        for unit in section["units"]:
            source_units.extend(_split_long_unit(unit, body_token_budget))

        current_units: List[Dict[str, Any]] = []
        for unit in source_units:
            if not current_units:
                current_units = [unit]
                continue

            candidate_units = current_units + [unit]
            if _token_count(_join_units(candidate_units)) <= body_token_budget:
                current_units = candidate_units
                continue

            raw_chunk_records.append({
                "body": _join_units(current_units),
                "units": current_units,
                "section_heading": section_heading,
                "section_index": section["section_index"],
                "prefix": prefix,
            })

            # Retain overlap only within the same section. A neighboring chunk
            # from a different paper section is usually less useful than a
            # smaller, semantically clean passage.
            available_overlap = max(0, body_token_budget - _token_count(unit["text"]))
            current_units = _tail_overlap_units(
                current_units,
                min(chunk_overlap_tokens, available_overlap),
            )
            current_units.append(unit)

        if current_units:
            raw_chunk_records.append({
                "body": _join_units(current_units),
                "units": current_units,
                "section_heading": section_heading,
                "section_index": section["section_index"],
                "prefix": prefix,
            })

    chunks: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []
    for chunk_index, record in enumerate(raw_chunk_records):
        body = record["body"]
        if not body:
            continue
        units = record["units"]
        section_index = record["section_index"]
        parent_chunk_id = f"{normalized_document_key}_section_{section_index}"
        chunk_id = f"{normalized_document_key}_chunk_{chunk_index}"
        chunk_text = f"{record['prefix']}{body}"

        chunks.append(chunk_text)
        chunk_metadatas.append({
            "content_type": "text",
            "document_key": normalized_document_key,
            "parent_document_id": normalized_document_key,
            "parent_chunk_id": parent_chunk_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section_index": section_index,
            "section_heading": record["section_heading"],
            "page_number": int(units[0]["page_number"]),
            "page_start": int(min(unit["page_number"] for unit in units)),
            "page_end": int(max(unit["page_number"] for unit in units)),
            "chunk_start": int(min(unit["start"] for unit in units)),
            "chunk_end": int(max(unit["end"] for unit in units)),
            "token_count": _token_count(chunk_text),
        })

    # Add adjacency metadata after all chunks are known. Restrict adjacency to
    # the same parent section so a later retrieval expansion does not pull a
    # Methods passage into an answer about a Results finding.
    for index, metadata in enumerate(chunk_metadatas):
        previous_metadata = chunk_metadatas[index - 1] if index > 0 else None
        next_metadata = chunk_metadatas[index + 1] if index + 1 < len(chunk_metadatas) else None
        if previous_metadata and previous_metadata["parent_chunk_id"] == metadata["parent_chunk_id"]:
            metadata["previous_chunk_index"] = previous_metadata["chunk_index"]
            metadata["previous_chunk_id"] = previous_metadata["chunk_id"]
        else:
            metadata["previous_chunk_index"] = -1
            metadata["previous_chunk_id"] = ""
        if next_metadata and next_metadata["parent_chunk_id"] == metadata["parent_chunk_id"]:
            metadata["next_chunk_index"] = next_metadata["chunk_index"]
            metadata["next_chunk_id"] = next_metadata["chunk_id"]
        else:
            metadata["next_chunk_index"] = -1
            metadata["next_chunk_id"] = ""

    return chunks, chunk_metadatas


def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with metadata: title, author, subject, creation_date
    """
    try:
        from PyPDF2 import PdfReader

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


def extract_text_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract PDF text while retaining the source page for each text block.

    ``pdfplumber`` can return ``None`` for scanned or image-only pages. Those
    pages are retained as empty records so later page references remain true to
    the PDF's page numbering, but they naturally produce no text chunks.
    """
    try:
        import pdfplumber

        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                pages.append({
                    "page_number": page_number,
                    "text": page.extract_text() or "",
                })
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return []


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber (legacy flat-text helper)."""
    page_records = extract_text_pages_from_pdf(pdf_path)
    return "".join(f"{page['text']}\n" for page in page_records)


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


def process_pdf_file_with_metadata(
    pdf_path: str,
    chunk_size_tokens: int = DEFAULT_STRUCTURED_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_STRUCTURED_CHUNK_OVERLAP_TOKENS,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Process a PDF into structured chunks and Chroma-ready chunk metadata.

    This is the opt-in ingestion API for the improved retrieval pipeline. It
    intentionally does not alter :func:`process_pdf_file`, whose two-item return
    value and 400-word chunking remain available to existing callers.

    Returns:
        A tuple of ``(chunks, basic_metadata, chunk_metadatas)``. Each entry in
        ``chunk_metadatas`` aligns by index with its corresponding item in
        ``chunks`` and includes page, section, source-span, parent, and neighbor
        information.
    """
    page_records = extract_text_pages_from_pdf(pdf_path)
    basic_metadata = extract_metadata(pdf_path)
    # Keep the document parent independent of the file extension. The indexer
    # can translate the helper's internal chunk IDs to its storage IDs, while
    # this semantic key remains stable for paper-level grouping and matches the
    # figure chunks generated from the same PDF.
    document_key = sanitize_id(Path(pdf_path).stem) or "document"
    chunks, chunk_metadatas = chunk_pdf_pages(
        page_records,
        document_key=document_key,
        document_title=basic_metadata.get("title"),
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    logger.info(
        "Processed %s into %s structure-aware chunks across %s extracted pages",
        pdf_path,
        len(chunks),
        len(page_records),
    )
    return chunks, basic_metadata, chunk_metadatas


def _find_caption(page, image_rect, max_gap: float = 160.0) -> str:
    """
    Best-effort caption lookup for an image on a page.

    Prefers a nearby text block beginning with "Figure"/"Fig"/"Table"; otherwise
    falls back to the closest text block sitting just below the image.

    Args:
        page: A PyMuPDF page object
        image_rect: The image's bounding box (fitz.Rect), or None
        max_gap: Maximum vertical distance (points) below the image to search

    Returns:
        Caption text, or "" if none found
    """
    if image_rect is None:
        return ""

    try:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    except Exception:
        return ""

    keyword_re = re.compile(r"^\s*(figure|fig\.?|table)\b", re.IGNORECASE)
    best_below = None
    best_gap = max_gap

    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        text = (text or "").strip()
        if not text:
            continue

        # Caption keyword anywhere reasonably close to the image (above or below)
        if keyword_re.match(text) and abs(y0 - image_rect.y1) <= max_gap * 1.5:
            return " ".join(text.split())[:500]

        # Otherwise track the closest block directly below the image
        gap = y0 - image_rect.y1
        if 0 <= gap < best_gap:
            best_gap = gap
            best_below = text

    if best_below:
        return " ".join(best_below.split())[:500]
    return ""


def extract_images_from_pdf(
    pdf_path: str,
    min_width: int = 100,
    min_height: int = 100,
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF along with a best-effort caption.

    Filters out tiny images and extreme aspect ratios (rules, separators, icons)
    and de-duplicates images that repeat across pages (e.g. journal logos).

    Args:
        pdf_path: Path to PDF file
        min_width: Minimum image width in pixels to keep
        min_height: Minimum image height in pixels to keep

    Returns:
        List of dicts with keys: image_bytes, ext, page_number, caption
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) is not installed; skipping image extraction. "
            "Install with `pip install PyMuPDF` to enable figure descriptions."
        )
        return []

    images: List[Dict[str, Any]] = []
    seen_xrefs: set = set()

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening {pdf_path} for image extraction: {e}")
        return []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue

                width = base.get("width", 0)
                height = base.get("height", 0)
                if width < min_width or height < min_height:
                    continue

                aspect = width / height if height else 0
                if aspect == 0 or aspect > 20 or aspect < 0.05:
                    continue  # likely a rule, border, or thin separator

                try:
                    rects = page.get_image_rects(xref)
                    image_rect = rects[0] if rects else None
                except Exception:
                    image_rect = None

                images.append({
                    "image_bytes": base["image"],
                    "ext": base.get("ext", "png"),
                    "page_number": page_index + 1,
                    "caption": _find_caption(page, image_rect),
                })
    finally:
        doc.close()

    logger.info(f"Extracted {len(images)} candidate images from {os.path.basename(pdf_path)}")
    return images


def process_pdf_images(pdf_path: str, openai_service) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract images from a PDF and turn each into an embeddable text chunk
    using the vision model to describe the figure.

    Args:
        pdf_path: Path to PDF file
        openai_service: An initialized OpenAIService (provides describe_image)

    Returns:
        List of (chunk_text, image_info) tuples, one per informative image.
        image_info carries page_number and caption for downstream metadata.
    """
    image_chunks: List[Tuple[str, Dict[str, Any]]] = []
    images = extract_images_from_pdf(pdf_path)

    for image in images:
        try:
            description = openai_service.describe_image(
                image_bytes=image["image_bytes"],
                image_ext=image["ext"],
                caption=image["caption"],
            )
        except Exception as e:
            logger.error(
                f"Vision description failed for image on page {image['page_number']} "
                f"of {os.path.basename(pdf_path)}: {e}"
            )
            continue

        if not description:
            continue  # decorative / non-informative image

        page_no = image["page_number"]
        caption = image["caption"]
        header = f"[Figure on page {page_no}]"
        if caption:
            header += f" Caption: {caption}"
        chunk_text = f"{header}\nDescription: {description}"

        image_chunks.extend(
            (safe_chunk, {"page_number": page_no, "caption": caption})
            for safe_chunk in _enforce_token_limit(chunk_text)
        )

    logger.info(
        f"Generated {len(image_chunks)} figure description chunks from "
        f"{os.path.basename(pdf_path)}"
    )
    return image_chunks


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
    
    # Stable ordering makes duplicate-paper resolution reproducible across
    # rebuilds and platforms.
    for pdf_file in sorted(path.glob("**/*.pdf")):
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
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5"),
        "vision_model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.5"),
        "enable_image_descriptions": os.getenv("OPENAI_ENABLE_IMAGE_DESCRIPTIONS", "true").lower() == "true",
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        "top_k_results": int(os.getenv("OPENAI_TOP_K_RESULTS", "5"))
    }
