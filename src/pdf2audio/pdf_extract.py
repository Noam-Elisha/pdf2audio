"""Extract chapters and text from PDF files using PyMuPDF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class Chapter:
    """A chapter extracted from a PDF."""

    title: str
    text: str
    page_start: int
    page_end: int
    index: int  # 0-based chapter number


def extract_chapters(pdf_path: str) -> list[Chapter]:
    """Extract chapters from a PDF file.

    Tries multiple strategies in order:
    1. PDF table of contents (bookmarks/outlines)
    2. Heuristic heading detection (large/bold text patterns)
    3. Falls back to treating the entire document as one chapter
    """
    doc = fitz.open(pdf_path)

    chapters = _try_toc_extraction(doc)
    if not chapters:
        chapters = _try_heuristic_extraction(doc)
    if not chapters:
        chapters = _fallback_single_chapter(doc, pdf_path)

    doc.close()
    return chapters


def _try_toc_extraction(doc: fitz.Document) -> list[Chapter]:
    """Extract chapters using the PDF's table of contents."""
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    if not toc:
        return []

    # Only use top-level entries (level 1) or level 2 if no level 1
    min_level = min(entry[0] for entry in toc)
    top_entries = [(title.strip(), page - 1) for level, title, page in toc if level == min_level]

    if not top_entries:
        return []

    chapters = []
    for i, (title, start_page) in enumerate(top_entries):
        end_page = top_entries[i + 1][1] - 1 if i + 1 < len(top_entries) else len(doc) - 1
        text = _extract_text_range(doc, start_page, end_page)
        if text.strip():
            chapters.append(Chapter(
                title=_clean_title(title),
                text=text,
                page_start=start_page + 1,
                page_end=end_page + 1,
                index=len(chapters),
            ))

    return chapters


# Patterns that look like chapter headings
_CHAPTER_PATTERNS = [
    re.compile(r'^chapter\s+\d+', re.IGNORECASE),
    re.compile(r'^chapter\s+[IVXLCDM]+', re.IGNORECASE),
    re.compile(r'^part\s+\d+', re.IGNORECASE),
    re.compile(r'^part\s+[IVXLCDM]+', re.IGNORECASE),
    re.compile(r'^section\s+\d+', re.IGNORECASE),
    re.compile(r'^\d+\.\s+\S'),
]


def _try_heuristic_extraction(doc: fitz.Document) -> list[Chapter]:
    """Detect chapters by looking for heading-like text patterns."""
    headings: list[tuple[str, int]] = []  # (title, page_index)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text or len(line_text) > 100:
                    continue

                # Check if any span is large or bold
                is_heading = False
                for span in line["spans"]:
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    is_bold = flags & 2 ** 4  # bit 4 = bold
                    if size >= 16 or (is_bold and size >= 14):
                        is_heading = True
                        break

                # Also check against chapter patterns
                if not is_heading:
                    for pattern in _CHAPTER_PATTERNS:
                        if pattern.match(line_text):
                            is_heading = True
                            break

                if is_heading:
                    # Avoid duplicate headings on same page
                    if not headings or headings[-1][1] != page_idx or headings[-1][0] != line_text:
                        headings.append((line_text, page_idx))

    if len(headings) < 2:
        return []

    chapters = []
    for i, (title, start_page) in enumerate(headings):
        end_page = headings[i + 1][1] - 1 if i + 1 < len(headings) else len(doc) - 1
        end_page = max(end_page, start_page)
        text = _extract_text_range(doc, start_page, end_page)
        if text.strip():
            chapters.append(Chapter(
                title=_clean_title(title),
                text=text,
                page_start=start_page + 1,
                page_end=end_page + 1,
                index=len(chapters),
            ))

    return chapters


def _fallback_single_chapter(doc: fitz.Document, pdf_path: str) -> list[Chapter]:
    """Treat the entire PDF as a single chapter."""
    text = _extract_text_range(doc, 0, len(doc) - 1)
    if not text.strip():
        return []

    import os
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    return [Chapter(
        title=name,
        text=text,
        page_start=1,
        page_end=len(doc),
        index=0,
    )]


def _extract_text_range(doc: fitz.Document, start: int, end: int) -> str:
    """Extract and clean text from a range of pages."""
    parts = []
    for i in range(start, min(end + 1, len(doc))):
        page_text = doc[i].get_text("text")
        if page_text:
            parts.append(page_text)
    text = "\n".join(parts)
    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _clean_title(title: str) -> str:
    """Clean up a chapter title for use as a filename."""
    title = title.strip()
    # Remove leading/trailing punctuation
    title = title.strip('.:;,- ')
    return title if title else "Untitled"


def get_pdf_info(pdf_path: str) -> dict:
    """Get basic info about a PDF file."""
    doc = fitz.open(pdf_path)
    info = {
        "pages": len(doc),
        "metadata": doc.metadata,
        "has_toc": len(doc.get_toc()) > 0,
    }
    doc.close()
    return info
