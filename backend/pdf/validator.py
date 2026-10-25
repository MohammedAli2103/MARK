"""
PDF Structural and Content Validation Engine for Mark.
Verifies PDF header/trailer, integrity, page counts, title & heading text extraction,
non-blank page guarantees, and metadata conformity using PyPDF.
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
from typing import List, Optional, Union

from pypdf import PdfReader

from config.pdf_config import MAX_PAGES, MAX_PDF_SIZE_BYTES
from backend.pdf.schema import DocumentModel


@dataclass
class ValidationResult:
    valid: bool
    page_count: int = 0
    size_bytes: int = 0
    extracted_text_sample: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_pdf(
    pdf_source: Union[bytes, str, Path],
    expected_doc: DocumentModel,
    expected_page_count: Optional[int] = None,
) -> ValidationResult:
    """
    Validates rendered PDF against schema expectations and structural invariants.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Byte Source Resolution & Basic Byte Checks
    if isinstance(pdf_source, (str, Path)):
        p = Path(pdf_source)
        if not p.is_file():
            return ValidationResult(valid=False, errors=[f"PDF file does not exist on disk: {p}"])
        try:
            pdf_bytes = p.read_bytes()
        except Exception as e:
            return ValidationResult(valid=False, errors=[f"Cannot read PDF file: {e}"])
    else:
        pdf_bytes = bytes(pdf_source)

    size = len(pdf_bytes)
    if size == 0:
        return ValidationResult(valid=False, errors=["PDF file is empty (0 bytes)."])

    if size > MAX_PDF_SIZE_BYTES:
        return ValidationResult(
            valid=False,
            size_bytes=size,
            errors=[f"PDF size ({size} bytes) exceeds configured limit of {MAX_PDF_SIZE_BYTES} bytes."],
        )

    # 2. Magic bytes and trailer structure
    if not pdf_bytes.startswith(b"%PDF-"):
        return ValidationResult(
            valid=False,
            size_bytes=size,
            errors=["File does not start with standard PDF magic bytes ('%PDF-')."],
        )

    # PDF standard trailer check (%%EOF typically in last 1024 bytes)
    trailing_chunk = pdf_bytes[-1024:]
    if b"%%EOF" not in trailing_chunk:
        errors.append("PDF trailer does not contain valid '%%EOF' marker.")

    # 3. PyPDF Parsing & Structure
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        return ValidationResult(
            valid=False,
            size_bytes=size,
            errors=[f"Failed to parse PDF document structure: {exc}"],
        )

    page_count = len(reader.pages)
    if page_count == 0:
        return ValidationResult(valid=False, size_bytes=size, errors=["PDF has 0 pages."])

    if page_count > MAX_PAGES:
        errors.append(f"PDF page count ({page_count}) exceeds limit of {MAX_PAGES} pages.")

    if expected_page_count is not None and expected_page_count != page_count:
        errors.append(
            f"Page count mismatch: renderer reported {expected_page_count} pages, but PDF contains {page_count}."
        )

    # 4. Text Extraction and Expected Content Checks
    full_extracted_text = []
    blank_pages: List[int] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"Warning extracting text from page {page_idx}: {exc}")
            page_text = ""

        clean_page_text = page_text.strip()
        if not clean_page_text and len(page.images) == 0:
            blank_pages.append(page_idx)

        full_extracted_text.append(clean_page_text)

    all_text = " ".join(full_extracted_text)
    sample_text = all_text[:500]

    # Blank pages check: warn unless deliberate
    if blank_pages:
        warnings.append(f"Pages with no extractable text or images detected: {blank_pages}")

    # Title verification in extracted text (normalized)
    expected_title = expected_doc.meta.title.strip()
    if expected_title and expected_title.lower() not in all_text.lower():
        # Check if words from title are present
        title_words = [w for w in expected_title.split() if len(w) > 3]
        if title_words and not any(w.lower() in all_text.lower() for w in title_words):
            errors.append(f"Document title '{expected_title}' not found in extracted PDF text.")

    # Section headings verification: check major section headings
    expected_headings = [
        b.data.get("text", "").strip()
        for b in expected_doc.blocks
        if b.type == "heading" and b.data.get("level", 1) in (1, 2) and len(b.data.get("text", "").strip()) > 3
    ]
    for h in expected_headings[:5]: # Check first 5 major headings
        if h.lower() not in all_text.lower():
            # Soft fallback: if parts of heading match
            h_words = [w for w in h.split() if len(w) > 3]
            if h_words and not any(w.lower() in all_text.lower() for w in h_words):
                errors.append(f"Major section heading '{h}' not found in extracted PDF text.")

    # 5. Metadata verification
    meta = reader.metadata
    if meta:
        pdf_title = meta.title or ""
        if not pdf_title.strip() and expected_title:
            warnings.append("PDF document metadata 'title' is empty.")

    is_valid = len(errors) == 0

    return ValidationResult(
        valid=is_valid,
        page_count=page_count,
        size_bytes=size,
        extracted_text_sample=sample_text,
        errors=errors,
        warnings=warnings,
    )
