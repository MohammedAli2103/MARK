"""
Deterministic Tool Interface for Mark PDF Operations.
Exposes standard callable tool signatures returning sanitized metadata records (never raw filesystem paths).
"""

from typing import Any, Dict, List, Optional
from backend.pdf.storage import storage
from backend.pdf.generator import generate_pdf_from_prompt, revise_pdf_from_prompt


def create_pdf(brief: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a new PDF document from a user brief and context.
    Returns:
        Dict with artifact metadata: artifact_id, display_name, size_bytes, page_count, version, status, warnings.
    """
    if not brief or not brief.strip():
        return {
            "status": "failed",
            "error": "A topic or brief is required to generate a PDF.",
        }

    try:
        record, summary = generate_pdf_from_prompt(brief, conversation_context=context)
        return {
            "status": "ready",
            "artifact_id": record.artifact_id,
            "display_name": record.display_name,
            "size_bytes": record.size_bytes,
            "page_count": record.page_count,
            "version": record.version,
            "sha256": record.sha256,
            "warnings": record.warnings,
            "summary": summary,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


def revise_pdf(instruction: str, artifact_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Revises an existing PDF document based on user instructions.
    """
    if not instruction or not instruction.strip():
        return {
            "status": "failed",
            "error": "A revision instruction is required.",
        }

    try:
        record, summary = revise_pdf_from_prompt(instruction, target_artifact_id=artifact_id)
        return {
            "status": "ready",
            "artifact_id": record.artifact_id,
            "display_name": record.display_name,
            "size_bytes": record.size_bytes,
            "page_count": record.page_count,
            "version": record.version,
            "parent_artifact_id": record.parent_artifact_id,
            "sha256": record.sha256,
            "warnings": record.warnings,
            "summary": summary,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


def list_pdfs(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent PDF artifacts."""
    records = storage.list_artifacts(limit=limit)
    return [
        {
            "artifact_id": r.artifact_id,
            "display_name": r.display_name,
            "size_bytes": r.size_bytes,
            "page_count": r.page_count,
            "version": r.version,
            "parent_artifact_id": r.parent_artifact_id,
            "status": r.status,
            "created_at": r.created_at,
            "warnings": r.warnings,
        }
        for r in records
    ]


def get_pdf_info(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific PDF artifact."""
    r = storage.get_artifact(artifact_id)
    if not r:
        return None
    return {
        "artifact_id": r.artifact_id,
        "display_name": r.display_name,
        "size_bytes": r.size_bytes,
        "page_count": r.page_count,
        "version": r.version,
        "parent_artifact_id": r.parent_artifact_id,
        "status": r.status,
        "created_at": r.created_at,
        "warnings": r.warnings,
        "error_message": r.error_message,
    }
