"""
Subprocess worker entry point for Mark PDF Renderer.
Called by backend.pdf.renderer with hard timeout enforcement.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.pdf.schema import validate_and_normalize_document
from backend.pdf.renderer import render_in_process


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m backend.pdf.renderer_worker <in_json> <out_pdf> <meta_json>", file=sys.stderr)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    meta_path = sys.argv[3]

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            raw_doc = json.load(f)

        doc_model = validate_and_normalize_document(raw_doc)
        result = render_in_process(doc_model)

        if result.success:
            with open(out_path, "wb") as pf:
                pf.write(result.pdf_bytes)

            meta = {
                "success": True,
                "page_count": result.page_count,
                "byte_size": result.byte_size,
                "warnings": result.warnings,
            }
        else:
            meta = {
                "success": False,
                "error_code": result.error_code,
                "error_message": result.error_message,
                "warnings": result.warnings,
            }

        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf)

        sys.exit(0)

    except Exception as exc:
        meta = {
            "success": False,
            "error_code": "WORKER_EXCEPTION",
            "error_message": f"{type(exc).__name__}: {str(exc)}",
            "warnings": [],
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
