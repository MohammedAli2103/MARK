"""
Structured Document Model, Schema Validation, and Normalization for Mark PDF Engine.
"""

from dataclasses import dataclass, field
import html
import re
from typing import Any, Dict, List, Optional, Union

from config.pdf_config import (
    MAX_BLOCKS,
    MAX_BLOCK_TEXT_LENGTH,
    MAX_TABLE_ROWS,
    MAX_TABLE_COLS,
    MAX_NESTING_DEPTH,
    MAX_IMAGES,
)

SCHEMA_VERSION = "1.0"

# Complex scripts regex pattern (ReportLab does not support complex text shaping for these)
# Devanagari (\u0900-\u097F), Bengali (\u0980-\u09FF), Gurmukhi (\u0A00-\u0A7F),
# Gujarati (\u0A80-\u0AFF), Oriya (\u0B00-\u0B7F), Tamil (\u0B80-\u0BFF),
# Telugu (\u0C00-\u0C7F), Kannada (\u0C80-\u0CFF), Malayalam (\u0D00-\u0D7F),
# Sinhala (\u0D80-\u0DFF), Thai (\u0E00-\u0E7F), Lao (\u0E80-\u0EFF),
# Tibetan (\u0F00-\u0FFF), Myanmar (\u1000-\u109F), Arabic (\u0600-\u06FF, \u0750-\u077F, \u08A0-\u08FF),
# Hebrew (\u0590-\u05FF)
COMPLEX_SCRIPT_PATTERN = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"\u0900-\u0DFF\u0E00-\u0EFF\u0F00-\u0FFF\u1000-\u109F]"
)

# Safe URL pattern for hyperlinks
SAFE_LINK_PATTERN = re.compile(r"^(https?://|mailto:)", re.IGNORECASE)

# Control characters except \n, \r, \t
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class ValidationError(ValueError):
    """Raised when document structure or values fail validation."""
    pass


def normalize_text(text: Any, allow_xml_entities: bool = False) -> str:
    """Strip control characters, collapse pathological whitespace, and sanitize."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # Strip control characters
    text = CONTROL_CHAR_PATTERN.sub("", text)
    # Collapse pathological whitespace (preserve single newlines)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Remove excessive blank lines (>2)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    return result


def detect_complex_scripts(text: str) -> Optional[str]:
    """Detect presence of complex scripts that require shaping."""
    m = COMPLEX_SCRIPT_PATTERN.search(text)
    if m:
        char = m.group(0)
        return f"Complex script character detected ('{char}' / U+{ord(char):04X}). ReportLab cannot shape complex scripts (e.g. Arabic, Devanagari, Tamil)."
    return None


@dataclass
class MetaModel:
    title: str = "Untitled Document"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    language: str = "en"
    keywords: List[str] = field(default_factory=list)


@dataclass
class PageModel:
    size: str = "A4"          # "A4" or "Letter"
    orientation: str = "portrait"  # "portrait" or "landscape"
    margin_top: float = 40.0
    margin_bottom: float = 40.0
    margin_left: float = 45.0
    margin_right: float = 45.0


@dataclass
class StyleModel:
    template: str = "report"  # "report", "academic", "resume", "notes"
    base_font_size: float = 10.0
    primary_color: str = "#1A2530"
    accent_color: str = "#0A66C2"
    heading_style: str = "modern"


@dataclass
class CoverModel:
    enabled: bool = False
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    organization: Optional[str] = None


@dataclass
class HeaderFooterModel:
    header_template: str = "{title}"
    footer_template: str = "Page {page} of {pages}"
    show_page_numbers: bool = True
    include_disclaimer: Optional[bool] = None  # None = template default


@dataclass
class BlockModel:
    type: str  # heading, paragraph, list, table, code, quote, callout, image, page_break, reference_list
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentModel:
    schema_version: str = SCHEMA_VERSION
    meta: MetaModel = field(default_factory=MetaModel)
    page: PageModel = field(default_factory=PageModel)
    style: StyleModel = field(default_factory=StyleModel)
    cover: CoverModel = field(default_factory=CoverModel)
    toc: bool = False
    header_footer: HeaderFooterModel = field(default_factory=HeaderFooterModel)
    blocks: List[BlockModel] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document model to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "meta": {
                "title": self.meta.title,
                "subtitle": self.meta.subtitle,
                "author": self.meta.author,
                "date": self.meta.date,
                "language": self.meta.language,
                "keywords": self.meta.keywords,
            },
            "page": {
                "size": self.page.size,
                "orientation": self.page.orientation,
                "margin_top": self.page.margin_top,
                "margin_bottom": self.page.margin_bottom,
                "margin_left": self.page.margin_left,
                "margin_right": self.page.margin_right,
            },
            "style": {
                "template": self.style.template,
                "base_font_size": self.style.base_font_size,
                "primary_color": self.style.primary_color,
                "accent_color": self.style.accent_color,
                "heading_style": self.style.heading_style,
            },
            "cover": {
                "enabled": self.cover.enabled,
                "title": self.cover.title,
                "subtitle": self.cover.subtitle,
                "author": self.cover.author,
                "date": self.cover.date,
                "organization": self.cover.organization,
            },
            "toc": self.toc,
            "header_footer": {
                "header_template": self.header_footer.header_template,
                "footer_template": self.header_footer.footer_template,
                "show_page_numbers": self.header_footer.show_page_numbers,
                "include_disclaimer": self.header_footer.include_disclaimer,
            },
            "blocks": [{"type": b.type, "data": b.data} for b in self.blocks],
            "warnings": self.warnings,
        }


def validate_and_normalize_document(raw_doc: Dict[str, Any]) -> DocumentModel:
    """
    Validates and normalizes raw JSON/dict document input into DocumentModel.
    Enforces strict security checks, limits, and sanitization.
    """
    if not isinstance(raw_doc, dict):
        raise ValidationError("Document root must be a JSON object.")

    # 1. Schema version
    version = str(raw_doc.get("schema_version", SCHEMA_VERSION))

    # 2. Meta
    raw_meta = raw_doc.get("meta", {})
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    
    title = normalize_text(raw_meta.get("title", "Untitled Document"))
    if not title:
        title = "Untitled Document"
    if len(title) > 300:
        title = title[:300]

    subtitle = normalize_text(raw_meta.get("subtitle", "")) or None
    if subtitle and len(subtitle) > 300:
        subtitle = subtitle[:300]

    author = normalize_text(raw_meta.get("author", "")) or None
    if author and len(author) > 150:
        author = author[:150]

    date = normalize_text(raw_meta.get("date", "")) or None
    if date and len(date) > 50:
        date = date[:50]

    language = normalize_text(raw_meta.get("language", "en")).lower()
    if not language or len(language) > 10:
        language = "en"

    raw_keywords = raw_meta.get("keywords", [])
    keywords = []
    if isinstance(raw_keywords, list):
        for k in raw_keywords[:20]:
            nk = normalize_text(k)
            if nk:
                keywords.append(nk[:50])

    meta = MetaModel(
        title=title,
        subtitle=subtitle,
        author=author,
        date=date,
        language=language,
        keywords=keywords,
    )

    # 3. Page
    raw_page = raw_doc.get("page", {})
    if not isinstance(raw_page, dict):
        raw_page = {}
    
    page_size = str(raw_page.get("size", "A4")).upper()
    if page_size not in ("A4", "LETTER"):
        page_size = "A4"

    orientation = str(raw_page.get("orientation", "portrait")).lower()
    if orientation not in ("portrait", "landscape"):
        orientation = "portrait"

    def bound_margin(val, default):
        try:
            f = float(val)
            return max(18.0, min(144.0, f))
        except (ValueError, TypeError):
            return default

    page = PageModel(
        size=page_size,
        orientation=orientation,
        margin_top=bound_margin(raw_page.get("margin_top"), 40.0),
        margin_bottom=bound_margin(raw_page.get("margin_bottom"), 40.0),
        margin_left=bound_margin(raw_page.get("margin_left"), 45.0),
        margin_right=bound_margin(raw_page.get("margin_right"), 45.0),
    )

    # 4. Style & Template
    raw_style = raw_doc.get("style", {})
    if not isinstance(raw_style, dict):
        raw_style = {}
    
    template = str(raw_style.get("template", "report")).lower()
    if template not in ("report", "academic", "resume", "notes"):
        template = "report"

    try:
        base_font_size = float(raw_style.get("base_font_size", 10.0))
        base_font_size = max(8.0, min(18.0, base_font_size))
    except (ValueError, TypeError):
        base_font_size = 10.0

    color_hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
    primary_color = str(raw_style.get("primary_color", "#1A2530"))
    if not color_hex_pattern.match(primary_color):
        primary_color = "#1A2530"

    accent_color = str(raw_style.get("accent_color", "#0A66C2"))
    if not color_hex_pattern.match(accent_color):
        accent_color = "#0A66C2"

    style = StyleModel(
        template=template,
        base_font_size=base_font_size,
        primary_color=primary_color,
        accent_color=accent_color,
        heading_style="modern",
    )

    # 5. Cover
    raw_cover = raw_doc.get("cover", {})
    if not isinstance(raw_cover, dict):
        raw_cover = {}
    
    cover_enabled = bool(raw_cover.get("enabled", False))
    cover = CoverModel(
        enabled=cover_enabled,
        title=normalize_text(raw_cover.get("title", meta.title)) or meta.title,
        subtitle=normalize_text(raw_cover.get("subtitle", meta.subtitle or "")) or None,
        author=normalize_text(raw_cover.get("author", meta.author or "")) or None,
        date=normalize_text(raw_cover.get("date", meta.date or "")) or None,
        organization=normalize_text(raw_cover.get("organization", "")) or None,
    )

    # 6. TOC
    toc = bool(raw_doc.get("toc", False))

    # 7. Header & Footer
    raw_hf = raw_doc.get("header_footer", {})
    if not isinstance(raw_hf, dict):
        raw_hf = {}
    
    header_template = normalize_text(raw_hf.get("header_template", "{title}")) or "{title}"
    footer_template = normalize_text(raw_hf.get("footer_template", "Page {page} of {pages}")) or "Page {page} of {pages}"
    show_page_numbers = bool(raw_hf.get("show_page_numbers", True))
    
    # Disclaimer footer logic:
    # Rule 6: Default ON for report and academic, OFF for resume and notes. NEVER on resume!
    include_disclaimer = raw_hf.get("include_disclaimer", None)
    if template == "resume":
        include_disclaimer = False
    elif include_disclaimer is None:
        include_disclaimer = True if template in ("report", "academic") else False
    else:
        include_disclaimer = bool(include_disclaimer)

    hf = HeaderFooterModel(
        header_template=header_template,
        footer_template=footer_template,
        show_page_numbers=show_page_numbers,
        include_disclaimer=include_disclaimer,
    )

    # 8. Blocks validation & normalization
    raw_blocks = raw_doc.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raise ValidationError("Document 'blocks' must be a list.")

    if not raw_blocks:
        raise ValidationError("Document must contain at least one content block.")

    if len(raw_blocks) > MAX_BLOCKS:
        raise ValidationError(f"Document exceeds maximum block limit of {MAX_BLOCKS} blocks.")

    warnings: List[str] = []
    normalized_blocks: List[BlockModel] = []
    image_count = 0

    # Complex script detection on title
    title_warning = detect_complex_scripts(meta.title)
    if title_warning:
        warnings.append(title_warning)

    for idx, b in enumerate(raw_blocks):
        if not isinstance(b, dict):
            continue
        
        b_type = str(b.get("type", "")).strip().lower()
        b_data = b.get("data", {})
        if not isinstance(b_data, dict):
            b_data = {}

        if b_type == "heading":
            level = b_data.get("level", 1)
            try:
                level = max(1, min(4, int(level)))
            except (ValueError, TypeError):
                level = 1
            text = normalize_text(b_data.get("text", ""))
            if len(text) > MAX_BLOCK_TEXT_LENGTH:
                text = text[:MAX_BLOCK_TEXT_LENGTH]
            cs_warn = detect_complex_scripts(text)
            if cs_warn and cs_warn not in warnings:
                warnings.append(cs_warn)
            normalized_blocks.append(BlockModel(type="heading", data={"level": level, "text": text}))

        elif b_type == "paragraph":
            text = normalize_text(b_data.get("text", ""))
            if len(text) > MAX_BLOCK_TEXT_LENGTH:
                text = text[:MAX_BLOCK_TEXT_LENGTH]
            cs_warn = detect_complex_scripts(text)
            if cs_warn and cs_warn not in warnings:
                warnings.append(cs_warn)

            # Optional inline spans
            raw_spans = b_data.get("spans")
            clean_spans = []
            if isinstance(raw_spans, list):
                for s in raw_spans:
                    if isinstance(s, dict):
                        stext = normalize_text(s.get("text", ""))
                        link = s.get("link")
                        if link:
                            link = str(link).strip()
                            if not SAFE_LINK_PATTERN.match(link):
                                link = None  # Reject unsafe link protocols
                        clean_spans.append({
                            "text": stext,
                            "bold": bool(s.get("bold", False)),
                            "italic": bool(s.get("italic", False)),
                            "code": bool(s.get("code", False)),
                            "link": link,
                            "sub": bool(s.get("sub", False)),
                            "sup": bool(s.get("sup", False)),
                        })
            
            p_data = {"text": text}
            if clean_spans:
                p_data["spans"] = clean_spans
            normalized_blocks.append(BlockModel(type="paragraph", data=p_data))

        elif b_type == "list":
            ordered = bool(b_data.get("ordered", False))
            raw_items = b_data.get("items", [])
            if not isinstance(raw_items, list):
                raw_items = []
            
            def clean_list_items(items: list, depth: int) -> list:
                if depth > MAX_NESTING_DEPTH:
                    warnings.append(f"List nesting exceeds maximum depth {MAX_NESTING_DEPTH}. Flattening nested items.")
                    return []
                cleaned = []
                for it in items:
                    if isinstance(it, str):
                        cit = normalize_text(it)
                        if cit:
                            cleaned.append(cit[:MAX_BLOCK_TEXT_LENGTH])
                    elif isinstance(it, dict) and "text" in it:
                        sub_text = normalize_text(it.get("text", ""))
                        sub_items = it.get("items", [])
                        item_obj = {"text": sub_text[:MAX_BLOCK_TEXT_LENGTH]}
                        if isinstance(sub_items, list) and sub_items:
                            item_obj["items"] = clean_list_items(sub_items, depth + 1)
                        cleaned.append(item_obj)
                    elif isinstance(it, list):
                        cleaned.extend(clean_list_items(it, depth + 1))
                return cleaned

            items = clean_list_items(raw_items, depth=1)
            normalized_blocks.append(BlockModel(type="list", data={"ordered": ordered, "items": items}))

        elif b_type == "table":
            raw_headers = b_data.get("headers") or b_data.get("columns", [])
            raw_rows = b_data.get("rows", [])
            if not isinstance(raw_headers, list):
                raw_headers = []
            if not isinstance(raw_rows, list):
                raw_rows = []

            # Truncate columns and rows to limits
            clean_headers = [normalize_text(h)[:500] for h in raw_headers[:MAX_TABLE_COLS]]
            clean_rows = []
            for r in raw_rows[:MAX_TABLE_ROWS]:
                if isinstance(r, list):
                    clean_rows.append([normalize_text(c)[:1000] for c in r[:MAX_TABLE_COLS]])

            alignments = b_data.get("alignments", [])
            if not isinstance(alignments, list):
                alignments = []
            clean_aligns = [str(a).lower() if str(a).lower() in ("left", "center", "right") else "left" for a in alignments]

            caption = normalize_text(b_data.get("caption") or b_data.get("title", "")) or None
            normalized_blocks.append(BlockModel(type="table", data={
                "headers": clean_headers,
                "rows": clean_rows,
                "alignments": clean_aligns,
                "caption": caption,
            }))

        elif b_type == "code":
            code_text = b_data.get("code", "")
            if not isinstance(code_text, str):
                code_text = str(code_text)
            # Remove control chars except newlines and tabs
            code_text = CONTROL_CHAR_PATTERN.sub("", code_text)
            if len(code_text) > MAX_BLOCK_TEXT_LENGTH * 2:
                code_text = code_text[:MAX_BLOCK_TEXT_LENGTH * 2]
            lang = normalize_text(b_data.get("language", "")) or "text"
            normalized_blocks.append(BlockModel(type="code", data={"code": code_text, "language": lang[:30]}))

        elif b_type == "quote":
            text = normalize_text(b_data.get("text", ""))
            author = normalize_text(b_data.get("author", "")) or None
            normalized_blocks.append(BlockModel(type="quote", data={"text": text[:MAX_BLOCK_TEXT_LENGTH], "author": author}))

        elif b_type == "callout":
            text = normalize_text(b_data.get("text", ""))
            kind = str(b_data.get("kind", "note")).lower()
            if kind not in ("info", "warning", "tip", "note"):
                kind = "note"
            normalized_blocks.append(BlockModel(type="callout", data={"text": text[:MAX_BLOCK_TEXT_LENGTH], "kind": kind}))

        elif b_type == "image":
            image_count += 1
            if image_count > MAX_IMAGES:
                warnings.append(f"Maximum image limit ({MAX_IMAGES}) exceeded. Dropping additional images.")
                continue
            asset_id = str(b_data.get("asset_id", "")).strip()
            # Only allow alphanumeric / UUID characters for asset_id (no paths or URLs)
            if not re.match(r"^[a-zA-Z0-9_\-]+$", asset_id):
                warnings.append(f"Invalid image asset ID '{asset_id}'. Image dropped.")
                continue
            caption = normalize_text(b_data.get("caption", "")) or None
            width = b_data.get("width")
            height = b_data.get("height")
            try:
                width = float(width) if width else None
            except (ValueError, TypeError):
                width = None
            try:
                height = float(height) if height else None
            except (ValueError, TypeError):
                height = None
            normalized_blocks.append(BlockModel(type="image", data={
                "asset_id": asset_id,
                "width": width,
                "height": height,
                "caption": caption,
            }))

        elif b_type == "page_break":
            normalized_blocks.append(BlockModel(type="page_break", data={}))

        elif b_type == "reference_list":
            raw_refs = b_data.get("items", [])
            if not isinstance(raw_refs, list):
                raw_refs = []
            clean_refs = [normalize_text(r)[:MAX_BLOCK_TEXT_LENGTH] for r in raw_refs if normalize_text(r)]
            normalized_blocks.append(BlockModel(type="reference_list", data={"items": clean_refs[:50]}))

        else:
            # Unsupported block element: Rule 3.1: Explicit warning, do not silently drop
            warnings.append(f"Unsupported block element type '{b_type}'. Ignored.")

    if not normalized_blocks:
        raise ValidationError("Document contains no valid content blocks.")

    return DocumentModel(
        schema_version=version,
        meta=meta,
        page=page,
        style=style,
        cover=cover,
        toc=toc,
        header_footer=hf,
        blocks=normalized_blocks,
        warnings=warnings,
    )
