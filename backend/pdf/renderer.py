"""
Deterministic PDF Renderer for Mark Local AI Assistant using ReportLab Platypus.
Supports templates, NumberedCanvas (Page X of Y), keep-with-next headings,
auto-wrapped tables, embedded DejaVu fonts, clickable TOC, and hard timeout via subprocess.
"""

from dataclasses import dataclass, field
import html
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, landscape, portrait
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import inch, cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from config.pdf_config import (
    FONTS_DIR,
    MAX_PAGES,
    MAX_PDF_SIZE_BYTES,
    RENDER_TIMEOUT_SECONDS,
)
from backend.pdf.schema import DocumentModel, validate_and_normalize_document
from backend.pdf.templates import TemplateConfig, get_template


# Register DejaVu Fonts once
_FONTS_REGISTERED = False

def register_fonts():
    """Register embedded DejaVu TrueType fonts with ReportLab."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return

    fonts = [
        ("DejaVuSans", "DejaVuSans.ttf"),
        ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
        ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
        ("DejaVuSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
        ("DejaVuSerif", "DejaVuSerif.ttf"),
        ("DejaVuSerif-Bold", "DejaVuSerif-Bold.ttf"),
        ("DejaVuSerif-Italic", "DejaVuSerif-Italic.ttf"),
        ("DejaVuSerif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
        ("DejaVuSansMono", "DejaVuSansMono.ttf"),
        ("DejaVuSansMono-Bold", "DejaVuSansMono-Bold.ttf"),
        ("DejaVuSansMono-Oblique", "DejaVuSansMono-Oblique.ttf"),
        ("DejaVuSansMono-BoldOblique", "DejaVuSansMono-BoldOblique.ttf"),
    ]

    for name, filename in fonts:
        path = FONTS_DIR / filename
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
            except Exception:
                pass

    try:
        pdfmetrics.registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique",
            boldItalic="DejaVuSans-BoldOblique",
        )
        pdfmetrics.registerFontFamily(
            "DejaVuSerif",
            normal="DejaVuSerif",
            bold="DejaVuSerif-Bold",
            italic="DejaVuSerif-Italic",
            boldItalic="DejaVuSerif-BoldItalic",
        )
        pdfmetrics.registerFontFamily(
            "DejaVuSansMono",
            normal="DejaVuSansMono",
            bold="DejaVuSansMono-Bold",
            italic="DejaVuSansMono-Oblique",
            boldItalic="DejaVuSansMono-BoldOblique",
        )
    except Exception:
        pass

    _FONTS_REGISTERED = True


def wrap_long_words(text: str, max_word_len: int = 35) -> str:
    """Insert zero-width space &#8203; into words longer than max_word_len to prevent clipping."""
    if not text:
        return ""
    words = text.split(" ")
    out_words = []
    for w in words:
        if len(w) > max_word_len and "http" not in w and not w.startswith("<"):
            chunks = [w[i:i + max_word_len] for i in range(0, len(w), max_word_len)]
            out_words.append("&#8203;".join(chunks))
        else:
            out_words.append(w)
    return " ".join(out_words)


def format_inline_spans(text: str, spans: Optional[List[Dict[str, Any]]] = None) -> str:
    """Safely format text and inline spans into ReportLab XML markup."""
    if not spans:
        escaped = html.escape(text)
        return wrap_long_words(escaped)

    out = []
    for s in spans:
        stext = html.escape(s.get("text", ""))
        stext = wrap_long_words(stext)
        if s.get("bold"):
            stext = f"<b>{stext}</b>"
        if s.get("italic"):
            stext = f"<i>{stext}</i>"
        if s.get("code"):
            stext = f"<font face='DejaVuSansMono'>{stext}</font>"
        if s.get("sub"):
            stext = f"<sub>{stext}</sub>"
        if s.get("sup"):
            stext = f"<sup>{stext}</sup>"
        if s.get("link"):
            link = s["link"]
            stext = f"<a href='{link}' color='blue'><u>{stext}</u></a>"
        out.append(stext)
    return "".join(out)


class SingleLineTOC(TableOfContents):
    """
    Custom single-line Table of Contents flowable.
    Renders each entry as 'Section title ..... page' on one line,
    with computed dot leaders and clickable intra-document anchors.
    """
    def __init__(self, usable_width: float, font_name: str = "DejaVuSans", font_size: float = 9.5, leading: float = 14.0):
        super().__init__()
        self.usable_width = usable_width
        self.font_name = font_name
        self.font_size = font_size
        self.leading = leading

    def wrap(self, availWidth, availHeight):
        if not self._lastEntries:
            self._table = Table([[""]], colWidths=(self.usable_width,))
            self.width, self.height = self._table.wrapOn(self.canv, availWidth, availHeight)
            return (self.width, self.height)

        style = ParagraphStyle(
            name="SingleLineTOCLine",
            fontName=self.font_name,
            fontSize=self.font_size,
            leading=self.leading,
        )

        table_data = []
        for (level, text, page_num, key) in self._lastEntries:
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;" if level > 1 else ""
            clean_indent = "    " if level > 1 else ""
            clean_title = clean_indent + text
            display_title = indent + text

            t_w = stringWidth(clean_title, self.font_name, self.font_size)
            p_w = stringWidth(f" {page_num}", self.font_name, self.font_size)
            dot_w = stringWidth(" .", self.font_name, self.font_size)
            avail = max(0, self.usable_width - t_w - p_w - 8)
            num_dots = max(3, int(avail / dot_w))
            dots_str = " ." * num_dots

            if key:
                line_html = f'<a href="#{key}">{display_title}</a> {dots_str} <a href="#{key}">{page_num}</a>'
            else:
                line_html = f'{display_title} {dots_str} {page_num}'

            p = Paragraph(line_html, style)
            table_data.append([p])

        self._table = Table(table_data, colWidths=(self.usable_width,))
        self.width, self.height = self._table.wrapOn(self.canv, availWidth, availHeight)
        return (self.width, self.height)

    def drawOn(self, canvas, x, y, _sW=0):
        if getattr(self, "_table", None) is not None:
            self._table.drawOn(canvas, x, y, _sW)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that accumulates pages and stamps headers, footers,
    page numbers ('Page X of Y'), and disclaimers on final save.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []
        self.doc_title = "Document"
        self.header_template = "{title}"
        self.footer_template = "Page {page} of {pages}"
        self.show_page_numbers = True
        self.has_cover_page = False
        self.include_disclaimer = False
        self.disclaimer_text = ""
        self.primary_color = "#1A2530"
        self.accent_color = "#0A66C2"
        self.page_limit = MAX_PAGES
        self.margin_left = 45.0
        self.margin_right = 45.0
        self.is_resume = False

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        if num_pages > self.page_limit:
            raise ValueError(
                f"Page limit exceeded: Document generated {num_pages} pages, but maximum allowed is {self.page_limit}."
            )

        for page_idx, state in enumerate(self._saved_page_states, start=1):
            self.__dict__.update(state)
            self._draw_decorations(page_idx, num_pages)
            super().showPage()

        super().save()

    def _draw_decorations(self, page_num: int, total_pages: int):
        self.saveState()
        page_width, page_height = self._pagesize

        # Skip headers/footers on cover page (if cover is page 1)
        if self.has_cover_page and page_num == 1:
            self.restoreState()
            return

        # Skip headers/footers on 1-page resumes
        if self.is_resume and total_pages == 1:
            self.restoreState()
            return

        font_name = "DejaVuSans"
        self.setFont(font_name, 8.5)
        self.setFillColor(colors.HexColor("#718096"))

        # --- Running Header ---
        header_y = page_height - 28.0
        header_text = self.header_template.replace("{title}", self.doc_title).replace("{page}", str(page_num)).replace("{pages}", str(total_pages))
        if header_text.strip():
            self.drawString(self.margin_left, header_y, header_text)
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.6)
            self.line(self.margin_left, header_y - 4.0, page_width - self.margin_right, header_y - 4.0)

        # --- Running Footer ---
        footer_y = 26.0
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.6)
        self.line(self.margin_left, footer_y + 12.0, page_width - self.margin_right, footer_y + 12.0)

        # Page number
        if self.show_page_numbers:
            p_text = self.footer_template.replace("{page}", str(page_num)).replace("{pages}", str(total_pages)).replace("{title}", self.doc_title)
            self.drawRightString(page_width - self.margin_right, footer_y, p_text)

        # Disclaimer note (Rule 6: Configurable per template, never on resume)
        if self.include_disclaimer and self.disclaimer_text:
            self.setFont(font_name, 7.5)
            self.setFillColor(colors.HexColor("#A0AEC0"))
            self.drawString(self.margin_left, footer_y, self.disclaimer_text)

        self.restoreState()


@dataclass
class RenderResult:
    success: bool
    page_count: int = 0
    byte_size: int = 0
    pdf_bytes: bytes = b""
    warnings: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None


def render_in_process(doc_model: DocumentModel, assets: Optional[Dict[str, Any]] = None) -> RenderResult:
    """
    Renders a DocumentModel directly in-process to PDF bytes.
    Pure and deterministic.
    """
    register_fonts()
    tmpl = get_template(doc_model.style.template)

    # Determine Page Size and Orientation
    if doc_model.page.size.upper() == "LETTER":
        base_size = LETTER
    else:
        base_size = A4

    if doc_model.page.orientation.lower() == "landscape":
        page_size = landscape(base_size)
    else:
        page_size = portrait(base_size)

    page_width, page_height = page_size
    is_resume = doc_model.style.template == "resume"
    margin_top = tmpl.margin_top if is_resume else doc_model.page.margin_top
    margin_bottom = tmpl.margin_bottom if is_resume else doc_model.page.margin_bottom
    margin_left = tmpl.margin_left if is_resume else doc_model.page.margin_left
    margin_right = tmpl.margin_right if is_resume else doc_model.page.margin_right
    usable_width = page_width - margin_left - margin_right

    # Styles
    stylesheet = StyleSheet1()

    # Base Body
    body_font_size = tmpl.body_size if is_resume else doc_model.style.base_font_size
    body_style = ParagraphStyle(
        name="DocBody",
        fontName=tmpl.body_font,
        fontSize=body_font_size,
        leading=body_font_size * tmpl.line_spacing_ratio,
        textColor=colors.HexColor(tmpl.neutral_dark),
        spaceAfter=3 if is_resume else 5,
        spaceBefore=0,
    )
    stylesheet.add(body_style)

    # Headings (All with keepWithNext=True to eliminate orphaned headings!)
    stylesheet.add(ParagraphStyle(
        name="DocH1",
        fontName=tmpl.body_font_bold,
        fontSize=tmpl.h1_size,
        leading=tmpl.h1_size * 1.2,
        textColor=colors.HexColor(tmpl.primary_color),
        spaceBefore=6 if is_resume else 13,
        spaceAfter=3 if is_resume else 6,
        keepWithNext=True,
    ))
    stylesheet.add(ParagraphStyle(
        name="DocH2",
        fontName=tmpl.body_font_bold,
        fontSize=tmpl.h2_size,
        leading=tmpl.h2_size * 1.2,
        textColor=colors.HexColor(tmpl.accent_color),
        spaceBefore=5 if is_resume else 9,
        spaceAfter=2 if is_resume else 4,
        keepWithNext=True,
    ))
    stylesheet.add(ParagraphStyle(
        name="DocH3",
        fontName=tmpl.body_font_bold,
        fontSize=tmpl.h3_size,
        leading=tmpl.h3_size * 1.2,
        textColor=colors.HexColor(tmpl.neutral_dark),
        spaceBefore=4 if is_resume else 7,
        spaceAfter=2 if is_resume else 3,
        keepWithNext=True,
    ))
    stylesheet.add(ParagraphStyle(
        name="DocH4",
        fontName=tmpl.body_font_bold,
        fontSize=tmpl.h4_size,
        leading=tmpl.h4_size * 1.2,
        textColor=colors.HexColor(tmpl.neutral_light),
        spaceBefore=3 if is_resume else 5,
        spaceAfter=2 if is_resume else 2,
        keepWithNext=True,
    ))

    # Code style
    code_style = ParagraphStyle(
        name="DocCode",
        fontName=tmpl.code_font,
        fontSize=tmpl.code_size,
        leading=tmpl.code_size * 1.35,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=0,
        spaceBefore=0,
    )

    # Table styles
    table_cell_style = ParagraphStyle(
        name="DocTableCell",
        fontName=tmpl.body_font,
        fontSize=doc_model.style.base_font_size * 0.92,
        leading=doc_model.style.base_font_size * 1.25,
        textColor=colors.HexColor(tmpl.neutral_dark),
    )
    table_hdr_style = ParagraphStyle(
        name="DocTableHdr",
        fontName=tmpl.body_font_bold,
        fontSize=doc_model.style.base_font_size * 0.92,
        leading=doc_model.style.base_font_size * 1.25,
        textColor=colors.white,
    )

    story: List[Any] = []
    warnings: List[str] = list(doc_model.warnings)

    # 1. Cover Page
    if doc_model.cover.enabled:
        story.append(Spacer(1, 40))
        # Top accent bar
        story.append(HRFlowable(
            width="100%", thickness=4, color=colors.HexColor(tmpl.accent_color),
            spaceBefore=0, spaceAfter=25
        ))

        cover_title_style = ParagraphStyle(
            name="CoverTitle",
            fontName=tmpl.body_font_bold,
            fontSize=28,
            leading=34,
            textColor=colors.HexColor(tmpl.primary_color),
            spaceAfter=12,
        )
        story.append(Paragraph(html.escape(doc_model.cover.title or doc_model.meta.title), cover_title_style))

        if doc_model.cover.subtitle:
            cover_sub_style = ParagraphStyle(
                name="CoverSub",
                fontName=tmpl.body_font,
                fontSize=14,
                leading=18,
                textColor=colors.HexColor(tmpl.accent_color),
                spaceAfter=25,
            )
            story.append(Paragraph(html.escape(doc_model.cover.subtitle), cover_sub_style))

        story.append(Spacer(1, 120))

        # Metadata block
        meta_items = []
        if doc_model.cover.author:
            meta_items.append(f"<b>Author:</b> {html.escape(doc_model.cover.author)}")
        if doc_model.cover.organization:
            meta_items.append(f"<b>Organization:</b> {html.escape(doc_model.cover.organization)}")
        if doc_model.cover.date:
            meta_items.append(f"<b>Date:</b> {html.escape(doc_model.cover.date)}")
        
        if meta_items:
            meta_style = ParagraphStyle(
                name="CoverMeta",
                fontName=tmpl.body_font,
                fontSize=10,
                leading=15,
                textColor=colors.HexColor(tmpl.neutral_light),
                spaceAfter=4,
            )
            for m in meta_items:
                story.append(Paragraph(m, meta_style))

        story.append(PageBreak())

    # 2. Table of Contents
    toc_flowable = None
    if doc_model.toc:
        story.append(Paragraph("Table of Contents", stylesheet["DocH1"]))
        story.append(Spacer(1, 8))
        toc_flowable = SingleLineTOC(usable_width, font_name=tmpl.body_font, font_size=9.5, leading=14.0)
        story.append(toc_flowable)
        story.append(PageBreak())

    # 2.5 Résumé Header (Mandatory top header block for resume template: name as large title, one contact line below)
    start_block_idx = 0
    if is_resume:
        resume_name_style = ParagraphStyle(
            name="ResumeHeaderName",
            fontName=tmpl.body_font_bold,
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(tmpl.primary_color),
            alignment=1,  # Centered
            spaceAfter=2,
            spaceBefore=0,
        )
        resume_contact_style = ParagraphStyle(
            name="ResumeHeaderContact",
            fontName=tmpl.body_font,
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor(tmpl.neutral_dark),
            alignment=1,  # Centered
            spaceAfter=5,
            spaceBefore=0,
        )

        name_text = doc_model.meta.title or "Curriculum Vitae"
        contact_line = doc_model.meta.subtitle or ""

        # If first block is candidate name (as heading of any level or matching title), extract it and skip from body blocks
        if (
            len(doc_model.blocks) > 0
            and doc_model.blocks[0].type == "heading"
            and (
                doc_model.blocks[0].data.get("level") == 1
                or doc_model.blocks[0].data.get("text", "").strip().lower() in name_text.lower()
                or name_text.lower() in doc_model.blocks[0].data.get("text", "").strip().lower()
            )
        ):
            name_text = doc_model.blocks[0].data.get("text", name_text)
            start_block_idx = 1
            # Check if second block is a contact info paragraph
            if (
                len(doc_model.blocks) > 1
                and doc_model.blocks[1].type == "paragraph"
                and any(sym in doc_model.blocks[1].data.get("text", "").lower() for sym in ("@", "|", "•", "phone", "email", "(555)"))
            ):
                contact_line = doc_model.blocks[1].data.get("text", contact_line)
                start_block_idx = 2

        story.append(Paragraph(html.escape(name_text), resume_name_style))
        if contact_line:
            story.append(Paragraph(html.escape(contact_line), resume_contact_style))
        story.append(HRFlowable(
            width="100%", thickness=1, color=colors.HexColor(tmpl.accent_color),
            spaceBefore=2, spaceAfter=8
        ))

    # 3. Document Content Blocks
    heading_counter = 0
    heading_map: Dict[Any, Tuple[int, str, str]] = {}
    for b in doc_model.blocks[start_block_idx:]:
        if is_resume and b.type == "heading" and b.data.get("text", "").strip().lower() == name_text.strip().lower():
            continue
        if is_resume and b.type == "paragraph" and contact_line and b.data.get("text", "").strip().lower() == contact_line.strip().lower():
            continue
        b_type = b.type
        b_data = b.data

        if b_type == "heading":
            level = b_data.get("level", 1)
            htext = b_data.get("text", "")
            hstyle_name = f"DocH{min(4, max(1, level))}"
            key = f"sec_dest_{heading_counter}"
            heading_counter += 1
            p = Paragraph(f'<a name="{key}"/>' + format_inline_spans(htext), stylesheet[hstyle_name])
            story.append(p)
            heading_map[p] = (level - 1, htext, key)

        elif b_type == "paragraph":
            ptext = b_data.get("text", "")
            spans = b_data.get("spans")
            p = Paragraph(format_inline_spans(ptext, spans), stylesheet["DocBody"])
            story.append(p)

        elif b_type == "list":
            ordered = b_data.get("ordered", False)
            items = b_data.get("items", [])
            for idx, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    it_text = item.get("text", "")
                else:
                    it_text = str(item)
                # Strip leading bullets, dashes, asterisks, numbers already emitted by LLM
                it_text = re.sub(r"^[\s•\-\*\–\—\.]+\s*", "", it_text)
                bullet_prefix = f"<b>{idx}.</b> " if ordered else "• "
                list_style = ParagraphStyle(
                    name=f"List_{idx}",
                    parent=stylesheet["DocBody"],
                    leftIndent=14 if is_resume else 16,
                    firstLineIndent=-10 if is_resume else -12,
                    spaceAfter=2 if is_resume else 3,
                )
                story.append(Paragraph(bullet_prefix + format_inline_spans(it_text), list_style))

        elif b_type == "table":
            headers = b_data.get("headers", [])
            rows = b_data.get("rows", [])
            num_cols = max(len(headers), max((len(r) for r in rows), default=0))

            if num_cols > 0:
                # Calculate column widths
                col_width = usable_width / num_cols
                col_widths = [col_width] * num_cols

                table_data = []
                if headers:
                    hdr_row = []
                    for h in headers:
                        hdr_row.append(Paragraph(format_inline_spans(h), table_hdr_style))
                    table_data.append(hdr_row)

                for r in rows:
                    row_cells = []
                    for c_idx in range(num_cols):
                        val = r[c_idx] if c_idx < len(r) else ""
                        row_cells.append(Paragraph(format_inline_spans(val), table_cell_style))
                    table_data.append(row_cells)

                t = Table(table_data, colWidths=col_widths, repeatRows=1 if headers else 0)
                t_style = [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ]
                if headers:
                    t_style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(tmpl.primary_color)))
                    for row_i in range(1, len(table_data)):
                        bg = "#FFFFFF" if row_i % 2 == 1 else tmpl.surface_bg
                        t_style.append(("BACKGROUND", (0, row_i), (-1, row_i), colors.HexColor(bg)))
                else:
                    for row_i in range(len(table_data)):
                        bg = "#FFFFFF" if row_i % 2 == 0 else tmpl.surface_bg
                        t_style.append(("BACKGROUND", (0, row_i), (-1, row_i), colors.HexColor(bg)))

                t.setStyle(TableStyle(t_style))
                story.append(Spacer(1, 3))
                # Keep small tables (up to ~8 rows) together on one page; split only larger ones
                if len(table_data) <= 8:
                    story.append(KeepTogether(t))
                else:
                    story.append(t)
                story.append(Spacer(1, 4))

        elif b_type == "code":
            code_text = b_data.get("code", "")
            code_lines = [html.escape(line) for line in code_text.splitlines()]
            code_html = "<br/>".join(code_lines) or "&nbsp;"
            code_p = Paragraph(code_html, code_style)
            code_table = Table([[code_p]], colWidths=[usable_width])
            code_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
            ]))
            story.append(Spacer(1, 4))
            story.append(code_table)
            story.append(Spacer(1, 6))

        elif b_type == "callout":
            ctext = b_data.get("text", "")
            kind = b_data.get("kind", "note").lower()
            callout_colors = {
                "info": ("#0284C7", "#F0F9FF"),
                "warning": ("#D97706", "#FFFBEB"),
                "tip": ("#059669", "#ECFDF5"),
                "note": (tmpl.accent_color, tmpl.surface_bg),
            }
            border_col, bg_col = callout_colors.get(kind, (tmpl.accent_color, tmpl.surface_bg))
            cp = Paragraph(format_inline_spans(ctext), stylesheet["DocBody"])
            ct = Table([[cp]], colWidths=[usable_width])
            ct.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_col)),
                ("LINELEFT", (0, 0), (-1, -1), 3.5, colors.HexColor(border_col)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 4))
            story.append(ct)
            story.append(Spacer(1, 6))

        elif b_type == "quote":
            qtext = b_data.get("text", "")
            author = b_data.get("author")
            qp = Paragraph(f"<i>“{format_inline_spans(qtext)}”</i>", stylesheet["DocBody"])
            flowables = [qp]
            if author:
                auth_p = Paragraph(f"— <b>{html.escape(author)}</b>", ParagraphStyle(
                    name="QAuth",
                    parent=stylesheet["DocBody"],
                    fontSize=doc_model.style.base_font_size * 0.9,
                    textColor=colors.HexColor(tmpl.neutral_light),
                    alignment=2,  # Right
                ))
                flowables.append(auth_p)
            qt = Table([[flowables]], colWidths=[usable_width])
            qt.setStyle(TableStyle([
                ("LINELEFT", (0, 0), (-1, -1), 3.0, colors.HexColor(tmpl.accent_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(Spacer(1, 4))
            story.append(qt)
            story.append(Spacer(1, 6))

        elif b_type == "page_break":
            story.append(PageBreak())

        elif b_type == "reference_list":
            story.append(Spacer(1, 6))
            story.append(Paragraph("References", stylesheet["DocH2"]))
            refs = b_data.get("items", [])
            for r_idx, ref in enumerate(refs, start=1):
                ref_p = Paragraph(f"[{r_idx}] {format_inline_spans(ref)}", ParagraphStyle(
                    name=f"Ref_{r_idx}",
                    parent=stylesheet["DocBody"],
                    fontSize=doc_model.style.base_font_size * 0.88,
                    leading=doc_model.style.base_font_size * 1.2,
                    leftIndent=14,
                    firstLineIndent=-14,
                    spaceAfter=4,
                ))
                story.append(ref_p)

    # Build Document with NumberedCanvas
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=page_size,
        leftMargin=margin_left,
        rightMargin=margin_right,
        topMargin=margin_top + 16,     # Space for running header
        bottomMargin=margin_bottom + 16, # Space for running footer
        title=doc_model.meta.title,
        author=doc_model.meta.author or "Mark AI Assistant",
        subject=doc_model.meta.subtitle or "",
    )

    # Set up NumberedCanvas properties
    canvas_maker = NumberedCanvas
    def custom_canvas_maker(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.doc_title = doc_model.meta.title
        c.header_template = doc_model.header_footer.header_template
        c.footer_template = doc_model.header_footer.footer_template
        c.show_page_numbers = doc_model.header_footer.show_page_numbers
        c.has_cover_page = doc_model.cover.enabled
        c.include_disclaimer = doc_model.header_footer.include_disclaimer if doc_model.header_footer.include_disclaimer is not None else tmpl.include_disclaimer_default
        c.disclaimer_text = tmpl.disclaimer_text
        c.primary_color = tmpl.primary_color
        c.accent_color = tmpl.accent_color
        c.page_limit = MAX_PAGES
        c.margin_left = margin_left
        c.margin_right = margin_right
        c.is_resume = (doc_model.style.template == "resume")
        return c

    # Table of contents notification callback
    if doc_model.toc:
        def after_flowable(flowable):
            if flowable in heading_map:
                level, text, key = heading_map[flowable]
                plain_text = text.strip()
                # Exclude the "Table of Contents" heading itself from TOC entries
                if plain_text.lower() in ("table of contents", "contents", "table of content"):
                    return
                if level in (0, 1):
                    doc.notify("TOCEntry", (level, plain_text, doc.page, key))
        doc.afterFlowable = after_flowable
        doc.multiBuild(story, canvasmaker=custom_canvas_maker)
    else:
        doc.build(story, canvasmaker=custom_canvas_maker)

    pdf_bytes = pdf_buffer.getvalue()
    byte_size = len(pdf_bytes)

    if byte_size > MAX_PDF_SIZE_BYTES:
        return RenderResult(
            success=False,
            error_code="SIZE_LIMIT_EXCEEDED",
            error_message=f"Rendered PDF size ({byte_size} bytes) exceeds maximum limit of {MAX_PDF_SIZE_BYTES} bytes.",
        )

    # Page count check via pypdf
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
    except Exception as e:
        return RenderResult(
            success=False,
            error_code="INVALID_PDF_OUTPUT",
            error_message=f"Failed to inspect rendered PDF structure: {e}",
        )

    return RenderResult(
        success=True,
        page_count=page_count,
        byte_size=byte_size,
        pdf_bytes=pdf_bytes,
        warnings=warnings,
    )


def render(
    doc_model: DocumentModel,
    assets: Optional[Dict[str, Any]] = None,
    timeout: float = RENDER_TIMEOUT_SECONDS,
) -> RenderResult:
    """
    Renders document in an isolated subprocess with hard timeout enforcement.
    Rule 2: ReportLab runs in a subprocess and is terminated/killed on timeout.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as in_f:
        json.dump(doc_model.to_dict(), in_f)
        in_path = in_f.name

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as out_f:
        out_path = out_f.name

    with tempfile.NamedTemporaryFile(suffix=".meta.json", delete=False) as meta_f:
        meta_path = meta_f.name

    cmd = [
        sys.executable,
        "-m", "backend.pdf.renderer_worker",
        in_path,
        out_path,
        meta_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
            return RenderResult(
                success=False,
                error_code="TIMEOUT",
                error_message=f"PDF rendering timed out after {timeout} seconds.",
            )

        if proc.returncode != 0:
            err_msg = stderr.strip() or stdout.strip() or f"Process exited with code {proc.returncode}"
            return RenderResult(
                success=False,
                error_code="SUBPROCESS_FAILURE",
                error_message=f"Subprocess rendering failed: {err_msg}",
            )

        # Read metadata and output PDF
        if not os.path.exists(meta_path) or not os.path.exists(out_path):
            return RenderResult(
                success=False,
                error_code="MISSING_OUTPUT",
                error_message="Subprocess did not produce output files.",
            )

        with open(meta_path, "r", encoding="utf-8") as mf:
            meta_res = json.load(mf)

        with open(out_path, "rb") as pf:
            pdf_bytes = pf.read()

        if not meta_res.get("success"):
            return RenderResult(
                success=False,
                error_code=meta_res.get("error_code", "UNKNOWN"),
                error_message=meta_res.get("error_message", "Rendering failed in worker"),
                warnings=meta_res.get("warnings", []),
            )

        return RenderResult(
            success=True,
            page_count=meta_res.get("page_count", 0),
            byte_size=len(pdf_bytes),
            pdf_bytes=pdf_bytes,
            warnings=meta_res.get("warnings", []),
        )

    except Exception as exc:
        return RenderResult(
            success=False,
            error_code="RENDER_EXCEPTION",
            error_message=str(exc),
        )

    finally:
        for p in (in_path, out_path, meta_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
