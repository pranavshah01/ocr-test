"""
Analyze textboxes/callouts in size_check.docx and compute capacity metrics.

For each textbox whose combined text starts with "Test", compute for a range of
font sizes:
- lines_fit (estimated number of rows without explicit newlines)
- max_chars_per_line
- total_chars_fit = lines_fit * max_chars_per_line
- chars_per_sq_in = total_chars_fit / area_sq_in

Outputs a CSV at reports/size_check_analysis_<timestamp>.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

from docx import Document

from app.utils.graphics_utils.graphics_docx_utils import (
    TextboxParser,
    GraphicsFontManager,
)


POINTS_PER_INCH = 72.0

# Width multipliers borrowed from graphics processor baseline
FONT_WIDTH_MULTIPLIERS: Dict[str, float] = {
    "Arial": 0.6,
    "Times New Roman": 0.55,
    "Calibri": 0.58,
    "PMingLiU": 0.65,
    "SimSun": 0.65,
    "Microsoft YaHei": 0.62,
    "Verdana": 0.62,
    "ACADEMY ENGRAVED LET PLAIN": 0.7,
}


def get_width_multiplier(font_family: str) -> float:
    if not font_family:
        return 0.6
    # Try exact family first, then a case-insensitive fallback
    if font_family in FONT_WIDTH_MULTIPLIERS:
        return FONT_WIDTH_MULTIPLIERS[font_family]
    for k, v in FONT_WIDTH_MULTIPLIERS.items():
        if k.lower() == font_family.lower():
            return v
    return 0.6


def compute_capacity(
    width_pt: float,
    height_pt: float,
    font_family: str,
    font_size_pt: float,
) -> Tuple[int, int, int, float]:
    """
    Compute (lines_fit, max_chars_per_line, total_chars_fit, chars_per_sq_in).
    """
    if width_pt <= 0 or height_pt <= 0 or font_size_pt <= 0:
        return 0, 0, 0, 0.0

    width_in = width_pt / POINTS_PER_INCH
    height_in = height_pt / POINTS_PER_INCH
    area_sq_in = width_in * height_in if width_in > 0 and height_in > 0 else 0.0

    width_multiplier = get_width_multiplier(font_family)
    # Character width in inches and line height in inches
    char_width_in = (font_size_pt * width_multiplier) / POINTS_PER_INCH
    line_height_in = (font_size_pt * 1.3) / POINTS_PER_INCH

    if char_width_in <= 0 or line_height_in <= 0:
        return 0, 0, 0, 0.0

    max_chars_per_line = int(width_in // char_width_in) if width_in > 0 else 0
    lines_fit = int(height_in // line_height_in) if height_in > 0 else 0

    total_chars_fit = max(0, max_chars_per_line * lines_fit)
    chars_per_sq_in = (total_chars_fit / area_sq_in) if area_sq_in > 0 else 0.0
    return lines_fit, max_chars_per_line, total_chars_fit, chars_per_sq_in


def main() -> None:
    # __file__ → app/tools/... so AWI root is parents[2]
    root = Path(__file__).resolve().parents[2]
    src_docx = root / "source_documents" / "size_check.docx"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not src_docx.exists():
        raise FileNotFoundError(f"Input file not found: {src_docx}")

    doc = Document(src_docx)
    textboxes = TextboxParser.find_textboxes(doc)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = reports_dir / f"size_check_analysis_{ts}.csv"

    # Candidate font sizes to evaluate
    candidate_sizes = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "textbox_index",
                "location",
                "width_pt",
                "height_pt",
                "width_in",
                "height_in",
                "area_sq_in",
                "font_family",
                "detected_sizes",
                "text_length",
                "sample_text",
                "font_size_pt",
                "lines_fit",
                "max_chars_per_line",
                "total_chars_fit",
                "chars_per_sq_in",
                "fits_text_length",
            ]
        )

        for i, tb in enumerate(textboxes):
            combined_text, wt_elements = TextboxParser.extract_text_from_textbox(tb)
            if not combined_text:
                continue
            if not combined_text.startswith("Test"):
                continue

            dims = TextboxParser.get_textbox_dimensions(tb)
            width_pt = float(dims.get("width", 0.0))
            height_pt = float(dims.get("height", 0.0))
            width_in = width_pt / POINTS_PER_INCH if width_pt > 0 else 0.0
            height_in = height_pt / POINTS_PER_INCH if height_pt > 0 else 0.0
            area_sq_in = width_in * height_in if width_in > 0 and height_in > 0 else 0.0

            font_info = GraphicsFontManager.get_font_info_from_wt_elements(wt_elements)
            primary_family = font_info.get("family", "Arial")
            detected_sizes = font_info.get("sizes", [])

            # Collect all font families present in wt_elements (rFonts ascii/hAnsi)
            families: List[str] = []
            try:
                from lxml import etree
                for wt in wt_elements:
                    parent = wt.getparent()
                    while parent is not None and not parent.tag.endswith('}r'):
                        parent = parent.getparent()
                    if parent is not None:
                        r_pr = parent.find('.//w:rPr', namespaces={
                            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
                        })
                        if r_pr is not None:
                            r_fonts = r_pr.find('.//w:rFonts', namespaces={
                                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
                            })
                            family = None
                            if r_fonts is not None:
                                ascii_font = r_fonts.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii')
                                h_ansi_font = r_fonts.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi')
                                family = ascii_font or h_ansi_font
                            if family and family not in families:
                                families.append(family)
            except Exception:
                # Fallback to primary family if detailed scan fails
                pass

            if primary_family and primary_family not in families:
                families.append(primary_family)
            if not families:
                families = ["Arial"]

            # Try detected sizes first (unique, sorted), then fill from candidate_sizes
            sizes_to_try: List[float] = sorted({float(s) for s in detected_sizes if isinstance(s, (int, float))})
            for s in candidate_sizes:
                if s not in sizes_to_try:
                    sizes_to_try.append(s)

            location = f"textbox_{i}"
            text_len = len(combined_text)
            sample_text = combined_text[:60]

            for family in families:
                for size_pt in sizes_to_try:
                    lines_fit, max_cpl, total_fit, cpsqi = compute_capacity(
                        width_pt, height_pt, family, float(size_pt)
                    )
                    writer.writerow(
                        [
                            i,
                            location,
                            f"{width_pt:.1f}",
                            f"{height_pt:.1f}",
                            f"{width_in:.2f}",
                            f"{height_in:.2f}",
                            f"{area_sq_in:.2f}",
                            family,
                            ", ".join(str(x) for x in detected_sizes),
                            text_len,
                            sample_text,
                            f"{float(size_pt):.1f}",
                            lines_fit,
                            max_cpl,
                            total_fit,
                            f"{cpsqi:.2f}",
                            "Y" if total_fit >= text_len else "N",
                        ]
                    )

    print(f"Wrote analysis to: {out_csv}")


if __name__ == "__main__":
    main()


