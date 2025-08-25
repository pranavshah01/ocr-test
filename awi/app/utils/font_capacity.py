"""
Font capacity utilities: estimate how much text fits in a textbox per font family and size.

Provides:
- get_width_multiplier(font_family)
- evaluate_capacity(width_pt, height_pt, font_family, font_size_pt, safety_margin=0.2)

All dimensions are points (1 inch = 72 points). Returns lines_fit, max_chars_per_line,
total_chars_fit, chars_per_sq_in, chars_per_sq_cm, area_in2, area_cm2.
"""

from __future__ import annotations

from typing import Dict, Tuple

POINTS_PER_INCH = 72.0
CM_PER_INCH = 2.54


FONT_WIDTH_MULTIPLIERS: Dict[str, float] = {
    "Arial": 0.58,
    "Times New Roman": 0.53,
    "Calibri": 0.58,
    "Verdana": 0.62,
    # Slightly narrower for CJK to better reflect dense packing in callouts
    "PMingLiU": 0.58,
    "SimSun": 0.60,
    "Microsoft YaHei": 0.60,
    "ACADEMY ENGRAVED LET PLAIN": 0.7,
}


def get_width_multiplier(font_family: str) -> float:
    if not font_family:
        return 0.6
    if font_family in FONT_WIDTH_MULTIPLIERS:
        return FONT_WIDTH_MULTIPLIERS[font_family]
    for name, mult in FONT_WIDTH_MULTIPLIERS.items():
        if name.lower() == font_family.lower():
            return mult
    return 0.6


FONT_LINE_HEIGHT_MULTIPLIERS: Dict[str, float] = {
    "Arial": 1.18,
    "Times New Roman": 1.25,
    # Tighter line-height observed for CJK in boxed callouts
    "PMingLiU": 1.10,
    "SimSun": 1.10,
    "Microsoft YaHei": 1.14,
}


def get_line_height_multiplier(font_family: str) -> float:
    if not font_family:
        return 1.25
    if font_family in FONT_LINE_HEIGHT_MULTIPLIERS:
        return FONT_LINE_HEIGHT_MULTIPLIERS[font_family]
    for name, mult in FONT_LINE_HEIGHT_MULTIPLIERS.items():
        if name.lower() == font_family.lower():
            return mult
    return 1.25


# Conservative headroom thresholds used during fitting (fraction of capacity)
CONSERVATIVE_HEADROOM: Dict[str, float] = {
    # CJK families: keep stricter headroom to avoid overflow
    "PMingLiU": 0.92,
    "SimSun": 0.92,
    "Microsoft YaHei": 0.92,
    # Latin defaults are handled by fallback below
}


def get_conservative_headroom(font_family: str) -> float:
    if not font_family:
        return 0.96
    # Direct match
    if font_family in CONSERVATIVE_HEADROOM:
        return CONSERVATIVE_HEADROOM[font_family]
    # Case-insensitive match
    for name, headroom in CONSERVATIVE_HEADROOM.items():
        if name.lower() == font_family.lower():
            return headroom
    # Default: slightly less conservative for Latin fonts
    return 0.96


# Per-font safety margins (fraction of width/height kept as padding)
# Latin fonts can use a smaller margin; CJK keeps a larger margin for glyph complexity
FONT_SAFETY_MARGINS: Dict[str, float] = {
    "Arial": 0.12,
    "Times New Roman": 0.12,
    "Verdana": 0.12,
    "Calibri": 0.12,
    "PMingLiU": 0.18,
    "SimSun": 0.18,
    "Microsoft YaHei": 0.18,
}


def get_safety_margin(font_family: str, default_margin: float = 0.15) -> float:
    if not font_family:
        return default_margin
    if font_family in FONT_SAFETY_MARGINS:
        return FONT_SAFETY_MARGINS[font_family]
    for name, margin in FONT_SAFETY_MARGINS.items():
        if name.lower() == font_family.lower():
            return margin
    return default_margin

def evaluate_capacity(
    width_pt: float,
    height_pt: float,
    font_family: str,
    font_size_pt: float,
    safety_margin: float = 0.15,
) -> Tuple[int, int, int, float, float, float, float]:
    """
    Compute capacity metrics for a given textbox and font settings.

    Returns:
        (lines_fit, max_chars_per_line, total_chars_fit,
         chars_per_sq_in, chars_per_sq_cm, area_in2, area_cm2)
    """
    if width_pt <= 0 or height_pt <= 0 or font_size_pt <= 0:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    width_in = width_pt / POINTS_PER_INCH
    height_in = height_pt / POINTS_PER_INCH
    area_in2 = max(0.0, width_in * height_in)
    area_cm2 = area_in2 * (CM_PER_INCH ** 2)

    # Apply padding based on safety margin
    usable_width_in = width_in * (1.0 - safety_margin)
    usable_height_in = height_in * (1.0 - safety_margin)

    width_multiplier = get_width_multiplier(font_family)
    line_multiplier = get_line_height_multiplier(font_family)
    char_width_in = (font_size_pt * width_multiplier) / POINTS_PER_INCH
    line_height_in = (font_size_pt * line_multiplier) / POINTS_PER_INCH

    if char_width_in <= 0 or line_height_in <= 0:
        return 0, 0, 0, 0.0, 0.0, area_in2, area_cm2

    max_chars_per_line = int(usable_width_in // char_width_in) if usable_width_in > 0 else 0
    lines_fit = int(usable_height_in // line_height_in) if usable_height_in > 0 else 0
    total_chars_fit = max(0, max_chars_per_line * lines_fit)

    chars_per_sq_in = (total_chars_fit / area_in2) if area_in2 > 0 else 0.0
    chars_per_sq_cm = (total_chars_fit / area_cm2) if area_cm2 > 0 else 0.0

    return (
        lines_fit,
        max_chars_per_line,
        total_chars_fit,
        chars_per_sq_in,
        chars_per_sq_cm,
        area_in2,
        area_cm2,
    )


# ----------------------
# Optional guideline system (per-font, per-dimension)
# ----------------------
# Standard schema (edit-friendly):
# - Always use 'size_range': [min_pt, max_pt]
# - Provide both 'max_lines' and 'max_cpl' (max characters per line) when possible
#
# Structure:
# FONT_CAPACITY_GUIDELINES = {
#   font_family: {
#     'dimensions': {
#        'WIDTHxHEIGHT' (points, 1-dec place): [
#           { 'size_range': [min_pt, max_pt], 'max_lines': int, 'max_cpl': int },
#           ...
#        ]
#     }
#   }
# }

FONT_CAPACITY_GUIDELINES: Dict[str, Dict] = {
    # Example based on user guidance for the box with dimensions 116.4x31.2 pt
    # that contains 77-110-0351301-00
    "Arial": {
        "dimensions": {
            "116.4x31.2": [
                {"size_range": [7.5, 11.0], "max_lines": 2, "max_cpl": 20},
                {"size_range": [7.0, 7.0],  "max_lines": 3, "max_cpl": 22},
                {"size_range": [11.5, 14.0], "max_lines": 1, "max_cpl": 16},
            ],
            "180.6x35.6":[
                {"size_range": [9.0,9.0], "max_lines": 2, "max_cpl": 34},  # Reduced from 34 to 30 for better fit
                {"size_range": [8.5,8.5], "max_lines": 3, "max_cpl": 36},  # Increased to 36 to allow the specific text to fit
                {"size_range": [8.0,8.0], "max_lines": 3, "max_cpl": 40},  # Added 8.0pt guideline for 3 lines
            ],
            "119.5x31.2":[
                {"size_range": [8.0,8.0], "max_lines": 2, "max_cpl": 22},  # Reduced from 34 to 30 for better fit
            ],
            "155.3x21.8":[
                {"size_range": [6.0,6.0], "max_lines": 2, "max_cpl": 45},  # Allow 6.0pt for 45-char text
                {"size_range": [5.0,5.0], "max_lines": 3, "max_cpl": 50},  # Fallback to 5.0pt if needed
            ]
        }
    },
    # Add more families/dimensions here as needed
    "Times New Roman": {
        "dimensions": {
            # Example placeholder; fill with rules if needed
        }
    },
    "PMingLiU": {
        "dimensions": {
        }
    },
}


def _dimension_key(width_pt: float, height_pt: float) -> str:
    return f"{round(float(width_pt), 1)}x{round(float(height_pt), 1)}"


def get_guideline_limits(
    font_family: str,
    width_pt: float,
    height_pt: float,
    font_size_pt: float,
) -> Tuple[int | None, int | None]:
    """Return (max_lines_allowed, max_cpl_allowed) for given settings if a guideline exists."""
    ruleset = FONT_CAPACITY_GUIDELINES.get(font_family)
    if not ruleset:
        # case-insensitive fallback
        for name, value in FONT_CAPACITY_GUIDELINES.items():
            if name.lower() == (font_family or "").lower():
                ruleset = value
                break
    if not ruleset:
        return None, None
    dims = ruleset.get("dimensions", {})
    key = _dimension_key(width_pt, height_pt)
    if key not in dims:
        return None, None
    for rule in dims[key]:
        r = rule or {}
        # Standardized: only size_range is supported; encode exact sizes as [x, x]
        srange = r.get("size_range")
        if not srange or len(srange) != 2:
            continue
        if float(srange[0]) <= float(font_size_pt) <= float(srange[1]):
            return r.get("max_lines"), r.get("max_cpl")
    return None, None


def apply_guidelines_to_capacity(
    lines_fit: int,
    max_cpl: int,
    width_pt: float,
    height_pt: float,
    font_family: str,
    font_size_pt: float,
) -> Tuple[int, int]:
    """Apply guideline limits to capacity numbers. Use guideline limits when available."""
    max_lines_allowed, max_cpl_allowed = get_guideline_limits(
        font_family, width_pt, height_pt, font_size_pt
    )
    # Use guideline limits when available, otherwise use calculated capacity
    effective_lines = max_lines_allowed if max_lines_allowed is not None else lines_fit
    effective_cpl = max_cpl_allowed if max_cpl_allowed is not None else max_cpl
    return effective_lines, effective_cpl

