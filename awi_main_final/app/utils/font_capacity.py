
from __future__ import annotations

from typing import Dict, Tuple
import json
from pathlib import Path
from config import (
    POINTS_PER_INCH,
    CM_PER_INCH,
)





FALLBACK_DEFAULTS = {
    'width_multiplier': 0.6,
    'line_height_multiplier': 1.25,
    'conservative_headroom': 0.96,
    'safety_margin': 0.15,
}


def _get_font_block(font_family: str) -> Dict | None:
    guidelines_db = _load_guidelines_db()
    fams = (guidelines_db.get('fonts') or {})
    for k, v in fams.items():
        if k == font_family or k.lower() == (font_family or '').lower():
            return v
    return None


def _get_defaults_block() -> Dict:
    guidelines_db = _load_guidelines_db()
    return guidelines_db.get('defaults') or {}


def get_width_multiplier(font_family: str) -> float:
    fam = _get_font_block(font_family)
    if fam:
        try:
            return float((fam.get('multipliers') or {}).get('width'))
        except Exception:
            pass
    try:
        return float(_get_defaults_block().get('width_multiplier', FALLBACK_DEFAULTS['width_multiplier']))
    except Exception:
        return FALLBACK_DEFAULTS['width_multiplier']


def get_line_height_multiplier(font_family: str) -> float:
    fam = _get_font_block(font_family)
    if fam:
        try:
            return float((fam.get('multipliers') or {}).get('line_height'))
        except Exception:
            pass
    try:
        return float(_get_defaults_block().get('line_height_multiplier', FALLBACK_DEFAULTS['line_height_multiplier']))
    except Exception:
        return FALLBACK_DEFAULTS['line_height_multiplier']


def get_conservative_headroom(font_family: str) -> float:
    fam = _get_font_block(font_family)
    if fam and 'conservative_headroom' in fam:
        try:
            return float(fam['conservative_headroom'])
        except Exception:
            pass
    try:
        return float(_get_defaults_block().get('conservative_headroom', FALLBACK_DEFAULTS['conservative_headroom']))
    except Exception:
        return FALLBACK_DEFAULTS['conservative_headroom']


def get_safety_margin(font_family: str, default_margin: float = FALLBACK_DEFAULTS['safety_margin']) -> float:
    fam = _get_font_block(font_family)
    if fam and 'safety_margin' in fam:
        try:
            return float(fam['safety_margin'])
        except Exception:
            pass
    try:
        return float(_get_defaults_block().get('safety_margin', default_margin))
    except Exception:
        return default_margin


def evaluate_capacity(
    width_pt: float,
    height_pt: float,
    font_family: str,
    font_size_pt: float,
    safety_margin: float = 0.15,
) -> Tuple[int, int, int, float, float, float, float]:
    if width_pt <= 0 or height_pt <= 0 or font_size_pt <= 0:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    width_in = width_pt / POINTS_PER_INCH
    height_in = height_pt / POINTS_PER_INCH
    area_in2 = max(0.0, width_in * height_in)
    area_cm2 = area_in2 * (CM_PER_INCH ** 2)

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




# ===== Unified JSON guidelines (height->lines, width->cpl) =====
# FIXED: Removed global variable to prevent memory leaks
# Guidelines are now loaded on-demand and cached per-process
_GUIDELINES_CACHE_TIMESTAMP: float = 0.0
_GUIDELINES_CACHE_PATH: Path = None
_GUIDELINES_CACHE_DATA: Dict = {}

def _json_path() -> Path:
    try:
        base_dir = Path(__file__).resolve().parents[2]
    except Exception:
        base_dir = Path('.')
    return base_dir / 'font_guidelines.json'


def _round_key(value: float) -> str:
    try:
        return f"{round(float(value), 1)}"
    except Exception:
        return str(value)


def _nearest_numeric_key(target: float, keys) -> str | None:
    try:
        numeric = []
        for k in keys:
            try:
                numeric.append((abs(float(k) - float(target)), k))
            except Exception:
                continue
        if not numeric:
            return None
        numeric.sort(key=lambda x: x[0])
        return numeric[0][1]
    except Exception:
        return None


def _interp_conservative(x: float, mapping: Dict[float, float], bias: float = 0.0) -> float:
    """Conservative interpolation between numeric-key mapping.

    - Never exceed the min of neighbor values (down-bias)
    - Apply additional multiplicative bias (e.g., 0.05 => 5% down)
    - Ensure minimum of 1.0 if input values are at least 1.0
    """
    xs = sorted(mapping.keys())
    if not xs:
        return 0.0
    if x <= xs[0]:
        base = mapping[xs[0]]
        result = max(0.0, float(base) * (1.0 - bias))
        # Ensure minimum of 1.0 if input was at least 1.0
        if float(base) >= 1.0:
            result = max(1.0, result)
        return result
    if x >= xs[-1]:
        base = mapping[xs[-1]]
        result = max(0.0, float(base) * (1.0 - bias))
        # Ensure minimum of 1.0 if input was at least 1.0
        if float(base) >= 1.0:
            result = max(1.0, result)
        return result
    import bisect as _bis
    i = _bis.bisect_left(xs, x)
    x1, x0 = xs[i], xs[i - 1]
    y1, y0 = float(mapping[x1]), float(mapping[x0])
    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    est = y0 * (1 - t) + y1 * t
    cons = min(est, y0, y1)
    result = max(0.0, cons * (1.0 - bias))
    # Ensure minimum of 1.0 if both input values are at least 1.0
    if y0 >= 1.0 and y1 >= 1.0:
        result = max(1.0, result)
    return result


def _size_keys_desc() -> list[str]:
    keys = []
    v = 12.0
    while v >= 6.0 - 1e-9:
        keys.append(f"{v:.1f}")
        v -= 0.5
    return keys


def _load_guidelines_db() -> Dict:
    """Load guidelines with file-based caching to avoid repeated 4MB+ file reads.
    FIXED: No longer uses global variable to prevent memory leaks."""
    global _GUIDELINES_CACHE_TIMESTAMP, _GUIDELINES_CACHE_PATH, _GUIDELINES_CACHE_DATA
    
    p = _json_path()
    if not p.exists():
        return {}
    
    try:
        # Check if we need to reload based on file modification time
        current_mtime = p.stat().st_mtime
        
        # Use cached data if file hasn't changed and we have data
        if (_GUIDELINES_CACHE_PATH == p and 
            _GUIDELINES_CACHE_TIMESTAMP == current_mtime and 
            _GUIDELINES_CACHE_DATA):
            return _GUIDELINES_CACHE_DATA
        
        # Load fresh data from file
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update cache (replaces previous data to prevent accumulation)
        _GUIDELINES_CACHE_DATA = data
        _GUIDELINES_CACHE_TIMESTAMP = current_mtime
        _GUIDELINES_CACHE_PATH = p
        
        return data
        
    except Exception as e:
        # Log error but don't crash - return cached data if available
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error loading guidelines from {p}: {e}")
        return _GUIDELINES_CACHE_DATA if _GUIDELINES_CACHE_DATA else {}

def get_guideline_limits(
    font_family: str,
    width_pt: float,
    height_pt: float,
    font_size_pt: float,
) -> Tuple[int | None, int | None]:
    # Prefer JSON DB with height/width separation
    guidelines_db = _load_guidelines_db()
    fam = None
    for k, v in (guidelines_db.get('fonts') or {}).items():
        if k == font_family or k.lower() == (font_family or '').lower():
            fam = v
            break
    if fam:
        size_key = _round_key(font_size_pt)
        lines_allowed = None
        cpl_allowed = None
        # lines depend on height only
        h_key = _round_key(height_pt)
        lines_map = (fam.get('lines_by_height') or {}).get(h_key)
        if lines_map:
            lines_allowed = lines_map.get(size_key)
        # cpl depends on width only
        w_key = _round_key(width_pt)
        cpl_map = (fam.get('cpl_by_width') or {}).get(w_key)
        if cpl_map:
            cpl_allowed = cpl_map.get(size_key)
        return lines_allowed, cpl_allowed

    # Return None if JSON format not available
    return None, None


def apply_guidelines_to_capacity(
    lines_fit: int,
    max_cpl: int,
    width_pt: float,
    height_pt: float,
    font_family: str,
    font_size_pt: float,
) -> Tuple[int, int]:
    # Use nearest dimension guideline limits scaled to the actual box size
    max_lines_allowed, max_cpl_allowed = get_guideline_limits_scaled(
        font_family, width_pt, height_pt, font_size_pt
    )

    effective_lines = (
        min(lines_fit, int(max_lines_allowed)) if max_lines_allowed is not None else lines_fit
    )
    effective_cpl = (
        min(max_cpl, int(max_cpl_allowed)) if max_cpl_allowed is not None else max_cpl
    )
    return effective_lines, effective_cpl


def get_guideline_limits_scaled(
    font_family: str,
    width_pt: float,
    height_pt: float,
    font_size_pt: float,
) -> Tuple[int | None, int | None]:
    """Find nearest guideline by dimensions, then scale its caps to our box size.

    - allowed_lines scales with height ratio
    - max_cpl scales with width ratio
    """
    # Prefer JSON DB: scale height-only and width-only separately
    guidelines_db = _load_guidelines_db()
    fam = None
    for k, v in (guidelines_db.get('fonts') or {}).items():
        if k == font_family or k.lower() == (font_family or '').lower():
            fam = v
            break
    size_key = _round_key(font_size_pt)
    if fam:
        h_key = _round_key(height_pt)
        w_key = _round_key(width_pt)
        base_lines = None
        base_cpl = None
        # If exact height/width not present, pick nearest numeric key
        lines_by_height = fam.get('lines_by_height') or {}
        if h_key not in lines_by_height and lines_by_height:
            nk = _nearest_numeric_key(float(h_key), lines_by_height.keys())
            h_key = nk or h_key
        cpl_by_width = fam.get('cpl_by_width') or {}
        if w_key not in cpl_by_width and cpl_by_width:
            nk = _nearest_numeric_key(float(w_key), cpl_by_width.keys())
            w_key = nk or w_key
        base_lines = (lines_by_height.get(h_key) or {}).get(size_key)
        base_cpl = (cpl_by_width.get(w_key) or {}).get(size_key)
        # If either is missing/None, derive via capacity using multipliers (best-fit estimate)
        if base_lines is None or base_cpl is None:
            try:
                safety = get_safety_margin(font_family, default_margin=FALLBACK_DEFAULTS['safety_margin'])
                lines_fit, max_cpl, *_ = evaluate_capacity(
                    width_pt, height_pt, font_family, float(size_key), safety_margin=safety
                )
                if base_lines is None:
                    base_lines = max(1, int(lines_fit))
                if base_cpl is None:
                    base_cpl = max(1, int(max_cpl))
            except Exception:
                pass
        # Return JSON-derived or evaluated values
        return base_lines, base_cpl

    # Return None if JSON format not available
    return None, None


def get_guideline_limits_interpolated(
    font_family: str,
    font_size_pt: float,
    height_pt: float,
    width_pt: float,
    conservative_bias: float = 0.0,
) -> Tuple[int | None, int | None, str]:

    # First, try exact match in new JSON format
    exact_lines, exact_cpl = get_guideline_limits(font_family, width_pt, height_pt, font_size_pt)
    
    # If we have exact matches, use them directly
    if exact_lines is not None and exact_cpl is not None:
        return exact_lines, exact_cpl, "Height: exact match; Width: exact match"
    
    # Use new JSON format for interpolation only if exact match failed
    fam = _get_font_block(font_family)
    if fam:
        lines = exact_lines  # Use exact lines if available
        cpl = exact_cpl      # Use exact CPL if available
        

        
        # If we don't have exact lines, interpolate height -> lines
        if lines is None:
            hmap = {}
            s_key = _round_key(font_size_pt)
            for h, m in (fam.get('lines_by_height') or {}).items():
                try:
                    val = m.get(s_key)
                    if val is not None:
                        hmap[float(h)] = float(val)
                except Exception:
                    continue
            if hmap:
                interpolated_lines = _interp_conservative(float(height_pt), hmap, 0.0)  # No bias for lines
                lines = int(interpolated_lines)

        # If we don't have exact CPL, interpolate width -> cpl
        if cpl is None:
            wmap = {}
            s_key = _round_key(font_size_pt)
            for w, m in (fam.get('cpl_by_width') or {}).items():
                try:
                    val = m.get(s_key)
                    if val is not None:
                        wmap[float(w)] = float(val)
                except Exception:
                    continue
            if wmap:
                interpolated_cpl = _interp_conservative(float(width_pt), wmap, conservative_bias)
                cpl = int(interpolated_cpl)

        # Build interpolation details
        interpolation_details = []
        if exact_lines is not None:
            interpolation_details.append("Height: exact match")
        else:
            interpolation_details.append("Height: interpolated")
            
        if exact_cpl is not None:
            interpolation_details.append("Width: exact match")
        else:
            interpolation_details.append("Width: interpolated")

        details_str = "; ".join(interpolation_details)
        return lines, cpl, details_str

    # Return None if JSON format not available
    return None, None, "No font guidelines available"