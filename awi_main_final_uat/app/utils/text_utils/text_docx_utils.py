
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from docx import Document
from docx.text.run import Run
from docx.text.paragraph import Paragraph

from config import DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY, DEFAULT_MAPPING

logger = logging.getLogger(__name__)


class TextReconstructor:

    @staticmethod
    def reconstruct_paragraph_text(paragraph: Paragraph) -> Tuple[str, List[Run]]:
        full_text = ""
        runs = []

        for run in paragraph.runs:
            full_text += run.text
            runs.append(run)

        return full_text, runs

    @staticmethod
    def find_text_in_runs(runs: List[Run], search_text: str, start_pos: int = 0) -> Optional[Tuple[int, int, List[Run]]]:
        full_text = "".join(run.text for run in runs)


        match_start = full_text.find(search_text, start_pos)
        if match_start == -1:
            return None

        match_end = match_start + len(search_text)


        current_pos = 0
        affected_runs = []

        for run in runs:
            run_start = current_pos
            run_end = current_pos + len(run.text)


            if run_start < match_end and run_end > match_start:
                affected_runs.append(run)

            current_pos = run_end


            if current_pos > match_end:
                break

        return match_start, match_end, affected_runs


class FontManager:

    @staticmethod
    def get_font_info(run: Run) -> Dict[str, Any]:
        font_info = {
            'font_family': run.font.name or DEFAULT_FONT_FAMILY,
            'font_size': run.font.size.pt if run.font.size else DEFAULT_FONT_SIZE,
            'is_bold': run.font.bold or False,
            'is_italic': run.font.italic or False,
            'is_underline': run.font.underline or False,
            'color': '000000',
            'highlight': None
        }


        if run.font.color and run.font.color.rgb:
            font_info['color'] = str(run.font.color.rgb)


        if run.font.highlight_color:
            font_info['highlight'] = str(run.font.highlight_color)

        return font_info

    @staticmethod
    def get_best_font_info_from_runs(runs: List[Run], document: Document = None) -> Dict[str, Any]:
        if not runs:
            return FontManager.get_default_font_info()


        best_font_info = None
        best_score = 0

        for run in runs:
            run_font_info = FontManager.get_font_info(run)


            score = 0
            if run_font_info.get('font_family') and run_font_info.get('font_family') != DEFAULT_FONT_FAMILY:
                score += 2
            if run_font_info.get('font_size') and run_font_info.get('font_size') != DEFAULT_FONT_SIZE:
                score += 2
            if run_font_info.get('font_family') == DEFAULT_FONT_FAMILY and run_font_info.get('font_size') == DEFAULT_FONT_SIZE:
                score = 0

            if score > best_score:
                best_score = score
                best_font_info = run_font_info


        if best_font_info and best_score > 0:
            return best_font_info
        else:
            return FontManager.get_font_info(runs[0])

    @staticmethod
    def get_default_font_info() -> Dict[str, Any]:
        return {
            'font_family': DEFAULT_FONT_FAMILY,
            'font_size': DEFAULT_FONT_SIZE,
            'color': '000000',
            'is_bold': False,
            'is_italic': False,
            'is_underline': False,
            'highlight': None
        }

    @staticmethod
    def extract_font_info_for_detection(runs: List[Run], matched_text: str, start_pos: int, document: Document = None) -> Dict[str, Any]:
        try:

            text_span = TextReconstructor.find_text_in_runs(runs, matched_text, start_pos)
            if text_span and text_span[2]:
                affected_runs = text_span[2]

                return FontManager.get_best_font_info_from_runs(affected_runs, document)
            else:

                return FontManager.get_default_font_info()
        except Exception as e:
            logger.debug(f"Could not extract font info for detection '{matched_text}': {e}")

            return FontManager.get_default_font_info()

    @staticmethod
    def apply_font_info(run: Run, font_info: Dict[str, Any]):
        try:

            if font_info.get('font_family'):
                run.font.name = font_info['font_family']


            if font_info.get('font_size'):
                from docx.shared import Pt
                run.font.size = Pt(font_info['font_size'])


            if font_info.get('is_bold') is not None:
                run.font.bold = font_info['is_bold']


            if font_info.get('is_italic') is not None:
                run.font.italic = font_info['is_italic']


            if font_info.get('is_underline') is not None:
                run.font.underline = font_info['is_underline']


            if font_info.get('color'):
                from docx.shared import RGBColor
                try:

                    color_hex = font_info['color'].lstrip('#')
                    r = int(color_hex[0:2], 16)
                    g = int(color_hex[2:4], 16)
                    b = int(color_hex[4:6], 16)
                    run.font.color.rgb = RGBColor(r, g, b)
                except (ValueError, IndexError):
                    logger.debug(f"Could not apply color {font_info['color']}")


            if font_info.get('highlight'):
                run.font.highlight_color = font_info['highlight']

        except Exception as e:
            logger.debug(f"Could not apply font info: {e}")


class PatternMatcher:

    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str]):
        self.patterns = patterns
        self.mappings = mappings
        self.compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        for name, pattern in self.patterns.items():
            if not name.startswith('_'):
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    self.compiled_patterns[name] = compiled_pattern
                    logger.debug(f"Compiled pattern '{name}': {pattern}")
                except re.error as e:
                    logger.error(f"Invalid pattern '{name}': {e}")

        logger.info(f"Compiled {len(self.compiled_patterns)} patterns")

    def find_all_pattern_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        all_matches = []

        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            for match in compiled_pattern.finditer(text):
                matched_text = match.group()
                all_matches.append((pattern_name, matched_text, match.start(), match.end()))


        all_matches.sort(key=lambda x: x[2])


        deduplicated_matches = []
        used_positions = set()

        for pattern_name, matched_text, start_pos, end_pos in all_matches:

            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                deduplicated_matches.append((pattern_name, matched_text, start_pos, end_pos))
                used_positions.update(position_range)

        return deduplicated_matches

    def find_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        all_matches = self.find_all_pattern_matches(text)
        matches_with_mappings = []

        for pattern_name, matched_text, start_pos, end_pos in all_matches:

            if matched_text in self.mappings:
                matches_with_mappings.append((pattern_name, matched_text, start_pos, end_pos))

        return matches_with_mappings

    def get_replacement(self, matched_text: str) -> Optional[str]:
        return self.mappings.get(matched_text)


def create_pattern_matcher(patterns: Dict[str, str], mappings: Dict[str, str]) -> PatternMatcher:
    return PatternMatcher(patterns, mappings)


# Cache for patterns and mappings to avoid repeated file I/O
_PATTERNS_CACHE = {}
_MAPPINGS_CACHE = {}
_CACHE_TIMESTAMPS = {}

def load_patterns_and_mappings(config) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load patterns and mappings with file-based caching."""
    import json

    patterns = {}
    mappings = {}

    try:
        # Load patterns with caching
        patterns_path = Path(config.patterns_file) if hasattr(config, 'patterns_file') else Path("patterns.json")
        if patterns_path.exists():
            patterns = _load_cached_json(patterns_path, 'patterns')
            if patterns:
                logger.debug(f"Loaded {len(patterns)} patterns from cache or {patterns_path}")
        else:
            logger.warning(f"Patterns file not found: {patterns_path}")

        # Load mappings with caching
        mappings_path = Path(config.mappings_file) if hasattr(config, 'mappings_file') else Path("mapping.json")
        if mappings_path.exists():
            mappings = _load_cached_json(mappings_path, 'mappings')
            if mappings:
                logger.debug(f"Loaded {len(mappings)} mappings from cache or {mappings_path}")
        else:
            logger.warning(f"Mappings file not found: {mappings_path}")

    except Exception as e:
        logger.error(f"Error loading patterns and mappings: {e}")

    return patterns, mappings


def _load_cached_json(file_path: Path, cache_key: str) -> Dict:
    """Load JSON with caching based on file modification time."""
    import json
    
    try:
        current_mtime = file_path.stat().st_mtime
        cache_entry_key = str(file_path)
        
        # Check if we have cached data that's still valid
        if (cache_entry_key in _CACHE_TIMESTAMPS and 
            _CACHE_TIMESTAMPS[cache_entry_key] == current_mtime):
            if cache_key == 'patterns' and cache_entry_key in _PATTERNS_CACHE:
                return _PATTERNS_CACHE[cache_entry_key]
            elif cache_key == 'mappings' and cache_entry_key in _MAPPINGS_CACHE:
                return _MAPPINGS_CACHE[cache_entry_key]
        
        # Load fresh data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update cache
        _CACHE_TIMESTAMPS[cache_entry_key] = current_mtime
        if cache_key == 'patterns':
            _PATTERNS_CACHE[cache_entry_key] = data
        elif cache_key == 'mappings':
            _MAPPINGS_CACHE[cache_entry_key] = data
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading cached JSON from {file_path}: {e}")
        # Return cached data if available, otherwise empty dict
        if cache_key == 'patterns' and str(file_path) in _PATTERNS_CACHE:
            return _PATTERNS_CACHE[str(file_path)]
        elif cache_key == 'mappings' and str(file_path) in _MAPPINGS_CACHE:
            return _MAPPINGS_CACHE[str(file_path)]
        return {}