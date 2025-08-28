"""
DOCX manipulation utilities for text processing and formatting preservation.
Handles text reconstruction, font management, and document structure operations.
Adapted for awi_main_final's text processing features.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import logging

from docx import Document
from docx.text.run import Run
from docx.text.paragraph import Paragraph

from config import DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY, DEFAULT_MAPPING

logger = logging.getLogger(__name__)


class TextReconstructor:
    """Reconstructs text across multiple <w:t> elements for pattern matching."""
    
    @staticmethod
    def reconstruct_paragraph_text(paragraph: Paragraph) -> Tuple[str, List[Run]]:
        """
        Reconstruct full text from paragraph runs for pattern matching.
        
        Args:
            paragraph: Paragraph to reconstruct
            
        Returns:
            Tuple of (full_text, list_of_runs)
        """
        full_text = ""
        runs = []
        
        for run in paragraph.runs:
            full_text += run.text
            runs.append(run)
        
        return full_text, runs
    
    @staticmethod
    def find_text_in_runs(runs: List[Run], search_text: str, start_pos: int = 0) -> Optional[Tuple[int, int, List[Run]]]:
        """
        Find text span across multiple runs.
        
        Args:
            runs: List of runs to search
            search_text: Text to find
            start_pos: Starting position in reconstructed text
            
        Returns:
            Tuple of (start_index, end_index, affected_runs) or None
        """
        full_text = "".join(run.text for run in runs)
        
        # Find the text in the reconstructed string
        match_start = full_text.find(search_text, start_pos)
        if match_start == -1:
            return None
        
        match_end = match_start + len(search_text)
        
        # Find which runs contain the matched text
        current_pos = 0
        affected_runs = []
        
        for run in runs:
            run_start = current_pos
            run_end = current_pos + len(run.text)
            
            # Check if this run overlaps with the match
            if run_start < match_end and run_end > match_start:
                affected_runs.append(run)
            
            current_pos = run_end
            
            # Stop if we've passed the match
            if current_pos > match_end:
                break
        
        return match_start, match_end, affected_runs


class FontManager:
    """Manages font information and formatting preservation."""
    
    @staticmethod
    def get_font_info(run: Run) -> Dict[str, Any]:
        """
        Extract font information from a run.
        
        Args:
            run: Run to extract font info from
            
        Returns:
            Dictionary with font information
        """
        font_info = {
            'font_family': run.font.name or DEFAULT_FONT_FAMILY,
            'font_size': run.font.size.pt if run.font.size else DEFAULT_FONT_SIZE,
            'is_bold': run.font.bold or False,
            'is_italic': run.font.italic or False,
            'is_underline': run.font.underline or False,
            'color': '000000',  # Default color
            'highlight': None
        }
        
        # Extract color information
        if run.font.color and run.font.color.rgb:
            font_info['color'] = str(run.font.color.rgb)
        
        # Extract highlight information
        if run.font.highlight_color:
            font_info['highlight'] = str(run.font.highlight_color)
        
        return font_info
    
    @staticmethod
    def get_best_font_info_from_runs(runs: List[Run], document: Document = None) -> Dict[str, Any]:
        """
        Get the best font information from a list of runs.
        
        Args:
            runs: List of runs to analyze
            document: Optional document for fallback font info
            
        Returns:
            Best font information dictionary
        """
        if not runs:
            return FontManager.get_default_font_info()
        
        # Find the best run with explicit font properties
        best_font_info = None
        best_score = 0
        
        for run in runs:
            run_font_info = FontManager.get_font_info(run)
            
            # Score this run's font info (higher score = better)
            score = 0
            if run_font_info.get('font_family') and run_font_info.get('font_family') != DEFAULT_FONT_FAMILY:
                score += 2  # Explicit font family
            if run_font_info.get('font_size') and run_font_info.get('font_size') != DEFAULT_FONT_SIZE:
                score += 2  # Explicit font size
            if run_font_info.get('font_family') == DEFAULT_FONT_FAMILY and run_font_info.get('font_size') == DEFAULT_FONT_SIZE:
                score = 0  # Default values, lowest priority
            
            if score > best_score:
                best_score = score
                best_font_info = run_font_info
        
        # Use the best font info found, or the first run's info if all are defaults
        if best_font_info and best_score > 0:
            return best_font_info
        else:
            return FontManager.get_font_info(runs[0])
    
    @staticmethod
    def get_default_font_info() -> Dict[str, Any]:
        """Get default font information."""
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
        """
        Extract font information for a specific text detection.
        
        Args:
            runs: List of runs to search
            matched_text: The matched text to find font info for
            start_pos: Starting position of the match
            document: Optional document for fallback
            
        Returns:
            Font information dictionary
        """
        try:
            # Find the text in the runs to get font info
            text_span = TextReconstructor.find_text_in_runs(runs, matched_text, start_pos)
            if text_span and text_span[2]:  # affected_runs
                affected_runs = text_span[2]
                # Use the utility class to get the best font info
                return FontManager.get_best_font_info_from_runs(affected_runs, document)
            else:
                # Fallback to default font info if no runs found
                return FontManager.get_default_font_info()
        except Exception as e:
            logger.debug(f"Could not extract font info for detection '{matched_text}': {e}")
            # Fallback to default font info
            return FontManager.get_default_font_info()
    
    @staticmethod
    def apply_font_info(run: Run, font_info: Dict[str, Any]):
        """
        Apply font information to a run.
        
        Args:
            run: Run to apply font info to
            font_info: Dictionary containing font properties
        """
        try:
            # Apply font family
            if font_info.get('font_family'):
                run.font.name = font_info['font_family']
            
            # Apply font size
            if font_info.get('font_size'):
                from docx.shared import Pt
                run.font.size = Pt(font_info['font_size'])
            
            # Apply bold
            if font_info.get('is_bold') is not None:
                run.font.bold = font_info['is_bold']
            
            # Apply italic
            if font_info.get('is_italic') is not None:
                run.font.italic = font_info['is_italic']
            
            # Apply underline
            if font_info.get('is_underline') is not None:
                run.font.underline = font_info['is_underline']
            
            # Apply color
            if font_info.get('color'):
                from docx.shared import RGBColor
                try:
                    # Convert hex color to RGB
                    color_hex = font_info['color'].lstrip('#')
                    r = int(color_hex[0:2], 16)
                    g = int(color_hex[2:4], 16)
                    b = int(color_hex[4:6], 16)
                    run.font.color.rgb = RGBColor(r, g, b)
                except (ValueError, IndexError):
                    logger.debug(f"Could not apply color {font_info['color']}")
            
            # Apply highlight
            if font_info.get('highlight'):
                run.font.highlight_color = font_info['highlight']
                
        except Exception as e:
            logger.debug(f"Could not apply font info: {e}")


class PatternMatcher:
    """Pattern matcher for text processing."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str]):
        """
        Initialize pattern matcher.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
        """
        self.patterns = patterns
        self.mappings = mappings
        self.compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all patterns for efficient matching."""
        for name, pattern in self.patterns.items():
            if not name.startswith('_'):  # Skip metadata
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    self.compiled_patterns[name] = compiled_pattern
                    logger.debug(f"Compiled pattern '{name}': {pattern}")
                except re.error as e:
                    logger.error(f"Invalid pattern '{name}': {e}")
        
        logger.info(f"Compiled {len(self.compiled_patterns)} patterns")
    
    def find_all_pattern_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find ALL pattern matches in text, regardless of whether they have mappings.
        
        Args:
            text: Text to search
            
        Returns:
            List of tuples (pattern_name, matched_text, start_pos, end_pos)
        """
        all_matches = []
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            for match in compiled_pattern.finditer(text):
                matched_text = match.group()
                all_matches.append((pattern_name, matched_text, match.start(), match.end()))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x[2])
        
        # Deduplicate overlapping matches - keep only the first match for each position
        deduplicated_matches = []
        used_positions = set()
        
        for pattern_name, matched_text, start_pos, end_pos in all_matches:
            # Check if this position range overlaps with any already used position
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                deduplicated_matches.append((pattern_name, matched_text, start_pos, end_pos))
                used_positions.update(position_range)
        
        return deduplicated_matches
    
    def get_replacement(self, matched_text: str) -> Optional[str]:
        """
        Get replacement text for matched text.
        
        Args:
            matched_text: The text that was matched
            
        Returns:
            Replacement text or None if no mapping exists
        """
        return self.mappings.get(matched_text)


def create_pattern_matcher(patterns: Dict[str, str], mappings: Dict[str, str]) -> PatternMatcher:
    """Factory function to create a PatternMatcher instance."""
    return PatternMatcher(patterns, mappings)


def load_patterns_and_mappings(config) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load patterns and mappings from configuration files.
    
    Args:
        config: Configuration object containing file paths
        
    Returns:
        Tuple of (patterns_dict, mappings_dict)
    """
    import json
    from pathlib import Path
    
    patterns = {}
    mappings = {}
    
    try:
        # Load patterns
        patterns_path = Path(config.patterns_file) if hasattr(config, 'patterns_file') else Path("patterns.json")
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded {len(patterns)} patterns from {patterns_path}")
        else:
            logger.warning(f"Patterns file not found: {patterns_path}")
        
        # Load mappings
        mappings_path = Path(config.mappings_file) if hasattr(config, 'mappings_file') else Path("mapping.json")
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
            logger.info(f"Loaded {len(mappings)} mappings from {mappings_path}")
        else:
            logger.warning(f"Mappings file not found: {mappings_path}")
            
    except Exception as e:
        logger.error(f"Error loading patterns and mappings: {e}")
    
    return patterns, mappings
