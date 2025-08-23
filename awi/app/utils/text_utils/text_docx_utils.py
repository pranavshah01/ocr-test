"""
DOCX manipulation utilities for text processing and formatting preservation.
Handles text reconstruction, font management, and document structure operations.
Adapted for awi's enhanced features including layout analysis and page mapping.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from docx import Document
from docx.text.run import Run
from docx.text.paragraph import Paragraph
from docx.shared import RGBColor, Pt
from docx.enum.text import WD_COLOR_INDEX

from ..shared_constants import XML_NAMESPACES, DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY, DEFAULT_MAPPING, DEFAULT_SEPARATOR, PROCESSING_MODES

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
            'color': None,
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
    def apply_font_info(run: Run, font_info: Dict[str, Any]):
        """
        Apply font information to a run.
        
        Args:
            run: Run to apply font info to
            font_info: Font information dictionary
        """
        try:
            if 'font_family' in font_info and font_info['font_family']:
                run.font.name = font_info['font_family']
            
            if 'font_size' in font_info and font_info['font_size']:
                run.font.size = Pt(font_info['font_size'])
            
            if 'is_bold' in font_info:
                run.font.bold = font_info['is_bold']
            
            if 'is_italic' in font_info:
                run.font.italic = font_info['is_italic']
            
            if 'is_underline' in font_info:
                run.font.underline = font_info['is_underline']
            
            if 'color' in font_info and font_info['color']:
                try:
                    # Parse RGB color string
                    color_str = font_info['color'].replace('#', '')
                    if len(color_str) == 6:
                        r = int(color_str[0:2], 16)
                        g = int(color_str[2:4], 16)
                        b = int(color_str[4:6], 16)
                        run.font.color.rgb = RGBColor(r, g, b)
                except (ValueError, AttributeError):
                    logger.warning(f"Failed to apply color: {font_info['color']}")
            
        except Exception as e:
            logger.warning(f"Failed to apply font formatting: {e}")
    
    @staticmethod
    def get_smallest_font_size(runs: List[Run]) -> float:
        """
        Get the smallest font size from a list of runs.
        
        Args:
            runs: List of runs to check
            
        Returns:
            Smallest font size in points
        """
        sizes = []
        for run in runs:
            if run.font.size:
                sizes.append(run.font.size.pt)
            else:
                sizes.append(DEFAULT_FONT_SIZE)
        
        return min(sizes) if sizes else DEFAULT_FONT_SIZE
    
    @staticmethod
    def normalize_font_sizes(runs: List[Run], target_size: float):
        """
        Normalize all runs to the same font size.
        
        Args:
            runs: List of runs to normalize
            target_size: Target font size in points
        """
        for run in runs:
            run.font.size = Pt(target_size)
    
    @staticmethod
    def get_document_default_font_info(document: Document) -> Dict[str, Any]:
        """
        Get default font information from document styles.
        
        Args:
            document: Document to extract default font info from
            
        Returns:
            Dictionary with default font information
        """
        try:
            # Try to get default font from document styles
            default_paragraph_style = document.styles['Normal']
            default_font = default_paragraph_style.font
            
            return {
                'font_family': default_font.name or DEFAULT_FONT_FAMILY,
                'font_size': default_font.size.pt if default_font.size else DEFAULT_FONT_SIZE,
                'is_bold': default_font.bold or False,
                'is_italic': default_font.italic or False
            }
        except Exception as e:
            logger.debug(f"Could not extract default font info: {e}")
            return {
                'font_family': DEFAULT_FONT_FAMILY,
                'font_size': DEFAULT_FONT_SIZE,
                'is_bold': False,
                'is_italic': False
            }
    
    @staticmethod
    def get_best_font_info_from_runs(runs: List[Run], document=None) -> Dict[str, Any]:
        """
        Get the best font information from a list of runs.
        
        Args:
            runs: List of runs to extract font info from
            document: Optional document for fallback font info
            
        Returns:
            Dictionary with the best font information found
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
            font_info = best_font_info
        else:
            font_info = FontManager.get_font_info(runs[0])
        
        # If still no explicit font info found, try to get from document styles
        if (font_info.get('font_family') == DEFAULT_FONT_FAMILY and 
            font_info.get('font_size') == DEFAULT_FONT_SIZE and 
            document):
            default_font_info = FontManager.get_document_default_font_info(document)
            # Only use default if we don't have explicit font info
            if font_info.get('font_family') == DEFAULT_FONT_FAMILY:
                font_info['font_family'] = default_font_info.get('font_family', DEFAULT_FONT_FAMILY)
            if font_info.get('font_size') == DEFAULT_FONT_SIZE:
                font_info['font_size'] = default_font_info.get('font_size', DEFAULT_FONT_SIZE)
        
        return font_info
    
    @staticmethod
    def get_default_font_info() -> Dict[str, Any]:
        """
        Get default font information.
        
        Returns:
            Dictionary with default font information
        """
        return {
            'font_family': DEFAULT_FONT_FAMILY,
            'font_size': DEFAULT_FONT_SIZE,
            'color': '000000',
            'is_bold': False,
            'is_italic': False,
            'is_underline': False
        }

class TextReplacer:
    """Handles text replacement while preserving formatting."""
    
    def __init__(self, mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING):
        """
        Initialize text replacer.
        
        Args:
            mode: Replacement mode ('append' or 'replace')
            separator: Separator between original and appended text in append mode
            default_mapping: Default text to append when no mapping is found
        """
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
    
    def replace_text_in_runs(self, runs: List[Run], original_text: str, replacement_text: str, 
                           start_pos: int, end_pos: int) -> bool:
        """
        Replace text across multiple runs while preserving formatting.
        
        Args:
            runs: List of runs containing the text
            original_text: Original text to replace
            replacement_text: Replacement text
            start_pos: Start position in reconstructed text
            end_pos: End position in reconstructed text
            
        Returns:
            True if replacement was successful
        """
        try:
            # Find which runs are affected
            current_pos = 0
            affected_runs = []
            
            for run in runs:
                run_start = current_pos
                run_end = current_pos + len(run.text)
                
                if run_start < end_pos and run_end > start_pos:
                    affected_runs.append((run, run_start, run_end))
                
                current_pos = run_end
            
            if not affected_runs:
                return False
            
            # Determine replacement text based on mode
            if self.mode == PROCESSING_MODES['APPEND']:
                # Check if the replacement text already exists after the original text
                # This prevents duplicate appending when the same text appears in multiple cells
                expected_append = f"{original_text}{self.separator}{replacement_text}"
                
                # Get the current text in the affected runs to check for existing append
                current_text = ""
                for run in runs:
                    current_text += run.text
                
                # Check if the expected append already exists
                if expected_append in current_text:
                    logger.info(f"APPEND MODE: Skipping duplicate append for '{original_text}' -> '{replacement_text}' (already exists)")
                    return True  # Return True to indicate "success" (no action needed)
                
                final_text = expected_append
            else:  # replace mode
                final_text = replacement_text
            
            # Get font info from the first affected run
            first_run = affected_runs[0][0]
            font_info = FontManager.get_font_info(first_run)
            
            # Clear text from all affected runs except the first
            for i, (run, run_start, run_end) in enumerate(affected_runs):
                if i == 0:
                    # Calculate the portion of text to replace in the first run
                    run_match_start = max(0, start_pos - run_start)
                    run_match_end = min(len(run.text), end_pos - run_start)
                    
                    # Replace the text in the first run
                    before_text = run.text[:run_match_start]
                    after_text = run.text[run_match_end:]
                    run.text = before_text + final_text + after_text
                    
                    # Apply original formatting
                    FontManager.apply_font_info(run, font_info)
                else:
                    # Clear subsequent runs that were part of the match
                    run_match_start = max(0, start_pos - run_start)
                    run_match_end = min(len(run.text), end_pos - run_start)
                    
                    before_text = run.text[:run_match_start]
                    after_text = run.text[run_match_end:]
                    run.text = before_text + after_text
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to replace text: {e}")
            return False

def create_text_replacer(mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING) -> TextReplacer:
    """
    Factory function to create a TextReplacer instance.
    
    Args:
        mode: Replacement mode ('append' or 'replace')
        separator: Separator between original and appended text in append mode
        default_mapping: Default text to append when no mapping is found
        
    Returns:
        TextReplacer instance
    """
    return TextReplacer(mode, separator, default_mapping)
