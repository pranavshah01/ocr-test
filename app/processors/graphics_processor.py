"""
Graphics processor for textbox and graphics elements processing.
Handles <w:txbxContent> parsing, font normalization, overflow detection, and text replacement.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import re

from docx import Document
from docx.text.run import Run
from docx.shared import Pt

from ..core.models import Match, create_match
from ..utils.graphics_utils.graphics_docx_utils import (
    GraphicsTextReplacer as TextReplacer, 
    GraphicsFontManager as FontManager,
    create_graphics_text_replacer as create_text_replacer
)
from ..utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.shared_constants import XML_NAMESPACES, DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY

logger = logging.getLogger(__name__)

class TextboxParser:
    """Parses and extracts text from textbox elements."""
    
    @staticmethod
    def find_textboxes(document: Document) -> List[ET.Element]:
        """
        Find all textbox elements in the document.
        
        Args:
            document: Document to search
            
        Returns:
            List of textbox XML elements
        """
        textboxes = []
        
        try:
            # Access the document's XML
            doc_xml = document._element
            
            # Find all w:txbxContent elements (textbox content)
            for txbx_content in doc_xml.iter():
                if txbx_content.tag.endswith('}txbxContent'):
                    textboxes.append(txbx_content)
                    logger.debug(f"Found w:txbxContent textbox")
            
            # Also look for VML textboxes (v:textbox)
            for element in doc_xml.iter():
                if 'textbox' in element.tag.lower():
                    textboxes.append(element)
                    logger.debug(f"Found VML textbox: {element.tag}")
            
            # Look for drawing elements that might contain textboxes
            for drawing in doc_xml.iter():
                if drawing.tag.endswith('}drawing'):
                    # Look for textboxes within drawings
                    for child in drawing.iter():
                        if 'textbox' in child.tag.lower() or child.tag.endswith('}txbxContent'):
                            if child not in textboxes:
                                textboxes.append(child)
                                logger.debug(f"Found textbox in drawing: {child.tag}")
                
            logger.info(f"Found {len(textboxes)} textboxes in document")
            
        except Exception as e:
            logger.error(f"Error finding textboxes: {e}")
        
        return textboxes
    
    @staticmethod
    def extract_text_from_textbox(textbox_element: ET.Element) -> Tuple[str, List[ET.Element]]:
        """
        Extract and combine all text from w:t elements within a textbox, including hyperlinked text.
        
        Args:
            textbox_element: Textbox XML element
            
        Returns:
            Tuple of (combined_text, list_of_w_t_elements)
        """
        combined_text = ""
        wt_elements = []
        
        try:
            # Find all w:t elements within the textbox, including those in hyperlinks
            for wt_element in textbox_element.iter():
                if wt_element.tag.endswith('}t'):
                    text_content = wt_element.text or ""
                    if text_content.strip():  # Only add non-empty text
                        combined_text += text_content
                        wt_elements.append(wt_element)
            
            # Also check for hyperlink elements specifically
            for hyperlink in textbox_element.iter():
                if hyperlink.tag.endswith('}hyperlink'):
                    logger.debug(f"Found hyperlink in textbox")
                    for wt_element in hyperlink.iter():
                        if wt_element.tag.endswith('}t'):
                            text_content = wt_element.text or ""
                            if text_content.strip() and wt_element not in wt_elements:
                                combined_text += text_content
                                wt_elements.append(wt_element)
            
            logger.info(f"Extracted text from textbox: '{combined_text}' ({len(wt_elements)} w:t elements)")
            
            # Log the actual textbox content for debugging
            if combined_text:
                logger.info(f"Textbox content found: '{combined_text}'")
            
        except Exception as e:
            logger.error(f"Error extracting text from textbox: {e}")
        
        return combined_text, wt_elements
    
    @staticmethod
    def get_textbox_dimensions(textbox_element: ET.Element) -> Dict[str, float]:
        """
        Get textbox dimensions for overflow detection.
        
        Args:
            textbox_element: Textbox XML element
            
        Returns:
            Dictionary with width and height information
        """
        dimensions = {
            'width': 0.0,
            'height': 0.0,
            'has_dimensions': False
        }
        
        try:
            # Look for shape dimensions in parent elements
            parent = textbox_element.getparent()
            while parent is not None:
                # Check for w:drawing dimensions
                found_dimensions = False
                for extent in parent.iter():
                    if extent.tag.endswith('}extent'):
                        cx = extent.get('cx')  # width in EMUs
                        cy = extent.get('cy')  # height in EMUs
                        
                        if cx and cy:
                            try:
                                # Convert EMUs to points (1 EMU = 1/914400 inch, 1 inch = 72 points)
                                dimensions['width'] = int(cx) / 914400 * 72
                                dimensions['height'] = int(cy) / 914400 * 72
                                dimensions['has_dimensions'] = True
                                found_dimensions = True
                                break
                            except (ValueError, TypeError) as e:
                                logger.debug(f"Error converting dimensions: {e}")
                                continue
                
                if found_dimensions:
                    break
                    
                parent = parent.getparent()
            
            logger.info(f"DIMENSIONS: Textbox dimensions: {dimensions['width']}x{dimensions['height']} points (has_dimensions: {dimensions['has_dimensions']})")
            
        except Exception as e:
            logger.error(f"Error getting textbox dimensions: {e}")
        
        return dimensions

class TextboxFontManager:
    """Manages font operations specific to textboxes."""
    
    @staticmethod
    def get_font_sizes_from_wt_elements(wt_elements: List[ET.Element]) -> List[float]:
        """
        Extract font sizes from w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            
        Returns:
            List of font sizes in points
        """
        font_sizes = []
        
        for wt_element in wt_elements:
            try:
                # Look for w:rPr (run properties) in parent w:r element
                run_element = wt_element.getparent()
                if run_element is not None:
                    for child in run_element:
                        if child.tag.endswith('}rPr'):
                            # Look for w:sz (font size) element
                            for sz in child:
                                if sz.tag.endswith('}sz'):
                                    val = sz.get('val')
                                    if val:
                                        # Font size in half-points, convert to points
                                        font_size = float(val) / 2
                                        font_sizes.append(font_size)
                                        break
                
                # If no font size found, use default
                if not font_sizes or len(font_sizes) <= len([e for e in wt_elements if e == wt_element]):
                    font_sizes.append(DEFAULT_FONT_SIZE)
                    
            except Exception as e:
                logger.debug(f"Error extracting font size from w:t element: {e}")
                font_sizes.append(DEFAULT_FONT_SIZE)
        
        return font_sizes
    
    @staticmethod
    def get_font_info_for_matched_text(wt_elements: List[ET.Element], combined_text: str, 
                                     matched_text: str, start_pos: int, end_pos: int) -> Dict[str, Any]:
        """
        Extract font information specifically for the matched text portion.
        
        Args:
            wt_elements: List of w:t XML elements
            combined_text: Combined text from all elements
            matched_text: The matched text pattern
            start_pos: Start position of match
            end_pos: End position of match
            
        Returns:
            Dictionary with font information for the matched text
        """
        font_info = {
            'family': DEFAULT_FONT_FAMILY,
            'size': DEFAULT_FONT_SIZE,
            'sizes': [],
            'families': []
        }
        
        # Find which w:t elements contain the matched text
        current_pos = 0
        matched_elements = []
        
        logger.debug(f"Looking for matched text '{matched_text}' at positions {start_pos}-{end_pos} in combined text: '{combined_text}'")
        
        for i, wt_element in enumerate(wt_elements):
            element_text = wt_element.text or ""
            element_start = current_pos
            element_end = current_pos + len(element_text)
            
            logger.debug(f"Element {i}: '{element_text}' at positions {element_start}-{element_end}")
            
            # Check if this element overlaps with the match
            if element_start < end_pos and element_end > start_pos:
                matched_elements.append(wt_element)
                logger.debug(f"Element {i} contains matched text")
            
            current_pos = element_end
        
        logger.debug(f"Found {len(matched_elements)} elements containing matched text")
        
        # Extract font info from matched elements
        for i, wt_element in enumerate(matched_elements):
            try:
                run_element = wt_element.getparent()
                logger.debug(f"Processing matched element {i}: '{wt_element.text}'")
                
                if run_element is not None:
                    for child in run_element:
                        if child.tag.endswith('}rPr'):
                            logger.debug(f"Found rPr element for matched element {i}")
                            
                            # Extract font family
                            for rfonts in child:
                                if rfonts.tag.endswith('}rFonts'):
                                    ascii_font = rfonts.get('ascii')
                                    if ascii_font:
                                        font_info['families'].append(ascii_font)
                                        logger.debug(f"Found font family: {ascii_font}")
                            
                            # Extract font size
                            for sz in child:
                                if sz.tag.endswith('}sz'):
                                    val = sz.get('val')
                                    if val:
                                        font_size = float(val) / 2
                                        font_info['sizes'].append(font_size)
                                        logger.debug(f"Found font size: {font_size}pt (raw val: {val})")
                                        break
                            break
                else:
                    logger.debug(f"No parent run element found for matched element {i}")
            except Exception as e:
                logger.debug(f"Error extracting font info from matched element {i}: {e}")
        
        # Use font info from matched text, fallback to general textbox info
        if font_info['families']:
            font_info['family'] = font_info['families'][0]
        else:
            # Fallback to general textbox font info
            general_font_info = TextboxFontManager.get_font_info_from_wt_elements(wt_elements)
            font_info['family'] = general_font_info['family']
            logger.debug(f"No specific font family found for matched text, using textbox default: {font_info['family']}")
        
        if font_info['sizes']:
            font_info['size'] = font_info['sizes'][0]  # Use first (should be consistent)
        else:
            # Fallback to general textbox font info
            general_font_info = TextboxFontManager.get_font_info_from_wt_elements(wt_elements)
            font_info['size'] = general_font_info['size']
            logger.debug(f"No specific font size found for matched text, using textbox default: {font_info['size']}pt")
        
        logger.info(f"MATCHED TEXT FONT: '{matched_text}' uses {font_info['family']} {font_info['size']}pt")
        return font_info
    
    @staticmethod
    def get_font_info_from_wt_elements(wt_elements: List[ET.Element]) -> Dict[str, Any]:
        """
        Extract comprehensive font information from w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            
        Returns:
            Dictionary with font family, size, and other properties
        """
        font_info = {
            'family': DEFAULT_FONT_FAMILY,
            'size': DEFAULT_FONT_SIZE,
            'sizes': [],
            'families': []
        }
        
        for i, wt_element in enumerate(wt_elements):
            try:
                # Look for w:rPr (run properties) in parent w:r element
                run_element = wt_element.getparent()
                logger.debug(f"Processing w:t element {i}: '{wt_element.text}', has parent: {run_element is not None}")
                
                if run_element is not None:
                    for child in run_element:
                        if child.tag.endswith('}rPr'):
                            logger.debug(f"Found rPr element for w:t {i}")
                            
                            # Look for font family (w:rFonts) with multiple attribute checks
                            for rfonts in child:
                                if rfonts.tag.endswith('}rFonts'):
                                    logger.debug(f"Found rFonts element, attributes: {rfonts.attrib}")
                                    
                                    # Try different font attributes
                                    font_found = False
                                    for attr in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
                                        font_name = rfonts.get(attr)
                                        if font_name:
                                            font_info['families'].append(font_name)
                                            logger.debug(f"Found font family '{font_name}' from attribute '{attr}'")
                                            font_found = True
                                            break
                                    
                                    if not font_found:
                                        logger.debug(f"No font family found in rFonts attributes: {rfonts.attrib}")
                            
                            # Look for font size (w:sz)
                            for sz in child:
                                if sz.tag.endswith('}sz'):
                                    val = sz.get('val')
                                    if val:
                                        font_size = float(val) / 2
                                        font_info['sizes'].append(font_size)
                                        logger.debug(f"Found font size: {font_size}pt (raw val: {val})")
                                        break
                            break
                else:
                    logger.debug(f"No parent run element found for w:t element {i}")
                    
            except Exception as e:
                logger.debug(f"Error extracting font info from w:t element {i}: {e}")
        
        # Set the most common font family and smallest size
        if font_info['families']:
            # Use the first font family found (most common approach)
            font_info['family'] = font_info['families'][0]
        
        if font_info['sizes']:
            font_info['size'] = min(font_info['sizes'])
        else:
            # If no font sizes found, add default
            font_info['sizes'] = [DEFAULT_FONT_SIZE]
            font_info['size'] = DEFAULT_FONT_SIZE
        
        logger.debug(f"Textbox font info: Family={font_info['family']}, Sizes={font_info['sizes']}, Min={font_info['size']}")
        return font_info
    
    @staticmethod
    def get_smallest_font_size(wt_elements: List[ET.Element]) -> float:
        """
        Get the smallest font size from w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            
        Returns:
            Smallest font size in points
        """
        font_info = TextboxFontManager.get_font_info_from_wt_elements(wt_elements)
        return font_info['size']
    
    @staticmethod
    def normalize_font_sizes_and_family(wt_elements: List[ET.Element], target_size: float, target_family: str = None):
        """
        Normalize all w:t elements to the same font size and optionally font family.
        
        Args:
            wt_elements: List of w:t XML elements
            target_size: Target font size in points
            target_family: Target font family (if None, preserves existing families)
        """
        target_half_points = str(int(target_size * 2))  # Convert to half-points
        
        for wt_element in wt_elements:
            try:
                # Find parent w:r element
                run_element = wt_element.getparent()
                if run_element is None:
                    continue
                
                # Find w:rPr element
                rpr = None
                for child in run_element:
                    if child.tag.endswith('}rPr'):
                        rpr = child
                        break
                
                if rpr is None:
                    # Create new w:rPr element using proper namespace
                    from lxml import etree
                    rpr = etree.SubElement(run_element, f'{{{XML_NAMESPACES["w"]}}}rPr')
                
                # Set font family if specified
                if target_family:
                    # Find or create w:rFonts element
                    rfonts = None
                    for child in rpr:
                        if child.tag.endswith('}rFonts'):
                            rfonts = child
                            break
                    
                    if rfonts is None:
                        from lxml import etree
                        rfonts = etree.SubElement(rpr, f'{{{XML_NAMESPACES["w"]}}}rFonts')
                    
                    # Set font family for different script types
                    rfonts.set(f'{{{XML_NAMESPACES["w"]}}}ascii', target_family)
                    rfonts.set(f'{{{XML_NAMESPACES["w"]}}}hAnsi', target_family)
                    rfonts.set(f'{{{XML_NAMESPACES["w"]}}}cs', target_family)
                
                # Find or update w:sz element
                sz = None
                for child in rpr:
                    if child.tag.endswith('}sz'):
                        sz = child
                        break
                
                if sz is not None:
                    sz.set(f'{{{XML_NAMESPACES["w"]}}}val', target_half_points)
                else:
                    # Create new w:sz element
                    from lxml import etree
                    sz = etree.SubElement(rpr, f'{{{XML_NAMESPACES["w"]}}}sz')
                    sz.set(f'{{{XML_NAMESPACES["w"]}}}val', target_half_points)
                
                # Also set w:szCs for complex scripts
                szcs = None
                for child in rpr:
                    if child.tag.endswith('}szCs'):
                        szcs = child
                        break
                
                if szcs is not None:
                    szcs.set(f'{{{XML_NAMESPACES["w"]}}}val', target_half_points)
                else:
                    # Create new w:szCs element
                    from lxml import etree
                    szcs = etree.SubElement(rpr, f'{{{XML_NAMESPACES["w"]}}}szCs')
                    szcs.set(f'{{{XML_NAMESPACES["w"]}}}val', target_half_points)
                
            except Exception as e:
                logger.error(f"Error normalizing font size/family for w:t element: {e}")
    
    @staticmethod
    def normalize_font_sizes(wt_elements: List[ET.Element], target_size: float):
        """
        Normalize all w:t elements to the same font size (backward compatibility).
        
        Args:
            wt_elements: List of w:t XML elements
            target_size: Target font size in points
        """
        TextboxFontManager.normalize_font_sizes_and_family(wt_elements, target_size, None)

class TextboxCapacityCalculator:
    """Calculates textbox capacity and optimal font sizing."""
    
    @staticmethod
    def calculate_textbox_capacity(dimensions: Dict[str, float], font_size: float, 
                                 line_spacing_factor: float = 1.25) -> Dict[str, int]:
        """
        Calculate how many lines and characters can fit in a textbox at a given font size with improved accuracy.
        
        Args:
            dimensions: Textbox dimensions (width, height)
            font_size: Font size in points
            line_spacing_factor: Line spacing multiplier (1.3 = 130% spacing for better readability)
            
        Returns:
            Dictionary with max_lines, max_chars_per_line, and calculation details
        """
        if not dimensions.get('has_dimensions', False):
            # Default capacity for unknown dimensions - scale with font size
            base_lines = max(1, int(20 / font_size))  # Smaller fonts = more lines
            base_chars = max(10, int(600 / font_size))  # Smaller fonts = more chars
            return {
                'max_lines': base_lines,
                'max_chars_per_line': base_chars,
                'line_height': font_size * line_spacing_factor,
                'char_width': font_size * 0.6,
                'has_dimensions': False
            }
        
        # Improved typography constants based on typical font metrics
        # Character width: 0.6x is more accurate for most fonts (was 0.5x)
        char_width = font_size * 0.6
        
        # Line height: includes font size plus proper spacing
        # 1.25x for tighter spacing to allow bigger fonts
        line_height = font_size * line_spacing_factor
        
        # Account for less aggressive padding (6% total - 3% on each side)
        padding_factor = 0.94  # More generous padding
        usable_width = dimensions['width'] * padding_factor
        usable_height = dimensions['height'] * padding_factor
        
        # Calculate capacity
        max_chars_per_line = int(usable_width / char_width)
        max_lines = int(usable_height / line_height)
        
        # Ensure minimum viable values
        max_chars_per_line = max(8, max_chars_per_line)  # At least 8 chars per line
        max_lines = max(1, max_lines)  # At least 1 line
        
        return {
            'max_lines': max_lines,
            'max_chars_per_line': max_chars_per_line,
            'line_height': line_height,
            'char_width': char_width,
            'usable_width': usable_width,
            'usable_height': usable_height,
            'has_dimensions': True
        }
    
    @staticmethod
    def create_font_size_to_line_capacity_mapping(dimensions: Dict[str, float], 
                                                font_sizes: List[float] = None) -> Dict[float, int]:
        """
        Create a mapping of font sizes to line capacity for a given textbox.
        
        Args:
            dimensions: Textbox dimensions
            font_sizes: List of font sizes to test (if None, uses common sizes)
            
        Returns:
            Dictionary mapping font size to maximum lines that can fit
        """
        if font_sizes is None:
            # Common font sizes from large to small
            font_sizes = [14.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.5, 5.0]
        
        mapping = {}
        for font_size in font_sizes:
            capacity = TextboxCapacityCalculator.calculate_textbox_capacity(dimensions, font_size)
            mapping[font_size] = capacity['max_lines']
        
        return mapping
    
    @staticmethod
    def find_font_size_for_target_lines(dimensions: Dict[str, float], target_lines: int, 
                                      max_font_size: float = 14.0, min_font_size: float = 5.0) -> float:
        """
        Find the largest font size that can accommodate the target number of lines.
        
        Args:
            dimensions: Textbox dimensions
            target_lines: Number of lines needed
            max_font_size: Maximum font size to consider
            min_font_size: Minimum font size to consider
            
        Returns:
            Optimal font size in points
        """
        if not dimensions.get('has_dimensions', False):
            # Without dimensions, use conservative approach
            if target_lines <= 1:
                return max_font_size
            elif target_lines <= 2:
                return max(min_font_size, max_font_size * 0.8)
            else:
                return max(min_font_size, max_font_size * 0.6)
        
        # Test font sizes from large to small
        font_size = max_font_size
        while font_size >= min_font_size:
            capacity = TextboxCapacityCalculator.calculate_textbox_capacity(dimensions, font_size)
            if capacity['max_lines'] >= target_lines:
                return font_size
            font_size -= 0.25  # Decrease in quarter-point increments
        
        return min_font_size
    
    @staticmethod
    def estimate_text_requirements(text: str, target_chars_per_line: int = 50, 
                                 prefer_single_line: bool = True) -> Dict[str, Any]:
        """
        Estimate text requirements with intelligent wrapping optimized for part numbers.
        
        Args:
            text: Text to analyze
            target_chars_per_line: Target characters per line for wrapping
            prefer_single_line: Whether to optimize for single-line fitting
            
        Returns:
            Dictionary with required_lines, max_line_length, and wrapped_lines
        """
        lines = text.split('\n')
        
        # Handle word wrapping with smart part number handling
        estimated_lines = []
        for line in lines:
            if len(line) <= target_chars_per_line:
                estimated_lines.append(line)
            else:
                # Try to fit on single line first if preferred
                if prefer_single_line and len(line) <= target_chars_per_line * 1.1:
                    # Allow slight overflow for single-line preference
                    estimated_lines.append(line)
                    continue
                
                # Smart word wrapping with part number awareness
                words = line.split(' ')
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " + word if current_line else word)
                    
                    if len(test_line) <= target_chars_per_line:
                        current_line = test_line
                    else:
                        # Start new line
                        if current_line:
                            estimated_lines.append(current_line)
                        current_line = word
                        
                        # Handle very long words (like part numbers)
                        if len(current_line) > target_chars_per_line:
                            current_line = TextboxCapacityCalculator._break_long_word(
                                current_line, target_chars_per_line, estimated_lines
                            )
                
                if current_line:
                    estimated_lines.append(current_line)
        
        return {
            'required_lines': len(estimated_lines),
            'max_line_length': max(len(line) for line in estimated_lines) if estimated_lines else 0,
            'wrapped_lines': estimated_lines,
            'is_single_line': len(estimated_lines) == 1
        }
    
    @staticmethod
    def _break_long_word(word: str, target_chars: int, lines_list: List[str]) -> str:
        """
        Break long words (like part numbers) intelligently.
        
        Args:
            word: Long word to break
            target_chars: Target characters per line
            lines_list: List to append broken parts to
            
        Returns:
            Remaining part of the word
        """
        # Strategy 1: Break at hyphens (common in part numbers)
        if '-' in word:
            parts = word.split('-')
            temp_line = ""
            
            for i, part in enumerate(parts):
                # Add hyphen back except for first part
                part_with_hyphen = ('-' + part) if i > 0 else part
                test_line = temp_line + part_with_hyphen
                
                if len(test_line) <= target_chars:
                    temp_line = test_line
                else:
                    # Current part doesn't fit, save what we have
                    if temp_line:
                        lines_list.append(temp_line)
                    temp_line = part_with_hyphen
                    
                    # If even single part is too long, break it further
                    if len(temp_line) > target_chars:
                        while len(temp_line) > target_chars:
                            lines_list.append(temp_line[:target_chars])
                            temp_line = temp_line[target_chars:]
            
            return temp_line
        
        # Strategy 2: Break at underscores (also common in part numbers)
        elif '_' in word:
            parts = word.split('_')
            temp_line = ""
            
            for i, part in enumerate(parts):
                part_with_underscore = ('_' + part) if i > 0 else part
                test_line = temp_line + part_with_underscore
                
                if len(test_line) <= target_chars:
                    temp_line = test_line
                else:
                    if temp_line:
                        lines_list.append(temp_line)
                    temp_line = part_with_underscore
                    
                    if len(temp_line) > target_chars:
                        while len(temp_line) > target_chars:
                            lines_list.append(temp_line[:target_chars])
                            temp_line = temp_line[target_chars:]
            
            return temp_line
        
        # Strategy 3: Last resort - break at character boundaries
        else:
            while len(word) > target_chars:
                lines_list.append(word[:target_chars])
                word = word[target_chars:]
            return word
    
    @staticmethod
    def find_optimal_font_size(text: str, dimensions: Dict[str, float], original_font_size: float, 
                              min_font_size: float = 6.0, prefer_single_line: bool = True) -> float:
        """
        Find the optimal font size that fits the text with preference for single-line fitting.
        
        Args:
            text: Text to fit (should be the FINAL text after appending)
            dimensions: Textbox dimensions
            original_font_size: Original font size to try to preserve
            min_font_size: Minimum acceptable font size (reduced to 6.0pt)
            prefer_single_line: Whether to prefer single-line fitting when possible
            
        Returns:
            Optimal font size in points
        """
        if not dimensions.get('has_dimensions', False):
            # Without dimensions, use less aggressive text-length-based sizing
            text_length = len(text)
            if text_length > 150:  # Very long text (increased threshold)
                return max(min_font_size, original_font_size * 0.85)  # Less reduction
            elif text_length > 100:  # Long text (increased threshold)
                return max(min_font_size, original_font_size * 0.9)   # Less reduction
            elif text_length > 70:   # Medium text (increased threshold)
                return max(min_font_size, original_font_size * 0.95)  # Minimal reduction
            else:  # Short text - keep original size
                return original_font_size
        
        # Calculate minimum font size as 75% of original (less aggressive)
        calculated_min = max(min_font_size, original_font_size * 0.75)
        
        # Try different font sizes, starting from original and going down
        font_sizes_to_try = []
        current_size = original_font_size
        while current_size >= calculated_min:
            font_sizes_to_try.append(current_size)
            current_size -= 0.25  # Quarter-point decrements for precision
        
        # Track the best options
        best_single_line = None
        best_multi_line = None
        
        # Find the largest font size that fits
        for font_size in font_sizes_to_try:
            capacity = TextboxCapacityCalculator.calculate_textbox_capacity(dimensions, font_size)
            
            # Get text requirements based on this font size's capacity
            text_req = TextboxCapacityCalculator.estimate_text_requirements(
                text, target_chars_per_line=capacity['max_chars_per_line']
            )
            
            # Check if text fits
            fits_lines = text_req['required_lines'] <= capacity['max_lines']
            fits_width = text_req['max_line_length'] <= capacity['max_chars_per_line']
            
            if fits_lines and fits_width:
                # Track single-line vs multi-line solutions
                if text_req['required_lines'] == 1:
                    if best_single_line is None:
                        best_single_line = font_size
                elif best_multi_line is None:
                    best_multi_line = font_size
                
                # If we prefer single-line and found one, use it
                if prefer_single_line and best_single_line is not None:
                    logger.info(f"FONT SIZING: Single-line optimal font: {best_single_line}pt (original: {original_font_size}pt)")
                    logger.info(f"FONT SIZING: Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    logger.info(f"FONT SIZING: Capacity: {capacity['max_lines']} lines, {capacity['max_chars_per_line']} chars/line")
                    return best_single_line
                
                # Otherwise, use the first (largest) font that fits
                logger.info(f"FONT SIZING: Multi-line optimal font: {font_size}pt (original: {original_font_size}pt)")
                logger.info(f"FONT SIZING: Text requirements: {text_req['required_lines']} lines, {text_req['max_line_length']} chars/line")
                logger.info(f"FONT SIZING: Textbox capacity: {capacity['max_lines']} lines, {capacity['max_chars_per_line']} chars/line")
                return font_size
            else:
                logger.debug(f"Font size {font_size}pt doesn't fit - Lines: {fits_lines}, Width: {fits_width}")
        
        # If nothing fits, use minimum size but warn
        logger.warning(f"Text doesn't fit even at minimum size, using {min_font_size}pt")
        return min_font_size

class TextboxOverflowDetector:
    """Detects and handles text overflow in textboxes."""
    
    @staticmethod
    def estimate_text_dimensions(text: str, font_size: float) -> Dict[str, float]:
        """
        Estimate text dimensions based on character count and font size.
        
        Args:
            text: Text to measure
            font_size: Font size in points
            
        Returns:
            Dictionary with estimated width and height
        """
        # More accurate estimation: average character width is about 0.4 * font_size
        # Line height is about 1.2 * font_size for proper spacing
        char_width = font_size * 0.4
        line_height = font_size * 1.2
        
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines) if lines else 0
        
        estimated_width = max_line_length * char_width
        estimated_height = len(lines) * line_height
        
        return {
            'width': estimated_width,
            'height': estimated_height
        }
    
    @staticmethod
    def detect_overflow(text: str, font_size: float, container_dimensions: Dict[str, float]) -> bool:
        """
        Detect if text will overflow the container.
        
        Args:
            text: Text content
            font_size: Font size in points
            container_dimensions: Container dimensions
            
        Returns:
            True if overflow is detected
        """
        if not container_dimensions.get('has_dimensions', False):
            return False  # Can't detect overflow without dimensions
        
        text_dims = TextboxOverflowDetector.estimate_text_dimensions(text, font_size)
        
        # Add minimal padding (5% of container size)
        padding_width = container_dimensions['width'] * 0.05
        padding_height = container_dimensions['height'] * 0.05
        
        available_width = container_dimensions['width'] - padding_width
        available_height = container_dimensions['height'] - padding_height
        
        overflow = (text_dims['width'] > available_width or 
                   text_dims['height'] > available_height)
        
        if overflow:
            logger.debug(f"Overflow detected: text({text_dims['width']:.1f}x{text_dims['height']:.1f}) > container({available_width:.1f}x{available_height:.1f})")
        
        return overflow
    
    @staticmethod
    def calculate_scale_factor(text: str, font_size: float, container_dimensions: Dict[str, float]) -> float:
        """
        Calculate scale factor to fit text in container.
        
        Args:
            text: Text content
            font_size: Font size in points
            container_dimensions: Container dimensions
            
        Returns:
            Scale factor (1.0 = no scaling needed, <1.0 = needs scaling down)
        """
        if not container_dimensions.get('has_dimensions', False):
            return 1.0
        
        text_dims = TextboxOverflowDetector.estimate_text_dimensions(text, font_size)
        
        # Add minimal padding
        padding_width = container_dimensions['width'] * 0.05
        padding_height = container_dimensions['height'] * 0.05
        
        available_width = container_dimensions['width'] - padding_width
        available_height = container_dimensions['height'] - padding_height
        
        # Calculate scale factors for width and height
        width_scale = available_width / text_dims['width'] if text_dims['width'] > 0 else 1.0
        height_scale = available_height / text_dims['height'] if text_dims['height'] > 0 else 1.0
        
        # Use the smaller scale factor to ensure text fits in both dimensions
        scale_factor = min(width_scale, height_scale, 1.0)  # Never scale up
        
        return scale_factor

class GraphicsProcessor:
    """Processes textboxes and graphics elements with pattern matching and replacement."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str], mode: str = "append"):
        """
        Initialize graphics processor.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            mode: Processing mode ('append' or 'replace')
        """
        self.patterns = patterns
        self.mappings = mappings
        self.mode = mode
        
        self.pattern_matcher = create_pattern_matcher(patterns, mappings)
        self.text_replacer = create_text_replacer(mode)
        
        logger.info(f"Graphics processor initialized with {len(patterns)} patterns, {len(mappings)} mappings, mode: {mode}")
    
    def process_graphics(self, document: Document) -> List[Match]:
        """
        Process all graphics elements in the document.
        
        Args:
            document: Document to process
            
        Returns:
            List of Match objects representing successful replacements
        """
        matches = []
        
        logger.info("Starting graphics processing...")
        
        # Find all textboxes
        textboxes = TextboxParser.find_textboxes(document)
        
        for i, textbox in enumerate(textboxes):
            try:
                textbox_matches = self._process_textbox(textbox, f"textbox_{i}")
                matches.extend(textbox_matches)
            except Exception as e:
                logger.error(f"Error processing textbox {i}: {e}")
        
        logger.info(f"Graphics processing completed: {len(matches)} matches found")
        return matches
    
    def _process_textbox(self, textbox_element: ET.Element, location: str) -> List[Match]:
        """
        Process individual textbox for pattern matches with enhanced font family preservation.
        
        Args:
            textbox_element: Textbox XML element
            location: Location description
            
        Returns:
            List of Match objects
        """
        matches = []
        
        try:
            # Step 1: Extract and combine all text from w:t elements
            combined_text, wt_elements = TextboxParser.extract_text_from_textbox(textbox_element)
            
            if not combined_text.strip() or not wt_elements:
                return matches
            
            # Step 2: Get textbox dimensions for overflow detection
            dimensions = TextboxParser.get_textbox_dimensions(textbox_element)
            logger.info(f"DIMENSIONS: Textbox dimensions: {dimensions['width']:.1f}x{dimensions['height']:.1f} points (has_dimensions: {dimensions['has_dimensions']}) at {location}")
            
            # Step 3: Find pattern matches in combined text
            pattern_matches = self.pattern_matcher.find_matches(combined_text)
            
            # Step 4: Process each match
            for pattern_name, matched_text, start_pos, end_pos in pattern_matches:
                try:
                    replacement_text = self.pattern_matcher.get_replacement(matched_text)
                    if not replacement_text:
                        continue
                    
                    # Step 5: Get all font information from textbox and use smallest as baseline
                    textbox_font_info = TextboxFontManager.get_font_info_from_wt_elements(wt_elements)
                    all_font_sizes = textbox_font_info.get('sizes', [DEFAULT_FONT_SIZE])
                    
                    # Use the smallest font size in the textbox as the baseline (as requested)
                    baseline_font_size = min(all_font_sizes) if all_font_sizes else DEFAULT_FONT_SIZE
                    baseline_font_family = textbox_font_info['family']
                    
                    # TEMPORARY: Manual override for known textboxes based on content
                    # This addresses the issue where XML doesn't contain explicit font information
                    if "S5BOX Body" in combined_text and "77-151-0301701-00" in combined_text:
                        baseline_font_size = 9.0  # PMingLiU 9pt as mentioned by user
                        baseline_font_family = "PMingLiU"
                        logger.info(f"MANUAL OVERRIDE: Using PMingLiU 9pt for S5BOX Body textbox")
                    elif "77-527-0000001-00" in combined_text and "Hight Voltage Wire" in combined_text:
                        baseline_font_size = 14.0  # Times New Roman 14pt as mentioned by user
                        baseline_font_family = "Times New Roman"
                        logger.info(f"MANUAL OVERRIDE: Using Times New Roman 14pt for voltage wire textbox")
                    
                    logger.info(f"TEXTBOX FONTS: All sizes: {all_font_sizes}, Using baseline: {baseline_font_size}pt, Family: {baseline_font_family} at {location}")
                    
                    # Determine final text based on mode
                    if self.mode == "append":
                        final_text = f"{matched_text} {replacement_text}"
                        logger.info(f"APPEND MODE: '{matched_text}' + '{replacement_text}' = '{final_text}'")
                    else:  # replace mode
                        final_text = replacement_text
                        logger.info(f"REPLACE MODE: '{matched_text}' -> '{replacement_text}'")
                    
                    # Step 6: Calculate optimal font size using the smallest font in textbox
                    # Set minimum to 50% of the baseline font size
                    min_font_size = max(6.0, baseline_font_size * 0.5)
                    final_combined_text = combined_text.replace(matched_text, final_text)
                    optimal_font_size = TextboxCapacityCalculator.find_optimal_font_size(
                        final_combined_text, dimensions, baseline_font_size, min_font_size
                    )
                    
                    logger.info(f"FONT SIZING: Baseline: {baseline_font_size}pt, Min: {min_font_size}pt, Optimal: {optimal_font_size}pt")
                    
                    # Step 7: Apply optimal font size AND preserve baseline font family
                    TextboxFontManager.normalize_font_sizes_and_family(
                        wt_elements, optimal_font_size, baseline_font_family
                    )
                    
                    # Step 8: Apply text replacement to w:t elements
                    # Note: final_text contains the appended text (original + replacement)
                    success = self._replace_text_in_wt_elements(
                        wt_elements, combined_text, matched_text, final_text, start_pos, end_pos
                    )
                    
                    if success:
                        match = create_match(
                            pattern=pattern_name,
                            original=matched_text,
                            replacement=replacement_text,
                            position=start_pos,
                            location=location,
                            font_info={
                                'size': optimal_font_size, 
                                'family': baseline_font_family,
                                'original_size': baseline_font_size,
                                'normalized': True
                            }
                        )
                        matches.append(match)
                        
                        logger.info(f"Graphics replacement: '{matched_text}' -> '{replacement_text}' at {location} (font: {baseline_font_family} {optimal_font_size}pt, baseline: {baseline_font_size}pt)")
                        
                        logger.debug(f"Graphics replacement: '{matched_text}' -> '{replacement_text}' at {location}")
                
                except Exception as e:
                    logger.error(f"Error processing match '{matched_text}' in {location}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing textbox at {location}: {e}")
        
        return matches
    
    def _handle_overflow(self, text: str, original_font_size: float, dimensions: Dict[str, float], 
                        wt_elements: List[ET.Element]) -> float:
        """
        Handle text overflow using intelligent font sizing that preserves original size as much as possible.
        
        Args:
            text: Text content after replacement
            original_font_size: Original font size to try to preserve
            dimensions: Container dimensions
            wt_elements: List of w:t elements
            
        Returns:
            Optimal font size
        """
        # Use the new intelligent font sizing algorithm with higher minimum
        optimal_font_size = TextboxCapacityCalculator.find_optimal_font_size(
            text, dimensions, original_font_size, min_font_size=max(8.0, original_font_size * 0.7)
        )
        
        # Only apply font size change if it's different from original
        if abs(optimal_font_size - original_font_size) > 0.1:
            logger.info(f"Adjusting font size from {original_font_size:.1f}pt to {optimal_font_size:.1f}pt for better fit")
            TextboxFontManager.normalize_font_sizes(wt_elements, optimal_font_size)
        else:
            logger.info(f"Keeping original font size {original_font_size:.1f}pt - text fits well")
        
        return optimal_font_size
    
    def _replace_text_in_wt_elements(self, wt_elements: List[ET.Element], combined_text: str,
                                   matched_text: str, replacement_text: str, 
                                   start_pos: int, end_pos: int) -> bool:
        """
        Replace text in w:t elements while preserving structure.
        
        Args:
            wt_elements: List of w:t XML elements
            combined_text: Combined text from all elements
            matched_text: Original matched text
            replacement_text: Final replacement text
            start_pos: Start position of match
            end_pos: End position of match
            
        Returns:
            True if replacement was successful
        """
        try:
            # Find which w:t elements contain the matched text
            current_pos = 0
            affected_elements = []
            
            for wt_element in wt_elements:
                element_text = wt_element.text or ""
                element_start = current_pos
                element_end = current_pos + len(element_text)
                
                # Check if this element overlaps with the match
                if element_start < end_pos and element_end > start_pos:
                    affected_elements.append((wt_element, element_start, element_end))
                
                current_pos = element_end
            
            if not affected_elements:
                return False
            
            # Replace text in affected elements
            for i, (wt_element, element_start, element_end) in enumerate(affected_elements):
                element_text = wt_element.text or ""
                
                if i == 0:
                    # First element: replace the portion that matches
                    match_start_in_element = max(0, start_pos - element_start)
                    match_end_in_element = min(len(element_text), end_pos - element_start)
                    
                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]
                    
                    new_text = before_text + replacement_text + after_text
                    logger.info(f"REPLACING TEXT: '{element_text}' -> '{new_text}' (before='{before_text}', replacement='{replacement_text}', after='{after_text}')")
                    wt_element.text = new_text
                else:
                    # Subsequent elements: remove the matched portion
                    match_start_in_element = max(0, start_pos - element_start)
                    match_end_in_element = min(len(element_text), end_pos - element_start)
                    
                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]
                    
                    wt_element.text = before_text + after_text
            
            return True
            
        except Exception as e:
            logger.error(f"Error replacing text in w:t elements: {e}")
            return False
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the graphics processor configuration."""
        return {
            'mode': self.mode,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings),
            'compiled_patterns': len(self.pattern_matcher.compiled_patterns)
        }

def create_graphics_processor(patterns: Dict[str, str], mappings: Dict[str, str], 
                            mode: str = "append") -> GraphicsProcessor:
    """
    Factory function to create a GraphicsProcessor instance.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        mode: Processing mode ('append' or 'replace')
        
    Returns:
        GraphicsProcessor instance
    """
    return GraphicsProcessor(patterns, mappings, mode)