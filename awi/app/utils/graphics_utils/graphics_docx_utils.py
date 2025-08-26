"""
Graphics utilities for DOCX document processing.
Handles textbox content parsing, font management, and graphics text replacement.
"""

import xml.etree.ElementTree as ET
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_COLOR_INDEX

from ..shared_constants import XML_NAMESPACES, DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY, DEFAULT_MAPPING, DEFAULT_SEPARATOR, PROCESSING_MODES

logger = logging.getLogger(__name__)

class GraphicsTextReconstructor:
    """Reconstructs text across multiple <w:t> elements for graphics pattern matching."""
    
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

class GraphicsFontManager:
    """Manages font information and formatting preservation for graphics elements."""
    
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
            'font_family': DEFAULT_FONT_FAMILY,
            'font_size': DEFAULT_FONT_SIZE,
            'color': '000000',
            'is_bold': False,
            'is_italic': False,
            'is_underline': False
        }
        
        try:
            if hasattr(run, 'font') and run.font:
                # Get font family
                if hasattr(run.font, 'name') and run.font.name:
                    font_name = str(run.font.name)
                    if ':' in font_name:
                        font_info['font_family'] = font_name.split(':')[0]
                    else:
                        font_info['font_family'] = font_name
                elif hasattr(run.font, 'ascii') and run.font.ascii:
                    font_info['font_family'] = str(run.font.ascii)
                
                # Get font size
                if hasattr(run.font, 'size') and run.font.size:
                    if hasattr(run.font.size, 'pt'):
                        font_info['font_size'] = float(run.font.size.pt)
                    else:
                        # Convert EMUs to points (1 point = 12700 EMUs)
                        emu_size = float(run.font.size)
                        font_info['font_size'] = emu_size / 12700.0
                elif hasattr(run.font, 'sz') and run.font.sz:
                    # Convert half-points to points
                    font_info['font_size'] = float(run.font.sz) / 2.0
                
                # Get font color
                if hasattr(run.font, 'color') and run.font.color:
                    if hasattr(run.font.color, 'rgb') and run.font.color.rgb:
                        font_info['color'] = str(run.font.color.rgb)
                
                # Get font styles
                font_info['is_bold'] = bool(getattr(run.font, 'bold', False))
                font_info['is_italic'] = bool(getattr(run.font, 'italic', False))
                font_info['is_underline'] = bool(getattr(run.font, 'underline', False))
        
        except Exception as e:
            logger.debug(f"Error extracting font info: {e}")
        
        return font_info
    
    @staticmethod
    def get_font_info_from_wt_elements(wt_elements: List[ET.Element]) -> Dict[str, Any]:
        """
        Extract comprehensive font information from w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            
        Returns:
            Dictionary with font information including all sizes found
        """
        font_info = {
            'family': DEFAULT_FONT_FAMILY,
            'sizes': [],
            'colors': [],
            'styles': []
        }
        
        try:
            for wt_element in wt_elements:
                # Get parent rPr (run properties) element
                parent = wt_element.getparent()
                while parent is not None and not parent.tag.endswith('}r'):
                    parent = parent.getparent()
                
                if parent is not None:
                    r_pr = parent.find('.//w:rPr', namespaces=XML_NAMESPACES)
                    if r_pr is not None:
                        # Extract font family
                        r_fonts = r_pr.find('.//w:rFonts', namespaces=XML_NAMESPACES)
                        if r_fonts is not None:
                            ascii_font = r_fonts.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii')
                            h_ansi_font = r_fonts.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi')
                            if ascii_font:
                                font_info['family'] = ascii_font
                            elif h_ansi_font:
                                font_info['family'] = h_ansi_font
                        
                        # Extract font size
                        sz_element = r_pr.find('.//w:sz', namespaces=XML_NAMESPACES)
                        if sz_element is not None:
                            sz_val = sz_element.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                            if sz_val:
                                font_size = float(sz_val) / 2.0  # Convert half-points to points
                                if font_size not in font_info['sizes']:
                                    font_info['sizes'].append(font_size)
                        
                        # Extract font color
                        color_element = r_pr.find('.//w:color', namespaces=XML_NAMESPACES)
                        if color_element is not None:
                            color_val = color_element.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                            if color_val and color_val not in font_info['colors']:
                                font_info['colors'].append(color_val)
                        
                        # Extract font styles
                        if r_pr.find('.//w:b', namespaces=XML_NAMESPACES) is not None:
                            if 'bold' not in font_info['styles']:
                                font_info['styles'].append('bold')
                        if r_pr.find('.//w:i', namespaces=XML_NAMESPACES) is not None:
                            if 'italic' not in font_info['styles']:
                                font_info['styles'].append('italic')
                        if r_pr.find('.//w:u', namespaces=XML_NAMESPACES) is not None:
                            if 'underline' not in font_info['styles']:
                                font_info['styles'].append('underline')
        
        except Exception as e:
            logger.debug(f"Error extracting font info from w:t elements: {e}")
        
        # Ensure we have at least one font size
        if not font_info['sizes']:
            font_info['sizes'] = [DEFAULT_FONT_SIZE]
        
        return font_info
    
    @staticmethod
    def normalize_font_sizes(wt_elements: List[ET.Element], target_size: float):
        """
        Normalize font sizes across all w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            target_size: Target font size in points
        """
        try:
            for wt_element in wt_elements:
                # Get parent rPr (run properties) element
                parent = wt_element.getparent()
                while parent is not None and not parent.tag.endswith('}r'):
                    parent = parent.getparent()
                
                if parent is not None:
                    # Ensure paragraph-level size exists (w:pPr/w:rPr/w:szCs)
                    GraphicsFontManager._ensure_paragraph_level_size(parent, target_size)
                    
                    r_pr = parent.find('.//w:rPr', namespaces=XML_NAMESPACES)
                    if r_pr is None:
                        # BUG FIX: Create w:rPr element if it doesn't exist
                        # This ensures font size normalization applies to ALL text runs, not just those with existing formatting
                        from lxml import etree
                        r_pr = etree.SubElement(parent, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
                        logger.debug(f"Created missing w:rPr element for text run to enable font size normalization")
                    # Ensure rPr precedes w:t inside the run
                    GraphicsFontManager._ensure_rpr_before_text(parent, r_pr)
                    
                    # Update or create sz element
                    sz_element = r_pr.find('.//w:sz', namespaces=XML_NAMESPACES)
                    if sz_element is not None:
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:
                        # Create new sz element
                        from lxml import etree
                        sz_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz')
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    
                    # Also update szCs element if it exists
                    sz_cs_element = r_pr.find('.//w:szCs', namespaces=XML_NAMESPACES)
                    if sz_cs_element is not None:
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:
                        # Create new szCs element
                        from lxml import etree
                        sz_cs_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
        
        except Exception as e:
            logger.error(f"Error normalizing font sizes: {e}")
    
    @staticmethod
    def normalize_font_sizes_and_family(wt_elements: List[ET.Element], target_size: float, target_family: str):
        """
        Normalize font sizes and family across all w:t elements.
        
        Args:
            wt_elements: List of w:t XML elements
            target_size: Target font size in points
            target_family: Target font family
        """
        try:
            for wt_element in wt_elements:
                # Get parent rPr (run properties) element
                parent = wt_element.getparent()
                while parent is not None and not parent.tag.endswith('}r'):
                    parent = parent.getparent()
                
                if parent is not None:
                    # Ensure paragraph-level size exists (w:pPr/w:rPr/w:szCs)
                    GraphicsFontManager._ensure_paragraph_level_size(parent, target_size)
                    
                    r_pr = parent.find('.//w:rPr', namespaces=XML_NAMESPACES)
                    if r_pr is None:
                        # BUG FIX: Create w:rPr element if it doesn't exist
                        # This ensures font normalization applies to ALL text runs, not just those with existing formatting
                        from lxml import etree
                        r_pr = etree.SubElement(parent, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
                        logger.debug(f"Created missing w:rPr element for text run to enable font family and size normalization")
                    # Ensure rPr precedes w:t inside the run
                    GraphicsFontManager._ensure_rpr_before_text(parent, r_pr)
                    
                    # Update font family
                    r_fonts = r_pr.find('.//w:rFonts', namespaces=XML_NAMESPACES)
                    if r_fonts is not None:
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', target_family)
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', target_family)
                    else:
                        # Create new rFonts element using the proper namespace
                        from lxml import etree
                        r_fonts = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', target_family)
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', target_family)
                    
                    # Update font size - do it directly here since we already have r_pr
                    sz_element = r_pr.find('.//w:sz', namespaces=XML_NAMESPACES)
                    if sz_element is not None:
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:
                        # Create new sz element using the proper namespace
                        from lxml import etree
                        sz_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz')
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    
                    # Also update szCs element if it exists
                    sz_cs_element = r_pr.find('.//w:szCs', namespaces=XML_NAMESPACES)
                    if sz_cs_element is not None:
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:
                        # Create new szCs element using the proper namespace  
                        from lxml import etree
                        sz_cs_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
        
        except Exception as e:
            logger.error(f"Error normalizing font sizes and family: {e}")

    @staticmethod
    def _ensure_paragraph_level_size(run_element: ET.Element, target_size: float) -> None:
        """Ensure ancestor w:p has w:pPr/w:rPr with w:szCs=target size (half-points)."""
        try:
            # Find ancestor paragraph element
            p = run_element.getparent()
            while p is not None and not p.tag.endswith('}p'):
                p = p.getparent()
            if p is None:
                return
            # Ensure w:pPr exists
            p_pr = p.find('.//w:pPr', namespaces=XML_NAMESPACES)
            if p_pr is None:
                from lxml import etree
                p_pr = etree.SubElement(p, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
            # Ensure w:rPr under w:pPr exists
            from lxml import etree
            p_rpr = p_pr.find('.//w:rPr', namespaces=XML_NAMESPACES)
            if p_rpr is None:
                p_rpr = etree.SubElement(p_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
            # Ensure w:szCs exists and set value
            szcs = p_rpr.find('.//w:szCs', namespaces=XML_NAMESPACES)
            if szcs is None:
                szcs = etree.SubElement(p_rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
            szcs.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
        except Exception as e:
            logger.debug(f"Failed to ensure paragraph-level size: {e}")

    @staticmethod
    def _ensure_rpr_before_text(run_element: ET.Element, r_pr: ET.Element) -> None:
        """Ensure w:rPr comes before the first w:t within the w:r element."""
        try:
            # Find first w:t in run
            t_elem = None
            for child in list(run_element):
                if child.tag.endswith('}t'):
                    t_elem = child
                    break
            if t_elem is None:
                return
            # If r_pr is after t_elem, move it before
            if run_element.index(r_pr) > run_element.index(t_elem):
                run_element.remove(r_pr)
                run_element.insert(run_element.index(t_elem), r_pr)
        except Exception as e:
            logger.debug(f"Failed to order rPr before text: {e}")

class GraphicsTextReplacer:
    """Handles text replacement in graphics elements while preserving formatting."""
    
    def __init__(self, mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING):
        """
        Initialize graphics text replacer.
        
        Args:
            mode: Processing mode ('append' or 'replace')
            separator: Separator between original and appended text in append mode
            default_mapping: Default text to append when no mapping is found
        """
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
    
    def replace_text_in_runs(self, runs: List[Run], original_text: str, replacement_text: str, 
                           start_pos: int, end_pos: int) -> bool:
        """
        Replace text in runs while preserving formatting.
        
        Args:
            runs: List of runs to modify
            original_text: Original text to replace
            replacement_text: Replacement text
            start_pos: Start position in combined text
            end_pos: End position in combined text
            
        Returns:
            True if replacement was successful
        """
        try:
            # Find affected runs
            text_span = GraphicsTextReconstructor.find_text_in_runs(runs, original_text, start_pos)
            if not text_span:
                return False
            
            span_start, span_end, affected_runs = text_span
            
            # Replace text in the first affected run
            if affected_runs:
                first_run = affected_runs[0]
                run_text = first_run.text
                
                # Calculate position within the first run
                current_pos = 0
                for run in runs:
                    if run == first_run:
                        break
                    current_pos += len(run.text)
                
                match_start_in_run = span_start - current_pos
                match_end_in_run = min(len(run_text), span_end - current_pos)
                
                # Replace the text
                before_text = run_text[:match_start_in_run]
                after_text = run_text[match_end_in_run:]
                
                # Apply mode-specific replacement
                if self.mode == PROCESSING_MODES['APPEND']:
                    final_text = f"{original_text}{self.separator}{replacement_text}"
                else:  # replace mode
                    final_text = replacement_text
                
                first_run.text = before_text + final_text + after_text
                
                # Clear text from subsequent affected runs
                for run in affected_runs[1:]:
                    run.text = ""
                
                return True
        
        except Exception as e:
            logger.error(f"Error replacing text in runs: {e}")
            return False

def create_graphics_text_replacer(mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR, 
                                                                  default_mapping: str = DEFAULT_MAPPING) -> GraphicsTextReplacer:
    """
    Factory function to create a GraphicsTextReplacer instance.
    
    Args:
        mode: Processing mode ('append' or 'replace')
        separator: Separator between original and appended text in append mode
        default_mapping: Default text to append when no mapping is found
        
    Returns:
        GraphicsTextReplacer instance
    """
    return GraphicsTextReplacer(mode, separator, default_mapping)


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
            
            # Helper: check if element is inside mc:Fallback (AlternateContent fallback)
            def _is_inside_fallback(el: ET.Element) -> bool:
                try:
                    parent = el.getparent()
                except Exception:
                    parent = None
                while parent is not None:
                    tag_lower = parent.tag.lower()
                    if tag_lower.endswith('}fallback') or tag_lower.endswith('mc:fallback'):
                        return True
                    parent = parent.getparent() if hasattr(parent, 'getparent') else None
                return False

            # Also look for VML textboxes (v:textbox), but skip those under mc:Fallback
            for element in doc_xml.iter():
                if 'textbox' in element.tag.lower():
                    if _is_inside_fallback(element):
                        logger.debug(f"Skipping VML textbox inside mc:Fallback: {element.tag}")
                        continue
                    textboxes.append(element)
                    logger.debug(f"Found VML textbox: {element.tag}")
            
            # Look for drawing elements that might contain textboxes, skipping mc:Fallback
            for drawing in doc_xml.iter():
                if drawing.tag.endswith('}drawing'):
                    # Look for textboxes within drawings
                    for child in drawing.iter():
                        if 'textbox' in child.tag.lower() or child.tag.endswith('}txbxContent'):
                            if _is_inside_fallback(child):
                                logger.debug(f"Skipping textbox under mc:Fallback within drawing: {child.tag}")
                                continue
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
                    # Always add the element, even if text is empty (preserves structure)
                    combined_text += text_content
                    wt_elements.append(wt_element)
            
            # Also check for hyperlink elements specifically
            for hyperlink in textbox_element.iter():
                if hyperlink.tag.endswith('}hyperlink'):
                    for wt_element in hyperlink.iter():
                        if wt_element.tag.endswith('}t'):
                            text_content = wt_element.text or ""
                            combined_text += text_content
                            wt_elements.append(wt_element)
            
            logger.debug(f"Extracted {len(wt_elements)} w:t elements with {len(combined_text)} characters")
            
        except Exception as e:
            logger.error(f"Error extracting text from textbox: {e}")
        
        return combined_text, wt_elements
    
    @staticmethod
    def get_textbox_dimensions(textbox_element: ET.Element) -> Dict[str, Any]:
        """
        Extract textbox dimensions from the XML structure.
        
        Args:
            textbox_element: Textbox XML element
            
        Returns:
            Dictionary with width, height, and has_dimensions flag
        """
        dimensions = {
            'width': 0.0,
            'height': 0.0,
            'has_dimensions': False
        }
        
        try:
            # Look for shape properties that contain dimensions
            for element in textbox_element.iter():
                if element.tag.endswith('}spPr') or element.tag.endswith('}shapePr'):
                    # Look for xfrm (transform) element that contains dimensions
                    xfrm = element.find('.//a:xfrm', namespaces=XML_NAMESPACES)
                    if xfrm is not None:
                        # Get width and height from ext (extent) element
                        ext = xfrm.find('.//a:ext', namespaces=XML_NAMESPACES)
                        if ext is not None:
                            width_emu = ext.get('cx')
                            height_emu = ext.get('cy')
                            
                            if width_emu and height_emu:
                                # Convert EMUs to points (1 point = 12700 EMUs)
                                dimensions['width'] = float(width_emu) / 12700.0
                                dimensions['height'] = float(height_emu) / 12700.0
                                dimensions['has_dimensions'] = True
                                logger.debug(f"Found dimensions from spPr: {width_emu}x{height_emu} EMUs = {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
                                break
                
                # Also check for VML dimensions
                elif element.tag.endswith('}shape'):
                    width_attr = element.get('style')
                    if width_attr:
                        # Parse style attribute for width and height
                        width_match = re.search(r'width:\s*([\d.]+)pt', width_attr)
                        height_match = re.search(r'height:\s*([\d.]+)pt', width_attr)
                        
                        if width_match and height_match:
                            dimensions['width'] = float(width_match.group(1))
                            dimensions['height'] = float(height_match.group(1))
                            dimensions['has_dimensions'] = True
                            logger.debug(f"Found dimensions from VML shape: {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
                            break
            
            if not dimensions['has_dimensions']:
                # Method 3: Look for w:drawing dimensions in parent elements (like awi_main)
                parent = textbox_element.getparent()
                while parent is not None:
                    # Check for w:drawing dimensions
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
                                    logger.debug(f"Found dimensions from w:drawing extent: {cx}x{cy} EMUs = {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
                                    break
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Error converting dimensions: {e}")
                                    continue
                    
                    if dimensions['has_dimensions']:
                        break
                        
                    parent = parent.getparent()
            
            # Method 4: Fallback - estimate dimensions based on content
            if not dimensions['has_dimensions']:
                combined_text, _ = TextboxParser.extract_text_from_textbox(textbox_element)
                if combined_text:
                    # Rough estimation: 1 character ≈ 6 points width, 1 line ≈ 12 points height
                    lines = combined_text.count('\n') + 1
                    dimensions['width'] = len(combined_text) * 6.0
                    dimensions['height'] = lines * 12.0
                    dimensions['has_dimensions'] = True
                    logger.debug(f"Estimated dimensions: {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
        
        except Exception as e:
            logger.error(f"Error extracting textbox dimensions: {e}")
        
        return dimensions





class TextboxCapacityCalculator:
    """Calculates optimal font size for textbox content."""
    
    @staticmethod
    def estimate_text_dimensions(text: str, font_size: float) -> Dict[str, float]:
        """
        Estimate text dimensions based on font size.
        
        Args:
            text: Text content
            font_size: Font size in points
            
        Returns:
            Dictionary with estimated width and height
        """
        # Rough estimation: 1 character ≈ 0.6 * font_size points width
        # 1 line ≈ 1.2 * font_size points height
        lines = text.count('\n') + 1
        max_line_length = max(len(line) for line in text.split('\n'))
        
        width = max_line_length * font_size * 0.6
        height = lines * font_size * 1.2
        
        return {'width': width, 'height': height}
    
    @staticmethod
    def find_optimal_font_size(text: str, container_dimensions: Dict[str, Any],
                              max_font_size: float, min_font_size: float = 5.0,
                              font_family: str = "Arial") -> float:
        """Find optimal font size using shared capacity evaluator."""
        if not container_dimensions.get('has_dimensions', False):
            return max_font_size

        width_pt = float(container_dimensions['width'])
        height_pt = float(container_dimensions['height'])

        for font_size in range(int(max_font_size), int(min_font_size) - 1, -1):
            test_size = float(font_size)
            _lines, _cpl, total_fit, *_ = evaluate_capacity(
                width_pt, height_pt, font_family, test_size, safety_margin=0.2
            )
            if total_fit >= len(text):
                logger.debug(f"Text fits with font size {test_size}pt via capacity evaluator")
                return test_size

        logger.warning(f"Text doesn't fit even with minimum font size {min_font_size}pt")
        return min_font_size
