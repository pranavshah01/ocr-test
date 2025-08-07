#!/usr/bin/env python3
"""
Enhanced Textbox OCR Processor for comprehensive font and shape management.
Handles text spanning multiple w:t tags, font size normalization, and intelligent sizing
to ensure all content (original + appended) fits within textbox/callout box boundaries.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
import logging

try:
    from .shared_constants import (XML_NAMESPACES, EMU_TO_PIXELS, DEFAULT_FONT_FAMILY, 
                                  DEFAULT_BOX_WIDTH, DEFAULT_BOX_HEIGHT, SharedUtilities)
    from .ocr_text_replacement import OCRTextReplacementProcessor
    from .comprehensive_image_detector import ComprehensiveImageDetector
except ImportError:
    from shared_constants import (XML_NAMESPACES, EMU_TO_PIXELS, DEFAULT_FONT_FAMILY, 
                                 DEFAULT_BOX_WIDTH, DEFAULT_BOX_HEIGHT, SharedUtilities)
    from ocr_text_replacement import OCRTextReplacementProcessor
    from comprehensive_image_detector import ComprehensiveImageDetector

logger = logging.getLogger(__name__)


class EnhancedTextboxOCRProcessor:
    """
    Enhanced processor for textboxes and callout boxes with comprehensive OCR text replacement.
    Handles font management, text spanning multiple w:t tags, and intelligent sizing.
    """
    
    def __init__(self):
        self.namespaces = XML_NAMESPACES
        self.ocr_processor = OCRTextReplacementProcessor()
        self.image_detector = ComprehensiveImageDetector()
        
        # Font size to box area ratios (characters per square pixel)
        self.font_area_ratios = {
            8: 0.0008,   # Very small font
            10: 0.0010,  # Small font
            12: 0.0012,  # Normal font
            14: 0.0016,  # Medium font
            16: 0.0020,  # Large font
            18: 0.0024,  # Very large font
            20: 0.0028   # Extra large font
        }
        
        # Processing statistics
        self.stats = {
            'textboxes_processed': 0,
            'font_normalizations': 0,
            'shape_resizes': 0,
            'text_overflow_fixes': 0,
            'multi_tag_consolidations': 0
        }
    
    def process_textbox_with_ocr(self, 
                                textbox_element: etree.Element,
                                mapping: Dict[str, str],
                                patterns: List[str],
                                ocr_mode: str = "append") -> Dict[str, Any]:
        """
        Process a textbox element with OCR text replacement and intelligent font management.
        
        Args:
            textbox_element: XML element representing the textbox
            mapping: Text replacement mapping
            patterns: Regex patterns to match
            ocr_mode: OCR processing mode ('replace', 'append', 'append-image')
            
        Returns:
            Processing results with font adjustments and modifications
        """
        results = {
            'textbox_id': self._get_textbox_id(textbox_element),
            'original_text_segments': [],
            'consolidated_text': '',
            'font_sizes_found': [],
            'minimum_font_size': None,
            'optimized_font_size': None,
            'box_dimensions': {},
            'text_fits': False,
            'modifications_made': [],
            'processing_errors': []
        }
        
        try:
            # Step 1: Extract all text segments and their font information
            text_segments = self._extract_text_segments_with_fonts(textbox_element)
            results['original_text_segments'] = text_segments
            
            # Step 2: Consolidate text from multiple w:t tags
            consolidated_text = self._consolidate_text_segments(text_segments)
            results['consolidated_text'] = consolidated_text
            
            # Step 3: Find all font sizes and determine minimum
            font_sizes = [seg['font_size'] for seg in text_segments if seg['font_size']]
            results['font_sizes_found'] = font_sizes
            
            if font_sizes:
                min_font_size = min(font_sizes)
                results['minimum_font_size'] = min_font_size
            else:
                min_font_size = 12  # Default font size
                results['minimum_font_size'] = min_font_size
            
            # Step 4: Get textbox dimensions
            box_dimensions = self._get_textbox_dimensions(textbox_element)
            results['box_dimensions'] = box_dimensions
            
            # Step 5: Apply OCR text replacement to consolidated text
            if consolidated_text.strip():
                ocr_matches = self._find_ocr_matches_in_text(
                    consolidated_text, mapping, patterns
                )
                
                if ocr_matches:
                    # Step 6: Calculate text after OCR replacement
                    modified_text = self._apply_text_replacements(
                        consolidated_text, ocr_matches, ocr_mode
                    )
                    
                    # Step 7: Calculate optimal font size for all content
                    optimal_font_size = self._calculate_optimal_font_size(
                        modified_text, box_dimensions, min_font_size
                    )
                    results['optimized_font_size'] = optimal_font_size
                    
                    # Step 8: Check if text fits and apply modifications
                    text_fits = self._text_fits_in_box(
                        modified_text, box_dimensions, optimal_font_size
                    )
                    results['text_fits'] = text_fits
                    
                    if text_fits:
                        # Apply font normalization to all text segments
                        self._normalize_all_font_sizes(textbox_element, optimal_font_size)
                        
                        # Update text content with OCR replacements
                        self._update_textbox_content(textbox_element, modified_text)
                        
                        results['modifications_made'].append(
                            f"Font normalized to {optimal_font_size}pt"
                        )
                        results['modifications_made'].append(
                            f"Text updated with {len(ocr_matches)} OCR replacements"
                        )
                        
                        self.stats['font_normalizations'] += 1
                        self.stats['textboxes_processed'] += 1
                    else:
                        # Text doesn't fit - need to resize textbox or reduce font further
                        resize_result = self._handle_text_overflow(
                            textbox_element, modified_text, box_dimensions, optimal_font_size
                        )
                        results['modifications_made'].extend(resize_result['modifications'])
                        self.stats['text_overflow_fixes'] += 1
            
        except Exception as e:
            error_msg = f"Error processing textbox: {e}"
            logger.error(error_msg)
            results['processing_errors'].append(error_msg)
        
        return results
    
    def process_all_textboxes(self, docx_path: Path, mapping: Dict[str, str], patterns: List[str]) -> Tuple[int, Dict[str, Any]]:
        """
        Process all textboxes in a DOCX file with OCR text replacement.
        
        Args:
            docx_path: Path to DOCX file
            mapping: Text replacement mapping
            patterns: Regex patterns to match
            
        Returns:
            Tuple of (total_replacements, processing_results)
        """
        results = {
            'textboxes_found': 0,
            'textboxes_processed': 0,
            'total_replacements': 0,
            'font_adjustments': 0,
            'shape_resizes': 0,
            'overflow_detections': 0,
            'processing_errors': [],
            'textbox_details': []
        }
        
        try:
            # Extract and process textboxes from DOCX
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                # Read document.xml
                doc_xml = docx_zip.read('word/document.xml')
                root = etree.fromstring(doc_xml)
                
                # Find all textbox elements
                textboxes = root.xpath('.//w:txbxContent', namespaces=self.namespaces)
                results['textboxes_found'] = len(textboxes)
                
                if not textboxes:
                    return 0, results
                
                # Process each textbox
                for i, textbox in enumerate(textboxes):
                    try:
                        textbox_result = self.process_textbox_with_ocr(
                            textbox, mapping, patterns, ocr_mode="replace"
                        )
                        
                        results['textboxes_processed'] += 1
                        results['textbox_details'].append(textbox_result)
                        
                        # Count modifications
                        if textbox_result.get('modifications_made'):
                            results['total_replacements'] += len(textbox_result['modifications_made'])
                        
                        if textbox_result.get('optimized_font_size'):
                            results['font_adjustments'] += 1
                        
                        if not textbox_result.get('text_fits', True):
                            results['overflow_detections'] += 1
                            
                    except Exception as e:
                        error_msg = f"Error processing textbox {i}: {e}"
                        results['processing_errors'].append(error_msg)
                        continue
                
                # Save modified document if changes were made
                if results['total_replacements'] > 0:
                    self._save_modified_document(docx_path, root)
                
        except Exception as e:
            error_msg = f"Error processing textboxes in {docx_path}: {e}"
            results['processing_errors'].append(error_msg)
        
        return results['total_replacements'], results
    
    def _save_modified_document(self, docx_path: Path, modified_root: etree.Element):
        """
        Save the modified document with textbox changes.
        
        Args:
            docx_path: Original DOCX file path
            modified_root: Modified XML root element
        """
        try:
            # Create output path
            output_dir = Path("./processed")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{docx_path.stem}_textbox_processed.docx"
            
            # Copy original file and update document.xml
            import shutil
            shutil.copy2(docx_path, output_path)
            
            with zipfile.ZipFile(output_path, 'a') as docx_zip:
                # Update document.xml with modifications
                modified_xml = etree.tostring(modified_root, encoding='utf-8', xml_declaration=True)
                docx_zip.writestr('word/document.xml', modified_xml)
                
        except Exception as e:
            print(f"Warning: Could not save modified textbox document: {e}")
    
    def _extract_text_segments_with_fonts(self, textbox_element: etree.Element) -> List[Dict[str, Any]]:
        """
        Extract all text segments from textbox with their font information.
        Handles text spanning multiple w:t tags.
        
        Args:
            textbox_element: Textbox XML element
            
        Returns:
            List of text segments with font information
        """
        text_segments = []
        
        try:
            # Find all w:t elements (text runs)
            text_runs = textbox_element.xpath('.//w:t', namespaces=self.namespaces)
            
            for text_run in text_runs:
                # Get the parent w:r (run) element for formatting info
                run_element = text_run.getparent()
                
                # Extract text content
                text_content = text_run.text or ''
                
                # Extract font information from w:rPr (run properties)
                font_info = self._extract_font_properties(run_element)
                
                segment = {
                    'text': text_content,
                    'font_family': font_info.get('font_family', DEFAULT_FONT_FAMILY),
                    'font_size': font_info.get('font_size', 12),
                    'bold': font_info.get('bold', False),
                    'italic': font_info.get('italic', False),
                    'color': font_info.get('color', '000000'),
                    'xml_element': text_run
                }
                
                text_segments.append(segment)
                
            self.stats['multi_tag_consolidations'] += len(text_segments)
            
        except Exception as e:
            logger.error(f"Error extracting text segments: {e}")
        
        return text_segments
    
    def _extract_font_properties(self, run_element: etree.Element) -> Dict[str, Any]:
        """
        Extract font properties from a w:r (run) element.
        
        Args:
            run_element: Run XML element
            
        Returns:
            Dictionary with font properties
        """
        font_props = {
            'font_family': DEFAULT_FONT_FAMILY,
            'font_size': 12,
            'bold': False,
            'italic': False,
            'color': '000000'
        }
        
        try:
            # Find run properties (w:rPr)
            rpr = run_element.find('.//w:rPr', namespaces=self.namespaces)
            if rpr is not None:
                # Font family
                font_elem = rpr.find('w:rFonts', namespaces=self.namespaces)
                if font_elem is not None:
                    font_props['font_family'] = font_elem.get(f'{{{self.namespaces["w"]}}}ascii', DEFAULT_FONT_FAMILY)
                
                # Font size (in half-points)
                sz_elem = rpr.find('w:sz', namespaces=self.namespaces)
                if sz_elem is not None:
                    half_points = int(sz_elem.get(f'{{{self.namespaces["w"]}}}val', '24'))
                    font_props['font_size'] = half_points // 2
                
                # Bold
                bold_elem = rpr.find('w:b', namespaces=self.namespaces)
                font_props['bold'] = bold_elem is not None
                
                # Italic
                italic_elem = rpr.find('w:i', namespaces=self.namespaces)
                font_props['italic'] = italic_elem is not None
                
                # Color
                color_elem = rpr.find('w:color', namespaces=self.namespaces)
                if color_elem is not None:
                    font_props['color'] = color_elem.get(f'{{{self.namespaces["w"]}}}val', '000000')
        
        except Exception as e:
            logger.warning(f"Error extracting font properties: {e}")
        
        return font_props
    
    def _consolidate_text_segments(self, text_segments: List[Dict[str, Any]]) -> str:
        """
        Consolidate text from multiple segments into a single string.
        
        Args:
            text_segments: List of text segments
            
        Returns:
            Consolidated text string
        """
        return ''.join(segment['text'] for segment in text_segments)
    
    def _get_textbox_dimensions(self, textbox_element: etree.Element) -> Dict[str, int]:
        """
        Extract textbox dimensions from XML element.
        
        Args:
            textbox_element: Textbox XML element
            
        Returns:
            Dictionary with width and height in pixels
        """
        dimensions = {
            'width': DEFAULT_BOX_WIDTH,
            'height': DEFAULT_BOX_HEIGHT,
            'area': DEFAULT_BOX_WIDTH * DEFAULT_BOX_HEIGHT
        }
        
        try:
            # Look for shape properties with dimensions
            # This is a simplified approach - real implementation would need more comprehensive parsing
            extent_elem = textbox_element.xpath('.//wp:extent', namespaces=self.namespaces)
            if extent_elem:
                cx = int(extent_elem[0].get('cx', str(DEFAULT_BOX_WIDTH * EMU_TO_PIXELS)))
                cy = int(extent_elem[0].get('cy', str(DEFAULT_BOX_HEIGHT * EMU_TO_PIXELS)))
                
                dimensions['width'] = int(cx / EMU_TO_PIXELS)
                dimensions['height'] = int(cy / EMU_TO_PIXELS)
                dimensions['area'] = dimensions['width'] * dimensions['height']
        
        except Exception as e:
            logger.warning(f"Error getting textbox dimensions: {e}")
        
        return dimensions
    
    def _find_ocr_matches_in_text(self, 
                                 text: str, 
                                 mapping: Dict[str, str], 
                                 patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Find OCR matches in consolidated text using mapping and patterns.
        
        Args:
            text: Consolidated text to search
            mapping: Text replacement mapping
            patterns: Regex patterns
            
        Returns:
            List of matches found
        """
        matches = []
        
        # Check direct mapping matches
        for from_text, to_text in mapping.items():
            if self._text_matches(text, from_text):
                matches.append({
                    'type': 'mapping',
                    'from_text': from_text,
                    'to_text': to_text,
                    'original_text': text
                })
        
        # Check pattern matches
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append({
                        'type': 'pattern',
                        'pattern': pattern,
                        'from_text': text,
                        'to_text': text,  # Pattern matches keep original
                        'original_text': text
                    })
            except re.error:
                continue
        
        return matches
    
    def _text_matches(self, text1: str, text2: str) -> bool:
        """Check if two text strings match with normalization."""
        norm1 = ''.join(text1.lower().split())
        norm2 = ''.join(text2.lower().split())
        return norm1 == norm2 or text1.lower() == text2.lower()
    
    def _apply_text_replacements(self, 
                                text: str, 
                                matches: List[Dict[str, Any]], 
                                ocr_mode: str) -> str:
        """
        Apply text replacements based on OCR mode.
        
        Args:
            text: Original text
            matches: List of matches to apply
            ocr_mode: OCR processing mode
            
        Returns:
            Modified text after replacements
        """
        modified_text = text
        
        for match in matches:
            from_text = match['from_text']
            to_text = match['to_text']
            
            if ocr_mode == "replace":
                modified_text = modified_text.replace(from_text, to_text)
            elif ocr_mode == "append":
                # Append with space and newline
                modified_text = modified_text.replace(from_text, f"{from_text}\n{to_text}")
            # append-image mode doesn't modify the original textbox
        
        return modified_text
    
    def _calculate_optimal_font_size(self, 
                                   text: str, 
                                   box_dimensions: Dict[str, int], 
                                   min_font_size: int) -> int:
        """
        Calculate optimal font size to fit all text in the textbox.
        Uses font size/box area ratios to estimate text capacity.
        
        Args:
            text: Text content to fit
            box_dimensions: Box dimensions
            min_font_size: Minimum font size from existing text
            
        Returns:
            Optimal font size in points
        """
        try:
            text_length = len(text)
            box_area = box_dimensions['area']
            
            # Start with minimum font size and work down if needed
            optimal_size = min_font_size
            
            # Calculate required area per character for different font sizes
            for font_size in sorted(self.font_area_ratios.keys(), reverse=True):
                chars_per_pixel = self.font_area_ratios[font_size]
                max_chars = int(box_area * chars_per_pixel)
                
                if text_length <= max_chars and font_size <= min_font_size:
                    optimal_size = font_size
                    break
            
            # Ensure minimum readable font size
            optimal_size = max(optimal_size, 8)
            
            # Further reduce if still too large
            while optimal_size > 6:
                if self._text_fits_in_box(text, box_dimensions, optimal_size):
                    break
                optimal_size -= 1
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal font size: {e}")
            return max(min_font_size - 2, 8)
    
    def _text_fits_in_box(self, 
                         text: str, 
                         box_dimensions: Dict[str, int], 
                         font_size: int) -> bool:
        """
        Check if text fits in the given box dimensions with specified font size.
        
        Args:
            text: Text to check
            box_dimensions: Box dimensions
            font_size: Font size in points
            
        Returns:
            True if text fits, False otherwise
        """
        try:
            # Create temporary image to measure text
            img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(img)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
            
            # Measure text dimensions
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Add padding (10% of box dimensions)
            available_width = box_dimensions['width'] * 0.9
            available_height = box_dimensions['height'] * 0.9
            
            return text_width <= available_width and text_height <= available_height
            
        except Exception as e:
            logger.warning(f"Error checking text fit: {e}")
            return True  # Assume it fits if we can't measure
    
    def _normalize_all_font_sizes(self, textbox_element: etree.Element, target_font_size: int):
        """
        Normalize all font sizes in the textbox to the target size.
        
        Args:
            textbox_element: Textbox XML element
            target_font_size: Target font size in points
        """
        try:
            # Find all w:sz elements (font size)
            size_elements = textbox_element.xpath('.//w:sz', namespaces=self.namespaces)
            
            target_half_points = str(target_font_size * 2)
            
            for size_elem in size_elements:
                size_elem.set(f'{{{self.namespaces["w"]}}}val', target_half_points)
            
            # Also update w:szCs (complex script font size) if present
            cs_size_elements = textbox_element.xpath('.//w:szCs', namespaces=self.namespaces)
            for cs_size_elem in cs_size_elements:
                cs_size_elem.set(f'{{{self.namespaces["w"]}}}val', target_half_points)
                
        except Exception as e:
            logger.error(f"Error normalizing font sizes: {e}")
    
    def _update_textbox_content(self, textbox_element: etree.Element, new_text: str):
        """
        Update textbox content with new text.
        
        Args:
            textbox_element: Textbox XML element
            new_text: New text content
        """
        try:
            # Find first w:t element and update its content
            text_elements = textbox_element.xpath('.//w:t', namespaces=self.namespaces)
            
            if text_elements:
                # Update first text element with new content
                text_elements[0].text = new_text
                
                # Remove additional text elements if any
                for text_elem in text_elements[1:]:
                    parent = text_elem.getparent()
                    if parent is not None:
                        parent.remove(text_elem)
                        
        except Exception as e:
            logger.error(f"Error updating textbox content: {e}")
    
    def _handle_text_overflow(self, 
                             textbox_element: etree.Element,
                             text: str,
                             box_dimensions: Dict[str, int],
                             font_size: int) -> Dict[str, Any]:
        """
        Handle text overflow by resizing textbox or further reducing font size.
        
        Args:
            textbox_element: Textbox XML element
            text: Text content
            box_dimensions: Current box dimensions
            font_size: Current font size
            
        Returns:
            Dictionary with modifications made
        """
        result = {'modifications': []}
        
        try:
            # Try reducing font size further
            reduced_font_size = max(font_size - 2, 6)
            
            if self._text_fits_in_box(text, box_dimensions, reduced_font_size):
                self._normalize_all_font_sizes(textbox_element, reduced_font_size)
                result['modifications'].append(f"Font reduced to {reduced_font_size}pt to fit text")
            else:
                # Try resizing textbox (increase height by 20%)
                new_height = int(box_dimensions['height'] * 1.2)
                if self._resize_textbox(textbox_element, box_dimensions['width'], new_height):
                    result['modifications'].append(f"Textbox resized to fit content")
                    self.stats['shape_resizes'] += 1
                else:
                    # Last resort: use minimum font size
                    self._normalize_all_font_sizes(textbox_element, 6)
                    result['modifications'].append("Font set to minimum size (6pt)")
                    
        except Exception as e:
            logger.error(f"Error handling text overflow: {e}")
            result['modifications'].append(f"Error handling overflow: {e}")
        
        return result
    
    def _resize_textbox(self, textbox_element: etree.Element, width: int, height: int) -> bool:
        """
        Resize textbox to new dimensions.
        
        Args:
            textbox_element: Textbox XML element
            width: New width in pixels
            height: New height in pixels
            
        Returns:
            True if resize was successful
        """
        try:
            # Find extent element and update dimensions
            extent_elem = textbox_element.xpath('.//wp:extent', namespaces=self.namespaces)
            if extent_elem:
                extent_elem[0].set('cx', str(int(width * EMU_TO_PIXELS)))
                extent_elem[0].set('cy', str(int(height * EMU_TO_PIXELS)))
                return True
                
        except Exception as e:
            logger.error(f"Error resizing textbox: {e}")
        
        return False
    
    def _get_textbox_id(self, textbox_element: etree.Element) -> str:
        """Get unique identifier for textbox."""
        try:
            # Try to find shape ID or create one based on position
            shape_id = textbox_element.get('id', '')
            if not shape_id:
                # Use element hash as fallback
                shape_id = f"textbox_{hash(etree.tostring(textbox_element)) % 10000}"
            return shape_id
        except:
            return "unknown_textbox"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'textbox_processing_stats': self.stats,
            'ocr_processing_stats': self.ocr_processor.get_processing_statistics()
        }


def process_docx_textboxes_with_ocr(docx_path: Path,
                                   mapping: Dict[str, str],
                                   patterns: List[str] = None,
                                   ocr_mode: str = "append") -> Dict[str, Any]:
    """
    Convenience function to process all textboxes in a DOCX file with enhanced OCR.
    
    Args:
        docx_path: Path to DOCX file
        mapping: Text replacement mapping
        patterns: Optional regex patterns
        ocr_mode: OCR processing mode
        
    Returns:
        Processing results
    """
    processor = EnhancedTextboxOCRProcessor()
    
    # This would need integration with DOCX parsing to find all textboxes
    # For now, return a placeholder structure
    return {
        'docx_path': str(docx_path),
        'textboxes_processed': 0,
        'font_normalizations': 0,
        'shape_resizes': 0,
        'processing_errors': []
    }


if __name__ == "__main__":
    # Test the enhanced textbox OCR processor
    print("Enhanced Textbox OCR Processor - Font and Shape Management")
    print("Features:")
    print("- Text spanning multiple w:t tags handling")
    print("- Font size normalization to smallest size in textbox")
    print("- Intelligent font sizing based on text content and box area")
    print("- Font size/box area ratios for optimal text fitting")
    print("- Shape resizing when text doesn't fit")
    print("- Comprehensive OCR text replacement integration")
