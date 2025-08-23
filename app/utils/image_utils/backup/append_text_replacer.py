"""
Append mode text replacer for OCR-based image text replacement.
Intelligently places original and replacement text using top/bottom or side-by-side layout
based on available space in the image, ensuring font readability.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from ...core.models import OCRResult, OCRMatch
from ..shared_constants import FALLBACK_FONTS
from .text_analyzer import TextAnalyzer, TextProperties, create_text_analyzer
from ..pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)

class AppendTextReplacer:
    """
    Independent text replacer that appends original and replacement text using smart positioning.
    Uses top/bottom or side-by-side layout based on available space in the image.
    Designed to not impact the existing replace mode functionality.
    """
    
    def __init__(self, pattern_matcher: PatternMatcher):
        """
        Initialize append text replacer.
        
        Args:
            pattern_matcher: Pattern matcher for finding text replacements
        """
        self.pattern_matcher = pattern_matcher
        self.text_analyzer = create_text_analyzer()
        
        logger.info("Append text replacer initialized for top-bottom text layout")
    
    def replace_text_in_image(self, image_path: Path, ocr_results: List[OCRResult], 
                            matches: List[OCRMatch]) -> Path:
        """
        Replace text in image using append mode (top-bottom layout).
        
        Args:
            image_path: Path to the image file
            ocr_results: OCR detection results
            matches: Pattern matches to replace
            
        Returns:
            Path to the modified image
        """
        logger.info(f"🔧 APPEND DEBUG: replace_text_in_image called with {len(matches)} matches for {image_path.name}")
        try:
            # Load the image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return image_path
            
            # Convert to PIL for text rendering
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Process each match
            replacements_made = 0
            logger.info(f"🔧 APPEND DEBUG: Starting to process {len(matches)} matches")
            for i, match in enumerate(matches):
                logger.info(f"🔧 APPEND DEBUG: Processing match {i}: {match}")
                try:
                    logger.info(f"🔧 APPEND DEBUG: About to process match {i}")
                    # Use the OCR result from the match object
                    ocr_result = match.ocr_result
                    
                    # Use the replacement text from the match object (already computed)
                    replacement_text = match.replacement_text
                    logger.info(f"🔧 APPEND DEBUG: Using replacement from match: '{replacement_text}'")
                    if not replacement_text:
                        logger.warning(f"No replacement found for: {match.matched_text}")
                        continue
                    
                    # Get the original text from the OCR result
                    original_text = match.ocr_result.text
                    logger.info(f"🔧 APPEND DEBUG: Processing '{original_text}' -> '{replacement_text}'")
                    
                    # Apply append mode replacement
                    success = self._apply_append_replacement(
                        pil_image, draw, ocr_result, original_text, replacement_text
                    )
                    
                    logger.info(f"🔧 APPEND DEBUG: Replacement success: {success}")
                    
                    if success:
                        replacements_made += 1
                        logger.debug(f"Applied append replacement: '{original_text}' -> '{replacement_text}'")
                    
                except Exception as e:
                    logger.error(f"🔧 APPEND DEBUG: Exception processing match: {e}")
                    import traceback
                    logger.error(f"🔧 APPEND DEBUG: Traceback: {traceback.format_exc()}")
            
            # Save the modified image
            if replacements_made > 0:
                output_path = self._generate_output_path(image_path, "append")
                
                # Convert back to BGR for OpenCV
                modified_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), modified_cv_image)
                
                logger.info(f"Applied {replacements_made} append replacements to {image_path.name}")
                return output_path
            else:
                logger.info(f"No replacements applied to {image_path.name}")
                return image_path
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return image_path
    
    def _find_ocr_result_for_match(self, match: OCRMatch, ocr_results: List[OCRResult]) -> Optional[OCRResult]:
        """Find the OCR result that corresponds to a pattern match."""
        logger.info(f"🔧 APPEND DEBUG: Looking for match '{match.matched_text}' in {len(ocr_results)} OCR results")
        for i, ocr_result in enumerate(ocr_results):
            logger.info(f"🔧 APPEND DEBUG: OCR result {i}: '{ocr_result.text}'")
            if match.matched_text in ocr_result.text or ocr_result.text in match.matched_text:
                logger.info(f"🔧 APPEND DEBUG: Found matching OCR result: '{ocr_result.text}'")
                return ocr_result
        logger.warning(f"🔧 APPEND DEBUG: No matching OCR result found for '{match.matched_text}'")
        return None
    
    def _apply_append_replacement(self, pil_image: Image.Image, draw: ImageDraw.Draw,
                                ocr_result: OCRResult, original_text: str, replacement_text: str) -> bool:
        """
        Apply append mode replacement with vertical text stacking (top-half/bottom-half positioning).
        Uses precise pattern-based bounding box to avoid overwriting surrounding text.
        
        Args:
            pil_image: PIL image to modify
            draw: ImageDraw object
            ocr_result: OCR result with bounding box
            original_text: Original text to replace
            replacement_text: Replacement text
            
        Returns:
            True if replacement was successful
        """
        try:
            full_ocr_bbox = ocr_result.bounding_box
            
            # Find the precise bounding box of just the pattern match within the OCR text
            pattern_bbox = self._find_pattern_bbox_in_ocr_text(
                pil_image, ocr_result, original_text, replacement_text
            )
            
            if not pattern_bbox:
                logger.warning(f"Could not find precise pattern bbox, using full OCR bbox")
                pattern_bbox = full_ocr_bbox
            
            logger.info(f"🔧 APPEND DEBUG: Full OCR bbox: {full_ocr_bbox}, Pattern bbox: {pattern_bbox}")
            
            # Convert PIL image to numpy array for text analyzer
            image_array = np.array(pil_image)
            
            # Analyze text properties using the pattern-specific bbox
            text_properties = self.text_analyzer.analyze_text_properties(
                image_array, pattern_bbox
            )
            
            # Import PreciseTextReplacer for vertical stacking functionality
            from .precise_text_replacer import PreciseTextReplacer
            precise_replacer = PreciseTextReplacer()
            
            # For append mode, we want to show both the original pattern and the replacement
            # Find the matched pattern text from the pattern bbox detection
            pattern_matches = self.pattern_matcher.find_matches_universal(original_text)
            matched_pattern_text = None
            
            for match in pattern_matches:
                if self.pattern_matcher.get_replacement(match.matched_text) == replacement_text:
                    matched_pattern_text = match.matched_text
                    break
            
            if matched_pattern_text:
                # Create the combined text that should be stacked: original + replacement
                combined_replacement_text = f"{matched_pattern_text} {replacement_text}"
            else:
                # Fallback to just the replacement text
                combined_replacement_text = replacement_text
            
            # Split the combined text for optimal vertical stacking
            text_lines = precise_replacer.split_replacement_text_for_stacking_append_mode(
                combined_replacement_text, pattern_bbox, text_properties
            )
            
            # If we have multiple lines, use vertical stacking
            if len(text_lines) > 1:
                logger.info(f"🔧 APPEND DEBUG: Using vertical stacking for {len(text_lines)} lines: {text_lines}")
                
                # Use the precise text replacer's stacking functionality with pattern-specific bbox
                result_image = precise_replacer.render_stacked_text_append_mode(
                    pil_image, text_lines, pattern_bbox, text_properties
                )
                
                logger.info(f"🔧 APPEND DEBUG: Applied vertical stacking at pattern bbox {pattern_bbox}: {text_lines}")
                return True
            else:
                # Single line - use traditional append mode layout with pattern bbox
                logger.info(f"🔧 APPEND DEBUG: Using traditional layout for single line: '{replacement_text}'")
                return self._apply_traditional_append_layout_with_pattern_bbox(
                    pil_image, draw, pattern_bbox, original_text, replacement_text, text_properties
                )
            
        except Exception as e:
            logger.error(f"Failed to apply append replacement: {e}")
            return False
    
    def _find_pattern_bbox_in_ocr_text(self, image: Image.Image, ocr_result: OCRResult, 
                                     original_text: str, replacement_text: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the precise bounding box of the pattern match within the OCR text.
        This ensures we only replace the specific pattern, not the entire OCR text.
        
        Args:
            image: PIL image
            ocr_result: OCR result with full text bounding box
            original_text: Original OCR text (e.g., "Heat ShrinkHeat:77-531-116BLK-245")
            replacement_text: Replacement text (e.g., "77-531-116BLK-245 4022-325-927260-30")
            
        Returns:
            Tuple of (x, y, width, height) for just the pattern match, or None if not found
        """
        try:
            # Find what pattern was actually matched using the pattern matcher
            pattern_matches = self.pattern_matcher.find_matches_universal(original_text)
            
            if not pattern_matches:
                logger.warning(f"No pattern matches found in OCR text: '{original_text}'")
                return None
            
            # Find the match that corresponds to our replacement
            target_match = None
            for match in pattern_matches:
                if self.pattern_matcher.get_replacement(match.matched_text) == replacement_text:
                    target_match = match
                    break
            
            if not target_match:
                logger.warning(f"Could not find target pattern match for replacement: '{replacement_text}'")
                return None
            
            # Calculate the position of the matched pattern within the OCR text
            full_x, full_y, full_w, full_h = ocr_result.bounding_box
            
            # Estimate the position of the pattern within the full OCR text
            pattern_start = target_match.start_pos
            pattern_end = target_match.end_pos
            pattern_text = target_match.matched_text
            
            # Calculate approximate character width
            text_length = len(original_text)
            if text_length > 0:
                char_width = full_w / text_length
                
                # Calculate pattern position
                pattern_x = full_x + int(pattern_start * char_width)
                pattern_width = int((pattern_end - pattern_start) * char_width)
                
                # Use full height but adjust width to pattern
                pattern_bbox = (pattern_x, full_y, pattern_width, full_h)
                
                logger.info(f"🔧 PATTERN BBOX: Found pattern '{pattern_text}' at positions {pattern_start}-{pattern_end}")
                logger.info(f"🔧 PATTERN BBOX: Calculated bbox {pattern_bbox} from full bbox {ocr_result.bounding_box}")
                
                return pattern_bbox
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find pattern bbox: {e}")
            return None
    
    def _apply_traditional_append_layout_with_pattern_bbox(self, pil_image: Image.Image, draw: ImageDraw.Draw,
                                                         pattern_bbox: Tuple[int, int, int, int], 
                                                         original_text: str, replacement_text: str,
                                                         text_properties) -> bool:
        """
        Apply traditional append mode layout using the precise pattern bounding box.
        This preserves existing functionality while using the correct bounding box.
        """
        try:
            x, y, w, h = pattern_bbox
            image_width, image_height = pil_image.size
            
            # Calculate optimal font size for append mode (maintain readability)
            original_font_size = text_properties.font_size
            append_font_size = max(10, int(original_font_size * 0.8))  # 80% of original, minimum 10pt for readability
            
            # Get font for rendering
            font = self._get_font_for_size(append_font_size)
            
            # Calculate text dimensions
            original_bbox = draw.textbbox((0, 0), original_text, font=font)
            replacement_bbox = draw.textbbox((0, 0), replacement_text, font=font)
            
            original_text_width = original_bbox[2] - original_bbox[0]
            original_text_height = original_bbox[3] - original_bbox[1]
            replacement_text_width = replacement_bbox[2] - replacement_bbox[0]
            replacement_text_height = replacement_bbox[3] - replacement_bbox[1]
            
            # Determine optimal layout based on available space
            layout_info = self._determine_optimal_layout(
                x, y, w, h, image_width, image_height,
                original_text_width, original_text_height,
                replacement_text_width, replacement_text_height
            )
            
            # Apply the determined layout
            success = self._apply_layout(
                draw, layout_info, original_text, replacement_text,
                font, text_properties
            )
            
            if success:
                logger.info(f"🔧 APPEND DEBUG: Applied {layout_info['layout_type']} layout at pattern bbox ({x}, {y}): "
                           f"'{original_text}' + '{replacement_text}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply traditional append layout with pattern bbox: {e}")
            return False
    
    def _apply_traditional_append_layout(self, pil_image: Image.Image, draw: ImageDraw.Draw,
                                       ocr_result: OCRResult, original_text: str, replacement_text: str,
                                       text_properties) -> bool:
        """
        Apply traditional append mode layout (side-by-side or top-bottom) for single-line replacements.
        This preserves the existing functionality for cases that don't need vertical stacking.
        """
        try:
            x, y, w, h = ocr_result.bounding_box
            image_width, image_height = pil_image.size
            
            # Calculate optimal font size for append mode (maintain readability)
            original_font_size = text_properties.font_size
            append_font_size = max(10, int(original_font_size * 0.8))  # 80% of original, minimum 10pt for readability
            
            # Get font for rendering
            font = self._get_font_for_size(append_font_size)
            
            # Calculate text dimensions
            original_bbox = draw.textbbox((0, 0), original_text, font=font)
            replacement_bbox = draw.textbbox((0, 0), replacement_text, font=font)
            
            original_text_width = original_bbox[2] - original_bbox[0]
            original_text_height = original_bbox[3] - original_bbox[1]
            replacement_text_width = replacement_bbox[2] - replacement_bbox[0]
            replacement_text_height = replacement_bbox[3] - replacement_bbox[1]
            
            # Determine optimal layout based on available space
            layout_info = self._determine_optimal_layout(
                x, y, w, h, image_width, image_height,
                original_text_width, original_text_height,
                replacement_text_width, replacement_text_height
            )
            
            # Apply the determined layout
            success = self._apply_layout(
                draw, layout_info, original_text, replacement_text,
                font, text_properties
            )
            
            if success:
                logger.info(f"🔧 APPEND DEBUG: Applied {layout_info['layout_type']} layout at ({x}, {y}): "
                           f"'{original_text}' + '{replacement_text}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply traditional append layout: {e}")
            return False
    
    def _get_font_for_size(self, font_size: int) -> ImageFont.ImageFont:
        """Get a font for the specified size, ensuring readability."""
        try:
            # Ensure minimum readable font size
            readable_font_size = max(font_size, 10)
            
            # Try to use a system font
            for font_name in FALLBACK_FONTS:
                try:
                    return ImageFont.truetype(font_name, readable_font_size)
                except (OSError, IOError):
                    continue
            
            # Fallback to default font with readable size
            try:
                return ImageFont.load_default()
            except Exception:
                # Ultimate fallback
                return ImageFont.load_default()
            
        except Exception as e:
            logger.debug(f"Font loading failed, using default: {e}")
            return ImageFont.load_default()
    
    def _determine_optimal_layout(self, x: int, y: int, w: int, h: int, 
                                image_width: int, image_height: int,
                                orig_text_width: int, orig_text_height: int,
                                repl_text_width: int, repl_text_height: int) -> Dict[str, Any]:
        """
        Determine the optimal layout (top/bottom or side-by-side) based on available space.
        
        Args:
            x, y, w, h: Original text bounding box
            image_width, image_height: Image dimensions
            orig_text_width, orig_text_height: Original text dimensions
            repl_text_width, repl_text_height: Replacement text dimensions
            
        Returns:
            Dictionary with layout information
        """
        padding = 4
        spacing = 3
        
        # Calculate space requirements for each layout option
        
        # Top/Bottom layout requirements
        tb_total_width = max(orig_text_width, repl_text_width)
        tb_total_height = orig_text_height + repl_text_height + spacing
        
        # Side-by-side layout requirements
        sb_total_width = orig_text_width + repl_text_width + spacing
        sb_total_height = max(orig_text_height, repl_text_height)
        
        # Calculate available space in each direction
        space_right = image_width - (x + w) - padding
        space_left = x - padding
        space_below = image_height - (y + h) - padding
        space_above = y - padding
        
        # Evaluate layout options
        layouts = []
        
        # Option 1: Top/Bottom - Original on top, replacement below
        if space_below >= (repl_text_height + spacing):
            layouts.append({
                'layout_type': 'top_bottom',
                'priority': 1,  # Preferred layout
                'clear_rect': (x - padding, y - padding, 
                             x + tb_total_width + padding, y + tb_total_height + padding),
                'original_pos': (x, y),
                'replacement_pos': (x, y + orig_text_height + spacing),
                'fits_well': tb_total_width <= (w + 2 * padding)
            })
        
        # Option 2: Side-by-side - Original left, replacement right
        if space_right >= (repl_text_width + spacing):
            layouts.append({
                'layout_type': 'side_by_side_right',
                'priority': 2,
                'clear_rect': (x - padding, y - padding,
                             x + sb_total_width + padding, y + sb_total_height + padding),
                'original_pos': (x, y),
                'replacement_pos': (x + orig_text_width + spacing, y),
                'fits_well': sb_total_height <= (h + 2 * padding)
            })
        
        # Option 3: Side-by-side - Original right, replacement left
        if space_left >= (repl_text_width + spacing):
            layouts.append({
                'layout_type': 'side_by_side_left',
                'priority': 3,
                'clear_rect': (x - repl_text_width - spacing - padding, y - padding,
                             x + orig_text_width + padding, y + sb_total_height + padding),
                'original_pos': (x, y),
                'replacement_pos': (x - repl_text_width - spacing, y),
                'fits_well': sb_total_height <= (h + 2 * padding)
            })
        
        # Option 4: Bottom/Top - Replacement on top, original below
        if space_above >= (repl_text_height + spacing):
            layouts.append({
                'layout_type': 'bottom_top',
                'priority': 4,
                'clear_rect': (x - padding, y - repl_text_height - spacing - padding,
                             x + tb_total_width + padding, y + orig_text_height + padding),
                'original_pos': (x, y),
                'replacement_pos': (x, y - repl_text_height - spacing),
                'fits_well': tb_total_width <= (w + 2 * padding)
            })
        
        # Select the best layout
        if not layouts:
            # Fallback: force top/bottom layout even if it goes outside image bounds
            return {
                'layout_type': 'top_bottom_forced',
                'priority': 99,
                'clear_rect': (x - padding, y - padding,
                             x + tb_total_width + padding, y + tb_total_height + padding),
                'original_pos': (x, y),
                'replacement_pos': (x, y + orig_text_height + spacing),
                'fits_well': False
            }
        
        # Sort by priority and preference for layouts that fit well
        layouts.sort(key=lambda l: (l['priority'], not l['fits_well']))
        selected_layout = layouts[0]
        
        logger.debug(f"Selected layout: {selected_layout['layout_type']} "
                    f"(fits_well: {selected_layout['fits_well']})")
        
        return selected_layout
    
    def _apply_layout(self, draw: ImageDraw.Draw, layout_info: Dict[str, Any],
                     original_text: str, replacement_text: str,
                     font: ImageFont.ImageFont, text_properties: TextProperties) -> bool:
        """
        Apply the determined layout to place both texts.
        
        Args:
            draw: ImageDraw object
            layout_info: Layout information from _determine_optimal_layout
            original_text: Original text
            replacement_text: Replacement text
            font: Font to use for rendering
            text_properties: Original text properties
            
        Returns:
            True if layout was applied successfully
        """
        try:
            # Clear the area with white rectangle
            draw.rectangle(layout_info['clear_rect'], fill='white')
            
            # Get colors
            original_color = text_properties.color
            replacement_color = self._get_replacement_text_color(original_color)
            
            # Draw original text
            draw.text(
                layout_info['original_pos'],
                original_text,
                fill=original_color,
                font=font
            )
            
            # Draw replacement text
            draw.text(
                layout_info['replacement_pos'],
                replacement_text,
                fill=replacement_color,
                font=font
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply layout {layout_info['layout_type']}: {e}")
            return False

    def _get_replacement_text_color(self, original_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get a slightly different color for replacement text to distinguish it from original.
        
        Args:
            original_color: RGB tuple of original text color
            
        Returns:
            RGB tuple for replacement text color
        """
        r, g, b = original_color
        
        # If original is very dark, make replacement slightly lighter
        if r + g + b < 150:  # Dark text
            return (min(255, r + 30), min(255, g + 30), min(255, b + 30))
        # If original is light, make replacement slightly darker
        else:
            return (max(0, r - 30), max(0, g - 30), max(0, b - 30))
    
    def process_image_append_mode(self, image_path: Path, patterns: Dict[str, str], 
                                mappings: Dict[str, str]) -> Path:
        """
        Independent append mode processing that doesn't interfere with replace mode.
        
        Args:
            image_path: Path to the image file
            patterns: Pattern definitions for matching
            mappings: Text replacement mappings
            
        Returns:
            Path to the processed image with append mode replacements
        """
        try:
            # Create a temporary pattern matcher for this processing
            temp_pattern_matcher = PatternMatcher(patterns, mappings)
            
            # Extract text using a basic OCR approach (independent of main processor)
            ocr_results = self._extract_text_for_append_mode(image_path)
            
            if not ocr_results:
                logger.info(f"No text detected in {image_path.name} for append mode")
                return image_path
            
            # Debug: log detected text
            detected_texts = [result.text for result in ocr_results]
            logger.info(f"Detected texts in {image_path.name}: {detected_texts[:5]}...")  # Show first 5
            
            # Find pattern matches
            matches = []
            for ocr_result in ocr_results:
                pattern_matches = temp_pattern_matcher.find_matches(ocr_result.text)
                for pattern_name, matched_text, start_pos, end_pos in pattern_matches:
                    # Create a simple match object for compatibility
                    match_obj = type('Match', (), {
                        'pattern_name': pattern_name,
                        'matched_text': matched_text,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'ocr_result': ocr_result
                    })()
                    matches.append(match_obj)
            
            if not matches:
                logger.info(f"No pattern matches found in {image_path.name} for append mode")
                # Debug: show what we were looking for
                all_text = " ".join([result.text for result in ocr_results])
                logger.debug(f"Full text content: {all_text[:200]}...")
                return image_path
            
            # Apply append mode replacements
            return self.replace_text_in_image(image_path, ocr_results, matches)
            
        except Exception as e:
            logger.error(f"Failed to process image in append mode: {e}")
            return image_path
    
    def _extract_text_for_append_mode(self, image_path: Path) -> List[OCRResult]:
        """
        Extract text using a simple OCR approach for append mode processing.
        Independent of the main OCR engine to avoid conflicts.
        """
        try:
            # Use a simple Tesseract approach for append mode
            import pytesseract
            from PIL import Image as PILImage
            
            image = PILImage.open(image_path)
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            results = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = float(ocr_data['conf'][i]) / 100.0
                
                if text and confidence >= 0.5:  # Lower threshold for append mode
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    width = ocr_data['width'][i]
                    height = ocr_data['height'][i]
                    
                    # Create a simple OCR result
                    ocr_result = type('OCRResult', (), {
                        'text': text,
                        'confidence': confidence,
                        'bounding_box': (x, y, width, height)
                    })()
                    results.append(ocr_result)
            
            logger.debug(f"Append mode OCR extracted {len(results)} text regions")
            return results
            
        except Exception as e:
            logger.error(f"Append mode OCR extraction failed: {e}")
            return []

    def _generate_output_path(self, input_path: Path, mode: str) -> Path:
        """Generate output path for processed image."""
        stem = input_path.stem
        suffix = input_path.suffix
        output_dir = input_path.parent
        
        return output_dir / f"{stem}_{mode}_modified{suffix}"

def create_append_text_replacer(pattern_matcher: PatternMatcher) -> AppendTextReplacer:
    """
    Factory function to create an AppendTextReplacer instance.
    
    Args:
        pattern_matcher: Pattern matcher for finding text replacements
        
    Returns:
        AppendTextReplacer instance
    """
    return AppendTextReplacer(pattern_matcher)