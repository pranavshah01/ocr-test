"""
Hybrid replace mode replacer combining OpenCV + Tesseract with existing mapping logic.
Replaces original text with new text, calculating font size based only on replacement text.
"""

import logging
import time
import cv2
import pytesseract
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from ...core.models import OCRResult, OCRMatch
from ..pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)

# Minimum font scale that roughly corresponds to ~6px height for cv2 putText in this project
MIN_FONT_SCALE = 0.3

class PreciseTextReplacer:
    """
    Hybrid replace mode replacer that combines OpenCV + Tesseract with existing mapping logic.
    Uses direct OpenCV text replacement while preserving our pattern matching system.
    """
    
    def __init__(self, pattern_matcher: PatternMatcher):
        """Initialize hybrid replace replacer."""
        self.pattern_matcher = pattern_matcher
        logger.info("PreciseTextReplacer initialized with hybrid OpenCV+mapping replace approach")
    
    def replace_text_in_image(self, image_path: Path, ocr_results: List[OCRResult], ocr_matches: List[OCRMatch]) -> Optional[Path]:
        """Replace text in image using hybrid OpenCV + mapping approach."""
        try:
            logger.info(f"HYBRID REPLACE: Processing {image_path} with {len(ocr_matches)} matches")
            
            # Convert to OpenCV format
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                logger.error(f"HYBRID REPLACE: Could not load image {image_path}")
                return None
            
            # Process each match using hybrid approach
            for i, match in enumerate(ocr_matches):
                logger.info(f"HYBRID REPLACE: Processing match {i}: '{match.ocr_result.text}' -> '{match.replacement_text}'")
                
                # Apply hybrid replace replacement
                cv_image = self._apply_hybrid_replace(cv_image, match, ocr_results)
            
            # Save result
            output_path = self._generate_output_path(image_path)
            cv2.imwrite(str(output_path), cv_image)
            
            logger.info(f"HYBRID REPLACE: Saved result to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"HYBRID REPLACE: Error processing {image_path}: {e}")
            return None
    
    def _apply_hybrid_replace(self, cv_image: np.ndarray, match: OCRMatch, all_ocr_results: List[OCRResult]) -> np.ndarray:
        """
        Apply hybrid replace replacement combining OpenCV + our mapping logic.
        
        Args:
            cv_image: OpenCV image to modify
            match: OCR match with bounding box and replacement text
            
        Returns:
            Modified OpenCV image
        """
        try:
            # Get bounding box and text from our mapping system
            bbox = match.ocr_result.bounding_box
            original_full_text = match.ocr_result.text
            replacement_text = match.replacement_text
            
            x, y, width, height = bbox
            
            logger.info(f" HYBRID REPLACE: Bbox: {bbox}, Full OCR: '{original_full_text}', Replacement: '{replacement_text}'")
            
            # Use our pattern matcher to find the exact pattern within the OCR text
            pattern_matches = self.pattern_matcher.find_matches_universal(original_full_text)
            if not pattern_matches:
                logger.warning(f" HYBRID REPLACE: No pattern found in '{original_full_text}'")
                return cv_image
            
            # Use the first match found
            first_match = pattern_matches[0]
            matched_pattern = first_match.matched_text
            logger.info(f" HYBRID REPLACE: Found pattern: '{matched_pattern}' (pattern: {first_match.pattern_name})")
            
            # Calculate wipe boundaries and store them in the match
            wipe_start_pos, wipe_end_pos = self._find_precise_pattern_boundaries(original_full_text, matched_pattern)
            pattern_rect = self._calculate_precise_pattern_rect(
                (x, y, width, height), original_full_text, wipe_start_pos, wipe_end_pos, cv_image
            )
            
            # Store wipe boundary information in the match
            match.wipe_boundaries = (wipe_start_pos, wipe_end_pos)
            match.calculated_text_boundary = pattern_rect
            match.wipe_area_info = {
                'matched_pattern': matched_pattern,
                'pattern_name': first_match.pattern_name,
                'original_text': original_full_text,
                'replacement_text': replacement_text,
                'wipe_start_char': wipe_start_pos,
                'wipe_end_char': wipe_end_pos,
                'wipe_area_pixels': pattern_rect,
                'processing_timestamp': time.time()
            }
            
            logger.info(f" HYBRID REPLACE: Stored wipe boundaries - chars: {wipe_start_pos}-{wipe_end_pos}, pixels: {pattern_rect}")
            
            # WIPE ONLY: Just clear the pattern area without replacement
            return self._wipe_pattern_area_only(cv_image, bbox, original_full_text, matched_pattern, all_ocr_results)
            
        except Exception as e:
            logger.error(f" HYBRID REPLACE: Error in hybrid replace: {e}")
            return cv_image
    
    def _draw_replaced_text(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int],
                           original_full_text: str, matched_pattern: str, replacement_text: str,
                           all_ocr_results: List[OCRResult]) -> np.ndarray:
        """
        Replace the matched pattern with replacement text, calculating font size based only on replacement text.
        
        Strategy:
        - Find the exact position of the matched pattern within the original text
        - Clear only the matched pattern area
        - Place replacement text in the cleared area, sized to fit optimally
        - Preserve any prefix and suffix text from the original
        """
        try:
            x, y, width, height = bbox
            img_h, img_w = cv_image.shape[:2]

            logger.info(f" REPLACE: Processing '{original_full_text}' -> replace '{matched_pattern}' with '{replacement_text}'")

            # Debug the text analysis
            self._debug_text_analysis(original_full_text, matched_pattern)

            # Find precise boundaries of the matched pattern
            wipe_start_pos, wipe_end_pos = self._find_precise_pattern_boundaries(original_full_text, matched_pattern)
            
            # Extract the parts: prefix (preserve) + pattern area (replace) + suffix (preserve)
            prefix_text = original_full_text[:wipe_start_pos] if wipe_start_pos > 0 else ""
            suffix_text = original_full_text[wipe_end_pos:] if wipe_end_pos < len(original_full_text) else ""
            
            # Calculate precise pixel boundaries for the pattern area
            pattern_rect = self._calculate_precise_pattern_rect(
                (x, y, width, height), original_full_text, wipe_start_pos, wipe_end_pos, cv_image
            )
            
            if pattern_rect is None:
                logger.warning(" REPLACE: Could not calculate precise pattern rect, using fallback")
                # Fallback to proportional calculation
                pattern_rect = self._calculate_fallback_pattern_rect((x, y, width, height), original_full_text, matched_pattern)

            # Clear the pattern area
            px, py, pw, ph = pattern_rect
            logger.info(f" REPLACE: Clearing pattern area: ({px}, {py}, {pw}, {ph})")
            
            # Enhanced clearing to prevent ghosting
            cv2.rectangle(cv_image, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
            # Additional pass for thorough clearing
            cv2.rectangle(cv_image, (px+1, py+1), (px + pw - 1, py + ph - 1), (255, 255, 255), -1)

            # Calculate optimal font size for replacement text in the pattern area
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            
            # Calculate font scale based ONLY on replacement text fitting in pattern area
            scale, text_size = self._compute_fitting_scale(
                replacement_text, pw - 4, ph - 4,  # Leave small padding
                base_scale=1.5, min_scale=MIN_FONT_SCALE, thickness=thickness, font=font
            )
            
            # Center replacement text in the pattern area
            text_x = px + (pw - text_size[0]) // 2
            text_y = py + (ph + text_size[1]) // 2  # Baseline positioning
            
            # Ensure text stays within bounds
            text_x = max(px + 2, min(text_x, px + pw - text_size[0] - 2))
            text_y = max(py + text_size[1] + 2, min(text_y, py + ph - 2))
            
            logger.info(f" REPLACE: Placing replacement text '{replacement_text}' at ({text_x}, {text_y}) with scale {scale:.2f}")
            
            # WIPE ONLY: Comment out replacement text drawing
            # self._draw_text_with_bg(cv_image, replacement_text, (text_x, text_y), scale, thickness, font, padding=2)
            logger.info(f" WIPE ONLY: Skipping replacement text drawing - only wiping pattern area")
            
            # Ensure prefix and suffix are preserved (redraw if necessary)
            if prefix_text.strip():
                logger.info(f" REPLACE: Ensuring prefix '{prefix_text}' is preserved")
                self._ensure_prefix_preserved(cv_image, (x, y, width, height), prefix_text, wipe_start_pos, original_full_text)
            
            if suffix_text.strip():
                logger.info(f" REPLACE: Ensuring suffix '{suffix_text}' is preserved")
                self._ensure_suffix_preserved(cv_image, (x, y, width, height), suffix_text, wipe_end_pos, original_full_text)
            
            return cv_image

        except Exception as e:
            logger.error(f" REPLACE: Error drawing replaced text: {e}")
            return cv_image
    
    def _wipe_pattern_area_only(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int],
                              original_full_text: str, matched_pattern: str,
                              all_ocr_results: List[OCRResult]) -> np.ndarray:
        """
        WIPE ONLY: Clear the pattern area character-by-character without drawing replacement text.
        Uses precise character-level OCR to wipe only the exact matching characters.
        
        Args:
            cv_image: OpenCV image to modify
            bbox: Bounding box of the text area
            original_full_text: Full original text
            matched_pattern: Pattern that was matched
            all_ocr_results: All OCR results for context
            
        Returns:
            Modified OpenCV image with pattern area wiped
        """
        try:
            x, y, width, height = bbox
            img_h, img_w = cv_image.shape[:2]

            logger.info(f" WIPE ONLY: Processing '{original_full_text}' -> wipe '{matched_pattern}'")

            # Find precise boundaries of the matched pattern
            wipe_start_pos, wipe_end_pos = self._find_precise_pattern_boundaries(original_full_text, matched_pattern)
            
            # Use character-level OCR for precise wiping
            pattern_rect = self._calculate_precise_pattern_rect_character_level(
                (x, y, width, height), original_full_text, wipe_start_pos, wipe_end_pos, cv_image, matched_pattern
            )
            
            if pattern_rect is None:
                logger.warning(" WIPE ONLY: Could not calculate precise pattern rect, using fallback")
                # Fallback to proportional calculation
                pattern_rect = self._calculate_fallback_pattern_rect((x, y, width, height), original_full_text, matched_pattern)

            # Clear the pattern area with white rectangle
            px, py, pw, ph = pattern_rect
            logger.info(f" WIPE ONLY: Clearing pattern area: ({px}, {py}, {pw}, {ph})")
            
            # Make wipe area taller to ensure full text coverage
            x, y, width, height = bbox
            wipe_height = max(ph, height)  # Use at least the full text height
            wipe_y = min(py, y)  # Start from the top of the text area
            
            # Enhanced clearing to prevent ghosting - make it more visible
            cv2.rectangle(cv_image, (px, wipe_y), (px + pw, wipe_y + wipe_height), (255, 255, 255), -1)
            # Additional passes for thorough clearing
            cv2.rectangle(cv_image, (px+1, wipe_y+1), (px + pw - 1, wipe_y + wipe_height - 1), (255, 255, 255), -1)
            cv2.rectangle(cv_image, (px+2, wipe_y+2), (px + pw - 2, wipe_y + wipe_height - 2), (255, 255, 255), -1)
            
            logger.info(f" WIPE ONLY: Enhanced wipe area: ({px}, {wipe_y}, {pw}, {wipe_height})")
            
            logger.info(f" WIPE ONLY: Successfully wiped pattern area '{matched_pattern}' from '{original_full_text}'")
            
            return cv_image

        except Exception as e:
            logger.error(f" WIPE ONLY: Error wiping pattern area: {e}")
            return cv_image
    
    def _find_precise_pattern_boundaries(self, full_text: str, matched_pattern: str) -> Tuple[int, int]:
        """
        Find precise start and end positions of the matched pattern.
        """
        try:
            # Find the matched pattern position
            pattern_start = full_text.find(matched_pattern)
            if pattern_start == -1:
                logger.warning(f" PATTERN BOUNDARIES: Pattern '{matched_pattern}' not found in '{full_text}'")
                return 0, len(full_text)
            
            pattern_end = pattern_start + len(matched_pattern)
            
            # Extract the parts for logging
            prefix_text = full_text[:pattern_start]
            pattern_text = full_text[pattern_start:pattern_end]
            suffix_text = full_text[pattern_end:]
            
            logger.info(f" PATTERN BOUNDARIES: Full text: '{full_text}'")
            logger.info(f" PATTERN BOUNDARIES: Pattern: '{matched_pattern}' at {pattern_start}-{pattern_end}")
            logger.info(f" PATTERN BOUNDARIES: PRESERVE prefix: '{prefix_text}'")
            logger.info(f" PATTERN BOUNDARIES: REPLACE pattern: '{pattern_text}'")
            logger.info(f" PATTERN BOUNDARIES: PRESERVE suffix: '{suffix_text}'")
            
            return pattern_start, pattern_end
            
        except Exception as e:
            logger.error(f"Failed to find precise pattern boundaries: {e}")
            return 0, len(full_text)
    
    def _calculate_precise_pattern_rect(self, bbox: Tuple[int, int, int, int], full_text: str, 
                                       pattern_start: int, pattern_end: int, cv_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate precise pixel rectangle for the pattern area based on character positions.
        """
        try:
            x, y, width, height = bbox
            
            if len(full_text) == 0:
                return None
            
            # Calculate character width proportionally
            char_width = width / len(full_text)
            
            # Calculate precise pixel positions for the pattern
            pattern_start_x = x + int(pattern_start * char_width)
            pattern_end_x = x + int(pattern_end * char_width)
            
            # Add minimal padding to ensure complete coverage
            padding_x = max(2, int(char_width * 0.1))
            pattern_start_x = max(x, pattern_start_x - padding_x)
            pattern_end_x = min(x + width, pattern_end_x + padding_x)
            
            # Calculate pattern rectangle dimensions
            pattern_width = max(10, pattern_end_x - pattern_start_x)
            pattern_height = height  # Use full height of original text
            
            # Ensure we don't exceed image bounds
            img_h, img_w = cv_image.shape[:2]
            pattern_width = min(pattern_width, img_w - pattern_start_x)
            pattern_height = min(pattern_height, img_h - y)
            
            logger.info(f" PATTERN RECT: Calculated pattern area: ({pattern_start_x}, {y}, {pattern_width}, {pattern_height})")
            logger.info(f" PATTERN RECT: Character positions {pattern_start}-{pattern_end} -> pixels {pattern_start_x}-{pattern_end_x}")
            
            return (pattern_start_x, y, pattern_width, pattern_height)
            
        except Exception as e:
            logger.error(f"Failed to calculate precise pattern rect: {e}")
            return None
    
    def _calculate_precise_pattern_rect_character_level(self, bbox: Tuple[int, int, int, int], full_text: str, 
                                                      pattern_start: int, pattern_end: int, cv_image: np.ndarray, matched_pattern: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate precise pixel rectangle for the pattern area using character-level OCR.
        Similar to wipe_text.py approach with pytesseract.image_to_boxes().
        """
        try:
            x, y, width, height = bbox
            img_h, img_w = cv_image.shape[:2]
            
            # Extract the text region from the image
            text_region = cv_image[y:y+height, x:x+width]
            
            # Get character-level bounding boxes using Tesseract
            try:
                char_boxes = pytesseract.image_to_boxes(text_region, lang="eng")
            except Exception as e:
                logger.warning(f" CHARACTER OCR: Failed to get character boxes: {e}")
                return None
            
            if not char_boxes.strip():
                logger.debug(" CHARACTER OCR: No character boxes found (this is normal for some text regions)")
                return None
            
            # Parse character boxes and build character list with coordinates
            chars = []
            coords = []
            
            for line in char_boxes.strip().split("\n"):
                try:
                    ch, x1, y1, x2, y2, _ = line.split()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Convert from Tesseract coordinates to OpenCV coordinates
                    # Tesseract: (0,0) at bottom-left, OpenCV: (0,0) at top-left
                    y1, y2 = height - y1, height - y2
                    
                    # Convert to absolute image coordinates
                    abs_x1 = x + x1
                    abs_y1 = y + y1
                    abs_x2 = x + x2
                    abs_y2 = y + y2
                    
                    chars.append(ch)
                    coords.append((abs_x1, abs_y1, abs_x2, abs_y2))  # top-left and bottom-right
                    
                except Exception as e:
                    logger.debug(f" CHARACTER OCR: Failed to parse line '{line}': {e}")
                    continue
            
            if not chars:
                logger.warning(" CHARACTER OCR: No valid characters found")
                return None
            
            # Build the full text from detected characters
            detected_text = "".join(chars)
            logger.debug(f" CHARACTER OCR: Detected text: '{detected_text}' (expected: '{full_text}')")
            
            # Find the pattern in the detected text
            pattern_match = None
            for i in range(len(detected_text) - len(matched_pattern) + 1):
                if detected_text[i:i+len(matched_pattern)] == matched_pattern:
                    pattern_match = (i, i + len(matched_pattern))
                    break
            
            if pattern_match is None:
                logger.debug(f" CHARACTER OCR: Pattern '{matched_pattern}' not found in detected text '{detected_text}' (this is normal for some text regions)")
                return None
            
            start_char, end_char = pattern_match
            
            # Calculate the bounding box for the pattern characters
            if start_char >= len(coords) or end_char > len(coords):
                logger.warning(f" CHARACTER OCR: Character indices out of range: {start_char}-{end_char}, coords: {len(coords)}")
                return None
            
            # Get coordinates of the first and last characters in the pattern
            first_char_coords = coords[start_char]
            last_char_coords = coords[end_char - 1]
            
            # Calculate pattern rectangle
            pattern_x1 = first_char_coords[0]  # leftmost x
            pattern_y1 = min(first_char_coords[1], last_char_coords[1])  # topmost y
            pattern_x2 = last_char_coords[2]   # rightmost x
            pattern_y2 = max(first_char_coords[3], last_char_coords[3])  # bottommost y
            
            # Ensure we don't exceed image bounds
            pattern_x1 = max(0, pattern_x1)
            pattern_y1 = max(0, pattern_y1)
            pattern_x2 = min(img_w, pattern_x2)
            pattern_y2 = min(img_h, pattern_y2)
            
            pattern_width = pattern_x2 - pattern_x1
            pattern_height = pattern_y2 - pattern_y1
            
            # Ensure minimum dimensions and use full text height for better visibility
            pattern_width = max(10, pattern_width)
            # Use the full text height to ensure complete coverage
            pattern_height = max(height, pattern_height)  # Use at least the full text height
            
            logger.info(f" CHARACTER OCR: Pattern '{matched_pattern}' at chars {start_char}-{end_char}")
            logger.info(f" CHARACTER OCR: Pattern rect: ({pattern_x1}, {pattern_y1}, {pattern_width}, {pattern_height})")
            
            return (pattern_x1, pattern_y1, pattern_width, pattern_height)
            
        except Exception as e:
            logger.error(f" CHARACTER OCR: Error in character-level calculation: {e}")
            return None
    
    def _calculate_fallback_pattern_rect(self, bbox: Tuple[int, int, int, int], full_text: str, matched_pattern: str) -> Tuple[int, int, int, int]:
        """
        Fallback calculation for pattern rectangle using proportional estimation.
        """
        x, y, width, height = bbox
        
        # Estimate pattern position proportionally
        pattern_pos = max(0, full_text.find(matched_pattern))
        pattern_len = len(matched_pattern)
        full_len = max(len(full_text), 1)
        
        # Calculate proportional positions
        start_ratio = pattern_pos / full_len
        end_ratio = min(1.0, (pattern_pos + pattern_len) / full_len)
        
        # Convert to pixel coordinates
        pattern_start_x = x + int(start_ratio * width)
        pattern_end_x = x + int(end_ratio * width)
        
        pattern_width = max(20, pattern_end_x - pattern_start_x)
        
        logger.info(f" FALLBACK RECT: Using fallback calculation: ({pattern_start_x}, {y}, {pattern_width}, {height})")
        
        return (pattern_start_x, y, pattern_width, height)
    
    def _ensure_prefix_preserved(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int], 
                               prefix_text: str, prefix_end_pos: int, full_text: str) -> None:
        """
        Ensure prefix text is preserved and visible.
        """
        try:
            if not prefix_text.strip():
                return
                
            x, y, width, height = bbox
            
            # Calculate prefix position
            char_width = width / len(full_text) if len(full_text) > 0 else 10
            prefix_width = int(prefix_end_pos * char_width)
            
            # Position prefix at the start of the original bbox
            prefix_x = x
            prefix_y = y + int(height * 0.7)  # Approximate baseline position
            
            # Use appropriate font size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(1.0, height / 25.0))
            thickness = 1
            
            logger.info(f" PREFIX PRESERVE: Drawing '{prefix_text}' at ({prefix_x}, {prefix_y})")
            
            # Draw prefix (it should already be there, but ensure it's clear)
            self._draw_text_with_bg(cv_image, prefix_text, (prefix_x, prefix_y), font_scale, thickness, font, padding=1)
            
        except Exception as e:
            logger.error(f"Failed to preserve prefix: {e}")
    
    def _ensure_suffix_preserved(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int], 
                               suffix_text: str, suffix_start_pos: int, full_text: str) -> None:
        """
        Ensure suffix text is preserved and visible.
        """
        try:
            if not suffix_text.strip():
                return
                
            x, y, width, height = bbox
            
            # Calculate suffix position
            char_width = width / len(full_text) if len(full_text) > 0 else 10
            suffix_x = x + int(suffix_start_pos * char_width)
            
            # Position suffix at the original text baseline
            suffix_y = y + int(height * 0.7)  # Approximate baseline position
            
            # Use appropriate font size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(1.0, height / 25.0))
            thickness = 1
            
            # Ensure suffix doesn't go outside image bounds
            img_h, img_w = cv_image.shape[:2]
            suffix_x = max(0, min(suffix_x, img_w - 50))
            suffix_y = max(15, min(suffix_y, img_h - 5))
            
            logger.info(f" SUFFIX PRESERVE: Drawing '{suffix_text}' at ({suffix_x}, {suffix_y})")
            
            # Draw suffix (redraw to ensure it's not accidentally wiped)
            self._draw_text_with_bg(cv_image, suffix_text, (suffix_x, suffix_y), font_scale, thickness, font, padding=1)
            
        except Exception as e:
            logger.error(f"Failed to preserve suffix: {e}")
    
    def _debug_text_analysis(self, full_text: str, matched_pattern: str) -> None:
        """Debug helper to analyze text parsing."""
        try:
            logger.info(f" DEBUG: Full text analysis:")
            logger.info(f" DEBUG: Full text: '{full_text}' (length: {len(full_text)})")
            logger.info(f" DEBUG: Matched pattern: '{matched_pattern}' (length: {len(matched_pattern)})")
            
            pattern_pos = full_text.find(matched_pattern)
            if pattern_pos != -1:
                before = full_text[:pattern_pos]
                after = full_text[pattern_pos + len(matched_pattern):]
                logger.info(f" DEBUG: Before pattern: '{before}'")
                logger.info(f" DEBUG: Pattern: '{matched_pattern}'")
                logger.info(f" DEBUG: After pattern: '{after}'")
                logger.info(f" DEBUG: Pattern position: {pattern_pos} to {pattern_pos + len(matched_pattern)}")
            else:
                logger.warning(f" DEBUG: Pattern '{matched_pattern}' not found in '{full_text}'")
                
        except Exception as e:
            logger.error(f"Debug analysis failed: {e}")

    def _compute_fitting_scale(self, text: str, max_width: int, max_height: int,
                               base_scale: float = 1.0, min_scale: float = 0.3,
                               thickness: int = 1, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> Tuple[float, Tuple[int, int]]:
        """Compute the largest font scale that fits text inside the given area."""
        scale = base_scale
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        while (size[0] > max_width or size[1] > max_height) and scale > min_scale:
            scale -= 0.05
            size = cv2.getTextSize(text, font, scale, thickness)[0]
        return max(scale, min_scale), size

    def _draw_text_with_bg(self, image: np.ndarray, text: str, origin: Tuple[int, int],
                            font_scale: float, thickness: int = 1,
                            font: int = cv2.FONT_HERSHEY_SIMPLEX,
                            padding: int = 2) -> None:
        """
        Draw text with a white background rectangle behind it for readability.
        """
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x0, y_baseline = origin
        
        top_left = (x0 - padding, y_baseline - text_size[1] - padding)
        bottom_right = (x0 + text_size[0] + padding, y_baseline + padding)
        
        h, w = image.shape[:2]
        tlx = max(0, min(top_left[0], w))
        tly = max(0, min(top_left[1], h))
        brx = max(0, min(bottom_right[0], w))
        bry = max(0, min(bottom_right[1], h))
        
        # Clear background
        cv2.rectangle(image, (tlx, tly), (brx, bry), (255, 255, 255), -1)
        
        # Draw text with anti-aliasing
        cv2.putText(image, text, (x0, y_baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        logger.debug(f" DRAW TEXT: '{text}' at ({x0}, {y_baseline}) with bg ({tlx}, {tly}) to ({brx}, {bry})")

    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path for processed image."""
        stem = input_path.stem
        suffix = input_path.suffix
        parent = input_path.parent
        
        return parent / f"{stem}_hybrid_replace{suffix}"

def create_precise_text_replacer(pattern_matcher: PatternMatcher) -> PreciseTextReplacer:
    """Factory function to create PreciseTextReplacer."""
    return PreciseTextReplacer(pattern_matcher)