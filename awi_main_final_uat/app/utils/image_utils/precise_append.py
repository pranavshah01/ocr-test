"""
Hybrid append mode replacer combining OpenCV + Tesseract with existing mapping logic.
Merges simple, direct OpenCV text replacement with our pattern matching system.
"""

import logging
import cv2
import pytesseract
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .ocr_models import OCRResult, OCRMatch
from .pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)

# Minimum font scale that roughly corresponds to ~6px height for cv2 putText in this project
MIN_FONT_SCALE = 0.3

class PreciseAppendReplacer:
    """
    Hybrid append mode replacer that combines OpenCV + Tesseract with existing mapping logic.
    Uses direct OpenCV text replacement while preserving our pattern matching system.
    """
    
    def __init__(self, pattern_matcher: PatternMatcher):
        """Initialize hybrid append replacer."""
        self.pattern_matcher = pattern_matcher
        logger.info(" PreciseAppendReplacer initialized with hybrid OpenCV+mapping approach")
    
    def replace_text_in_image(self, image_path: Path, ocr_results: List[OCRResult], ocr_matches: List[OCRMatch]) -> Optional[Path]:
        """Replace text in image using hybrid OpenCV + mapping approach."""
        try:
            logger.info(f" HYBRID APPEND: Processing {image_path} with {len(ocr_matches)} matches")
            
            # Convert to OpenCV format
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                logger.error(f" HYBRID APPEND: Could not load image {image_path}")
                return None
            
            # Process each match using hybrid approach
            for i, match in enumerate(ocr_matches):
                logger.info(f" HYBRID APPEND: Processing match {i}: '{match.ocr_result.text}' -> '{match.replacement_text}'")
                
                # Apply hybrid append replacement
                cv_image = self._apply_hybrid_append(cv_image, match, ocr_results)
            
            # Save result
            output_path = self._generate_output_path(image_path)
            cv2.imwrite(str(output_path), cv_image)
            
            logger.info(f" HYBRID APPEND: Saved result to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f" HYBRID APPEND: Error processing {image_path}: {e}")
            return None
    
    def _apply_hybrid_append(self, cv_image: np.ndarray, match: OCRMatch, all_ocr_results: List[OCRResult]) -> np.ndarray:
        """
        Apply hybrid append replacement combining OpenCV + our mapping logic.
        
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
            
            logger.info(f" HYBRID APPEND: Bbox: {bbox}, Full OCR: '{original_full_text}', Replacement: '{replacement_text}'")
            
            # Use our pattern matcher to find the pattern within the OCR text
            # In append mode we must handle unmapped patterns too, so allow all matches here
            raw_matches = self.pattern_matcher.find_all_pattern_matches(original_full_text)
            if not raw_matches:
                logger.warning(f" HYBRID APPEND: No pattern found in '{original_full_text}'")
                return cv_image
            
            # Use the first match found (pattern_name, matched_text, start_pos, end_pos)
            pattern_name, matched_pattern, _, _ = raw_matches[0]
            logger.info(f" HYBRID APPEND: Found pattern: '{matched_pattern}' (pattern: {pattern_name})")
            
            # Draw the appended text either side-by-side or top/down without erasing originals
            return self._draw_appended_text(cv_image, bbox, original_full_text, matched_pattern, replacement_text, all_ocr_results)
            
        except Exception as e:
            logger.error(f" HYBRID APPEND: Error in hybrid append: {e}")
            return cv_image
    
    def _apply_opencv_replacement(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int],
                                original_text: str, new_text: str) -> np.ndarray:
        """
        Apply OpenCV-based text replacement ensuring text fits inside the original bbox.
        """
        try:
            x, y, width, height = bbox
            logger.info(f" OPENCV REPLACE: Replacing '{original_text}' with '{new_text}' at bbox {bbox}")

            # Wipe original text area
            cv2.rectangle(cv_image, (x, y), (x + width, y + height), (255, 255, 255), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1

            # Start with a base scale and adjust down if it overflows
            font_scale = 1.0
            text_size = cv2.getTextSize(new_text, font, font_scale, thickness)[0]

            # Dynamically shrink font until it fits inside bbox (minimum font size 6)
            while (text_size[0] > width or text_size[1] > height) and font_scale > 0.3:
                font_scale -= 0.05
                text_size = cv2.getTextSize(new_text, font, font_scale, thickness)[0]

            text_width, text_height = text_size

            # Center text in the bounding box
            text_x = x + (width - text_width) // 2
            text_y = y + (height + text_height) // 2  # baseline alignment

            # Clip coordinates to image boundaries
            img_h, img_w = cv_image.shape[:2]
            text_x = max(0, min(text_x, img_w - text_width))
            text_y = max(text_height, min(text_y, img_h))

            cv2.putText(cv_image, new_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            logger.info(f" OPENCV REPLACE: Successfully placed text at ({text_x}, {text_y}), font_scale={font_scale}")
            return cv_image
        except Exception as e:
            logger.error(f" OPENCV REPLACE: Error in OpenCV replacement: {e}")
            return cv_image

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
        ENHANCED: More aggressive background clearing to prevent ghosting.
        """
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x0, y_baseline = origin
        
        # ENHANCED: Larger padding for more complete background clearing
        enhanced_padding = max(padding, 3)
        
        top_left = (x0 - enhanced_padding, y_baseline - text_size[1] - enhanced_padding)
        bottom_right = (x0 + text_size[0] + enhanced_padding, y_baseline + enhanced_padding)
        
        h, w = image.shape[:2]
        tlx = max(0, min(top_left[0], w))
        tly = max(0, min(top_left[1], h))
        brx = max(0, min(bottom_right[0], w))
        bry = max(0, min(bottom_right[1], h))
        
        # ENHANCED: Double-clear the background to ensure complete wiping
        # First pass: slightly larger area
        cv2.rectangle(image, (tlx-1, tly-1), (brx+1, bry+1), (255, 255, 255), -1)
        # Second pass: exact area
        cv2.rectangle(image, (tlx, tly), (brx, bry), (255, 255, 255), -1)
        
        # Draw text with enhanced anti-aliasing
        cv2.putText(image, text, (x0, y_baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        logger.debug(f" DRAW TEXT: '{text}' at ({x0}, {y_baseline}) with bg ({tlx}, {tly}) to ({brx}, {bry})")

    def _draw_text_plain(self, image: np.ndarray, text: str, origin: Tuple[int, int],
                         font_scale: float, thickness: int = 1,
                         font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        """Draw text without any background rectangle (used after area is wiped)."""
        cv2.putText(image, text, origin, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def _draw_appended_text(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int],
                            original_full_text: str, matched_pattern: str, replacement_text: str,
                            all_ocr_results: List[OCRResult]) -> np.ndarray:
        """
        Draw the appended text near the matched pattern without erasing any original content.

        Strategy:
        - Try side-by-side layout to the right of the matched pattern if it fits.
        - Otherwise, place the appended text below (or above if not enough space below),
          allowing slight usage of top/bottom margins.
        """
        try:
            x, y, width, height = bbox
            img_h, img_w = cv_image.shape[:2]

            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1

            # Try to precisely localize the matched substring within the bbox using Tesseract
            pattern_rect, next_token_left_x = self._estimate_pattern_rect_via_tesseract(
                cv_image, (x, y, width, height), matched_pattern
            )
            if pattern_rect is None:
                # Fallback to proportional estimate if OCR tokenization fails
                full_len = max(len(original_full_text), 1)
                start_idx = original_full_text.find(matched_pattern)
                if start_idx < 0:
                    logger.warning(" HYBRID APPEND: matched pattern not found in full text; fallback to below layout")
                    start_idx = 0
                end_idx = start_idx + len(matched_pattern)
                proportion_end = min(max(end_idx / full_len, 0.0), 1.0)
                side_by_side_x = int(x + proportion_end * width) + 5
                # Approximate anchor band spanning the matched substring horizontally
                proportion_start = min(max(start_idx / full_len, 0.0), 1.0)
                anchor_left = int(x + proportion_start * width)
                anchor_right = int(x + proportion_end * width)
                # Ensure a reasonable minimum width to allow legible placement
                min_band_w = max(10, int(0.2 * width))
                if anchor_right - anchor_left < min_band_w:
                    center = (anchor_left + anchor_right) // 2
                    anchor_left = max(x, center - min_band_w // 2)
                    anchor_right = min(x + width, anchor_left + min_band_w)
                anchor_band_x = anchor_left
                anchor_band_w = max(10, anchor_right - anchor_left)
            else:
                px, py, pw, ph = pattern_rect
                side_by_side_x = px + pw + 5
                # Use the exact matched width as the anchor band to horizontally align top/down
                anchor_band_x = px
                anchor_band_w = max(10, pw)

            # NEW: Exact character-level wipe using OCR boxes (like user's script)
            wiped_rect = self._wipe_exact_pattern_chars(
                cv_image, (x, y, width, height), [matched_pattern]
            )
            if wiped_rect is not None:
                token_77 = matched_pattern
                token_4022 = self._extract_4022_token(replacement_text) or replacement_text
                # Decide layout using the wiped rectangle width at minimum readable scale
                rx, ry, rw, rh = wiped_rect
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1
                min_scale = MIN_FONT_SCALE
                size1_min = cv2.getTextSize(token_77, font, min_scale, thickness)[0]
                size2_min = cv2.getTextSize(token_4022, font, min_scale, thickness)[0]
                can_horizontal_at_min = (size1_min[0] + 8 + size2_min[0] <= rw - 4) and (max(size1_min[1], size2_min[1]) <= rh)
                prefer_vertical_choice = not can_horizontal_at_min
                # Try to expand into surrounding whitespace before placement
                expanded_rect = self._expand_rect_into_whitespace(cv_image, wiped_rect, all_ocr_results, anchor_bbox=bbox)
                # Expose precise and reconstruction rects on the match for reporting
                try:
                    setattr(match, 'precise_bbox', (rx, ry, rw, rh))
                    setattr(match, 'reconstruction_bbox', expanded_rect)
                    setattr(match, 'reconstruction_reasoning', {
                        'reason': 'Character-level wipe + whitespace expansion',
                        'expanded_from': (rx, ry, rw, rh),
                        'expanded_to': expanded_rect
                    })
                except Exception:
                    pass
                placed = self._place_two_tokens_in_rect(cv_image, expanded_rect, token_77, token_4022, prefer_vertical=prefer_vertical_choice, anchor_limit_rect=bbox)
                if not placed:
                    joined = f"{token_77} {token_4022}".strip()
                    self._place_single_text_in_rect(cv_image, expanded_rect, joined)
                # Mark reconstructed
                try:
                    setattr(match, 'reconstructed', True)
                except Exception:
                    pass
                return cv_image

            # ENHANCED: Use precise wipe boundaries to avoid wiping suffix
            logger.info(f" PRECISE APPEND: Processing '{original_full_text}' -> pattern '{matched_pattern}' + replacement '{replacement_text}'")

            # Debug the text analysis
            self._debug_text_analysis(original_full_text, matched_pattern)

            # Find precise wipe boundaries (ONLY the matched pattern)
            wipe_start_pos, wipe_end_pos = self._find_precise_wipe_boundaries(original_full_text, matched_pattern)
            
            # Extract the parts: prefix (preserve) + wipe area + suffix (preserve)
            prefix_text = original_full_text[:wipe_start_pos] if wipe_start_pos > 0 else ""
            suffix_text = original_full_text[wipe_end_pos:] if wipe_end_pos < len(original_full_text) else ""
            
            # Calculate precise pixel boundaries for wiping
            precise_wipe_rect = self._calculate_precise_wipe_rect(
                (x, y, width, height), original_full_text, wipe_start_pos, wipe_end_pos, cv_image
            )
            
            if precise_wipe_rect is None:
                logger.warning(" PRECISE APPEND: Could not calculate precise wipe rect, using fallback")
                # Fallback to original method
                clear_rect = self._compute_clear_rect(
                    (x, y, width, height), pattern_rect, next_token_left_x, all_ocr_results,
                    cv_image.shape[:2], original_full_text, matched_pattern,
                )
                precise_wipe_rect = clear_rect

            # ULTRA-CONSERVATIVE WIPE: Only clear the exact pattern area
            wx, wy, ww, wh = precise_wipe_rect
            logger.info(f" PRECISE APPEND: Wiping ONLY pattern area: ({wx}, {wy}, {ww}, {wh})")
            logger.info(f" PRECISE APPEND: Original bbox: ({x}, {y}, {width}, {height})")
            
            # CONSERVATIVE: Single-pass clearing to avoid over-wiping
            cv2.rectangle(cv_image, (wx, wy), (wx + ww, wy + wh), (255, 255, 255), -1)
            
            # Additional clearing only if needed for ghosting
            if ww > 20 and wh > 15:  # Only for reasonably sized areas
                cv2.rectangle(cv_image, (wx+1, wy+1), (wx + ww - 1, wy + wh - 1), (255, 255, 255), -1)

            # Build tokens for vertical stacking
            token_77 = matched_pattern
            token_4022 = self._extract_4022_token(replacement_text) or replacement_text

            # VERTICAL STACKING: Place tokens in the wiped area
            logger.info(f" PRECISE APPEND: Placing tokens '{token_77}' and '{token_4022}' vertically in wiped area")
            
            expanded_precise_rect = self._expand_rect_into_whitespace(cv_image, precise_wipe_rect, all_ocr_results, anchor_bbox=bbox)
            # Attach rectangles for reporting
            try:
                setattr(match, 'precise_bbox', (wx, wy, ww, wh))
                setattr(match, 'reconstruction_bbox', expanded_precise_rect)
                setattr(match, 'reconstruction_reasoning', {
                    'reason': 'Precise proportional wipe + whitespace expansion',
                    'expanded_from': (wx, wy, ww, wh),
                    'expanded_to': expanded_precise_rect
                })
            except Exception:
                pass
            placed = self._place_two_tokens_in_rect(cv_image, expanded_precise_rect, token_77, token_4022, prefer_vertical=False, anchor_limit_rect=bbox)
            if not placed:
                logger.warning(f" PRECISE APPEND: Two-token placement failed, trying single text fallback")
                joined = f"{token_77} {token_4022}".strip()
                self._place_single_text_in_rect(cv_image, expanded_precise_rect, joined)
            try:
                setattr(match, 'reconstructed', True)
            except Exception:
                pass
            else:
                logger.info(f" PRECISE APPEND: Successfully placed two tokens vertically")

            # PRESERVE SUFFIX: Always redraw suffix to ensure it's visible
            if suffix_text.strip():
                logger.info(f" PRECISE APPEND: Redrawing suffix '{suffix_text}' to ensure preservation")
                self._redraw_suffix_if_needed(cv_image, (x, y, width, height), suffix_text, wipe_end_pos, original_full_text)
            else:
                logger.info(f" PRECISE APPEND: No suffix to preserve")
            
            return cv_image

        except Exception as e:
            logger.error(f" HYBRID APPEND: Error drawing appended text: {e}")
            return cv_image
    
    def _find_precise_wipe_boundaries(self, full_text: str, matched_pattern: str) -> Tuple[int, int]:
        """
        Find precise start and end positions for wiping.
        ONLY wipe the exact matched pattern, preserve everything else.
        """
        try:
            # Find the matched pattern position
            pattern_start = full_text.find(matched_pattern)
            if pattern_start == -1:
                logger.warning(f" WIPE BOUNDARIES: Pattern '{matched_pattern}' not found in '{full_text}'")
                return 0, len(full_text)
            
            pattern_end = pattern_start + len(matched_pattern)
            
            # CRITICAL: Only wipe the exact pattern, nothing more
            wipe_start = pattern_start
            wipe_end = pattern_end
            
            # Extract the parts for logging
            prefix_text = full_text[:wipe_start]
            pattern_text = full_text[wipe_start:wipe_end]
            suffix_text = full_text[wipe_end:]
            
            logger.info(f" WIPE BOUNDARIES: Full text: '{full_text}'")
            logger.info(f" WIPE BOUNDARIES: Pattern: '{matched_pattern}' at {pattern_start}-{pattern_end}")
            logger.info(f" WIPE BOUNDARIES: PRESERVE prefix: '{prefix_text}'")
            logger.info(f" WIPE BOUNDARIES: WIPE pattern: '{pattern_text}'")
            logger.info(f" WIPE BOUNDARIES: PRESERVE suffix: '{suffix_text}'")
            logger.info(f" WIPE BOUNDARIES: Wipe boundaries: {wipe_start} to {wipe_end}")
            
            return wipe_start, wipe_end
            
        except Exception as e:
            logger.error(f"Failed to find precise wipe boundaries: {e}")
            return 0, len(full_text)
    
    def _calculate_precise_wipe_rect(self, bbox: Tuple[int, int, int, int], full_text: str, 
                                   wipe_start: int, wipe_end: int, cv_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate precise pixel rectangle for wiping based on character positions.
        CONSERVATIVE: Only wipe the exact pattern area with minimal padding.
        """
        try:
            x, y, width, height = bbox
            
            if len(full_text) == 0:
                return None
            
            # Calculate character width more precisely
            char_width = width / len(full_text)
            
            # Calculate precise pixel positions for ONLY the pattern
            wipe_start_x = x + int(wipe_start * char_width)
            wipe_end_x = x + int(wipe_end * char_width)
            
            # CONSERVATIVE: Add minimal padding to ensure complete clearing but not too much
            padding_x = max(2, int(char_width * 0.1))  # Very small horizontal padding
            wipe_start_x = max(x, wipe_start_x - padding_x)
            wipe_end_x = min(x + width, wipe_end_x + padding_x)
            
            # Calculate wipe rectangle - be conservative with dimensions
            wipe_width = max(10, wipe_end_x - wipe_start_x)
            
            # CONSERVATIVE: Only add minimal vertical padding for stacking
            vertical_padding = max(8, int(height * 0.2))  # 20% of height for stacking
            wipe_height = height + vertical_padding
            wipe_y = max(0, y - 4)  # Minimal top padding
            
            # Ensure we don't exceed image bounds
            img_h, img_w = cv_image.shape[:2]
            wipe_height = min(wipe_height, img_h - wipe_y)
            wipe_width = min(wipe_width, img_w - wipe_start_x)
            
            logger.info(f" WIPE RECT: Calculated precise wipe area: ({wipe_start_x}, {wipe_y}, {wipe_width}, {wipe_height})")
            logger.info(f" WIPE RECT: Character positions {wipe_start}-{wipe_end} -> pixels {wipe_start_x}-{wipe_end_x}")
            logger.info(f" WIPE RECT: Original bbox: ({x}, {y}, {width}, {height})")
            
            return (wipe_start_x, wipe_y, wipe_width, wipe_height)
            
        except Exception as e:
            logger.error(f"Failed to calculate precise wipe rect: {e}")
            return None
    
    def _redraw_suffix_if_needed(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int], 
                               suffix_text: str, suffix_start_pos: int, full_text: str) -> None:
        """
        Redraw suffix text if it was accidentally wiped.
        ENHANCED: More robust suffix preservation.
        """
        try:
            if not suffix_text.strip():
                logger.info(" SUFFIX REDRAW: No suffix text to redraw")
                return
                
            x, y, width, height = bbox
            
            # Calculate suffix position more precisely
            char_width = width / len(full_text) if len(full_text) > 0 else 10
            suffix_x = x + int(suffix_start_pos * char_width)
            
            # Position suffix at the original text baseline
            suffix_y = y + int(height * 0.7)  # Approximate baseline position
            
            # Use appropriate font size based on original text height
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(1.0, height / 25.0))  # Scale based on text height
            thickness = 1
            
            # Ensure suffix doesn't go outside image bounds
            img_h, img_w = cv_image.shape[:2]
            suffix_x = max(0, min(suffix_x, img_w - 50))  # Leave some margin
            suffix_y = max(15, min(suffix_y, img_h - 5))
            
            logger.info(f" SUFFIX REDRAW: Drawing '{suffix_text}' at ({suffix_x}, {suffix_y}) with scale {font_scale}")
            
            # Draw suffix with enhanced background clearing
            self._draw_text_with_bg(cv_image, suffix_text, (suffix_x, suffix_y), font_scale, thickness, font, padding=3)
            
            logger.info(f" SUFFIX REDRAW: Successfully drew suffix '{suffix_text}'")
            
        except Exception as e:
            logger.error(f"Failed to redraw suffix: {e}")
    
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

    def _wipe_exact_pattern_chars(self, cv_image: np.ndarray, region_bbox: Tuple[int, int, int, int],
                                  wipe_texts: List[str]) -> Optional[Tuple[int, int, int, int]]:
        """
        Wipe exactly the characters that match any of the given texts using
        pytesseract.image_to_boxes, similar to the user's reference script.
        Returns the union rectangle of the wiped area in absolute image coords.
        """
        try:
            x, y, w, h = region_bbox
            crop = cv_image[y:y+h, x:x+w]
            if crop is None or crop.size == 0:
                return None

            boxes_str = pytesseract.image_to_boxes(crop, lang="eng")
            if not boxes_str:
                return None

            crop_h = crop.shape[0]
            chars: List[str] = []
            coords: List[Tuple[int, int, int, int]] = []
            for line in boxes_str.strip().split("\n"):
                parts = line.split()
                if len(parts) < 6:
                    continue
                ch, x1, y1, x2, y2, _ = parts[:6]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Convert to OpenCV coords for the crop
                top = crop_h - y2
                bottom = crop_h - y1
                chars.append(ch)
                coords.append((x1, top, x2, bottom))

            if not chars:
                return None

            full_text = "".join(chars)

            # Prepare regex patterns (hyphen or en-dash)
            patterns = [re.compile(re.escape(t).replace(r"\-", "[-–]"), re.IGNORECASE) for t in wipe_texts]

            union_left, union_top, union_right, union_bottom = None, None, None, None
            pad = 1

            for pat in patterns:
                m = pat.search(full_text)
                if not m:
                    continue
                start, end = m.span()
                for i in range(start, end):
                    cx1, cy1, cx2, cy2 = coords[i]
                    # Expand slightly for cleanliness
                    rx1 = x + max(0, cx1 - pad)
                    ry1 = y + max(0, cy1 - pad)
                    rx2 = x + min(w, cx2 + pad)
                    ry2 = y + min(h, cy2 + pad)
                    cv2.rectangle(cv_image, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
                    # Update union
                    union_left = rx1 if union_left is None else min(union_left, rx1)
                    union_top = ry1 if union_top is None else min(union_top, ry1)
                    union_right = rx2 if union_right is None else max(union_right, rx2)
                    union_bottom = ry2 if union_bottom is None else max(union_bottom, ry2)

            if union_left is None:
                return None
            return (union_left, union_top, max(1, union_right - union_left), max(1, union_bottom - union_top))
        except Exception as e:
            logger.error(f"Exact pattern wipe failed: {e}")
            return None
    
    def _shrink_right_width_to_avoid_overlap(self, anchor_bbox: Tuple[int, int, int, int], start_x: int,
                                             all_ocr_results: List[OCRResult], img_shape: Tuple[int, int]) -> int:
        """Compute safe width to the right of anchor bbox without intersecting other OCR boxes."""
        x, y, width, height = anchor_bbox
        img_h, img_w = img_shape
        max_right = min(img_w, x + width)
        safe_right = max_right
        for res in all_ocr_results:
            bx, by, bw, bh = res.bounding_box
            horizontally_overlaps = not (by + bh < y or by > y + height)
            if horizontally_overlaps and bx >= start_x:
                safe_right = min(safe_right, bx - 2)
        return max(0, safe_right - start_x)

    def _shrink_vertical_band_to_avoid_overlap(self, anchor_bbox: Tuple[int, int, int, int],
                                               band_bbox: Tuple[int, int, int, int],
                                               all_ocr_results: List[OCRResult], img_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Given a vertical band (x, top, width, height) aligned to anchor bbox width,
        shrink its height so it does not intersect other OCR boxes.
        Returns (new_top, new_height).
        """
        x_anchor, y_anchor, w_anchor, h_anchor = anchor_bbox
        x, top, width, height = band_bbox
        img_h, img_w = img_shape
        bottom = min(img_h - 1, top + height)
        safe_top, safe_bottom = top, bottom
        for res in all_ocr_results:
            bx, by, bw, bh = res.bounding_box
            horizontal_overlap = not (bx + bw < x or bx > x + width)
            if horizontal_overlap:
                if by <= safe_bottom and by + bh >= safe_top:
                    if by < y_anchor:
                        safe_top = max(safe_top, by + bh + 2)
                    else:
                        safe_bottom = min(safe_bottom, by - 2)
        new_height = max(0, safe_bottom - safe_top)
        return safe_top, new_height

    def _estimate_pattern_rect_via_tesseract(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                                             pattern_text: str) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
        """
        Use Tesseract TSV to estimate the rectangle of the given pattern text within bbox.
        Returns (pattern_rect, next_token_left_x). If not found, both may be None.
        """
        try:
            x, y, w, h = bbox
            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                return None, None
            tsv = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DATAFRAME)
            if tsv is None or len(tsv) == 0:
                return None, None
            # Normalize pattern
            target = pattern_text.strip()
            candidates = tsv.dropna(subset=['text'])
            # Build full line strings and track token positions to find a contiguous match
            best_rect = None
            next_left_x = None
            for line_num in candidates['line_num'].unique():
                line_df = candidates[candidates['line_num'] == line_num]
                line_text = ' '.join([str(t) for t in line_df['text'] if isinstance(t, str)])
                if target in line_text:
                    # Find first token that starts this sequence by cumulative matching
                    cumulative = ''
                    start_token_idx = None
                    end_token_idx = None
                    tokens = list(line_df[['text','left','top','width','height']].itertuples(index=False, name=None))
                    for i, (tok, l, t, ww, hh) in enumerate(tokens):
                        if not isinstance(tok, str):
                            continue
                        if target.startswith((cumulative + tok).strip()):
                            if start_token_idx is None:
                                start_token_idx = i
                            cumulative = (cumulative + ' ' + tok).strip()
                            if cumulative == target:
                                end_token_idx = i
                                break
                    if start_token_idx is not None and end_token_idx is not None:
                        l0, t0, w0, h0 = tokens[start_token_idx][1:]
                        l1, t1, w1, h1 = tokens[end_token_idx][1:]
                        rect = (x + int(l0), y + int(min(t0, t1)), int((l1 + w1) - l0), int(max(h0, h1)))
                        # next token's left
                        if end_token_idx + 1 < len(tokens):
                            next_left_x = x + int(tokens[end_token_idx + 1][1])
                        best_rect = rect
                        break
            return best_rect, next_left_x
        except Exception:
            return None, None

    def _parse_part_numbers(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (77-..., 4022-...) tokens if both are present, else (None, None)."""
        try:
            patt77 = re.search(r"(77-[0-9A-Za-z\-]+)", text)
            patt4022 = re.search(r"(4022-[0-9A-Za-z\-]+)", text)
            if patt77 and patt4022:
                return patt77.group(1), patt4022.group(1)
            return None, None
        except Exception:
            return None, None

    def _extract_4022_token(self, text: str) -> Optional[str]:
        try:
            m = re.search(r"(4022-[0-9A-Za-z\-]+)", text)
            return m.group(1) if m else None
        except Exception:
            return None

    def _compute_clear_rect(self, anchor_bbox: Tuple[int, int, int, int],
                            pattern_rect: Optional[Tuple[int, int, int, int]],
                            next_token_left_x: Optional[int],
                            all_ocr_results: List[OCRResult],
                            img_shape: Tuple[int, int],
                            original_full_text: str,
                            matched_pattern: str) -> Tuple[int, int, int, int]:
        """
        Compute a safe rectangle to clear for replacement around the matched pattern.
        ENHANCED: Provides more aggressive clearing to prevent ghosting and supports vertical stacking.
        """
        x, y, w, h = anchor_bbox
        img_h, img_w = img_shape
        
        logger.info(f" CLEAR RECT: Computing clear area for pattern '{matched_pattern}' in bbox {anchor_bbox}")
        
        if pattern_rect is None:
            # Estimate exact horizontal span using character proportions inside bbox
            full_len = max(len(original_full_text), 1)
            start_idx = max(0, original_full_text.find(matched_pattern))
            end_idx = min(full_len, start_idx + len(matched_pattern))
            proportion_start = min(max(start_idx / full_len, 0.0), 1.0)
            proportion_end = min(max(end_idx / full_len, 0.0), 1.0)
            est_left = x + int(proportion_start * w)
            est_right = x + int(proportion_end * w)
            
            # ENHANCED: Use more aggressive clearing for vertical stacking
            # Expand the clear area to accommodate both original and appended text
            clear_width = max(int(w * 0.8), est_right - est_left + 20)  # Ensure enough width
            clear_height = max(int(h * 1.2), h + 15)  # Extra height for stacking
            
            cx = max(x, est_left - 5)  # Small left padding
            cy = max(0, y - 5)  # Small top padding for stacking
            cw = min(clear_width, x + w - cx + 10)  # Allow slight overflow to right
            ch = min(clear_height, img_h - cy)  # Use available height
            
            logger.info(f" CLEAR RECT: Proportional estimate - left: {est_left}, right: {est_right}")
        else:
            px, py, pw, ph = pattern_rect
            
            # ENHANCED: More aggressive clearing for complete wipe
            # Expand horizontally to ensure complete text removal
            pad_x = max(5, int(pw * 0.15))  # Larger horizontal padding
            left = max(x, px - pad_x)
            right = min(x + w + 15, px + pw + pad_x)  # Allow slight overflow
            
            # Don't cross next token boundary if detected
            if next_token_left_x is not None:
                right = min(right, next_token_left_x - 3)
            
            # ENHANCED: Expand vertically for stacking layout
            pad_y = max(8, int(ph * 0.6))  # Larger vertical padding
            top = max(0, py - pad_y)
            bottom = min(img_h, py + ph + pad_y + 20)  # Extra space below for appended text
            
            cx, cy, cw, ch = left, top, max(10, right - left), max(10, bottom - top)
            
            logger.info(f" CLEAR RECT: Pattern-based - px: {px}, py: {py}, pw: {pw}, ph: {ph}")

        # ENHANCED: More conservative overlap avoidance - only avoid critical overlaps
        # Allow some overlap with distant text to ensure complete clearing
        for res in all_ocr_results:
            bx, by, bw, bh = res.bounding_box
            if (bx, by, bw, bh) == anchor_bbox:
                continue
                
            # Only avoid overlaps with text that's very close (within 10 pixels)
            if self._rects_intersect((cx, cy, cw, ch), (bx, by, bw, bh)):
                distance_x = min(abs(bx - (cx + cw)), abs((bx + bw) - cx))
                distance_y = min(abs(by - (cy + ch)), abs((by + bh) - cy))
                
                # Only trim if the overlap is significant and close
                if distance_x < 10 or distance_y < 10:
                    if bx > cx + cw/2:  # Other text is to the right
                        cw = max(10, bx - 3 - cx)
                    elif bx + bw < cx + cw/2:  # Other text is to the left
                        new_left = bx + bw + 3
                        cw = max(10, (cx + cw) - new_left)
                        cx = new_left
                    
                    if by > cy + ch/2:  # Other text is below
                        ch = max(10, by - 3 - cy)
                    elif by + bh < cy + ch/2:  # Other text is above
                        new_top = by + bh + 3
                        ch = max(10, (cy + ch) - new_top)
                        cy = new_top

        # Final bounds checking
        cx = max(0, min(cx, img_w - 10))
        cy = max(0, min(cy, img_h - 10))
        cw = max(10, min(cw, img_w - cx))
        ch = max(10, min(ch, img_h - cy))
        
        logger.info(f" CLEAR RECT: Final clear area: ({cx}, {cy}, {cw}, {ch})")
        return cx, cy, cw, ch

    def _rects_intersect(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    def _expand_rect_into_whitespace(self, image: np.ndarray, rect: Tuple[int, int, int, int],
                                     all_ocr_results: List[OCRResult], anchor_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Expand the placement rectangle into nearby whitespace without colliding with other OCR boxes or leaving the anchor band.
        We grow left/right/up/down while pixels are mostly white and we don't intersect other OCR regions. Conservative limits apply.
        """
        x, y, w, h = rect
        img_h, img_w = image.shape[:2]
        # Strict bounds: prefer detected red box surrounding anchor, else fallback to anchor bbox
        rb = self._detect_red_box_bounds(image, anchor_bbox)
        ax, ay, aw, ah = (rb if rb is not None else anchor_bbox)
        bound_left = max(0, ax)
        bound_top = max(0, ay)
        bound_right = min(img_w, ax + aw)
        bound_bottom = min(img_h, ay + ah)

        def is_white_run(px1: int, py1: int, px2: int, py2: int, threshold: int = 245) -> bool:
            sub = image[max(0, py1):min(img_h, py2), max(0, px1):min(img_w, px2)]
            if sub.size == 0:
                return False
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            white_ratio = np.mean(gray >= threshold)
            return white_ratio > 0.95

        # Compute forbidden regions from other OCR boxes
        forbidden: List[Tuple[int, int, int, int]] = []
        for res in all_ocr_results:
            fb = res.bounding_box
            if fb == anchor_bbox:
                continue
            forbidden.append(fb)

        # Grow horizontally to the right
        grow_step = max(2, w // 10)
        max_right = min(bound_right, x + w + 120)  # do not cross red box
        right = x + w
        while right + grow_step <= max_right and is_white_run(right, y, right + grow_step, y + h):
            candidate = (x, y, (right + grow_step) - x, h)
            if not any(self._rects_intersect(candidate, fb) for fb in forbidden):
                right += grow_step
            else:
                break

        # Grow horizontally to the left
        max_left = max(bound_left, x - 120)
        left = x
        while left - grow_step >= max_left and is_white_run(left - grow_step, y, left, y + h):
            candidate = (left - grow_step, y, right - (left - grow_step), h)
            if not any(self._rects_intersect(candidate, fb) for fb in forbidden):
                left -= grow_step
            else:
                break

        # Grow vertically up and down
        grow_step_v = max(2, h // 10)
        max_top = max(bound_top, y - 120)
        top = y
        while top - grow_step_v >= max_top and is_white_run(left, top - grow_step_v, right, top):
            candidate = (left, top - grow_step_v, right - left, h + (top - (top - grow_step_v)))
            if not any(self._rects_intersect(candidate, fb) for fb in forbidden):
                top -= grow_step_v
            else:
                break

        max_bottom = min(bound_bottom, y + h + 120)
        bottom = y + h
        while bottom + grow_step_v <= max_bottom and is_white_run(left, bottom, right, bottom + grow_step_v):
            candidate = (left, top, right - left, (bottom + grow_step_v) - top)
            if not any(self._rects_intersect(candidate, fb) for fb in forbidden):
                bottom += grow_step_v
            else:
                break

        return (left, top, max(1, right - left), max(1, bottom - top))

    def _detect_red_box_bounds(self, image: np.ndarray, anchor_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Detect the red annotation rectangle surrounding the anchor area.
        Returns bounding rect of the closest red box if found, else None.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # red in two ranges
            lower1 = np.array([0, 120, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 100])
            upper2 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            ax, ay, aw, ah = anchor_bbox
            anchor_center = (ax + aw // 2, ay + ah // 2)
            candidates: List[Tuple[int, int, int, int, float]] = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Heuristic: prefer wide thin rectangles
                if w < 30 or h < 10:
                    continue
                # Does this rect contain/overlap anchor center?
                if not (x <= anchor_center[0] <= x + w and y <= anchor_center[1] <= y + h):
                    continue
                # score: distance from center to rect edges (smaller is tighter)
                dx = min(anchor_center[0] - x, (x + w) - anchor_center[0])
                dy = min(anchor_center[1] - y, (y + h) - anchor_center[1])
                score = dx + dy
                candidates.append((x, y, w, h, score))
            if not candidates:
                return None
            candidates.sort(key=lambda t: t[4])
            x, y, w, h, _ = candidates[0]
            return (x, y, w, h)
        except Exception:
            return None

    def _place_single_text_in_rect(self, image: np.ndarray, rect: Tuple[int, int, int, int], text: str) -> None:
        """
        Place single text in rectangle with enhanced clearing and positioning.
        ENHANCED: Better font scaling and positioning for single text fallback.
        """
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        logger.info(f" SINGLE TEXT: Placing '{text}' in rect ({x}, {y}, {w}, {h})")
        
        # ENHANCED: Better font scaling with higher minimum size
        scale, size = self._compute_fitting_scale(text, w - 6, h - 6, base_scale=1.2, min_scale=0.4, thickness=thickness, font=font)
        
        # Center text in rectangle
        tx = x + (w - size[0]) // 2
        ty = y + (h + size[1]) // 2
        
        # Ensure text stays within bounds
        tx = max(x + 2, min(tx, x + w - size[0] - 2))
        ty = max(y + size[1] + 2, min(ty, y + h - 2))
        
        logger.info(f" SINGLE TEXT: Positioned at ({tx}, {ty}) with scale {scale:.2f}")
        
        # Draw with enhanced background clearing
        self._draw_text_with_bg(image, text, (tx, ty), scale, thickness, font, padding=4)

    def _place_two_tokens_in_rect(self, image: np.ndarray, rect: Tuple[int, int, int, int], t1: str, t2: str, prefer_vertical: bool = False, anchor_limit_rect: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Place two tokens in the given rectangle with enhanced vertical stacking preference.
        ENHANCED: Always prefer vertical stacking as per user requirements.
        """
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        logger.info(f" PLACE TOKENS: Placing '{t1}' and '{t2}' in rect ({x}, {y}, {w}, {h})")

        padding = 2
        gap_min = max(2, int(h * 0.04))

        def measure(text: str, scale: float) -> Tuple[int, int]:
            size = cv2.getTextSize(text, font, scale, thickness)[0]
            return size[0], size[1]

        # Maximise shared scale for 2-line vertical layout
        def max_scale_vertical() -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
            # Allow independent per-line scaling to maximize usage
            avail_w = w - 2 * padding
            avail_h = h - 2 * padding

            def search_scale_width_only(text: str) -> float:
                lo, hi = MIN_FONT_SCALE, 3.0
                best = MIN_FONT_SCALE
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    tw, th = measure(text, mid)
                    if tw <= avail_w and th <= avail_h:  # width bound dominant; height coarse-checked
                        best = mid
                        lo = mid
                    else:
                        hi = mid
                return best

            s1 = search_scale_width_only(t1)
            s2 = search_scale_width_only(t2)

            w1, h1 = measure(t1, s1)
            w2, h2 = measure(t2, s2)

            total_needed = h1 + gap_min + h2
            if total_needed > avail_h and (h1 + h2) > 0:
                factor = max(0.1, (avail_h - gap_min) / (h1 + h2))
                s1 *= factor
                s2 *= factor
                w1, h1 = measure(t1, s1)
                w2, h2 = measure(t2, s2)

            # Use a common scale so both lines look identical
            common = min(s1, s2)
            w1, h1 = measure(t1, common)
            w2, h2 = measure(t2, common)
            return common, (w1, h1), (w2, h2)

        # Maximise scale for single-line horizontal layout with joined text
        def max_scale_horizontal() -> Tuple[float, Tuple[int, int], str]:
            joined = f"{t1} {t2}"
            low, high = MIN_FONT_SCALE, 3.0
            best = MIN_FONT_SCALE
            best_sz = (0, 0)
            for _ in range(20):
                mid = (low + high) / 2.0
                wj, hj = measure(joined, mid)
                if wj <= w - 2 * padding and hj <= h - 2 * padding:
                    best = mid
                    best_sz = (wj, hj)
                    low = mid
                else:
                    high = mid
            return best, best_sz, joined

        v_scale, v_sz1, v_sz2 = max_scale_vertical()
        h_scale, h_sz, joined = max_scale_horizontal()

        # Heuristic: favor vertical when the rect is relatively tall; otherwise compare scales
        aspect_ratio = h / max(1, w)
        vertical_bias = aspect_ratio >= 0.22
        if prefer_vertical or vertical_bias:
            choose_vertical = not (h_scale >= v_scale * 1.25)
        else:
            choose_vertical = not (h_scale > v_scale * 1.05)

        if choose_vertical and max(v_sz1[0], v_sz2[0]) > 0:
            # Center-align two-line layout within the rect
            total_h = v_sz1[1] + gap_min + v_sz2[1]
            start_y = y + max(0, (h - total_h) // 2)
            ty1 = start_y + v_sz1[1]
            ty2 = ty1 + gap_min + v_sz2[1]
            tx1 = x + (w - v_sz1[0]) // 2
            tx2 = x + (w - v_sz2[0]) // 2

            tx1 = max(x + padding, min(tx1, x + w - v_sz1[0] - padding))
            tx2 = max(x + padding, min(tx2, x + w - v_sz2[0] - padding))
            ty1 = max(y + v_sz1[1] + 1, min(ty1, y + h - padding))
            ty2 = max(y + v_sz2[1] + 1, min(ty2, y + h - padding))

            logger.info(f" VERTICAL STACK (common font): scale={v_scale:.2f}, L1 at ({tx1},{ty1}), L2 at ({tx2},{ty2})")
            # Ensure the entire rect is white once; avoid per-line bg overlap
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
            self._draw_text_plain(image, t1, (tx1, ty1), v_scale, thickness, font)
            self._draw_text_plain(image, t2, (tx2, ty2), v_scale, thickness, font)
            # Attach reasoning for report exporters if upstream captures it
            reasoning = {
                'algorithm': 'Rect-fit (binary search)',
                'decision': 'vertical-two-line',
                'rect_used': {'x': x, 'y': y, 'w': w, 'h': h},
                'scales': {'vertical_common': round(v_scale, 4)},
                'layout_params': {'gap': gap_min, 'padding': padding}
            }
            try:
                # best-effort: store on image as attribute for later harvesting if pipeline supports
                setattr(self, '_last_reasoning', reasoning)
            except Exception:
                pass
            return True

        if h_sz[0] > 0:
            # Side-by-side must remain strictly within rect; align vertically centered
            # Slightly lower the font if text nearly touches the edges for visual comfort
            comfort_ratio = 0.96
            if h_sz[0] > (w - 2 * padding) * comfort_ratio:
                adjusted_scale = h_scale * comfort_ratio
                h_sz = (cv2.getTextSize(f"{t1} {t2}", font, adjusted_scale, thickness)[0])
                h_scale = adjusted_scale
            ty = y + (h + h_sz[1]) // 2
            tx = x + (w - h_sz[0]) // 2
            tx = max(x + padding, min(tx, x + w - h_sz[0] - padding))
            ty = max(y + h_sz[1] + padding, min(ty, y + h - padding))
            logger.info(f" HORIZONTAL (max font): scale={h_scale:.2f}, at ({tx},{ty})")
            self._draw_text_with_bg(image, joined, (tx, ty), h_scale, thickness, font, padding=3)
            reasoning = {
                'algorithm': 'Rect-fit (binary search)',
                'decision': 'horizontal-single-line',
                'rect_used': {'x': x, 'y': y, 'w': w, 'h': h},
                'scales': {'horizontal': round(h_scale, 4)},
                'layout_params': {'padding': padding}
            }
            try:
                setattr(self, '_last_reasoning', reasoning)
            except Exception:
                pass
            return True

        logger.warning(f" PLACE TOKENS: Could not fit both tokens in rect ({x}, {y}, {w}, {h})")
        return False

    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path for processed image."""
        stem = input_path.stem
        suffix = input_path.suffix
        parent = input_path.parent
        
        return parent / f"{stem}_hybrid_append{suffix}"

def create_precise_append_replacer(pattern_matcher: PatternMatcher) -> PreciseAppendReplacer:
    """Factory function to create PreciseAppendReplacer."""
    return PreciseAppendReplacer(pattern_matcher)
