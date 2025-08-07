#!/usr/bin/env python3
"""
Task 4.3: OCR Text Replacement/Append Logic.
Implements comprehensive OCR-based text replacement with OpenCV rendering for all three OCR modes:
- replace: Replace from_text with to_text at exact position, preserving other text
- append: Append to_text to from_text in two lines at same location
- append-image: Create new image with replaced text and append after original
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import math
import re
from io import BytesIO

try:
    from .shared_constants import SharedUtilities, DEFAULT_OCR_CONFIDENCE
    from .ocr_engine import EnhancedOCREngine
    from .comprehensive_image_detector import ComprehensiveImageDetector
except ImportError:
    from shared_constants import SharedUtilities, DEFAULT_OCR_CONFIDENCE
    from ocr_engine import EnhancedOCREngine
    from comprehensive_image_detector import ComprehensiveImageDetector

logger = logging.getLogger(__name__)


class OCRTextReplacementProcessor:
    """
    Comprehensive OCR text replacement processor supporting all three OCR modes:
    - replace: Replace from_text with to_text at exact position, preserving other text
    - append: Append to_text to from_text in two lines at same location  
    - append-image: Create new image with replaced text and append after original
    """
    
    def __init__(self, 
                 ocr_engine: Optional[EnhancedOCREngine] = None,
                 image_detector: Optional[ComprehensiveImageDetector] = None,
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE):
        """
        Initialize OCR text replacement processor.
        
        Args:
            ocr_engine: Enhanced OCR engine instance
            image_detector: Comprehensive image detector instance
            confidence_threshold: Minimum confidence for OCR results
        """
        self.ocr_engine = ocr_engine or EnhancedOCREngine()
        self.image_detector = image_detector or ComprehensiveImageDetector()
        self.confidence_threshold = confidence_threshold
        
        # Font fallbacks for different platforms
        self.font_fallbacks = [
            "arial.ttf", "Arial.ttf", "helvetica.ttf", "DejaVuSans.ttf",
            "Liberation Sans.ttf", "Roboto-Regular.ttf"
        ]
        
        # Processing statistics
        self.stats = {
            'images_processed': 0,
            'text_replacements': 0,
            'text_appends': 0,
            'image_appends': 0,
            'ocr_matches': 0,
            'processing_errors': []
        }
    
    def process_docx_images(self, 
                           docx_path: Path,
                           mapping: Dict[str, str],
                           patterns: List[str],
                           ocr_mode: str = "replace") -> Dict[str, Any]:
        """
        Process all images in a DOCX file for OCR text replacement.
        
        Args:
            docx_path: Path to DOCX file
            mapping: Dictionary of text replacements
            patterns: List of regex patterns to match
            ocr_mode: OCR processing mode ('replace', 'append', 'append-image')
            
        Returns:
            Processing results with statistics and modified images
        """
        results = {
            'docx_path': str(docx_path),
            'ocr_mode': ocr_mode,
            'total_images_found': 0,
            'images_processed': 0,
            'replacements_made': 0,
            'modified_images': [],
            'processing_errors': [],
            'ocr_statistics': {}
        }
        
        try:
            # Detect all images in DOCX
            logger.info(f"Starting OCR text replacement for {docx_path.name} in {ocr_mode} mode")
            detection_results = self.image_detector.detect_all_images(docx_path)
            
            results['total_images_found'] = detection_results['total_images']
            
            # Get OCR-ready images
            ocr_images = self.image_detector.get_images_for_ocr_processing(detection_results)
            
            # Process each image
            for image_data in ocr_images:
                try:
                    image_results = self._process_single_image(
                        image_data, mapping, patterns, ocr_mode
                    )
                    
                    if image_results['modified']:
                        results['modified_images'].append(image_results)
                        results['replacements_made'] += image_results['replacements_count']
                    
                    results['images_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing image {image_data.get('filename', 'unknown')}: {e}"
                    logger.error(error_msg)
                    results['processing_errors'].append(error_msg)
            
            # Compile OCR statistics
            results['ocr_statistics'] = self.ocr_engine.get_engine_statistics()
            
            logger.info(f"OCR processing completed: {results['images_processed']} images, "
                       f"{results['replacements_made']} replacements")
            
        except Exception as e:
            error_msg = f"Error in DOCX OCR processing: {e}"
            logger.error(error_msg)
            results['processing_errors'].append(error_msg)
        
        return results
    
    def _process_single_image(self, 
                             image_data: Dict[str, Any],
                             mapping: Dict[str, str],
                             patterns: List[str],
                             ocr_mode: str) -> Dict[str, Any]:
        """
        Process a single image for OCR text replacement.
        
        Args:
            image_data: Image data from comprehensive detector
            mapping: Text replacement mapping
            patterns: Regex patterns to match
            ocr_mode: Processing mode
            
        Returns:
            Processing results for this image
        """
        image_results = {
            'image_id': image_data['image_id'],
            'original_path': image_data['original_path'],
            'modified': False,
            'replacements_count': 0,
            'replacements_made': [],
            'modified_image': None,
            'append_images': [],
            'ocr_results': [],
            'processing_errors': []
        }
        
        try:
            pil_image = image_data['pil_image']
            
            # Run OCR on the image
            ocr_results = self.ocr_engine.run_ocr(pil_image)
            image_results['ocr_results'] = ocr_results.get('text_results', [])
            
            # Find matches and apply replacements
            matches_found = self._find_text_matches(
                image_results['ocr_results'], mapping, patterns
            )
            
            if matches_found:
                # Apply replacements based on OCR mode
                if ocr_mode == "replace":
                    modified_image = self._apply_replace_mode(pil_image, matches_found)
                    if modified_image:
                        image_results['modified_image'] = modified_image
                        image_results['modified'] = True
                
                elif ocr_mode == "append":
                    modified_image = self._apply_append_mode(pil_image, matches_found)
                    if modified_image:
                        image_results['modified_image'] = modified_image
                        image_results['modified'] = True
                
                elif ocr_mode == "append-image":
                    append_images = self._apply_append_image_mode(pil_image, matches_found)
                    if append_images:
                        image_results['append_images'] = append_images
                        image_results['modified'] = True
                
                image_results['replacements_count'] = len(matches_found)
                image_results['replacements_made'] = matches_found
                
        except Exception as e:
            error_msg = f"Error processing image {image_data['image_id']}: {e}"
            logger.error(error_msg)
            image_results['processing_errors'].append(error_msg)
        
        return image_results
    
    def _find_text_matches(self, 
                          ocr_results: List[Dict[str, Any]], 
                          mapping: Dict[str, str], 
                          patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Find text matches in OCR results using mapping and patterns.
        
        Args:
            ocr_results: OCR detection results
            mapping: Text replacement mapping
            patterns: Regex patterns to match
            
        Returns:
            List of matches with replacement information
        """
        matches = []
        
        for ocr_result in ocr_results:
            text = ocr_result.get('text', '').strip()
            confidence = ocr_result.get('confidence', 0)
            
            # Skip low confidence results
            if confidence < self.confidence_threshold:
                continue
            
            # Check direct mapping matches
            # Handle case where mapping might be passed as list instead of dict
            if isinstance(mapping, dict):
                mapping_items = mapping.items()
            elif isinstance(mapping, list):
                # Convert list of dicts to items if needed
                mapping_items = [(item.get('from', ''), item.get('to', '')) for item in mapping if isinstance(item, dict)]
            else:
                logger.warning(f"Unexpected mapping type: {type(mapping)}")
                mapping_items = []
            
            for from_text, to_text in mapping_items:
                if self._text_matches(text, from_text):
                    match = {
                        'original_text': text,
                        'from_text': from_text,
                        'to_text': to_text,
                        'match_type': 'mapping',
                        'confidence': confidence,
                        'bbox': ocr_result.get('bbox', []),
                        'position': ocr_result.get('position', {}),
                        'orientation': ocr_result.get('orientation', 0)
                    }
                    matches.append(match)
                    break
            
            # Check pattern matches
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        match = {
                            'original_text': text,
                            'from_text': text,
                            'to_text': text,  # Pattern matches keep original text
                            'match_type': 'pattern',
                            'pattern': pattern,
                            'confidence': confidence,
                            'bbox': ocr_result.get('bbox', []),
                            'position': ocr_result.get('position', {}),
                            'orientation': ocr_result.get('orientation', 0)
                        }
                        matches.append(match)
                        break
                except re.error:
                    continue
        
        return matches
    
    def _text_matches(self, text1: str, text2: str) -> bool:
        """
        Check if two text strings match with normalization.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            True if texts match
        """
        # Normalize both texts
        norm1 = ''.join(text1.lower().split())
        norm2 = ''.join(text2.lower().split())
        
        return norm1 == norm2 or text1.lower() == text2.lower()
    
    def _apply_replace_mode(self, 
                           image: Image.Image, 
                           matches: List[Dict[str, Any]]) -> Optional[Image.Image]:
        """
        Apply replace mode: Replace from_text with to_text at exact position.
        Preserves other text and image elements.
        
        Args:
            image: Original PIL Image
            matches: List of text matches to replace
            
        Returns:
            Modified PIL Image or None if no changes
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for match in matches:
                bbox = match['bbox']
                from_text = match['from_text']
                to_text = match['to_text']
                
                if not bbox or from_text == to_text:
                    continue
                
                # Extract bounding box coordinates
                x, y, width, height = self._extract_bbox_coords(bbox)
                
                # Create mask to remove original text
                mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
                
                # Inpaint to remove original text (fill with background)
                cv_image = cv2.inpaint(cv_image, mask, 3, cv2.INPAINT_TELEA)
                
                # Render new text at same position
                self._render_text_opencv(
                    cv_image, to_text, x, y, width, height, 
                    match.get('orientation', 0)
                )
            
            # Convert back to PIL
            modified_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return modified_image
            
        except Exception as e:
            logger.error(f"Error in replace mode: {e}")
            return None
    
    def _apply_append_mode(self, 
                          image: Image.Image, 
                          matches: List[Dict[str, Any]]) -> Optional[Image.Image]:
        """
        Apply append mode: Append to_text to from_text in two lines at same location.
        Preserves rest of image and doesn't overwrite anything else.
        
        Args:
            image: Original PIL Image
            matches: List of text matches to append
            
        Returns:
            Modified PIL Image or None if no changes
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for match in matches:
                bbox = match['bbox']
                from_text = match['from_text']
                to_text = match['to_text']
                
                if not bbox or from_text == to_text:
                    continue
                
                # Extract bounding box coordinates
                x, y, width, height = self._extract_bbox_coords(bbox)
                
                # Calculate positions for two lines
                line1_y = y
                line2_y = y + height + 5  # Small gap between lines
                
                # Create mask to remove original text
                mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
                
                # Inpaint to remove original text
                cv_image = cv2.inpaint(cv_image, mask, 3, cv2.INPAINT_TELEA)
                
                # Render original text (line 1)
                self._render_text_opencv(
                    cv_image, from_text, x, line1_y, width, height // 2,
                    match.get('orientation', 0)
                )
                
                # Render appended text (line 2)
                self._render_text_opencv(
                    cv_image, to_text, x, line2_y, width, height // 2,
                    match.get('orientation', 0)
                )
            
            # Convert back to PIL
            modified_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return modified_image
            
        except Exception as e:
            logger.error(f"Error in append mode: {e}")
            return None
    
    def _apply_append_image_mode(self, 
                                image: Image.Image, 
                                matches: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Apply append-image mode: Create new image with replaced text.
        Returns list of new images to append after original in DOCX.
        
        Args:
            image: Original PIL Image
            matches: List of text matches to replace
            
        Returns:
            List of new PIL Images to append
        """
        append_images = []
        
        try:
            for match in matches:
                from_text = match['from_text']
                to_text = match['to_text']
                
                if from_text == to_text:
                    continue
                
                # Create copy of original image
                new_image = image.copy()
                
                # Apply replace mode to the copy
                modified_image = self._apply_replace_mode(new_image, [match])
                
                if modified_image:
                    append_images.append(modified_image)
            
        except Exception as e:
            logger.error(f"Error in append-image mode: {e}")
        
        return append_images
    
    def _extract_bbox_coords(self, bbox) -> Tuple[int, int, int, int]:
        """
        Extract x, y, width, height from various bbox formats.
        
        Args:
            bbox: Bounding box in various formats
            
        Returns:
            Tuple of (x, y, width, height)
        """
        if isinstance(bbox, list):
            if len(bbox) == 4 and isinstance(bbox[0], list):
                # EasyOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = min(x_coords)
                y = min(y_coords)
                width = max(x_coords) - x
                height = max(y_coords) - y
            elif len(bbox) == 4:
                # Tesseract format: [x, y, x2, y2] or [x, y, width, height]
                x, y, w_or_x2, h_or_y2 = bbox
                if w_or_x2 > x and h_or_y2 > y:
                    # Assume it's [x, y, x2, y2]
                    width = w_or_x2 - x
                    height = h_or_y2 - y
                else:
                    # Assume it's [x, y, width, height]
                    width = w_or_x2
                    height = h_or_y2
        else:
            # Default fallback
            x, y, width, height = 0, 0, 100, 20
        
        return int(x), int(y), int(width), int(height)
    
    def _render_text_opencv(self, 
                           cv_image: np.ndarray,
                           text: str,
                           x: int, y: int,
                           width: int, height: int,
                           orientation: float = 0) -> None:
        """
        Render text on OpenCV image at specified position.
        
        Args:
            cv_image: OpenCV image array
            text: Text to render
            x, y: Position coordinates
            width, height: Bounding box dimensions
            orientation: Text orientation in degrees
        """
        try:
            # Calculate font size based on bounding box
            font_scale = min(width / (len(text) * 10), height / 20)
            font_scale = max(0.3, min(font_scale, 2.0))  # Clamp font scale
            
            # Choose font
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = max(1, int(font_scale))
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Center text in bounding box
            text_x = x + (width - text_width) // 2
            text_y = y + (height + text_height) // 2
            
            # Handle rotation if needed
            if abs(orientation) > 5:  # Only rotate if significant angle
                # Create rotated text (simplified approach)
                # For now, render without rotation to avoid complexity
                pass
            
            # Render text
            cv2.putText(
                cv_image, text, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )
            
        except Exception as e:
            logger.warning(f"Error rendering text '{text}': {e}")
    
    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """
        Load appropriate font for text rendering.
        
        Args:
            size: Font size
            
        Returns:
            PIL ImageFont object
        """
        for font_name in self.font_fallbacks:
            try:
                return ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        return ImageFont.load_default()
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        ocr_stats = self.ocr_engine.get_engine_statistics()
        
        return {
            'processor_stats': self.stats,
            'ocr_engine_stats': ocr_stats,
            'total_images_processed': self.stats['images_processed'],
            'total_replacements': self.stats['text_replacements'],
            'total_appends': self.stats['text_appends'],
            'total_image_appends': self.stats['image_appends']
        }


def process_docx_with_ocr(docx_path: Path,
                         mapping: Dict[str, str],
                         patterns: List[str] = None,
                         ocr_mode: str = "replace",
                         confidence_threshold: float = DEFAULT_OCR_CONFIDENCE) -> Dict[str, Any]:
    """
    Convenience function to process DOCX file with OCR text replacement.
    
    Args:
        docx_path: Path to DOCX file
        mapping: Text replacement mapping
        patterns: Optional regex patterns
        ocr_mode: OCR processing mode ('replace', 'append', 'append-image')
        confidence_threshold: Minimum OCR confidence
        
    Returns:
        Processing results
    """
    processor = OCRTextReplacementProcessor(confidence_threshold=confidence_threshold)
    return processor.process_docx_images(
        docx_path, mapping, patterns or [], ocr_mode
    )


if __name__ == "__main__":
    # Test the OCR text replacement processor
    import sys
    
    if len(sys.argv) > 1:
        docx_file = Path(sys.argv[1])
        if docx_file.exists():
            # Test mapping
            test_mapping = {
                "old text": "new text",
                "replace me": "replaced!"
            }
            
            # Test all three modes
            for mode in ["replace", "append", "append-image"]:
                print(f"\n=== TESTING {mode.upper()} MODE ===")
                results = process_docx_with_ocr(
                    docx_file, test_mapping, ocr_mode=mode
                )
                
                print(f"Images found: {results['total_images_found']}")
                print(f"Images processed: {results['images_processed']}")
                print(f"Replacements made: {results['replacements_made']}")
                print(f"Modified images: {len(results['modified_images'])}")
        else:
            print(f"File not found: {docx_file}")
    else:
        print("Usage: python ocr_text_replacement.py <docx_file>")
