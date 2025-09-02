import re
import logging
import time
import uuid
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from PIL import Image
from docx import Document

from ..core.models import ProcessingResult, MatchDetail, ProcessorType, MatchFlag, FallbackFlag
from ..utils.text_utils.text_docx_utils import create_pattern_matcher, load_patterns_and_mappings
from config import DEFAULT_MAPPING, DEFAULT_SEPARATOR, PROCESSING_MODES, DEFAULT_OCR_CONFIDENCE

# Import from the new utility file
from ..utils.image_utils.image_docx_utils import (
    extract_images_with_mapping,
    preprocess_image,
    format_bounding_box,
    get_bounding_box_xywh,
    calculate_sub_bounding_box,
    OCRResult,
    EasyOCRManager,
    TesseractManager,
    HybridOCRManager,
    BasicOCRManager
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Processes images within a DOCX document to find and map text patterns.
    This class orchestrates image extraction, OCR, and pattern matching.
    """

    def __init__(self, patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None,
                 mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR,
                 default_mapping: str = DEFAULT_MAPPING, ocr_engine: str = "hybrid",
                 confidence_min: float = DEFAULT_OCR_CONFIDENCE, use_gpu: bool = True):
        """Initializes the image processor with configuration."""
        self.patterns, self.mappings = patterns or {}, mappings or {}
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
        self.ocr_engine = ocr_engine
        self.confidence_min = confidence_min
        self.use_gpu = use_gpu

        self.pattern_matcher = None
        self.ocr_manager = None
        self.temp_dir = None
        logger.info(f"ImageProcessor initialized with OCR engine: {ocr_engine}")

    def initialize(self) -> bool:
        """Initializes the pattern matcher and OCR manager."""
        try:
            logger.info("Initializing ImageProcessor components...")
            self.pattern_matcher = create_pattern_matcher(self.patterns, self.mappings)
            self.ocr_manager = self._create_ocr_manager()
            self.temp_dir = Path(tempfile.mkdtemp(prefix="docx_images_"))
            return self.pattern_matcher is not None and self.ocr_manager is not None
        except Exception as e:
            logger.error(f"Error initializing ImageProcessor: {e}")
            return False

    def process_images(self, document: Document, processing_result: ProcessingResult, file_path: Path) -> ProcessingResult:
        """Main method to process all images in a document."""
        start_time = time.time()
        logger.info(f"Starting image processing for {file_path.name}...")

        if not self.initialize():
            processing_result.error_message = "Failed to initialize ImageProcessor"
            return processing_result

        # Use the new XML-based extraction method from image_utils
        image_info_list = extract_images_with_mapping(file_path, self.temp_dir)
        logger.info(f"Extracted {len(image_info_list)} images from document.")

        all_detections = []
        for i, image_info in enumerate(image_info_list):
            logger.info(f"Processing image {i+1}/{len(image_info_list)}: {image_info['media_filename']}")
            image_detections = self._process_single_image(image_info)
            all_detections.extend(image_detections)

        self._update_processing_result(processing_result, all_detections)
        processing_time = time.time() - start_time
        logger.info(f"Image processing completed in {processing_time:.2f}s with {len(all_detections)} detections.")
        self.cleanup()
        return processing_result

    def _process_single_image(self, image_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Performs OCR and pattern matching on a single image."""
        preprocessed_img = preprocess_image(image_info['temp_path'])
        if not preprocessed_img:
            return []

        ocr_results = self.ocr_manager.process_image_with_image(image_info['temp_path'], preprocessed_img)
        
        # Check if initial OCR found text or if pattern matching on results works
        initial_matches = self._find_pattern_matches_in_ocr(ocr_results)

        if not initial_matches:
            logger.info(f"No initial matches in {image_info['media_filename']}. Trying fallback OCR...")
            ocr_results = self._try_fallback_ocr(image_info['temp_path'])

        # Final pattern matching on the best available OCR results
        final_matches = self._find_pattern_matches_in_ocr(ocr_results)
        
        return self._create_detection_results(final_matches, image_info, ocr_results)
    
    def _find_pattern_matches_in_ocr(self, ocr_results: List[OCRResult]) -> List[tuple]:
        """Finds all pattern matches within a list of OCR results."""
        all_matches = []
        for result in ocr_results:
            text = result.text.strip()
            if not text:
                continue
            
            find_func = self.pattern_matcher.find_all_pattern_matches if self.mode == PROCESSING_MODES['APPEND'] else self.pattern_matcher.find_matches
            pattern_matches = find_func(text)
            
            for pm in pattern_matches:
                pattern_name, matched_text, start, end = pm
                # Preserve character indices for bounding-box refinement later
                all_matches.append((pattern_name, matched_text, start, end, result))
        return all_matches

    def _create_detection_results(self, pattern_matches: List[tuple], image_info: Dict[str, Any], ocr_results: List[OCRResult]) -> List[Dict[str, Any]]:
        """Creates a list of detection dictionaries from pattern matches."""
        detection_results = []
        # Deduplicate by normalized ID (strip non-alphanumerics), prefer entries with actual mapping
        seen: Dict[str, int] = {}
        
        try:
            with Image.open(image_info['temp_path']) as img:
                width, height = img.size
                image_dimension = f"{width}x{height} pixels"
        except Exception:
            image_dimension = "Unknown"

        for pattern_name, matched_text, start, end, ocr_result in pattern_matches:
            replacement_text = self.pattern_matcher.get_replacement(matched_text) or self.default_mapping
            has_mapping = replacement_text != self.default_mapping

            # Normalize to dedupe variants like 77-110-817895000 vs 77-110-817895-000
            normalized_key = re.sub(r"[^A-Za-z0-9]", "", matched_text).upper()

            # Compute a refined sub-bounding-box for the matched substring, if possible
            formatted_bbox = image_dimension
            try:
                bbox_xywh = get_bounding_box_xywh(getattr(ocr_result, 'bounding_box', []) or [])
                if bbox_xywh is not None:
                    sub_bbox_xywh = calculate_sub_bounding_box(
                        bbox_xywh,
                        getattr(ocr_result, 'text', matched_text) or "",
                        (matched_text, start, end)
                    )
                    formatted_bbox = format_bounding_box(sub_bbox_xywh, getattr(ocr_result, 'rotation_angle', 0))
            except Exception:
                # Fall back to coarse image dimensions if bbox refinement fails
                formatted_bbox = image_dimension

            new_entry = {
                'pattern_name': pattern_name,
                'matched_text': matched_text,
                'replacement_text': replacement_text,
                'is_matched': has_mapping,
                'location': image_info['location'],
                'media_filename': image_info['media_filename'],
                # Store formatted bounding box string in Src Dimension as requested
                'dimension': formatted_bbox,
                'bbox_info': formatted_bbox,
                'font_info': {'ocr_confidence': ocr_result.confidence}
            }

            if normalized_key in seen:
                idx = seen[normalized_key]
                existing = detection_results[idx]
                # Prefer the one that has an actual mapping; if both or neither, keep higher confidence
                existing_has_mapping = existing.get('is_matched', False)
                if (has_mapping and not existing_has_mapping) or (
                    has_mapping == existing_has_mapping and
                    new_entry['font_info'].get('ocr_confidence', 0.0) > existing.get('font_info', {}).get('ocr_confidence', 0.0)
                ):
                    detection_results[idx] = new_entry
                # else keep existing
            else:
                seen[normalized_key] = len(detection_results)
                detection_results.append(new_entry)
        return detection_results

    def _update_processing_result(self, processing_result: ProcessingResult, detections: List[Dict[str, Any]]):
        """Updates the main ProcessingResult object with found matches."""
        # Maintain sequential sr_no across all processors by continuing from existing details
        next_sr_no = len(processing_result.match_details) + 1
        # This implementation remains largely the same as your original, creating MatchDetail objects.
        for detection in detections:
            is_matched = detection.get('is_matched', False)
            if is_matched:
                processing_result.total_image_matches += 1
            else:
                processing_result.total_image_no_match += 1
            
            match_detail = MatchDetail(
                sr_no=next_sr_no,
                type=ProcessorType.IMAGE,
                orig_id_name=detection.get('media_filename', ''),
                src_text=detection.get('matched_text', ''),
                src_text_size=f"OCR Conf: {detection.get('font_info', {}).get('ocr_confidence', 0.0):.2f}",
                src_dimension=detection.get('dimension', ''),
                src_bbox=detection.get('bbox_info', ''),
                mapped_text=detection.get('replacement_text', ''),
                match_flag=MatchFlag.YES if is_matched else MatchFlag.NO,
                is_fallback=FallbackFlag.NO # Add logic if you distinguish fallback matches
            )
            processing_result.match_details.append(match_detail)
            next_sr_no += 1

    def _create_ocr_manager(self):
        """Factory method to create the configured OCR manager."""
        if self.ocr_engine == "easyocr":
            try:
                import easyocr
                return EasyOCRManager(easyocr.Reader(EASYOCR_LANGUAGES, gpu=self.use_gpu), self.confidence_min)
            except Exception:
                logger.error("Failed to initialize EasyOCR, using BasicOCRManager.")
                return BasicOCRManager(self.confidence_min)
        elif self.ocr_engine == "tesseract":
            try:
                import pytesseract
                return TesseractManager(pytesseract, self.confidence_min)
            except Exception:
                logger.error("Failed to initialize Tesseract, using BasicOCRManager.")
                return BasicOCRManager(self.confidence_min)
        elif self.ocr_engine == "hybrid":
            return HybridOCRManager(self.confidence_min, self.use_gpu)
        else:
            logger.warning(f"Unknown OCR engine '{self.ocr_engine}', using BasicOCRManager.")
            return BasicOCRManager(self.confidence_min)

    def _run_ocr_with_osd_rotation(self, image_path: Path) -> List[OCRResult]:
        """Efficiently corrects orientation using Tesseract OSD and runs OCR."""
        logger.info(f"Attempting Tesseract OSD orientation correction for {image_path.name}")
        try:
            import pytesseract
            with Image.open(image_path) as img:
                # Heuristic: skip OSD on very small images where OSD is unreliable
                width, height = img.size
                if (width * height) < 200_000:  # ~< 447x447
                    logger.debug(f"OSD skipped (image too small): {width}x{height}")
                    return []

                # Use OSD mode explicitly
                try:
                    osd_data = pytesseract.image_to_osd(img, config='--psm 0')
                except Exception as e:
                    msg = str(e)
                    # Downgrade common benign messages to debug
                    if 'Too few characters' in msg or 'Invalid resolution' in msg:
                        logger.debug(f"OSD fallback benign skip: {msg}")
                        return []
                    raise
                angle_match = re.search(r'(?<=Rotate: )\d+', osd_data)
                angle = int(angle_match.group(0)) if angle_match else 0

                if angle not in [0, 360]:
                    logger.info(f"  OSD detected {angle}° rotation. Correcting image...")
                    corrected_img = img.rotate(360 - angle, expand=True)
                    results = self.ocr_manager.process_image_with_image(image_path, corrected_img)
                    for res in results:
                        setattr(res, 'rotation_angle', angle) # Tag result with rotation info
                    return results
                else:
                    logger.info("  OSD found no rotation needed.")
                    return []
        except ImportError:
            logger.warning("Pytesseract is not installed; cannot perform OSD fallback.")
            return []
        except Exception as e:
            # Non-fatal: OSD fallback can legitimately fail on low-text/low-DPI images
            logger.warning(f"OSD fallback skipped: {e}")
            return []

    def _try_fallback_ocr(self, image_path: Path) -> List[OCRResult]:
        """Tries various fallback methods if initial OCR fails."""
        # Fallback 1: Tesseract OSD (most efficient for rotation)
        osd_results = self._run_ocr_with_osd_rotation(image_path)
        if osd_results:
            return osd_results

        # Fallback 2: Image enhancement (sharpen, contrast)
        try:
            logger.info("Trying fallback with image enhancement...")
            with Image.open(image_path) as img:
                from PIL import ImageEnhance, ImageFilter
                enhanced_img = img.filter(ImageFilter.SHARPEN)
                enhancer = ImageEnhance.Contrast(enhanced_img)
                enhanced_img = enhancer.enhance(1.5)
                enhanced_results = self.ocr_manager.process_image_with_image(image_path, enhanced_img)
                if enhanced_results:
                    return enhanced_results
        except Exception as e:
            logger.warning(f"Image enhancement fallback failed: {e}")

        logger.warning(f"All fallback OCR methods failed for {image_path.name}")
        return []

    def cleanup(self):
        """Removes the temporary directory used for image extraction."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {self.temp_dir}: {e}")
