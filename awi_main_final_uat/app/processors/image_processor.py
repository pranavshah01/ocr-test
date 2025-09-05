"""
Ported Image Processor (from UAT) with minimal changes:
- Imports remapped to app structure and config
- Result storage integrated with app/core/models.ProcessingResult and MatchDetail
- Added process_images(document, processing_result, file_path, pattern_matcher=None)
All processing logic remains line-by-line equivalent where possible.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from io import BytesIO
from PIL import Image, UnidentifiedImageError
from docx import Document

from config import (
    DEFAULT_MAPPING,
    DEFAULT_SEPARATOR,
    PROCESSING_MODES
)
from ..core.models import (
    ProcessingResult, MatchDetail, ImageReasoning, ProcessorType, MatchFlag
)
from ..core.processor_interface import BaseProcessor
from ..utils.image_utils.hybrid_ocr_manager import HybridOCRManager
from ..utils.image_utils.ocr_models import OCRResult, OCRMatch
from ..utils.image_utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.image_utils.platform_utils import PathManager
from ..utils.image_utils.precise_append import PreciseAppendReplacer

logger = logging.getLogger(__name__)

# Supported vector types we will skip (report-only)
VECTOR_CTYPES = {
    'image/wmf', 'image/x-wmf', 'application/x-msmetafile',
    'image/emf', 'image/x-emf', 'application/emf'
}
VECTOR_EXTS = {'.wmf', '.emf'}

@dataclass
class EnhancedImageMatch:
    pattern: str
    original_text: str
    replacement_text: str
    position: int
    location: str
    font_info: Dict[str, Any] = None
    confidence: Optional[float] = None
    actual_pattern: str = ""
    content_type: str = "Image"
    dimension: str = ""
    processor: str = "Image"
    ocr_result: Optional[Any] = None
    extracted_pattern_text: str = ""

    def __post_init__(self):
        if self.font_info is None:
            self.font_info = {}

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'pattern': self.pattern,
            'pattern_name': self.pattern,
            'original_text': self.original_text,
            'extracted_pattern_text': self.extracted_pattern_text,
            'replacement_text': self.replacement_text,
            'position': self.position,
            'location': self.location,
            'font_info': self.font_info,
            'confidence': self.confidence,
            'actual_pattern': self.actual_pattern,
            'content_type': self.content_type,
            'dimension': self.dimension,
            'processor': self.processor
        }
        if hasattr(self, 'reconstruction_reasoning') and self.reconstruction_reasoning:
            result['reconstruction_reasoning'] = self.reconstruction_reasoning
        return result

class ImageProcessor(BaseProcessor):
    def __init__(self, patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None,
                 mode: str = PROCESSING_MODES['REPLACE'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING,
                 ocr_engine: str = "easyocr", confidence_min: float = 0.4, use_gpu: bool = True, enable_debugging: bool = False):
        config = {
            'patterns': patterns or {},
            'mappings': mappings or {},
            'mode': mode,
            'separator': separator,
            'default_mapping': default_mapping,
            'ocr_engine': ocr_engine,
            'confidence_min': confidence_min,
            'use_gpu': use_gpu
        }
        super().__init__("image", config)

        self.patterns = patterns or {}
        self.mappings = mappings or {}
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
        self.ocr_engine = ocr_engine
        self.confidence_min = confidence_min
        self.use_gpu = use_gpu
        self.enable_debugging = enable_debugging
        
        # Document-level comments to surface in HTML report (e.g., unsupported images)
        self._file_comments: List[str] = []
        
        self.pattern_matcher = None
        self.ocr_manager = None
        self.ocr_comparison_data = []
        self._append_replacer_for_reconstruction = None

        logger.info(f"Image processor initialized with mode: {mode}, separator: '{separator}', "
                    f"default_mapping: '{default_mapping}', OCR engine: {ocr_engine}, "
                    f"confidence_min: {confidence_min}, use_gpu: {use_gpu}")

    def initialize(self, **kwargs) -> bool:
        try:
            logger.info("Initializing Image Processor...")

            patterns = kwargs.get('patterns', self.patterns)
            mappings = kwargs.get('mappings', self.mappings)
            mode = kwargs.get('mode', self.mode)
            separator = kwargs.get('separator', self.separator)
            default_mapping = kwargs.get('default_mapping', self.default_mapping)
            ocr_engine = kwargs.get('ocr_engine', self.ocr_engine)
            confidence_min = kwargs.get('confidence_min', self.confidence_min)
            use_gpu = kwargs.get('use_gpu', self.use_gpu)

            if not patterns or not mappings:
                patterns, mappings = self._load_patterns_and_mappings()

            self.patterns = patterns
            self.mappings = mappings
            self.mode = mode
            self.separator = separator
            self.default_mapping = default_mapping
            self.ocr_engine = ocr_engine
            self.confidence_min = confidence_min
            self.use_gpu = use_gpu

            self.pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
            self.ocr_manager = HybridOCRManager(confidence_threshold=confidence_min, use_gpu=use_gpu)
            self._append_replacer_for_reconstruction = PreciseAppendReplacer(self.pattern_matcher, mode=self.mode)

            self.initialized = True
            logger.info(f"Image processor initialized with {len(patterns)} patterns, {len(mappings)} mappings, mode: {mode}")
            return True
        except Exception as e:
            logger.error(f"Error initializing Image Processor: {e}")
            self.initialized = False
            return False

    # UAT-style public method adapted to app integration
    def process_images(self, document: Document, processing_result: ProcessingResult, file_path: Optional[Path] = None, pattern_matcher: Optional[PatternMatcher] = None) -> ProcessingResult:
        start_time = time.time()
        try:
            # Prefer shared matcher only if it supports the enhanced API used by image processor
            if pattern_matcher is not None and hasattr(pattern_matcher, 'find_matches_universal'):
                self.pattern_matcher = pattern_matcher
            elif self.pattern_matcher is None:
                # Fallback: ensure we have our own enhanced matcher
                self.pattern_matcher = create_pattern_matcher(self.patterns, self.mappings, enhanced_mode=True)
            image_matches = self._process_images(document, media_dir=None)

            # Convert matches to MatchDetail entries
            sr_no = len(processing_result.match_details) + 1
            match_details: List[MatchDetail] = []
            for m in image_matches:
                bbox = getattr(m, 'bounding_box', getattr(m.ocr_result, 'bounding_box', (0, 0, 0, 0)))
                mapped_text = m.replacement_text or self.default_mapping
                # Treat default mapping (e.g., 4022-NA) as no specific mapping found
                has_specific_mapping = (
                    m.replacement_text is not None and m.replacement_text != "" and m.replacement_text != self.default_mapping
                )
                # Get dimension from the match object
                src_dimension = getattr(m, 'dimension', '')
                
                match_detail = MatchDetail(
                    sr_no=sr_no,
                    type=ProcessorType.IMAGE,
                    orig_id_name=m.location,
                    src_text=m.extracted_pattern_text or m.original_text,
                    src_bbox=','.join(str(int(x)) for x in (
                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    )) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else "",
                    src_dimension=src_dimension,
                    mapped_text=mapped_text,
                    match_flag=MatchFlag.YES if has_specific_mapping else MatchFlag.NO,
                    reasoning=ImageReasoning(
                        available_size_for_new_text=str(getattr(m, 'dimension', '')),
                        total_characters=len(mapped_text),
                        line_reasoning=str(getattr(m, 'reconstruction_reasoning', ''))
                    ),
                    reconstructed=bool(getattr(m, 'reconstructed', False))
                )
                # Populate reconstruction-specific fields only when reconstruction occurred
                try:
                    if getattr(match_detail, 'reconstructed', False):
                        # Note: precise_bbox and reconstruction_bbox are not currently being set
                        # in the reconstruction process, so we skip those for now
                        if hasattr(m, 'reconstruction_reasoning'):
                            rr = getattr(m, 'reconstruction_reasoning')
                            match_detail.reconstruction_reasoning = rr if isinstance(rr, str) else str(rr)
                except Exception:
                    pass
                match_details.append(match_detail)
                sr_no += 1

            # Update processing_result aggregations
            processing_time = time.time() - start_time
            processing_result.total_image_matches += len(match_details)
            processing_result.matches_found += len(match_details)
            processing_result.match_details.extend(match_details)
            processing_result.processing_time += processing_time
            # Add metadata
            detailed_matches = [md.to_dict() for md in match_details]
            ocr_comp = self.get_ocr_comparison_data()
            processor_info = self.get_processing_info()
            processing_result.metadata.setdefault('image', {})
            processing_result.metadata['image'].update({
                'detailed_matches': detailed_matches,
                'ocr_comparison_data': ocr_comp,
                'processor_info': processor_info
            })
            # Attach document-level comments (e.g., unsupported images)
            if self._file_comments:
                try:
                    processing_result.metadata.setdefault('file_comments', [])
                    processing_result.metadata['file_comments'].extend(self._file_comments)
                except Exception:
                    pass
            processing_result.success = True
            return processing_result
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            processing_result.error_message = str(e)
            return processing_result

    # --- Below methods mirror UAT logic with minimal/no changes ---
    def _process_images(self, document: Document, **kwargs) -> List[EnhancedImageMatch]:
        logger.info("Starting 3-phase image processing...")
        temp_dir = None
        try:
            logger.info("=== PHASE 1: EXTRACTION ===")
            extraction_results = self._phase1_extraction(document, **kwargs)

            logger.info("=== PHASE 2: MATCH & WIPE ===")
            wipe_results = self._phase2_match_and_wipe(extraction_results)

            logger.info("=== PHASE 3: RECONSTRUCT ===")
            final_matches = self._phase3_reconstruct(document, wipe_results)

            logger.info(f"3-phase image processing completed: {len(final_matches)} matches found")
            return final_matches
        except Exception as e:
            logger.error(f"Error in 3-phase image processing: {e}")
            return []
        finally:
            # Clean up temp directory after all processing is complete
            if hasattr(self, '_current_temp_dir') and self._current_temp_dir and self._current_temp_dir.exists():
                self._safe_cleanup_temp_directory(self._current_temp_dir)
                self._current_temp_dir = None

    def _phase1_extraction(self, document: Document, **kwargs) -> List[Dict]:
        extraction_results: List[Dict[str, Any]] = []
        try:
            image_info_list = self._extract_images_with_mapping(document, **kwargs)
            for image_info in image_info_list:
                try:
                    logger.debug(f"Extracting text from: {image_info['location']}")
                    ocr_results = self.ocr_manager.process_hybrid(image_info['temp_path'])
                    
                    # Check if initial OCR found text or if pattern matching on results works
                    initial_matches = self._find_pattern_matches_in_ocr(ocr_results)
                    
                    if not initial_matches:
                        logger.info(f"No initial matches in {image_info['location']}. Trying fallback OCR...")
                        ocr_results = self._try_fallback_ocr(image_info['temp_path'])
                    
                    ocr_comparison = {}
                    if hasattr(self.ocr_manager, 'get_comparison_data'):
                        ocr_comparison = self.ocr_manager.get_comparison_data()
                    self._store_ocr_comparison_data(image_info['location'], image_info['temp_path'], ocr_comparison)
                    extraction_results.append({
                        'image_info': image_info,
                        'ocr_results': ocr_results,
                        'ocr_comparison': ocr_comparison
                    })
                except Exception as e:
                    logger.error(f"Error in extraction phase for {image_info['location']}: {e}")
            logger.info(f"Extraction phase completed: {len(extraction_results)} images processed")
        except Exception as e:
            logger.error(f"Error in extraction phase: {e}")
        return extraction_results

    def _phase2_match_and_wipe(self, extraction_results: List[Dict]) -> List[Dict]:
        wipe_results: List[Dict[str, Any]] = []
        for extraction_result in extraction_results:
            try:
                image_info = extraction_result['image_info']
                ocr_results = extraction_result['ocr_results']
                logger.debug(f"Matching and wiping: {image_info['location']}")
                if not ocr_results:
                    wipe_results.append({'image_info': image_info, 'matches': [], 'wiped_image_path': None})
                    continue
                matches: List[EnhancedImageMatch] = []
                logger.debug(f"Processing {len(ocr_results)} OCR results for {image_info['location']}")
                for ocr_result in ocr_results:
                    logger.debug(f"OCR Text: '{ocr_result.text}'")
                    try:
                        if self.mode in ["append", "append-image"]:
                            all_matches = self.pattern_matcher.find_all_pattern_matches(ocr_result.text)
                            for pattern_name, matched_text, start_pos, end_pos in all_matches:
                                replacement_text = self.pattern_matcher.get_replacement(matched_text)
                                if not replacement_text:
                                    replacement_text = self.default_mapping
                                font_info = getattr(ocr_result, 'font_info', {})
                                # Format bounding box as dimension string with orientation
                                bbox = ocr_result.bounding_box
                                rotation_angle = getattr(ocr_result, 'rotation_angle', 0)
                                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                    dimension_str = f"({bbox[0]},{bbox[1]})-({bbox[0]+bbox[2]},{bbox[1]+bbox[3]})"
                                    if rotation_angle and rotation_angle not in [0, 360]:
                                        dimension_str += f" (orientation: {rotation_angle}°)"
                                else:
                                    dimension_str = ""
                                
                                ocr_match = EnhancedImageMatch(
                                    pattern_name, matched_text, replacement_text,
                                    start_pos, image_info['location'],
                                    font_info, ocr_result.confidence, pattern_name,
                                    "Image", dimension_str, "OCR", ocr_result
                                )
                                ocr_match.extracted_pattern_text = matched_text
                                ocr_match.original_text = ocr_result.text
                                matches.append(ocr_match)
                        else:
                            # Replace mode: create matches for all patterns, but only process replacements for those with mappings
                            universal_matches = self.pattern_matcher.find_matches_universal(ocr_result.text, include_unmapped=True)
                            for um in universal_matches:
                                replacement_text = self.pattern_matcher.get_replacement(um.matched_text)
                                # In replace mode, create matches even without mappings for reporting purposes
                                font_info = getattr(ocr_result, 'font_info', {})
                                # Format bounding box as dimension string with orientation
                                bbox = ocr_result.bounding_box
                                rotation_angle = getattr(ocr_result, 'rotation_angle', 0)
                                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                    dimension_str = f"({bbox[0]},{bbox[1]})-({bbox[0]+bbox[2]},{bbox[1]+bbox[3]})"
                                    if rotation_angle and rotation_angle not in [0, 360]:
                                        dimension_str += f" (orientation: {rotation_angle}°)"
                                else:
                                    dimension_str = ""
                                
                                ocr_match = EnhancedImageMatch(
                                    um.pattern_name, um.matched_text, replacement_text,
                                    um.start_pos, image_info['location'],
                                    font_info, ocr_result.confidence, um.pattern_name,
                                    "Image", dimension_str, "OCR", ocr_result
                                )
                                ocr_match.extracted_pattern_text = um.matched_text
                                ocr_match.original_text = ocr_result.text
                                matches.append(ocr_match)
                    except Exception as e:
                        logger.error(f"Error in pattern matching for text '{ocr_result.text}': {e}")
                
                # Apply wipes with proper rotation handling
                wiped_image_path = None
                if matches:
                    wiped_image_path = self._apply_wipes_with_rotation(image_info['temp_path'], matches)
                wipe_results.append({'image_info': image_info, 'matches': matches, 'wiped_image_path': wiped_image_path})
                logger.debug(f"Match & wipe completed for {image_info['location']}: {len(matches)} matches, wiped: {wiped_image_path is not None}")
            except Exception as e:
                logger.error(f"Error in match & wipe phase for {extraction_result['image_info']['location']}: {e}")
                wipe_results.append({'image_info': extraction_result['image_info'], 'matches': [], 'wiped_image_path': None})
        logger.info(f"Match & wipe phase completed: {len([r for r in wipe_results if r['matches']])} images with matches")
        return wipe_results

    def _phase3_reconstruct(self, document: Document, wipe_results: List[Dict]) -> List[EnhancedImageMatch]:
        final_matches: List[EnhancedImageMatch] = []
        for wipe_result in wipe_results:
            try:
                image_info = wipe_result['image_info']
                # Get all matches for reporting purposes
                all_matches = wipe_result['matches']
                # Respect mode semantics: append always reconstructs; replace skips default-mapped
                matches_for_reconstruction = all_matches
                if str(self.mode).lower() == "replace":
                    matches_for_reconstruction = [m for m in all_matches if getattr(m, 'replacement_text', None) not in (None, "", self.default_mapping)]
                wiped_image_path = wipe_result['wiped_image_path']
                logger.debug(f"Reconstructing: {image_info['location']}")
                if wiped_image_path and matches_for_reconstruction:
                    try:
                        # Apply reconstruction with proper rotation handling
                        reconstructed_path = self._apply_reconstruction_with_rotation(wiped_image_path, matches_for_reconstruction)
                        image_to_insert = reconstructed_path or wiped_image_path
                    except Exception as e:
                        logger.error(f"Error in reconstruction for {image_info['location']}: {e}")
                        image_to_insert = wiped_image_path
                    success = self._replace_image_in_document(image_info['original_rel'], image_to_insert)
                    if success:
                        # Mark only reconstructed matches as reconstructed
                        for match in matches_for_reconstruction:
                            match.image_path = image_to_insert
                            # Mark as reconstructed for reporting
                            setattr(match, 'reconstructed', True)
                        logger.info(f"RECONSTRUCTED: {len(matches_for_reconstruction)} matches in {image_info['location']}")
                    else:
                        logger.warning(f"Failed to reconstruct image in document for {image_info['location']}")
                
                # Always include all matches in final results for reporting
                final_matches.extend(all_matches)
            except Exception as e:
                logger.error(f"Error in reconstruction phase for {wipe_result['image_info']['location']}: {e}")
        logger.info(f"Reconstruction phase completed: {len(final_matches)} total matches")
        return final_matches

    def _apply_wipes_with_rotation(self, image_path: Path, matches: List[EnhancedImageMatch]) -> Optional[Path]:
        """Apply wipes with proper rotation handling, following the working version's approach."""
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert('RGB')
            
            # Group actions by rotation angle to apply in aligned orientation
            actions_by_angle: Dict[int, List[EnhancedImageMatch]] = {}
            for match in matches:
                rotation_angle = getattr(match.ocr_result, 'rotation_angle', 0) if hasattr(match, 'ocr_result') and match.ocr_result else 0
                angle = int(rotation_angle) % 360
                actions_by_angle.setdefault(angle, []).append(match)

            # Apply wipes for each rotation angle
            for angle, angle_matches in actions_by_angle.items():
                if angle:
                    rotate_before = (360 - angle) % 360
                    img = img.rotate(rotate_before, expand=True)
                    logger.debug(f"Rotated image by {rotate_before}° for wiping at angle {angle}°")
                
                for match in angle_matches:
                    try:
                        img = self._apply_single_wipe_no_rotation(img, match)
                    except Exception as e:
                        logger.error(f"Failed to apply wipe for match '{match.original_text}': {e}")
                
                if angle:
                    img = img.rotate(angle, expand=True)
                    logger.debug(f"Rotated image back by {angle}° after wiping")

            output_path = self._generate_output_path(image_path)
            img.save(output_path)
            logger.debug(f"Wipe application with rotation completed: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to apply wipes with rotation to image {image_path}: {e}")
            return None

    def _apply_wipes_to_image(self, image_path: Path, matches: List[EnhancedImageMatch]) -> Optional[Path]:
        """Legacy method - now redirects to rotation-aware version."""
        return self._apply_wipes_with_rotation(image_path, matches)

    def _calculate_reconstruction_reasoning(self, image: Image.Image, match: EnhancedImageMatch) -> Dict[str, Any]:
        try:
            img_width, img_height = image.size
            bbox = getattr(match, 'bounding_box', (0, 0, 0, 0))
            if len(bbox) >= 4:
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = int(img_width * 0.3)
                text_height = int(img_height * 0.1)
            if text_height < 20:
                font_size = 12
                font_reasoning = "Small text area - using 12pt font"
            elif text_height < 40:
                font_size = 16
                font_reasoning = "Medium text area - using 16pt font"
            else:
                font_size = 20
                font_reasoning = "Large text area - using 20pt font"
            text_length = len(match.replacement_text)
            if text_length <= 15:
                line_reasoning = "Single line - text length <= 15 characters"
                num_lines = 1
            elif text_length <= 30:
                line_reasoning = "Single line - text length <= 30 characters"
                num_lines = 1
            else:
                line_reasoning = "Two lines - text length > 30 characters"
                num_lines = 2
            reconstruction_width = max(text_width + 20, 100)
            reconstruction_height = text_height * num_lines + 20
            reasoning = {
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height,
                    'text_area_width': text_width,
                    'text_area_height': text_height
                },
                'font_logic': {'font_size': font_size, 'reasoning': font_reasoning},
                'line_reasoning': {'num_lines': num_lines, 'text_length': text_length, 'reasoning': line_reasoning},
                'reconstruction_dimensions': {'width': reconstruction_width, 'height': reconstruction_height}
            }
            logger.debug(f"Reconstruction reasoning: {reasoning}")
            return reasoning
        except Exception as e:
            logger.error(f"Error calculating reconstruction reasoning: {e}")
            return {
                'image_dimensions': {'width': 0, 'height': 0, 'text_area_width': 0, 'text_area_height': 0},
                'font_logic': {'font_size': 12, 'reasoning': 'Default due to error'},
                'line_reasoning': {'num_lines': 1, 'text_length': 0, 'reasoning': 'Default due to error'},
                'reconstruction_dimensions': {'width': 100, 'height': 50}
            }

    def _apply_single_wipe_no_rotation(self, image: Image.Image, match: EnhancedImageMatch) -> Image.Image:
        """Apply wipe without rotation handling (image is already in correct orientation)."""
        try:
            reasoning = self._calculate_reconstruction_reasoning(image, match)
            bbox = getattr(match, 'bounding_box', (0, 0, 0, 0))
            
            original_full_text = match.original_text
            pattern_matches = self.pattern_matcher.find_matches_universal(original_full_text)
            if not pattern_matches:
                logger.warning(f"No pattern matches found in text: '{original_full_text}'")
                return image
            target_match = None
            for pm in pattern_matches:
                replacement = self.pattern_matcher.get_replacement(pm.matched_text)
                if not replacement:
                    if self.mode in ["append", "append-image"]:
                        replacement = self.default_mapping
                    else:
                        continue
                if replacement == match.replacement_text:
                    target_match = pm
                    break
            if not target_match:
                logger.warning(f"Could not find target match for replacement: '{match.replacement_text}'")
                return image
            
            import numpy as np
            import cv2
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            ocr_result = OCRResult(text=match.original_text, confidence=match.confidence or 0.0, bounding_box=bbox)
            # Simple wipe - just fill the bounding box with white
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            wiped_image = Image.fromarray(cv_image_rgb)
            
            match.reconstruction_reasoning = reasoning
            logger.debug(f"Applied wipe for '{original_full_text}' (no rotation)")
            return wiped_image
        except Exception as e:
            logger.error(f"Failed to apply single wipe: {e}")
            return image

    def _apply_single_wipe(self, image: Image.Image, match: EnhancedImageMatch) -> Image.Image:
        """Legacy method - now redirects to no-rotation version since rotation is handled at higher level."""
        return self._apply_single_wipe_no_rotation(image, match)

    def _apply_reconstruction_with_rotation(self, wiped_image_path: Path, matches: List[EnhancedImageMatch]) -> Optional[Path]:
        """Apply reconstruction with proper rotation handling, following the working version's approach."""
        try:
            from PIL import Image as PILImage
            img = PILImage.open(wiped_image_path).convert('RGB')
            
            # Group matches by rotation angle to apply in aligned orientation
            matches_by_angle: Dict[int, List[EnhancedImageMatch]] = {}
            for match in matches:
                rotation_angle = getattr(match.ocr_result, 'rotation_angle', 0) if hasattr(match, 'ocr_result') and match.ocr_result else 0
                angle = int(rotation_angle) % 360
                matches_by_angle.setdefault(angle, []).append(match)

            # Apply reconstruction for each rotation angle
            for angle, angle_matches in matches_by_angle.items():
                if angle:
                    rotate_before = (360 - angle) % 360
                    img = img.rotate(rotate_before, expand=True)
                    logger.debug(f"Rotated image by {rotate_before}° for reconstruction at angle {angle}°")
                
                # Apply reconstruction using the existing precise replacers
                try:
                    ocr_results_for_image = []
                    for m in angle_matches:
                        if hasattr(m, 'ocr_result') and m.ocr_result:
                            ocr_results_for_image.append(m.ocr_result)
                        else:
                            ocr_results_for_image.append(OCRResult(text=m.original_text, confidence=getattr(m, 'confidence', 0.0), bounding_box=getattr(m, 'bounding_box', (0, 0, 0, 0))))
                    
                    # Use append replacer for all modes (it will handle mode-specific behavior internally)
                    temp_path = wiped_image_path.parent / f"temp_recon_{wiped_image_path.name}"
                    try:
                        # Ensure proper file closure
                        with open(temp_path, 'wb') as temp_file:
                            img.save(temp_file, format='PNG')
                        
                        reconstructed_path = self._append_replacer_for_reconstruction.replace_text_in_image(
                            temp_path, ocr_results_for_image, angle_matches
                        )
                        if reconstructed_path and reconstructed_path.exists():
                            img = PILImage.open(reconstructed_path).convert('RGB')
                    finally:
                        # Robust cleanup with Windows-specific handling
                        self._safe_cleanup_temp_file(temp_path)
                except Exception as e:
                    logger.error(f"Error in reconstruction for angle {angle}: {e}")
                
                if angle:
                    img = img.rotate(angle, expand=True)
                    logger.debug(f"Rotated image back by {angle}° after reconstruction")

            # Save the final reconstructed image
            output_path = self._generate_output_path(wiped_image_path)
            img.save(output_path)
            logger.debug(f"Reconstruction with rotation completed: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to apply reconstruction with rotation to image {wiped_image_path}: {e}")
            return None

    def _extract_images_with_mapping(self, document: Document, **kwargs) -> List[Dict]:
        image_info_list: List[Dict[str, Any]] = []
        try:
            temp_dir = PathManager.get_temp_directory() / f"enhanced_docx_images_{uuid.uuid4().hex[:8]}"
            PathManager.ensure_directory(temp_dir)
            # Store temp directory reference for cleanup at the end of processing
            self._current_temp_dir = temp_dir
            for i, rel in enumerate(document.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        content_type = rel.target_part.content_type
                        # Prefer actual media filename from relationship if present
                        try:
                            media_name = Path(rel.target_ref).name
                        except Exception:
                            media_name = ""
                        # Derive extension from content type when absent
                        ext = Path(media_name).suffix.lower()
                        if not ext:
                            if 'jpeg' in content_type:
                                ext = '.jpg'
                            elif 'png' in content_type:
                                ext = '.png'
                            elif 'gif' in content_type:
                                ext = '.gif'
                            else:
                                ext = '.jpg'
                        if not media_name:
                            media_name = f"image_{i}{ext}"
                        # Capability-based check: try verifying with Pillow in-memory
                        try:
                            with Image.open(BytesIO(image_data)) as im:
                                im.verify()
                        except (UnidentifiedImageError, OSError, RuntimeError) as e:
                            reason = f"Skipped image '{media_name}' ({content_type or ext}): unsupported image format - {e}"
                            logger.warning(reason)
                            self._file_comments.append(reason)
                            continue
                        except Exception as e:
                            reason = f"Skipped image '{media_name}' ({content_type or ext}): image verification error - {e}"
                            logger.warning(reason)
                            self._file_comments.append(reason)
                            continue
                        image_filename = media_name
                        temp_image_path = temp_dir / image_filename
                        
                        # Ensure proper file closure and handle Windows file locking
                        try:
                            with open(temp_image_path, 'wb') as f:
                                f.write(image_data)
                                f.flush()  # Ensure data is written to disk
                                os.fsync(f.fileno())  # Force OS to write to disk
                            
                            image_info = {
                                'temp_path': temp_image_path,
                                'original_rel': rel,
                                'location': media_name,
                                'content_type': content_type
                            }
                            image_info_list.append(image_info)
                            logger.debug(f"Extracted image: {temp_image_path}")
                        except Exception as e:
                            logger.error(f"Failed to write image data to {temp_image_path}: {e}")
                            # Add doc-level comment for write failure
                            try:
                                self._file_comments.append(f"Skipped image '{media_name}' ({content_type}): write error - {e}")
                            except Exception:
                                pass
                            # Clean up partial file if it exists
                            if temp_image_path.exists():
                                self._safe_cleanup_temp_file(temp_image_path)
                    except Exception as e:
                        logger.error(f"Failed to extract image from {rel.target_ref}: {e}")
                        # Add doc-level comment for extraction failure
                        try:
                            failed_name = None
                            try:
                                failed_name = Path(rel.target_ref).name
                            except Exception:
                                failed_name = str(rel.target_ref)
                            ct = getattr(rel.target_part, 'content_type', '') if hasattr(rel, 'target_part') else ''
                            self._file_comments.append(f"Skipped image '{failed_name}' ({ct}): extraction failed - {e}")
                        except Exception:
                            pass
            logger.debug(f"Extracted {len(image_info_list)} images from document")
        except Exception as e:
            logger.error(f"Error extracting images from document: {e}")
        # Note: temp directory cleanup is handled at the end of _process_images method
        return image_info_list

    def _store_ocr_comparison_data(self, location: str, image_path: Path, ocr_comparison: Dict[str, List[OCRResult]]):
        try:
            # Store stable, document-relative image reference instead of temp absolute path
            comparison_entry = {
                'location': location,
                'image_path': location,  # Use original media name/path (e.g., image1.png)
                'image_name': image_path.name,
                'temp_image_path': str(image_path),  # Keep temp path for debugging only
                'easyocr_results': [
                    {'text': r.text, 'confidence': r.confidence, 'bounding_box': r.bounding_box}
                    for r in ocr_comparison.get('easyocr', [])
                ],
                'tesseract_results': [
                    {'text': r.text, 'confidence': r.confidence, 'bounding_box': r.bounding_box}
                    for r in ocr_comparison.get('tesseract', [])
                ],
                'easyocr_text_count': len(ocr_comparison.get('easyocr', [])),
                'tesseract_text_count': len(ocr_comparison.get('tesseract', [])),
                'easyocr_texts': [r.text for r in ocr_comparison.get('easyocr', [])],
                'tesseract_texts': [r.text for r in ocr_comparison.get('tesseract', [])]
            }
            self.ocr_comparison_data.append(comparison_entry)
        except Exception as e:
            logger.error(f"Failed to store OCR comparison data for {location}: {e}")

    def get_ocr_comparison_data(self) -> List[Dict]:
        return self.ocr_comparison_data

    def _replace_image_in_document(self, original_rel, modified_image_path: Path) -> bool:
        try:
            with open(modified_image_path, 'rb') as f:
                modified_image_data = f.read()
            original_rel.target_part._blob = modified_image_data
            logger.debug(f"Successfully replaced image data in document: {modified_image_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to replace image in document: {e}")
            return False

    def get_processing_info(self) -> Dict[str, Any]:
        base_info = {
            'mode': self.mode,
            'ocr_engine': self.ocr_engine,
            'use_gpu': self.use_gpu,
            'confidence_threshold': self.confidence_min,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings),
            'preprocessing_enabled': True,
            'debugging_enabled': self.enable_debugging
        }
        if hasattr(self.pattern_matcher, 'get_debug_info'):
            base_info['pattern_matcher_info'] = self.pattern_matcher.get_debug_info()
        return base_info

    def _generate_output_path(self, input_path: Path) -> Path:
        try:
            output_dir = input_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem
            suffix = input_path.suffix
            output_filename = f"{stem}_modified{suffix}"
            output_path = output_dir / output_filename
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate output path: {e}")
            return input_path

    def _find_pattern_matches_in_ocr(self, ocr_results: List[OCRResult]) -> List[tuple]:
        """Finds all pattern matches within a list of OCR results."""
        all_matches = []
        for result in ocr_results:
            text = result.text.strip()
            if not text:
                continue
            
            if self.mode in ["append", "append-image"]:
                pattern_matches = self.pattern_matcher.find_all_pattern_matches(text)
            else:
                universal_matches = self.pattern_matcher.find_matches_universal(text, include_unmapped=True)
                pattern_matches = [(um.pattern_name, um.matched_text, um.start_pos, um.end_pos) for um in universal_matches]
            
            for pm in pattern_matches:
                pattern_name, matched_text, start, end = pm
                # Preserve character indices for bounding-box refinement later
                all_matches.append((pattern_name, matched_text, start, end, result))
        return all_matches

    def _run_ocr_with_osd_rotation(self, image_path: Path) -> List[OCRResult]:
        """Efficiently corrects orientation using Tesseract OSD and runs OCR."""
        logger.info(f"Attempting Tesseract OSD orientation correction for {image_path.name}")
        try:
            import pytesseract
            import re
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
                    results = self.ocr_manager.process_hybrid_with_image(image_path, corrected_img)
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

    def _safe_cleanup_temp_file(self, temp_path: Path) -> None:
        """
        Safely clean up temporary files with Windows-specific handling.
        Handles file locking issues and permission problems.
        """
        if not temp_path or not temp_path.exists():
            return
        
        try:
            # Windows-specific: Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Try multiple cleanup strategies
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Standard deletion
                    temp_path.unlink()
                    logger.debug(f"Successfully cleaned up temp file: {temp_path.name}")
                    return
                except PermissionError as e:
                    if attempt < max_attempts - 1:
                        # Wait a bit and try again (Windows file locking)
                        import time
                        time.sleep(0.1)
                        logger.debug(f"Permission error on attempt {attempt + 1}, retrying: {e}")
                        continue
                    else:
                        logger.warning(f"Could not delete temp file after {max_attempts} attempts: {e}")
                        # Try to mark for deletion on Windows
                        try:
                            import os
                            os.remove(str(temp_path))
                        except:
                            pass
                except FileNotFoundError:
                    # File already deleted, that's fine
                    logger.debug(f"Temp file already deleted: {temp_path.name}")
                    return
                except Exception as e:
                    logger.warning(f"Unexpected error cleaning up temp file {temp_path.name}: {e}")
                    return
                    
        except Exception as e:
            logger.error(f"Failed to clean up temp file {temp_path.name}: {e}")

    def _safe_cleanup_temp_directory(self, temp_dir: Path) -> None:
        """
        Safely clean up temporary directories with Windows-specific handling.
        Handles file locking issues and permission problems.
        """
        if not temp_dir or not temp_dir.exists():
            return
        
        try:
            # Windows-specific: Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Try multiple cleanup strategies
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Standard deletion
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Successfully cleaned up temp directory: {temp_dir}")
                    return
                except PermissionError as e:
                    if attempt < max_attempts - 1:
                        # Wait a bit and try again (Windows file locking)
                        import time
                        time.sleep(0.2)
                        logger.debug(f"Permission error on attempt {attempt + 1}, retrying: {e}")
                        continue
                    else:
                        logger.warning(f"Could not delete temp directory after {max_attempts} attempts: {e}")
                        # Try to remove individual files first
                        try:
                            for root, dirs, files in os.walk(temp_dir, topdown=False):
                                for file in files:
                                    try:
                                        os.remove(os.path.join(root, file))
                                    except:
                                        pass
                                for dir in dirs:
                                    try:
                                        os.rmdir(os.path.join(root, dir))
                                    except:
                                        pass
                            os.rmdir(temp_dir)
                        except:
                            pass
                except FileNotFoundError:
                    # Directory already deleted, that's fine
                    logger.debug(f"Temp directory already deleted: {temp_dir}")
                    return
                except Exception as e:
                    logger.warning(f"Unexpected error cleaning up temp directory {temp_dir}: {e}")
                    return
                    
        except Exception as e:
            logger.error(f"Failed to clean up temp directory {temp_dir}: {e}")

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
                enhanced_results = self.ocr_manager.process_hybrid_with_image(image_path, enhanced_img)
                if enhanced_results:
                    return enhanced_results
        except Exception as e:
            logger.warning(f"Image enhancement fallback failed: {e}")

        logger.warning(f"All fallback OCR methods failed for {image_path.name}")
        return []
