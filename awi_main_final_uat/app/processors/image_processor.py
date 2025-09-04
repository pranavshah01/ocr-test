"""
Ported Image Processor (from UAT) with minimal changes:
- Imports remapped to app structure and config
- Result storage integrated with app/core/models.ProcessingResult and MatchDetail
- Added process_images(document, processing_result, file_path, pattern_matcher=None)
All processing logic remains line-by-line equivalent where possible.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from docx import Document

from ..core.processor_interface import BaseProcessor, ProcessingResult as UATProcessingResult
from ..core.models import (
    ProcessingResult, MatchDetail, ImageReasoning, ProcessorType, MatchFlag
)
from ..utils.image_utils.ocr_models import OCRResult, OCRMatch
from ..utils.image_utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.image_utils.hybrid_ocr_manager import HybridOCRManager
from ..utils.image_utils.precise_replace import create_precise_text_replacer
from ..utils.image_utils.precise_append import PreciseAppendReplacer
from config import (
    DEFAULT_MAPPING,
    DEFAULT_SEPARATOR,
    PROCESSING_MODES,
    DEFAULT_OCR_CONFIDENCE
)
from ..utils.image_utils.platform_utils import PathManager

logger = logging.getLogger(__name__)

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

        self.pattern_matcher = None
        self.ocr_manager = None
        self.ocr_comparison_data = []
        self.text_replacer = None
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
            self.text_replacer = create_precise_text_replacer(self.pattern_matcher)
            self._append_replacer_for_reconstruction = PreciseAppendReplacer(self.pattern_matcher)

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
                has_specific_mapping = m.replacement_text is not None and m.replacement_text != ""
                match_detail = MatchDetail(
                    sr_no=sr_no,
                    type=ProcessorType.IMAGE,
                    orig_id_name=m.location,
                    src_text=m.extracted_pattern_text or m.original_text,
                    src_bbox=','.join(str(int(x)) for x in (
                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    )) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else "",
                    mapped_text=mapped_text,
                    match_flag=MatchFlag.YES if has_specific_mapping else MatchFlag.NO,
                    reasoning=ImageReasoning(
                        available_size_for_new_text=str(getattr(m, 'dimension', '')),
                        total_characters=len(mapped_text),
                        line_reasoning=str(getattr(m, 'reconstruction_reasoning', ''))
                    ),
                    reconstructed=False
                )
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
            processing_result.success = True
            return processing_result
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            processing_result.error_message = str(e)
            return processing_result

    # --- Below methods mirror UAT logic with minimal/no changes ---
    def _process_images(self, document: Document, **kwargs) -> List[EnhancedImageMatch]:
        logger.info("Starting 3-phase image processing...")
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

    def _phase1_extraction(self, document: Document, **kwargs) -> List[Dict]:
        extraction_results: List[Dict[str, Any]] = []
        try:
            image_info_list = self._extract_images_with_mapping(document, **kwargs)
            for image_info in image_info_list:
                try:
                    logger.debug(f"Extracting text from: {image_info['location']}")
                    ocr_results = self.ocr_manager.process_hybrid(image_info['temp_path'])
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
                        universal_matches = self.pattern_matcher.find_matches_universal(ocr_result.text)
                        for um in universal_matches:
                            replacement_text = self.pattern_matcher.get_replacement(um.matched_text)
                            if not replacement_text:
                                if self.mode == "replace":
                                    continue
                                elif self.mode in ["append", "append-image"]:
                                    replacement_text = self.default_mapping
                                else:
                                    continue
                            font_info = getattr(ocr_result, 'font_info', {})
                            ocr_match = EnhancedImageMatch(
                                um.pattern_name, um.matched_text, replacement_text,
                                um.start_pos, image_info['location'],
                                font_info, ocr_result.confidence, um.pattern_name,
                                "Image", ocr_result.bounding_box, "OCR", ocr_result
                            )
                            ocr_match.extracted_pattern_text = um.matched_text
                            ocr_match.original_text = ocr_result.text
                            matches.append(ocr_match)
                    except Exception as e:
                        logger.error(f"Error in pattern matching for text '{ocr_result.text}': {e}")
                wiped_image_path = None
                if matches:
                    wiped_image_path = self._apply_wipes_to_image(image_info['temp_path'], matches)
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
                matches = wipe_result['matches']
                wiped_image_path = wipe_result['wiped_image_path']
                logger.debug(f"Reconstructing: {image_info['location']}")
                if wiped_image_path and matches:
                    try:
                        ocr_results_for_image = []
                        for m in matches:
                            if hasattr(m, 'ocr_result') and m.ocr_result:
                                ocr_results_for_image.append(m.ocr_result)
                            else:
                                ocr_results_for_image.append(OCRResult(text=m.original_text, confidence=getattr(m, 'confidence', 0.0), bounding_box=getattr(m, 'bounding_box', (0, 0, 0, 0))))
                        reconstructed_path = self._append_replacer_for_reconstruction.replace_text_in_image(
                            Path(wiped_image_path), ocr_results_for_image, matches
                        )
                        image_to_insert = reconstructed_path or wiped_image_path
                    except Exception as e:
                        logger.error(f"Error in reconstruction for {image_info['location']}: {e}")
                        image_to_insert = wiped_image_path
                    success = self._replace_image_in_document(image_info['original_rel'], image_to_insert)
                    if success:
                        for match in matches:
                            match.image_path = image_to_insert
                        final_matches.extend(matches)
                        logger.info(f"RECONSTRUCTED: {len(matches)} matches in {image_info['location']}")
                    else:
                        logger.warning(f"Failed to reconstruct image in document for {image_info['location']}")
                else:
                    logger.debug(f"No reconstruction needed for {image_info['location']}")
            except Exception as e:
                logger.error(f"Error in reconstruction phase for {wipe_result['image_info']['location']}: {e}")
        logger.info(f"Reconstruction phase completed: {len(final_matches)} total matches")
        return final_matches

    def _apply_wipes_to_image(self, image_path: Path, matches: List[EnhancedImageMatch]) -> Optional[Path]:
        try:
            image = Image.open(image_path)
            for match in matches:
                try:
                    image = self._apply_single_wipe(image, match)
                except Exception as e:
                    logger.error(f"Failed to apply wipe for match '{match.original_text}': {e}")
            output_path = self._generate_output_path(image_path)
            image.save(output_path)
            logger.debug(f"Wipe application completed: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to apply wipes to image {image_path}: {e}")
            return None

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

    def _apply_single_wipe(self, image: Image.Image, match: EnhancedImageMatch) -> Image.Image:
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
            if hasattr(self, 'text_replacer') and self.text_replacer:
                cv_image = self.text_replacer._apply_hybrid_replace(cv_image, OCRMatch(ocr_result=ocr_result, pattern=match.pattern, replacement_text=match.replacement_text, image_path=Path(""), processing_mode=self.mode), [ocr_result])
            else:
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            wiped_image = Image.fromarray(cv_image_rgb)
            match.reconstruction_reasoning = reasoning
            logger.debug(f"Applied wipe for '{original_full_text}'")
            return wiped_image
        except Exception as e:
            logger.error(f"Failed to apply single wipe: {e}")
            return image

    def _extract_images_with_mapping(self, document: Document, **kwargs) -> List[Dict]:
        image_info_list: List[Dict[str, Any]] = []
        try:
            temp_dir = PathManager.get_temp_directory() / f"enhanced_docx_images_{uuid.uuid4().hex[:8]}"
            PathManager.ensure_directory(temp_dir)
            for i, rel in enumerate(document.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        content_type = rel.target_part.content_type
                        if 'jpeg' in content_type:
                            ext = '.jpg'
                        elif 'png' in content_type:
                            ext = '.png'
                        elif 'gif' in content_type:
                            ext = '.gif'
                        else:
                            ext = '.jpg'
                        image_filename = f"image_{i}{ext}"
                        temp_image_path = temp_dir / image_filename
                        with open(temp_image_path, 'wb') as f:
                            f.write(image_data)
                        image_info = {
                            'temp_path': temp_image_path,
                            'original_rel': rel,
                            'location': f"image_{i}",
                            'content_type': content_type
                        }
                        image_info_list.append(image_info)
                        logger.debug(f"Extracted image: {temp_image_path}")
                    except Exception as e:
                        logger.error(f"Failed to extract image from {rel.target_ref}: {e}")
            logger.debug(f"Extracted {len(image_info_list)} images from document")
        except Exception as e:
            logger.error(f"Error extracting images from document: {e}")
        return image_info_list

    def _store_ocr_comparison_data(self, location: str, image_path: Path, ocr_comparison: Dict[str, List[OCRResult]]):
        try:
            comparison_entry = {
                'location': location,
                'image_path': str(image_path),
                'image_name': image_path.name,
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
