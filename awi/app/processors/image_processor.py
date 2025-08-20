"""
Enhanced image processor for OCR-based image text detection and replacement.
Handles image extraction, OCR processing, and text replacement with GPU support,
universal pattern matching, and orientation-aware text replacement.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import pytesseract

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Import existing components
from ..core.models import OCRResult, OCRMatch, Match, HybridOCRResult, create_ocr_result, create_ocr_match, create_hybrid_ocr_result
from ..utils.shared_constants import DEFAULT_OCR_CONFIDENCE, FALLBACK_FONTS
from ..utils.platform_utils import PathManager

# Import enhanced components from image_utils
from ..utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.image_utils.image_preprocessor import ImagePreprocessor, create_image_preprocessor
from ..utils.image_utils.preprocessing_strategies import PreprocessingStrategyManager, create_preprocessing_strategy_manager
from ..utils.image_utils.text_analyzer import TextAnalyzer, TextProperties, create_text_analyzer
from ..utils.image_utils.precise_replace import PreciseTextReplacer, create_precise_text_replacer
from ..utils.image_utils.pattern_debugger import PatternDebugger, create_pattern_debugger
from ..utils.image_utils.hybrid_ocr_manager import HybridOCRManager, create_hybrid_ocr_manager

from ..utils.image_utils.precise_append import PreciseAppendReplacer, create_precise_append_replacer

logger = logging.getLogger(__name__)

class OCREngine:
    """Enhanced OCR engine with preprocessing and hybrid mode support."""
    
    def __init__(self, engine: str = "easyocr", use_gpu: bool = True, 
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE, enable_preprocessing: bool = True):
        """
        Initialize enhanced OCR engine.
        
        Args:
            engine: OCR engine to use ('easyocr', 'tesseract', or 'hybrid')
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence threshold for OCR results
            enable_preprocessing: Whether to enable image preprocessing
        """
        self.engine = engine
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        self.reader = None
        self.hybrid_manager = None
        
        # Initialize enhanced components
        if self.enable_preprocessing:
            self.preprocessor = create_image_preprocessor()
            self.strategy_manager = create_preprocessing_strategy_manager()
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the OCR engine with GPU detection and fallback."""
        try:
            if self.engine == "easyocr":
                self._initialize_easyocr()
            elif self.engine == "tesseract":
                self._initialize_tesseract()
            elif self.engine == "hybrid":
                self._initialize_hybrid()
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
                
            logger.info(f"Enhanced OCR engine '{self.engine}' initialized successfully "
                       f"(GPU: {self.use_gpu}, Preprocessing: {self.enable_preprocessing})")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced OCR engine '{self.engine}': {e}")
            # Try fallback to CPU
            if self.use_gpu and self.engine != "hybrid":
                logger.info("Attempting fallback to CPU...")
                self.use_gpu = False
                self._initialize_engine()
            else:
                raise
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR with GPU support."""
        try:
            import torch
            
            # Detect GPU availability
            gpu_available = False
            if self.use_gpu:
                if torch.cuda.is_available():
                    gpu_available = True
                    logger.info("Using CUDA GPU for Enhanced EasyOCR")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    logger.info("Using MPS GPU for Enhanced EasyOCR")
                else:
                    logger.info("GPU requested but not available, using CPU for Enhanced EasyOCR")
            
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(['en'], gpu=gpu_available)
            self.use_gpu = gpu_available
            
        except Exception as e:
            logger.error(f"Enhanced EasyOCR initialization failed: {e}")
            raise
    
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            logger.info(f"Using Enhanced Tesseract version: {version}")
            
            # Tesseract doesn't use GPU, so set use_gpu to False
            self.use_gpu = False
            
        except Exception as e:
            logger.error(f"Enhanced Tesseract initialization failed: {e}")
            raise
    
    def _initialize_hybrid(self):
        """Initialize Hybrid OCR Manager."""
        try:
            self.hybrid_manager = create_hybrid_ocr_manager(
                max_workers=2,
                confidence_threshold=self.confidence_threshold,
                use_gpu=self.use_gpu,
                overlap_threshold=0.5
            )
            logger.info("Hybrid OCR Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Hybrid OCR Manager initialization failed: {e}")
            raise
    
    def extract_text(self, image_path: Path) -> List[OCRResult]:
        """
        Extract text using enhanced preprocessing and multiple strategies.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of OCRResult objects (converted from HybridOCRResult for hybrid mode)
        """
        try:
            if self.engine == "hybrid":
                return self._extract_with_hybrid(image_path)
            elif self.enable_preprocessing:
                return self._extract_with_preprocessing(image_path)
            else:
                return self._extract_without_preprocessing(image_path)
                
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed for {image_path}: {e}")
            # Try fallback engine
            return self._try_fallback_extraction(image_path)
    
    def extract_text_comparison(self, image_path: Path) -> Dict[str, List[OCRResult]]:
        """
        Extract text using both EasyOCR and Tesseract for comparison.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with 'easyocr' and 'tesseract' keys containing OCR results
        """
        comparison_results = {
            'easyocr': [],
            'tesseract': []
        }
        
        try:
            # For hybrid mode, use the hybrid manager to get individual results
            if self.engine == "hybrid" and self.hybrid_manager:
                try:
                    easyocr_results, tesseract_results = self.hybrid_manager.execute_ocr_parallel(image_path)
                    comparison_results['easyocr'] = easyocr_results
                    comparison_results['tesseract'] = tesseract_results
                    logger.debug(f"Hybrid comparison: EasyOCR={len(easyocr_results)}, Tesseract={len(tesseract_results)}")
                except Exception as e:
                    logger.error(f"Hybrid comparison extraction failed: {e}")
            else:
                # Extract with EasyOCR
                try:
                    if self.engine in ["easyocr", "hybrid"] and self.reader:
                        easyocr_results = self._extract_with_easyocr(image_path)
                        comparison_results['easyocr'] = easyocr_results
                        logger.debug(f"EasyOCR extracted {len(easyocr_results)} text regions from {image_path}")
                except Exception as e:
                    logger.error(f"EasyOCR extraction failed: {e}")
                
                # Extract with Tesseract
                try:
                    tesseract_results = self._extract_with_tesseract(image_path)
                    comparison_results['tesseract'] = tesseract_results
                    logger.debug(f"Tesseract extracted {len(tesseract_results)} text regions from {image_path}")
                except Exception as e:
                    logger.error(f"Tesseract extraction failed: {e}")
                
        except Exception as e:
            logger.error(f"OCR comparison extraction failed for {image_path}: {e}")
        
        return comparison_results
    
    def _extract_with_preprocessing(self, image_path: Path) -> List[OCRResult]:
        """Extract text with preprocessing strategies."""
        try:
            # Generate preprocessed image variants
            image_variants = self.strategy_manager.process_image_with_fallbacks(image_path, False)
            
            if not image_variants:
                logger.warning(f"No image variants generated for {image_path}")
                return []
            
            all_results = []
            best_results = []
            best_confidence = 0.0
            
            # Try OCR on each variant
            for i, variant in enumerate(image_variants):
                try:
                    # Log variant info for debugging
                    if variant is not None and hasattr(variant, 'shape'):
                        logger.debug(f"Processing variant {i}: shape={variant.shape}, dtype={variant.dtype}")
                    else:
                        logger.debug(f"Processing variant {i}: invalid variant (None or no shape)")
                    
                    # Extract text from variant
                    variant_results = self._extract_from_image_array(variant)
                    
                    # Calculate average confidence
                    if variant_results:
                        avg_confidence = sum(r.confidence for r in variant_results) / len(variant_results)
                        
                        # Keep track of best results
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_results = variant_results
                        
                        all_results.extend(variant_results)
                        
                        logger.debug(f"Variant {i}: {len(variant_results)} results, "
                                   f"avg confidence: {avg_confidence:.2f}")
                        
                except Exception as e:
                    logger.warning(f"OCR failed on variant {i}: {e}")
            
            # Return best results or all results if no clear winner
            final_results = best_results if best_results else all_results
            
            # Remove duplicates based on text and position
            final_results = self._deduplicate_ocr_results(final_results)
            
            logger.info(f"Enhanced OCR extracted {len(final_results)} unique text regions "
                       f"from {len(image_variants)} variants")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Preprocessing-based extraction failed: {e}")
            return self._extract_without_preprocessing(image_path)
    
    def _extract_with_hybrid(self, image_path: Path) -> List[OCRResult]:
        """Extract text using hybrid OCR manager."""
        try:
            if self.hybrid_manager is None:
                logger.error("Hybrid OCR manager not initialized")
                return []
            
            # Process with hybrid OCR manager
            hybrid_results = self.hybrid_manager.process_hybrid(image_path)
            
            # Convert HybridOCRResult objects to OCRResult objects for compatibility
            ocr_results = []
            for hybrid_result in hybrid_results:
                ocr_result = create_ocr_result(
                    text=hybrid_result.text,
                    confidence=hybrid_result.confidence,
                    bbox=hybrid_result.bounding_box
                )
                # Store hybrid information as metadata for debugging
                ocr_result._hybrid_info = {
                    'source_engine': hybrid_result.source_engine,
                    'selection_reason': hybrid_result.selection_reason,
                    'conflict_resolved': hybrid_result.conflict_resolved
                }
                ocr_results.append(ocr_result)
            
            # TODO: Text region merging needs refinement - disabled for now
            # merged_results = self._merge_adjacent_text_regions(ocr_results)
            
            logger.info(f"Hybrid OCR extracted {len(ocr_results)} text regions from {image_path}")
            return ocr_results
            
        except Exception as e:
            logger.error(f"Hybrid OCR extraction failed: {e}")
            return []
    
    def _extract_without_preprocessing(self, image_path: Path) -> List[OCRResult]:
        """Extract text without preprocessing (fallback method)."""
        try:
            if self.engine == "easyocr":
                return self._extract_with_easyocr(image_path)
            elif self.engine == "tesseract":
                return self._extract_with_tesseract(image_path)
            elif self.engine == "hybrid":
                return self._extract_with_hybrid(image_path)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
                
        except Exception as e:
            logger.error(f"Standard OCR extraction failed: {e}")
            return []
    
    def _extract_from_image_array(self, image_array: np.ndarray) -> List[OCRResult]:
        """Extract text from numpy image array."""
        try:
            # Validate image array
            if image_array is None:
                logger.warning("Image array is None, skipping OCR extraction")
                return []
            
            if not isinstance(image_array, np.ndarray):
                logger.warning(f"Invalid image array type: {type(image_array)}, skipping OCR extraction")
                return []
            
            if image_array.size == 0:
                logger.warning("Empty image array, skipping OCR extraction")
                return []
            
            # Check if image has valid dimensions
            if len(image_array.shape) < 2:
                logger.warning(f"Invalid image array shape: {image_array.shape}, skipping OCR extraction")
                return []
            
            if self.engine == "easyocr":
                return self._extract_with_easyocr_array(image_array)
            elif self.engine == "tesseract":
                return self._extract_with_tesseract_array(image_array)
            else:
                return []
                
        except Exception as e:
            logger.debug(f"Array-based OCR extraction failed: {e}")
            return []
    
    def _extract_with_easyocr(self, image_path: Path) -> List[OCRResult]:
        """Extract text using EasyOCR from file path."""
        results = []
        
        try:
            # Validate image path
            if not image_path.exists():
                logger.warning(f"Image file does not exist: {image_path}")
                return []
            
            if image_path.stat().st_size == 0:
                logger.warning(f"Image file is empty: {image_path}")
                return []
            
            # Read image and perform OCR
            ocr_results = self.reader.readtext(str(image_path))
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                # Filter by confidence threshold
                if confidence >= self.confidence_threshold:
                    # Convert bbox to (x, y, width, height) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    ocr_result = create_ocr_result(text, confidence, (x, y, width, height))
                    results.append(ocr_result)
            
            logger.debug(f"Enhanced EasyOCR extracted {len(results)} text regions from {image_path}")
            
        except Exception as e:
            logger.error(f"Enhanced EasyOCR extraction failed: {e}")
            raise
        
        return results
    
    def _extract_with_easyocr_array(self, image_array: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR from numpy array."""
        results = []
        
        try:
            # Additional validation for EasyOCR specific requirements
            if image_array.dtype not in [np.uint8, np.float32, np.float64]:
                logger.warning(f"Invalid image array dtype for EasyOCR: {image_array.dtype}")
                return []
            
            # EasyOCR can work with numpy arrays directly
            ocr_results = self.reader.readtext(image_array)
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                # Filter by confidence threshold
                if confidence >= self.confidence_threshold:
                    # Convert bbox to (x, y, width, height) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    ocr_result = create_ocr_result(text, confidence, (x, y, width, height))
                    results.append(ocr_result)
            
        except Exception as e:
            logger.debug(f"Enhanced EasyOCR array extraction failed: {e}")
        
        return results
    
    def _extract_with_tesseract(self, image_path: Path) -> List[OCRResult]:
        """Extract text using Tesseract OCR from file path."""
        results = []
        
        try:
            # Validate image path
            if not image_path.exists():
                logger.warning(f"Image file does not exist: {image_path}")
                return []
            
            if image_path.stat().st_size == 0:
                logger.warning(f"Image file is empty: {image_path}")
                return []
            
            # Check file extension for supported formats
            supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
            if image_path.suffix.lower() not in supported_formats:
                logger.warning(f"Unsupported image format: {image_path.suffix} for {image_path}")
                return []
            
            # Load image
            image = Image.open(image_path)
            
            # Validate loaded image
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return []
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Process OCR results
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = float(ocr_data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                if text and confidence >= self.confidence_threshold:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    width = ocr_data['width'][i]
                    height = ocr_data['height'][i]
                    
                    ocr_result = create_ocr_result(text, confidence, (x, y, width, height))
                    results.append(ocr_result)
            
            logger.debug(f"Enhanced Tesseract extracted {len(results)} text regions from {image_path}")
            
        except Exception as e:
            logger.error(f"Enhanced Tesseract extraction failed: {e}")
            raise
        
        return results
    
    def _extract_with_tesseract_array(self, image_array: np.ndarray) -> List[OCRResult]:
        """Extract text using Tesseract OCR from numpy array."""
        results = []
        
        try:
            # Convert numpy array to PIL Image
            if len(image_array.shape) == 3:
                # BGR to RGB conversion for OpenCV arrays
                try:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_rgb)
                except Exception as e:
                    logger.warning(f"Failed to convert BGR to RGB: {e}")
                    # Try direct conversion
                    image = Image.fromarray(image_array)
            else:
                image = Image.fromarray(image_array)
            
            # Validate converted image
            if image is None:
                logger.warning("Failed to convert numpy array to PIL Image")
                return []
            
            if image.size[0] == 0 or image.size[1] == 0:
                logger.warning(f"Invalid image dimensions: {image.size}")
                return []
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Process OCR results
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = float(ocr_data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                if text and confidence >= self.confidence_threshold:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    width = ocr_data['width'][i]
                    height = ocr_data['height'][i]
                    
                    ocr_result = create_ocr_result(text, confidence, (x, y, width, height))
                    results.append(ocr_result)
            
        except Exception as e:
            logger.debug(f"Enhanced Tesseract array extraction failed: {e}")
        
        return results
    
    def _deduplicate_ocr_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """Remove duplicate OCR results based on text and position."""
        if not results:
            return results
        
        unique_results = []
        seen_combinations = set()
        
        for result in results:
            # Create a key based on text and approximate position
            key = (result.text.strip().lower(), 
                   result.bounding_box[0] // 10,  # Quantize position to reduce minor differences
                   result.bounding_box[1] // 10)
            
            if key not in seen_combinations:
                unique_results.append(result)
                seen_combinations.add(key)
        
        logger.debug(f"Deduplicated OCR results: {len(results)} -> {len(unique_results)}")
        return unique_results
    
    def _try_fallback_extraction(self, image_path: Path) -> List[OCRResult]:
        """Try fallback OCR engine if primary fails."""
        try:
            # For hybrid mode, try individual engines as fallback
            if self.engine == "hybrid":
                logger.info("Trying fallback to individual OCR engines after hybrid failure")
                # Try EasyOCR first, then Tesseract
                for fallback_engine in ["easyocr", "tesseract"]:
                    try:
                        fallback = OCREngine(fallback_engine, use_gpu=False, 
                                           confidence_threshold=self.confidence_threshold,
                                           enable_preprocessing=False)
                        results = fallback.extract_text(image_path)
                        if results:
                            logger.info(f"Fallback {fallback_engine} succeeded with {len(results)} results")
                            return results
                    except Exception as e:
                        logger.warning(f"Fallback {fallback_engine} also failed: {e}")
                return []
            else:
                # For single engines, try the other engine
                fallback_engine = "tesseract" if self.engine == "easyocr" else "easyocr"
                logger.info(f"Trying fallback enhanced OCR engine: {fallback_engine}")
                
                # Create temporary fallback engine
                fallback = OCREngine(fallback_engine, use_gpu=False, 
                                   confidence_threshold=self.confidence_threshold,
                                   enable_preprocessing=False)  # Disable preprocessing for fallback
                return fallback.extract_text(image_path)
            
        except Exception as e:
            logger.error(f"Fallback enhanced OCR also failed: {e}")
            return []
    
    def _merge_adjacent_text_regions(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """
        Merge adjacent OCR text regions that are likely part of the same line.
        This helps capture full text lines that may be split by OCR engines.
        
        Args:
            ocr_results: List of OCR results to merge
            
        Returns:
            List of merged OCR results
        """
        if len(ocr_results) <= 1:
            return ocr_results
        
        try:
            # Sort results by vertical position (y-coordinate) first, then horizontal (x-coordinate)
            sorted_results = sorted(ocr_results, key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
            
            merged_results = []
            current_group = [sorted_results[0]]
            
            for i in range(1, len(sorted_results)):
                current = sorted_results[i]
                last_in_group = current_group[-1]
                
                # Check if current result should be merged with the group
                if self._should_merge_text_regions(last_in_group, current):
                    current_group.append(current)
                else:
                    # Finalize current group and start new one
                    if len(current_group) > 1:
                        merged_result = self._merge_text_group(current_group)
                        merged_results.append(merged_result)
                    else:
                        merged_results.append(current_group[0])
                    
                    current_group = [current]
            
            # Handle the last group
            if len(current_group) > 1:
                merged_result = self._merge_text_group(current_group)
                merged_results.append(merged_result)
            else:
                merged_results.append(current_group[0])
            
            logger.debug(f"Merged {len(ocr_results)} OCR regions into {len(merged_results)} regions")
            return merged_results
            
        except Exception as e:
            logger.error(f"Text region merging failed: {e}")
            return ocr_results
    
    def _should_merge_text_regions(self, region1: OCRResult, region2: OCRResult) -> bool:
        """
        Determine if two OCR regions should be merged based on proximity and alignment.
        
        Args:
            region1: First OCR region
            region2: Second OCR region
            
        Returns:
            True if regions should be merged
        """
        try:
            x1, y1, w1, h1 = region1.bounding_box
            x2, y2, w2, h2 = region2.bounding_box
            
            # Calculate vertical overlap and proximity
            y1_center = y1 + h1 / 2
            y2_center = y2 + h2 / 2
            vertical_distance = abs(y1_center - y2_center)
            
            # Calculate horizontal gap
            if x1 + w1 < x2:
                horizontal_gap = x2 - (x1 + w1)
            elif x2 + w2 < x1:
                horizontal_gap = x1 - (x2 + w2)
            else:
                horizontal_gap = 0  # Overlapping horizontally
            
            # Merge criteria:
            # 1. Regions are on approximately the same line (vertical distance < average height)
            # 2. Horizontal gap is reasonable (< 3x average height)
            avg_height = (h1 + h2) / 2
            
            same_line = vertical_distance < avg_height * 0.5
            reasonable_gap = horizontal_gap < avg_height * 3
            
            should_merge = same_line and reasonable_gap
            
            if should_merge:
                logger.debug(f"Merging regions: '{region1.text}' + '{region2.text}' "
                           f"(v_dist: {vertical_distance:.1f}, h_gap: {horizontal_gap:.1f})")
            
            return should_merge
            
        except Exception as e:
            logger.debug(f"Region merge check failed: {e}")
            return False
    
    def _merge_text_group(self, group: List[OCRResult]) -> OCRResult:
        """
        Merge a group of OCR results into a single result.
        
        Args:
            group: List of OCR results to merge
            
        Returns:
            Single merged OCR result
        """
        try:
            # Sort group by horizontal position
            sorted_group = sorted(group, key=lambda r: r.bounding_box[0])
            
            # Combine text with spaces
            combined_text = " ".join(result.text.strip() for result in sorted_group if result.text.strip())
            
            # Calculate combined bounding box
            min_x = min(r.bounding_box[0] for r in sorted_group)
            min_y = min(r.bounding_box[1] for r in sorted_group)
            max_x = max(r.bounding_box[0] + r.bounding_box[2] for r in sorted_group)
            max_y = max(r.bounding_box[1] + r.bounding_box[3] for r in sorted_group)
            
            combined_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # Use average confidence
            avg_confidence = sum(r.confidence for r in sorted_group) / len(sorted_group)
            
            merged_result = create_ocr_result(combined_text, avg_confidence, combined_bbox)
            
            logger.debug(f"Merged {len(sorted_group)} regions into: '{combined_text}' at {combined_bbox}")
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Text group merging failed: {e}")
            return group[0]  # Return first result as fallback

class ImageTextReplacer:
    """Enhanced image text replacer with precise positioning and orientation support."""
    
    def __init__(self, mode: str = "replace", pattern_matcher=None):
        """
        Initialize enhanced image text replacer.
        
        Args:
            mode: Processing mode ('replace', 'append', or 'append-image')
            pattern_matcher: Pattern matcher instance for finding patterns in text
        """
        self.mode = mode
        self.pattern_matcher = pattern_matcher
        self.text_analyzer = create_text_analyzer()
        self.precise_replacer = create_precise_text_replacer(self.pattern_matcher)
        
        # Initialize append mode replacer if needed
        if mode == "append" and pattern_matcher:
            self.precise_append_replacer = create_precise_append_replacer(pattern_matcher)
        else:
            self.precise_append_replacer = None
    
    def replace_text_in_image(self, image_path: Path, ocr_matches: List[OCRMatch]) -> Optional[Path]:
        """
        Replace text in image using enhanced precise positioning and orientation support.
        
        Args:
            image_path: Path to original image
            ocr_matches: List of OCR matches to replace
            
        Returns:
            Path to modified image or None if failed
        """
        if not ocr_matches:
            return None
        
        try:
            # Handle append mode separately to avoid impacting replace mode
            if self.mode == "append" and self.precise_append_replacer:
                logger.info(f"APPEND MODE: Using simplified append mode with {len(ocr_matches)} matches")
                
                # Extract OCR results from matches for the append replacer
                ocr_results = [match.ocr_result for match in ocr_matches]
                
                # Log the matches for debugging
                for i, match in enumerate(ocr_matches):
                    logger.info(f"APPEND MODE: Match {i} - OCR: '{match.ocr_result.text}' -> Replacement: '{match.replacement_text}'")
                
                # Use the simplified append replacer that actually works
                result = self.precise_append_replacer.replace_text_in_image(image_path, ocr_results, ocr_matches)
                logger.info(f"APPEND MODE: Simple append replacer result: {result}")
                return result
            
            # Original replace mode logic (unchanged)
            # Load image
            image = Image.open(image_path)
            
            logger.debug(f"Processing {len(ocr_matches)} text replacements in {image_path}")
            
            # Process each match with enhanced precision
            for match in ocr_matches:
                try:
                    if match.processing_mode == "replace":
                        image = self._replace_text_region(image, match)
                    elif match.processing_mode == "append":
                        # This should not be reached when mode is "append" since we handle it above
                        logger.warning("Append mode match found in replace mode processing - skipping")
                        continue
                except Exception as e:
                    logger.error(f"Failed to process match '{match.ocr_result.text}': {e}")
            
            # Save modified image
            output_path = self._generate_output_path(image_path)
            image.save(output_path)
            
            logger.debug(f"Enhanced image text replacement completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Enhanced image text replacement failed for {image_path}: {e}")
            return None
    
    def _create_partial_replacement_text(self, original_text: str, match: OCRMatch) -> str:
        """
        Create new text by replacing only the matched pattern within the original text.
        
        Args:
            original_text: Full original OCR text
            match: OCRMatch containing pattern and replacement info
            
        Returns:
            New text with only the pattern part replaced
        """
        try:
            # Use the pattern matcher to find the exact match within the text
            pattern_matches = self.pattern_matcher.find_matches_universal(original_text)
            
            if not pattern_matches:
                logger.warning(f"No pattern matches found in text: '{original_text}'")
                return original_text
            
            # Find the match that corresponds to our replacement
            target_match = None
            for pm in pattern_matches:
                if pm.matched_text in self.pattern_matcher.mappings:
                    replacement = self.pattern_matcher.get_replacement(pm.matched_text)
                    if replacement == match.replacement_text:
                        target_match = pm
                        break
            
            if not target_match:
                logger.warning(f"Could not find target match for replacement: '{match.replacement_text}'")
                return original_text
            
            # Replace only the matched pattern within the original text
            new_text = (original_text[:target_match.start_pos] + 
                       match.replacement_text + 
                       original_text[target_match.end_pos:])
            
            logger.debug(f"Partial replacement: '{original_text}' -> '{new_text}' "
                        f"(replaced '{target_match.matched_text}' with '{match.replacement_text}')")
            
            return new_text
            
        except Exception as e:
            logger.error(f"Failed to create partial replacement text: {e}")
            return original_text
    
    def _replace_text_region(self, image: Image.Image, match: OCRMatch) -> Image.Image:
        """Replace text region with enhanced precision and orientation support."""
        try:
            bbox = match.ocr_result.bounding_box
            original_full_text = match.ocr_result.text
            pattern_replacement = match.replacement_text
            
            # Find the pattern in the original text and create the new text
            # We need to find what pattern was matched and replace only that part
            new_full_text = self._create_partial_replacement_text(original_full_text, match)
            
            # Analyze text properties including orientation
            image_array = np.array(image)
            properties = self.text_analyzer.analyze_text_properties(image_array, bbox)
            
            # Convert PIL Image to cv2 format for precise replacement
            import numpy as np
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply the hybrid replace directly 
            cv_image = self.precise_replacer._apply_hybrid_replace(cv_image, match, [match.ocr_result])
            
            # Convert back to PIL
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            enhanced_image = Image.fromarray(cv_image_rgb)
            
            logger.debug(f"Enhanced replace: '{original_full_text}' -> '{new_full_text}' "
                        f"at {bbox} with {properties.orientation}° orientation")
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Enhanced text region replacement failed: {e}")
            return image
    
    def _append_text_region(self, image: Image.Image, match: OCRMatch) -> Image.Image:
        """Append text region with enhanced precision and orientation support."""
        try:
            bbox = match.ocr_result.bounding_box
            original_text = match.ocr_result.text
            replacement_text = match.replacement_text
            
            # Create combined text for append mode
            combined_text = f"{original_text} {replacement_text}"
            
            # Analyze text properties including orientation
            image_array = np.array(image)
            properties = self.text_analyzer.analyze_text_properties(image_array, bbox)
            
            # Adjust bounding box for combined text (rough estimation)
            x, y, width, height = bbox
            if abs(properties.orientation) < 45 or abs(properties.orientation - 180) < 45:
                # Horizontal text - extend width
                extended_bbox = (x, y, int(width * 1.5), height)
            else:
                # Vertical text - extend height
                extended_bbox = (x, y, width, int(height * 1.5))
            
            # Use precise text replacer with combined text
            enhanced_image = self.precise_replacer.replace_text_precise(
                image, original_text, combined_text, extended_bbox, properties
            )
            
            logger.debug(f"Enhanced append: '{original_text}' + '{replacement_text}' "
                        f"at {extended_bbox} with {properties.orientation}° orientation")
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Enhanced text region append failed: {e}")
            return image
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path for modified image."""
        output_dir = input_path.parent
        stem = input_path.stem
        suffix = input_path.suffix
        
        return output_dir / f"{stem}_modified{suffix}"

class ImageProcessor:
    """Enhanced image processor with universal pattern matching and orientation support."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str], 
                 mode: str = "replace", separator: str = ";", default_mapping: str = "4022-NA",
                 ocr_engine: str = "easyocr", use_gpu: bool = True, 
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                 enable_preprocessing: bool = True, enable_debugging: bool = False):
        """
        Initialize enhanced image processor with consolidated config support.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            mode: Processing mode ('replace', 'append', or 'append-image')
            separator: Separator between original and appended text in append mode
            default_mapping: Default text to append when no mapping is found
            ocr_engine: OCR engine to use
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum OCR confidence threshold
            enable_preprocessing: Whether to enable image preprocessing
            enable_debugging: Whether to enable debugging features
        """
        self.patterns = patterns
        self.mappings = mappings
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
        self.enable_preprocessing = enable_preprocessing
        self.enable_debugging = enable_debugging
        
        # Initialize enhanced components
        self.pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
        self.ocr_engine = OCREngine(ocr_engine, use_gpu, confidence_threshold, enable_preprocessing)
        self.text_replacer = ImageTextReplacer(mode, self.pattern_matcher)
        # Always have an append replacer available for reconstruction phase
        self._append_replacer_for_reconstruction = create_precise_append_replacer(self.pattern_matcher)
        
        # Initialize debugging if enabled
        if self.enable_debugging:
            self.debugger = create_pattern_debugger(self.pattern_matcher)
        
        # Initialize OCR comparison data storage for reporting
        self.ocr_comparison_data = []
        
        logger.info(f"Enhanced image processor initialized with {len(patterns)} patterns, "
                   f"{len(mappings)} mappings, mode: {mode}, separator: '{separator}', "
                   f"default_mapping: '{default_mapping}', OCR engine: {ocr_engine}, "
                   f"preprocessing: {enable_preprocessing}, debugging: {enable_debugging}")
    
    def process_images(self, document: Document, media_dir: Optional[Path] = None) -> List[OCRMatch]:
        """
        Process all images in the document with 3-phase approach:
        1. EXTRACTION: Extract text from images using OCR
        2. MATCH & WIPE: Find patterns and wipe them (no replacement text)
        3. RECONSTRUCT: Reconstruct document (currently just replacing images)
        
        Args:
            document: Document to process
            media_dir: Media directory path (optional)
            
        Returns:
            List of OCRMatch objects representing successful matches
        """
        logger.info("Starting 3-phase image processing...")
        
        try:
            # PHASE 1: EXTRACTION - Extract images and text from document
            logger.info("=== PHASE 1: EXTRACTION ===")
            extraction_results = self._phase1_extraction(document, media_dir)
            
            # PHASE 2: MATCH & WIPE - Find patterns and apply wipes
            logger.info("=== PHASE 2: MATCH & WIPE ===")
            wipe_results = self._phase2_match_and_wipe(extraction_results)
            
            # PHASE 3: RECONSTRUCT - Reconstruct document with wiped images
            logger.info("=== PHASE 3: RECONSTRUCT ===")
            final_matches = self._phase3_reconstruct(document, wipe_results)
            
            logger.info(f"3-phase image processing completed: {len(final_matches)} matches found")
            return final_matches
            
        except Exception as e:
            logger.error(f"Error in 3-phase image processing: {e}")
            return []
    
    def _phase1_extraction(self, document: Document, media_dir: Optional[Path] = None) -> List[Dict]:
        """
        PHASE 1: EXTRACTION
        Extract images from document and perform OCR to get text.
        
        Args:
            document: Document to process
            media_dir: Media directory path (optional)
            
        Returns:
            List of extraction results with image info and OCR results
        """
        extraction_results = []
        
        try:
            # Extract images from document
            image_info_list = self._extract_images_with_mapping(document, media_dir)
            
            for image_info in image_info_list:
                try:
                    logger.debug(f"Extracting text from: {image_info['location']}")
                    
                    # Extract text using enhanced OCR with preprocessing
                    ocr_results = self.ocr_engine.extract_text(image_info['temp_path'])
                    
                    # Collect OCR comparison data for reporting
                    ocr_comparison = self.ocr_engine.extract_text_comparison(image_info['temp_path'])
                    self._store_ocr_comparison_data(image_info['location'], image_info['temp_path'], ocr_comparison)
                    
                    # Store extraction result
                    extraction_result = {
                        'image_info': image_info,
                        'ocr_results': ocr_results,
                        'ocr_comparison': ocr_comparison
                    }
                    
                    extraction_results.append(extraction_result)
                    
                    logger.debug(f"Extraction completed for {image_info['location']}: {len(ocr_results)} text regions found")
                    
                except Exception as e:
                    logger.error(f"Error in extraction phase for {image_info['location']}: {e}")
            
            logger.info(f"Extraction phase completed: {len(extraction_results)} images processed")
            
        except Exception as e:
            logger.error(f"Error in extraction phase: {e}")
        
        return extraction_results
    
    def _phase2_match_and_wipe(self, extraction_results: List[Dict]) -> List[Dict]:
        """
        PHASE 2: MATCH & WIPE
        Find patterns in extracted text and apply wipes to images.
        
        Args:
            extraction_results: Results from extraction phase
            
        Returns:
            List of wipe results with matches and wiped images
        """
        wipe_results = []
        
        for extraction_result in extraction_results:
            try:
                image_info = extraction_result['image_info']
                ocr_results = extraction_result['ocr_results']
                
                logger.debug(f"Matching and wiping: {image_info['location']}")
                
                if not ocr_results:
                    logger.debug(f"No OCR results for {image_info['location']}, skipping match & wipe")
                    wipe_results.append({
                        'image_info': image_info,
                        'matches': [],
                        'wiped_image_path': None
                    })
                    continue
                
                # Find pattern matches
                matches = []
                for ocr_result in ocr_results:
                    try:
                        # Use enhanced pattern matching with universal character support
                        universal_matches = self.pattern_matcher.find_matches_universal(ocr_result.text)
                        
                        for universal_match in universal_matches:
                            replacement_text = self.pattern_matcher.get_replacement(universal_match.matched_text)
                            if replacement_text:
                                ocr_match = create_ocr_match(
                                    ocr_result, universal_match.pattern_name, replacement_text, 
                                    image_info['temp_path'], self.mode, universal_match.matched_text
                                )
                                matches.append(ocr_match)
                                
                                logger.debug(f"Pattern match found: '{universal_match.matched_text}' -> "
                                           f"'{replacement_text}' (pattern: {universal_match.pattern_name})")
                    
                    except Exception as e:
                        logger.error(f"Error in pattern matching for text '{ocr_result.text}': {e}")
                
                # Apply wipes to image (no replacement text, just clearing)
                wiped_image_path = None
                if matches:
                    wiped_image_path = self._apply_wipes_to_image(image_info['temp_path'], matches)
                
                # Store wipe result
                wipe_result = {
                    'image_info': image_info,
                    'matches': matches,
                    'wiped_image_path': wiped_image_path
                }
                
                wipe_results.append(wipe_result)
                
                logger.debug(f"Match & wipe completed for {image_info['location']}: {len(matches)} matches, wiped: {wiped_image_path is not None}")
                
            except Exception as e:
                logger.error(f"Error in match & wipe phase for {extraction_result['image_info']['location']}: {e}")
                # Add empty result to maintain structure
                wipe_results.append({
                    'image_info': extraction_result['image_info'],
                    'matches': [],
                    'wiped_image_path': None
                })
        
        logger.info(f"Match & wipe phase completed: {len([r for r in wipe_results if r['matches']])} images with matches")
        return wipe_results
    
    def _phase3_reconstruct(self, document: Document, wipe_results: List[Dict]) -> List[OCRMatch]:
        """
        PHASE 3: RECONSTRUCT
        Reconstruct document with wiped images.
        
        Args:
            document: Original document
            wipe_results: Results from match & wipe phase
            
        Returns:
            List of final OCRMatch objects
        """
        final_matches = []
        
        for wipe_result in wipe_results:
            try:
                image_info = wipe_result['image_info']
                matches = wipe_result['matches']
                wiped_image_path = wipe_result['wiped_image_path']
                
                logger.debug(f"Reconstructing: {image_info['location']}")
                
                if wiped_image_path and matches:
                    # RECONSTRUCTION-ONLY: Render appended text inside the original wiped bboxes
                    try:
                        ocr_results_for_image = [m.ocr_result for m in matches]
                        reconstructed_path = self._append_replacer_for_reconstruction.replace_text_in_image(
                            Path(wiped_image_path), ocr_results_for_image, matches
                        )
                        image_to_insert = reconstructed_path or wiped_image_path
                    except Exception:
                        image_to_insert = wiped_image_path

                    # Replace the image in the document with reconstructed version
                    success = self._replace_image_in_document(image_info['original_rel'], image_to_insert)
                    
                    if success:
                        # Update image paths in matches
                        for match in matches:
                            match.image_path = image_to_insert
                        
                        final_matches.extend(matches)
                        logger.info(f"RECONSTRUCTED: {len(matches)} matches in {image_info['location']}")
                    else:
                        logger.warning(f"Failed to reconstruct image in document for {image_info['location']}")
                else:
                    logger.debug(f"No reconstruction needed for {image_info['location']} (no matches or wipe failed)")
                
            except Exception as e:
                logger.error(f"Error in reconstruction phase for {wipe_result['image_info']['location']}: {e}")
        
        logger.info(f"Reconstruction phase completed: {len(final_matches)} total matches")
        return final_matches
    
    def _apply_wipes_to_image(self, image_path: Path, matches: List[OCRMatch]) -> Optional[Path]:
        """
        Apply wipes to image (clear pattern areas without replacement text).
        
        Args:
            image_path: Path to original image
            matches: List of OCRMatch objects with pattern information
            
        Returns:
            Path to wiped image, or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Apply wipes for each match
            for match in matches:
                try:
                    # Apply wipe to the image
                    image = self._apply_single_wipe(image, match)
                except Exception as e:
                    logger.error(f"Failed to apply wipe for match '{match.ocr_result.text}': {e}")
            
            # Save wiped image
            output_path = self._generate_output_path(image_path)
            image.save(output_path)
            
            logger.debug(f"Wipe application completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply wipes to image {image_path}: {e}")
            return None
    
    def _apply_single_wipe(self, image: Image.Image, match: OCRMatch) -> Image.Image:
        """
        Apply a single wipe to an image (clear pattern area without replacement text).
        
        Args:
            image: PIL Image to modify
            match: OCRMatch containing pattern information
            
        Returns:
            Modified PIL Image with pattern area wiped
        """
        try:
            bbox = match.ocr_result.bounding_box
            original_full_text = match.ocr_result.text
            
            # Find the pattern in the original text
            pattern_matches = self.pattern_matcher.find_matches_universal(original_full_text)
            
            if not pattern_matches:
                logger.warning(f"No pattern matches found in text: '{original_full_text}'")
                return image
            
            # Find the specific match
            target_match = None
            for pm in pattern_matches:
                if pm.matched_text in self.pattern_matcher.mappings:
                    replacement = self.pattern_matcher.get_replacement(pm.matched_text)
                    if replacement == match.replacement_text:
                        target_match = pm
                        break
            
            if not target_match:
                logger.warning(f"Could not find target match for replacement: '{match.replacement_text}'")
                return image
            
            # Convert PIL Image to cv2 format for precise wiping
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply wipe using precise replacer (wipe only, no replacement text)
            cv_image = self.text_replacer.precise_replacer._apply_hybrid_replace(cv_image, match, [match.ocr_result])
            
            # Convert back to PIL
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            wiped_image = Image.fromarray(cv_image_rgb)
            
            logger.debug(f"Applied wipe: '{target_match.matched_text}' in '{original_full_text}' at {bbox}")
            
            return wiped_image
            
        except Exception as e:
            logger.error(f"Failed to apply single wipe: {e}")
            return image
    
    def _extract_images_with_mapping(self, document: Document, media_dir: Optional[Path] = None) -> List[Dict]:
        """Extract images from DOCX document with mapping information for replacement."""
        image_info_list = []
        
        try:
            # Create temporary directory for extracted images
            temp_dir = PathManager.get_temp_directory() / f"enhanced_docx_images_{uuid.uuid4().hex[:8]}"
            PathManager.ensure_directory(temp_dir)
            
            # Access document parts to find images
            for i, rel in enumerate(document.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        # Extract image data
                        image_data = rel.target_part.blob
                        
                        # Determine file extension
                        content_type = rel.target_part.content_type
                        if 'jpeg' in content_type:
                            ext = '.jpg'
                        elif 'png' in content_type:
                            ext = '.png'
                        elif 'gif' in content_type:
                            ext = '.gif'
                        else:
                            ext = '.jpg'  # Default
                        
                        # Save image to temporary file
                        image_filename = f"image_{i}{ext}"
                        temp_image_path = temp_dir / image_filename
                        
                        with open(temp_image_path, 'wb') as f:
                            f.write(image_data)
                        
                        # Store mapping information
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
    
    # OLD METHOD REMOVED: _process_image_with_replacement has been replaced by the 3-phase approach
    # Phase 1: _phase1_extraction
    # Phase 2: _phase2_match_and_wipe  
    # Phase 3: _phase3_reconstruct
    
    def _store_ocr_comparison_data(self, location: str, image_path: Path, ocr_comparison: Dict[str, List[OCRResult]]):
        """Store OCR comparison data for reporting."""
        try:
            comparison_entry = {
                'location': location,
                'image_path': str(image_path),
                'image_name': image_path.name,
                'easyocr_results': [
                    {
                        'text': result.text,
                        'confidence': result.confidence,
                        'bounding_box': result.bounding_box
                    }
                    for result in ocr_comparison.get('easyocr', [])
                ],
                'tesseract_results': [
                    {
                        'text': result.text,
                        'confidence': result.confidence,
                        'bounding_box': result.bounding_box
                    }
                    for result in ocr_comparison.get('tesseract', [])
                ],
                'easyocr_text_count': len(ocr_comparison.get('easyocr', [])),
                'tesseract_text_count': len(ocr_comparison.get('tesseract', [])),
                'easyocr_texts': [result.text for result in ocr_comparison.get('easyocr', [])],
                'tesseract_texts': [result.text for result in ocr_comparison.get('tesseract', [])]
            }
            
            self.ocr_comparison_data.append(comparison_entry)
            
            logger.debug(f"Stored OCR comparison for {location}: "
                        f"EasyOCR={len(ocr_comparison.get('easyocr', []))} texts, "
                        f"Tesseract={len(ocr_comparison.get('tesseract', []))} texts")
            
        except Exception as e:
            logger.error(f"Failed to store OCR comparison data for {location}: {e}")
    
    def get_ocr_comparison_data(self) -> List[Dict]:
        """Get collected OCR comparison data for reporting."""
        return self.ocr_comparison_data
    

    
    def _replace_image_in_document(self, original_rel, modified_image_path: Path) -> bool:
        """
        Replace the original image in the document with the modified version.
        
        Args:
            original_rel: Original document relationship for the image
            modified_image_path: Path to the modified image
            
        Returns:
            True if replacement was successful, False otherwise
        """
        try:
            # Read the modified image data
            with open(modified_image_path, 'rb') as f:
                modified_image_data = f.read()
            
            # Replace the image data in the document's relationship
            original_rel.target_part._blob = modified_image_data
            
            logger.debug(f"Successfully replaced image data in document: {modified_image_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to replace image in document: {e}")
            return False
    
    def _run_debug_analysis(self, image_path: Path, ocr_results: List[OCRResult], matches: List[OCRMatch]):
        """Run debugging analysis on the processed image."""
        try:
            # Debug pattern matching for each OCR result
            for ocr_result in ocr_results:
                debug_info = self.debugger.debug_pattern_matching(ocr_result.text, verbose=False)
                
                if debug_info['final_match_count'] == 0 and ocr_result.text.strip():
                    logger.debug(f"No matches found for OCR text: '{ocr_result.text}' "
                               f"(confidence: {ocr_result.confidence:.2f})")
            
            # Save debug report if there were issues
            if not matches and ocr_results:
                debug_dir = image_path.parent / "enhanced_debug"
                debug_dir.mkdir(exist_ok=True)
                
                debug_report_path = debug_dir / f"{image_path.stem}_debug.json"
                
                # Run comprehensive tests
                test_results = self.debugger.run_comprehensive_tests()
                
                # Save debug report
                self.debugger.save_debug_report(debug_report_path, test_results)
                
                logger.info(f"Debug report saved: {debug_report_path}")
        
        except Exception as e:
            logger.warning(f"Debug analysis failed: {e}")
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the enhanced image processor configuration."""
        base_info = {
            'mode': self.mode,
            'ocr_engine': self.ocr_engine.engine,
            'use_gpu': self.ocr_engine.use_gpu,
            'confidence_threshold': self.ocr_engine.confidence_threshold,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings),
            'preprocessing_enabled': self.enable_preprocessing,
            'debugging_enabled': self.enable_debugging
        }
        
        # Add enhanced pattern matcher info
        if hasattr(self.pattern_matcher, 'get_debug_info'):
            base_info['pattern_matcher_info'] = self.pattern_matcher.get_debug_info()
        
        return base_info
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """
        Generate output path for modified image.
        
        Args:
            input_path: Original image path
            
        Returns:
            Path for the modified image
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = input_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename with _modified suffix
            stem = input_path.stem
            suffix = input_path.suffix
            output_filename = f"{stem}_modified{suffix}"
            output_path = output_dir / output_filename
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate output path: {e}")
            # Fallback to input path
            return input_path

class ProcessorImageWrapper:
    """Wrapper to make ImageProcessor compatible with BaseProcessor interface."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str], 
                 mode: str = "replace", separator: str = ";", default_mapping: str = "4022-NA",
                 ocr_engine: str = "easyocr", use_gpu: bool = True, 
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                 enable_preprocessing: bool = True, enable_debugging: bool = False):
        """Initialize the wrapper with ImageProcessor."""
        self.image_processor = ImageProcessor(patterns, mappings, mode, separator, default_mapping,
                                            ocr_engine, use_gpu, confidence_threshold, 
                                            enable_preprocessing, enable_debugging)
        self.processor_type = "image"
        self.config = {
            'mode': mode,
            'separator': separator,
            'default_mapping': default_mapping,
            'ocr_engine': ocr_engine,
            'use_gpu': use_gpu,
            'confidence_threshold': confidence_threshold,
            'patterns_count': len(patterns),
            'mappings_count': len(mappings)
        }
        self.initialized = False
        
    def initialize(self, **kwargs) -> bool:
        """Initialize the image processor."""
        try:
            # The ImageProcessor is ready to use after construction
            self.initialized = True
            logger.info("Image processor wrapper initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize image processor wrapper: {e}")
            return False
    
    def process(self, document, **kwargs) -> 'ProcessingResult':
        """Process document images and return standardized result."""
        import time
        from ..core.processor_interface import ProcessingResult
        
        start_time = time.time()
        
        try:
            # Process images in the document
            image_matches = self.image_processor.process_images(document)
            
            # Get OCR comparison data for reporting
            ocr_comparison_data = self.image_processor.get_ocr_comparison_data()
            
            # Create detailed matches data for reporting with wipe boundaries
            detailed_matches = []
            for match in image_matches:
                match_dict = match.to_dict()
                # Add wipe boundary information if available
                if hasattr(match, 'wipe_boundaries') and match.wipe_boundaries:
                    match_dict['wipe_boundaries'] = match.wipe_boundaries
                if hasattr(match, 'calculated_text_boundary') and match.calculated_text_boundary:
                    match_dict['calculated_text_boundary'] = match.calculated_text_boundary
                if hasattr(match, 'wipe_area_info') and match.wipe_area_info:
                    match_dict['wipe_area_info'] = match.wipe_area_info
                detailed_matches.append(match_dict)
            
            # Create metadata
            metadata = {
                'detailed_matches': detailed_matches,
                'image_matches': [match.to_dict() for match in image_matches],
                'ocr_comparison_data': ocr_comparison_data,
                'processor_info': self.image_processor.get_processing_info()
            }
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                processor_type=self.processor_type,
                matches_found=len(image_matches),
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.info(f"Image processing completed: {len(image_matches)} matches found in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Image processing failed: {e}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                processor_type=self.processor_type,
                matches_found=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats (document formats that can contain images)."""
        return ['.docx']
    
    def cleanup(self):
        """Clean up resources."""
        self.initialized = False
        logger.info("Image processor wrapper cleaned up")
    
    def is_initialized(self) -> bool:
        """Check if processor is initialized."""
        return self.initialized


def create_image_processor(patterns: Dict[str, str], mappings: Dict[str, str], 
                          mode: str = "replace", separator: str = ";", default_mapping: str = "4022-NA",
                          ocr_engine: str = "easyocr", use_gpu: bool = True, 
                          confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                          enable_preprocessing: bool = True, enable_debugging: bool = False) -> ProcessorImageWrapper:
    """
    Factory function to create a BaseProcessor-compatible image processor.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        mode: Processing mode ('replace' or 'append')
        separator: Separator between original and appended text in append mode
        default_mapping: Default text to append when no mapping is found
        ocr_engine: OCR engine to use
        use_gpu: Whether to use GPU acceleration
        confidence_threshold: Minimum OCR confidence threshold
        enable_preprocessing: Whether to enable image preprocessing
        enable_debugging: Whether to enable debugging features
        
    Returns:
        ProcessorImageWrapper instance that implements BaseProcessor interface
    """
    return ProcessorImageWrapper(patterns, mappings, mode, separator, default_mapping,
                               ocr_engine, use_gpu, confidence_threshold, 
                               enable_preprocessing, enable_debugging)