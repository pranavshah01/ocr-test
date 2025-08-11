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
from ..core.models import OCRResult, OCRMatch, Match, create_ocr_result, create_ocr_match
from ..utils.shared_constants import DEFAULT_OCR_CONFIDENCE, FALLBACK_FONTS
from ..utils.platform_utils import PathManager

# Import enhanced components from image_utils
from ..utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.image_utils.image_preprocessor import ImagePreprocessor, create_image_preprocessor
from ..utils.image_utils.preprocessing_strategies import PreprocessingStrategyManager, create_preprocessing_strategy_manager
from ..utils.image_utils.text_analyzer import TextAnalyzer, TextProperties, create_text_analyzer
from ..utils.image_utils.precise_text_replacer import PreciseTextReplacer, create_precise_text_replacer
from ..utils.image_utils.pattern_debugger import PatternDebugger, create_pattern_debugger

logger = logging.getLogger(__name__)

class OCREngine:
    """Enhanced OCR engine with preprocessing and multiple strategy support."""
    
    def __init__(self, engine: str = "easyocr", use_gpu: bool = True, 
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE, enable_preprocessing: bool = True):
        """
        Initialize enhanced OCR engine.
        
        Args:
            engine: OCR engine to use ('easyocr' or 'tesseract')
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence threshold for OCR results
            enable_preprocessing: Whether to enable image preprocessing
        """
        self.engine = engine
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        self.reader = None
        
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
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
                
            logger.info(f"Enhanced OCR engine '{self.engine}' initialized successfully "
                       f"(GPU: {self.use_gpu}, Preprocessing: {self.enable_preprocessing})")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced OCR engine '{self.engine}': {e}")
            # Try fallback to CPU
            if self.use_gpu:
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
    
    def extract_text(self, image_path: Path) -> List[OCRResult]:
        """
        Extract text using enhanced preprocessing and multiple strategies.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of OCRResult objects
        """
        try:
            if self.enable_preprocessing:
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
            # Extract with EasyOCR
            try:
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
    
    def _extract_without_preprocessing(self, image_path: Path) -> List[OCRResult]:
        """Extract text without preprocessing (fallback method)."""
        try:
            if self.engine == "easyocr":
                return self._extract_with_easyocr(image_path)
            elif self.engine == "tesseract":
                return self._extract_with_tesseract(image_path)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
                
        except Exception as e:
            logger.error(f"Standard OCR extraction failed: {e}")
            return []
    
    def _extract_from_image_array(self, image_array: np.ndarray) -> List[OCRResult]:
        """Extract text from numpy image array."""
        try:
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
            # Load image
            image = Image.open(image_path)
            
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
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
            else:
                image = Image.fromarray(image_array)
            
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

class ImageTextReplacer:
    """Enhanced image text replacer with precise positioning and orientation support."""
    
    def __init__(self, mode: str = "replace", pattern_matcher=None):
        """
        Initialize enhanced image text replacer.
        
        Args:
            mode: Processing mode ('replace' or 'append')
            pattern_matcher: Pattern matcher instance for finding patterns in text
        """
        self.mode = mode
        self.pattern_matcher = pattern_matcher
        self.text_analyzer = create_text_analyzer()
        self.precise_replacer = create_precise_text_replacer()
    
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
            # Load image
            image = Image.open(image_path)
            
            logger.debug(f"Processing {len(ocr_matches)} text replacements in {image_path}")
            
            # Process each match with enhanced precision
            for match in ocr_matches:
                try:
                    if match.processing_mode == "replace":
                        image = self._replace_text_region(image, match)
                    elif match.processing_mode == "append":
                        image = self._append_text_region(image, match)
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
            
            # Use precise text replacer with the full text replacement
            enhanced_image = self.precise_replacer.replace_text_precise(
                image, original_full_text, new_full_text, bbox, properties
            )
            
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
                 mode: str = "replace", ocr_engine: str = "easyocr", 
                 use_gpu: bool = True, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                 enable_preprocessing: bool = True, enable_debugging: bool = False):
        """
        Initialize enhanced image processor.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            mode: Processing mode ('replace' or 'append')
            ocr_engine: OCR engine to use
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum OCR confidence threshold
            enable_preprocessing: Whether to enable image preprocessing
            enable_debugging: Whether to enable debugging features
        """
        self.patterns = patterns
        self.mappings = mappings
        self.mode = mode
        self.enable_preprocessing = enable_preprocessing
        self.enable_debugging = enable_debugging
        
        # Initialize enhanced components
        self.pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
        self.ocr_engine = OCREngine(ocr_engine, use_gpu, confidence_threshold, enable_preprocessing)
        self.text_replacer = ImageTextReplacer(mode, self.pattern_matcher)
        
        # Initialize debugging if enabled
        if self.enable_debugging:
            self.debugger = create_pattern_debugger(self.pattern_matcher)
        
        # Initialize OCR comparison data storage for reporting
        self.ocr_comparison_data = []
        
        logger.info(f"Enhanced image processor initialized with {len(patterns)} patterns, "
                   f"{len(mappings)} mappings, mode: {mode}, preprocessing: {enable_preprocessing}, "
                   f"debugging: {enable_debugging}")
    
    def process_images(self, document: Document, media_dir: Optional[Path] = None) -> List[OCRMatch]:
        """
        Process all images in the document with enhanced capabilities.
        
        Args:
            document: Document to process
            media_dir: Media directory path (optional)
            
        Returns:
            List of OCRMatch objects representing successful replacements
        """
        matches = []
        
        logger.info("Starting enhanced image processing...")
        
        try:
            # Extract images from document and get mapping info
            image_info_list = self._extract_images_with_mapping(document, media_dir)
            
            for image_info in image_info_list:
                try:
                    image_matches = self._process_image_with_replacement(
                        image_info['temp_path'], 
                        image_info['original_rel'], 
                        image_info['location']
                    )
                    matches.extend(image_matches)
                except Exception as e:
                    logger.error(f"Error processing image {image_info['location']}: {e}")
            
        except Exception as e:
            logger.error(f"Error in enhanced image processing: {e}")
        
        logger.info(f"Enhanced image processing completed: {len(matches)} matches found")
        return matches
    
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
    
    def _process_image_with_replacement(self, temp_image_path: Path, original_rel, location: str) -> List[OCRMatch]:
        """
        Process individual image with enhanced OCR and replace in document.
        
        Args:
            temp_image_path: Path to temporary image file
            original_rel: Original document relationship for the image
            location: Location description
            
        Returns:
            List of OCRMatch objects
        """
        matches = []
        
        try:
            logger.debug(f"Processing image with enhanced capabilities: {temp_image_path}")
            
            # Step 1: Extract text using enhanced OCR with preprocessing
            ocr_results = self.ocr_engine.extract_text(temp_image_path)
            
            # Step 1.5: Collect OCR comparison data for reporting
            ocr_comparison = self.ocr_engine.extract_text_comparison(temp_image_path)
            self._store_ocr_comparison_data(location, temp_image_path, ocr_comparison)
            
            if not ocr_results:
                logger.debug(f"No OCR results found for {location}")
                return matches
            
            logger.debug(f"Enhanced OCR found {len(ocr_results)} text regions in {location}")
            
            # Step 2: Find pattern matches using enhanced pattern matcher
            ocr_matches = []
            for ocr_result in ocr_results:
                try:
                    # Use enhanced pattern matching with universal character support
                    universal_matches = self.pattern_matcher.find_matches_universal(ocr_result.text)
                    
                    for universal_match in universal_matches:
                        replacement_text = self.pattern_matcher.get_replacement(universal_match.matched_text)
                        if replacement_text:
                            ocr_match = create_ocr_match(
                                ocr_result, universal_match.pattern_name, replacement_text, 
                                temp_image_path, self.mode
                            )
                            ocr_matches.append(ocr_match)
                            
                            logger.debug(f"Enhanced pattern match: '{universal_match.matched_text}' -> "
                                       f"'{replacement_text}' (pattern: {universal_match.pattern_name}, "
                                       f"context: '{universal_match.preceding_context}|{universal_match.matched_text}|{universal_match.following_context}')")
                
                except Exception as e:
                    logger.error(f"Error in enhanced pattern matching for text '{ocr_result.text}': {e}")
            
            # Step 3: Apply enhanced text replacements to image and replace in document
            if ocr_matches:
                if self.mode == "replace":
                    # Replace mode: modify image and replace in document
                    modified_image_path = self.text_replacer.replace_text_in_image(temp_image_path, ocr_matches)
                    
                    if modified_image_path:
                        # Replace the image in the document
                        success = self._replace_image_in_document(original_rel, modified_image_path)
                        
                        if success:
                            # Update image paths in matches
                            for match in ocr_matches:
                                match.image_path = modified_image_path
                            
                            matches.extend(ocr_matches)
                            logger.info(f"✅ IMAGE REPLACED: {len(ocr_matches)} text replacements in {location}")
                        else:
                            logger.warning(f"Failed to replace image in document for {location}")
                    else:
                        logger.warning(f"Enhanced text replacement failed for {location}")
                        
                elif self.mode == "append":
                    # Append mode: not supported yet
                    logger.warning(f"APPEND MODE NOT SUPPORTED: Image processing in append mode is not yet implemented for {location}")
                    logger.info(f"Found {len(ocr_matches)} matches in {location} but cannot process in append mode")
            
            # Step 4: Run debugging if enabled
            if self.enable_debugging and ocr_results:
                self._run_debug_analysis(temp_image_path, ocr_results, matches)
        
        except Exception as e:
            logger.error(f"Error in enhanced image processing at {location}: {e}")
        
        return matches
    
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

def create_image_processor(patterns: Dict[str, str], mappings: Dict[str, str], 
                          mode: str = "replace", ocr_engine: str = "easyocr",
                          use_gpu: bool = True, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                          enable_preprocessing: bool = True, enable_debugging: bool = False) -> ImageProcessor:
    """
    Factory function to create an enhanced ImageProcessor instance.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        mode: Processing mode ('replace' or 'append')
        ocr_engine: OCR engine to use
        use_gpu: Whether to use GPU acceleration
        confidence_threshold: Minimum OCR confidence threshold
        enable_preprocessing: Whether to enable image preprocessing
        enable_debugging: Whether to enable debugging features
        
    Returns:
        Enhanced ImageProcessor instance
    """
    return ImageProcessor(patterns, mappings, mode, ocr_engine, use_gpu, 
                         confidence_threshold, enable_preprocessing, enable_debugging)