"""
Enhanced Image Processor for DOCX document processing.

This module provides comprehensive image processing capabilities for DOCX documents including:
- OCR-based text detection in images
- Pattern matching using regex patterns
- Text replacement with formatting preservation
- Layout impact analysis for image changes
- Support for both append and replace modes
- Font and formatting preservation during replacements

The processor analyzes document images, performs OCR to detect text, finds patterns in the detected text,
applies replacements while maintaining document formatting, and provides detailed analysis of potential
layout impacts from image modifications.
"""

import re
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.section import Section
from docx.enum.section import WD_SECTION_START
from docx.shape import InlineShape
from docx.shared import Inches

from ..core.processor_interface import BaseProcessor, ProcessingResult
from ..core.models import OCRResult, OCRMatch
from ..utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.image_utils.hybrid_ocr_manager import HybridOCRManager
from ..utils.image_utils.precise_replace import create_precise_text_replacer
from ..utils.image_utils.precise_append import PreciseAppendReplacer
from ..utils.shared_constants import (
    DEFAULT_MAPPING,
    DEFAULT_SEPARATOR,
    PROCESSING_MODES,
    DEFAULT_OCR_CONFIDENCE
)
from ..utils.platform_utils import PathManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedImageMatch:
    """
    Enhanced image match information with comprehensive details about text replacements.
    
    This class extends basic pattern matching with additional context including
    font details, and confidence scores. It provides a complete record of each text replacement
    operation for reporting and analysis purposes.
    
    Attributes:
        pattern: The regex pattern that was matched
        original_text: The original text found in the document
        replacement_text: The text that will replace the original
        position: Character position where the match was found
        location: Human-readable location description (e.g., "body_paragraph_5")
        font_info: Dictionary containing font properties (name, size, style, etc.)
        confidence: Confidence score for the match (0.0 to 1.0)
        actual_pattern: The actual regex pattern string
        content_type: Type of content (Paragraph, Table, Header, Footer, etc.)
        dimension: Element dimensions
        processor: Name of the processor that found this match
        ocr_result: OCR result object containing bounding box and text information
    """
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
    extracted_pattern_text: str = ""  # The exact pattern text that was matched
    
    def __post_init__(self):
        """
        Initialize default values after object creation.
        
        This method is automatically called after the dataclass is created to ensure
        that font_info is always a dictionary, even if not provided during initialization.
        """
        if self.font_info is None:
            self.font_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'pattern': self.pattern,
            'pattern_name': self.pattern,  # Add pattern_name for compatibility with report generator
            'original_text': self.original_text,
            'extracted_pattern_text': self.extracted_pattern_text,  # The exact matched pattern text
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
        
        # Add reconstruction reasoning if available
        if hasattr(self, 'reconstruction_reasoning') and self.reconstruction_reasoning:
            result['reconstruction_reasoning'] = self.reconstruction_reasoning
        
        return result

class ImageProcessor(BaseProcessor):
    """
    Enhanced image processor for DOCX documents with comprehensive analysis capabilities.
    
    This class provides the main interface for processing images in DOCX documents. It handles
    OCR-based text detection, pattern matching, text replacement within images, and maintains 
    detailed information about all image modifications for reporting purposes.
    
    The processor integrates with core infrastructure components including memory
    management, performance monitoring, and GPU acceleration for optimal performance.
    """
    
    def __init__(self, patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None, 
                 mode: str = PROCESSING_MODES['REPLACE'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING,
                 ocr_engine: str = "easyocr", confidence_min: float = 0.4, use_gpu: bool = True, enable_debugging: bool = False):
        """
        Initialize the image processor with patterns, mappings, and processing mode.
        
        Args:
            patterns: Dictionary containing regex patterns for text matching
            mappings: Dictionary containing text mappings for replacements
            mode: Processing mode - "append" or "replace"
            separator: Separator between original and appended text in append mode
            default_mapping: Default text to append when no mapping is found
            ocr_engine: OCR engine to use ("easyocr", "tesseract", "hybrid")
            confidence_min: Minimum confidence threshold for OCR text detection
            use_gpu: Whether to use GPU acceleration for OCR processing
        """
        # Initialize base processor
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
        
        # Initialize components
        self.pattern_matcher = None
        self.ocr_manager = None
        self.ocr_comparison_data = []
        
        logger.info(f"Image processor initialized with mode: {mode}, separator: '{separator}', "
                   f"default_mapping: '{default_mapping}', OCR engine: {ocr_engine}, "
                   f"confidence_min: {confidence_min}, use_gpu: {use_gpu}")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the processor with configuration parameters.
        
        This method loads patterns and mappings, configures the processing mode,
        and sets up all internal components for document processing. It can accept
        patterns and mappings directly or load them from JSON files.
        
        Args:
            **kwargs: Configuration parameters including:
                - patterns: Dictionary of pattern names to regex patterns
                - mappings: Dictionary of original text to replacement text
                - mode: Processing mode ('append' or 'replace')
                - separator: Separator between original and appended text in append mode
                - default_mapping: Default text to append when no mapping is found
                - ocr_engine: OCR engine to use ("easyocr", "tesseract", "hybrid")
                - confidence_min: Minimum confidence threshold for OCR text detection
                - use_gpu: Whether to use GPU acceleration for OCR processing
                - patterns_file: Path to patterns JSON file (if patterns not provided)
                - mappings_file: Path to mappings JSON file (if mappings not provided)
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing Image Processor...")
            
            # Extract configuration parameters
            patterns = kwargs.get('patterns', self.patterns)
            mappings = kwargs.get('mappings', self.mappings)
            mode = kwargs.get('mode', self.mode)
            separator = kwargs.get('separator', self.separator)
            default_mapping = kwargs.get('default_mapping', self.default_mapping)
            ocr_engine = kwargs.get('ocr_engine', self.ocr_engine)
            confidence_min = kwargs.get('confidence_min', self.confidence_min)
            use_gpu = kwargs.get('use_gpu', self.use_gpu)
            
            # Load patterns and mappings if not provided
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
            
            # Initialize components using utility classes
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
    
    def process(self, document, **kwargs) -> 'ProcessingResult':
        """
        Process document images and return standardized result.
        
        This method orchestrates the three-phase image processing pipeline:
        1. EXTRACTION: Extract text from images using OCR.
        2. MATCH & WIPE: Find patterns and wipe them (no replacement text).
        3. RECONSTRUCT: Reconstruct document with wiped images.
        
        Args:
            document: DOCX document to process.
            **kwargs: Additional arguments (e.g., media_dir for image extraction).
            
        Returns:
            ProcessingResult object containing match information and metadata.
        """
        import time
        from ..core.processor_interface import ProcessingResult
        
        start_time = time.time()
        
        try:
            # Process images in the document
            image_matches = self._process_images(document, **kwargs)
            
            # Get OCR comparison data for reporting
            ocr_comparison_data = self.get_ocr_comparison_data()
            
            # Create detailed matches data for reporting with wipe boundaries
            detailed_matches = []
            for match in image_matches:
                # Handle both objects with to_dict() method and dictionaries
                if hasattr(match, 'to_dict'):
                    match_dict = match.to_dict()
                else:
                    match_dict = match.copy() if isinstance(match, dict) else {}
                
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
                'image_matches': [match.to_dict() if hasattr(match, 'to_dict') else match for match in image_matches],
                'ocr_comparison_data': ocr_comparison_data,
                'processor_info': self.get_processing_info()
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
        logger.info("Image processor cleaned up")
    
    def is_initialized(self) -> bool:
        """Check if processor is initialized."""
        return self.initialized
    
    def _process_images(self, document: Document, **kwargs) -> List[EnhancedImageMatch]:
        """
        Process all images in the document with 3-phase approach:
        1. EXTRACTION: Extract text from images using OCR
        2. MATCH & WIPE: Find patterns and wipe them (no replacement text)
        3. RECONSTRUCT: Reconstruct document (currently just replacing images)
        
        Args:
            document: Document to process
            **kwargs: Additional arguments (e.g., media_dir for image extraction)
            
        Returns:
            List of EnhancedImageMatch objects representing successful matches
        """
        logger.info("Starting 3-phase image processing...")
        
        try:
            # PHASE 1: EXTRACTION - Extract images and text from document
            logger.info("=== PHASE 1: EXTRACTION ===")
            extraction_results = self._phase1_extraction(document, **kwargs)
            
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
    
    def _phase1_extraction(self, document: Document, **kwargs) -> List[Dict]:
        """
        PHASE 1: EXTRACTION
        Extract images from document and perform OCR to get text.
        
        Args:
            document: Document to process
            **kwargs: Additional arguments (e.g., media_dir for image extraction)
            
        Returns:
            List of extraction results with image info and OCR results
        """
        extraction_results = []
        
        try:
            # Extract images from document
            image_info_list = self._extract_images_with_mapping(document, **kwargs)
            
            for image_info in image_info_list:
                try:
                    logger.debug(f"Extracting text from: {image_info['location']}")
                    logger.debug(f"Image temp path: {image_info['temp_path']}")
                    logger.debug(f"Image location: {image_info['location']}")
                    logger.debug(f"Image info keys: {list(image_info.keys())}")
                    
                    # Extract text using enhanced OCR with preprocessing
                    ocr_results = self.ocr_manager.process_hybrid(image_info['temp_path'])
                    logger.debug(f"OCR Results Type: {type(ocr_results)}")
                    logger.debug(f"OCR Results Length: {len(ocr_results) if ocr_results else 0}")
                    if ocr_results:
                        logger.debug(f"First OCR Result Type: {type(ocr_results[0])}")
                        logger.debug(f"First OCR Result: {ocr_results[0]}")
                    
                    # Collect OCR comparison data for reporting
                    # Get comparison data from OCR manager if available
                    ocr_comparison = {}
                    if hasattr(self.ocr_manager, 'get_comparison_data'):
                        ocr_comparison = self.ocr_manager.get_comparison_data()
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
                logger.debug(f"Processing {len(ocr_results)} OCR results for {image_info['location']}")
                for i, ocr_result in enumerate(ocr_results):
                    logger.debug(f"OCR Result {i+1}: '{ocr_result.text}'")
                    
                    try:
                        # Use enhanced pattern matching with universal character support
                        universal_matches = self.pattern_matcher.find_matches_universal(ocr_result.text)
                        logger.debug(f"Found {len(universal_matches)} universal matches for text: '{ocr_result.text}'")
                        
                        for universal_match in universal_matches:
                            replacement_text = self.pattern_matcher.get_replacement(universal_match.matched_text)
                            
                            # Mode-specific behavior:
                            # - In "replace" mode: only process if we have a valid mapping (no default)
                            # - In "append" or "append-image" mode: use default mapping if no valid mapping found
                            if not replacement_text:
                                if self.mode == "replace":
                                    # In replace mode, skip if no mapping found
                                    logger.debug(f"REPLACE MODE: Skipping '{universal_match.matched_text}' - no mapping found")
                                    continue
                                elif self.mode in ["append", "append-image"]:
                                    # In append mode, use default mapping
                                    replacement_text = self.default_mapping
                                    logger.info(f"APPEND MODE: Using default mapping '{self.default_mapping}' for '{universal_match.matched_text}'")
                                else:
                                    # Unknown mode, skip
                                    logger.warning(f"Unknown mode '{self.mode}', skipping '{universal_match.matched_text}'")
                                    continue
                            
                            # Handle missing font_info attribute in HybridOCRResult
                            font_info = getattr(ocr_result, 'font_info', {})
                            
                            # Create EnhancedImageMatch with extracted pattern text
                            logger.debug(f"Creating EnhancedImageMatch for '{universal_match.matched_text}' -> '{replacement_text}'")
                            ocr_match = EnhancedImageMatch(
                                universal_match.pattern_name, universal_match.matched_text, replacement_text, 
                                universal_match.start_pos, image_info['location'],
                                font_info, ocr_result.confidence, universal_match.pattern_name,
                                "Image", ocr_result.bounding_box, "OCR", ocr_result
                            )
                            # Set the extracted pattern text (the exact matched part)
                            ocr_match.extracted_pattern_text = universal_match.matched_text
                            # Set the original text (the full OCR text)
                            ocr_match.original_text = ocr_result.text
                            matches.append(ocr_match)
                            logger.debug(f"Added match to matches list. Total matches: {len(matches)}")
                            
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
    
    def _phase3_reconstruct(self, document: Document, wipe_results: List[Dict]) -> List[EnhancedImageMatch]:
        """
        PHASE 3: RECONSTRUCT
        Reconstruct document with wiped images.
        
        Args:
            document: Original document
            wipe_results: Results from match & wipe phase
            
        Returns:
            List of final EnhancedImageMatch objects
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
                        # Get OCR results from matches, handling both EnhancedImageMatch and regular objects
                        ocr_results_for_image = []
                        for m in matches:
                            if hasattr(m, 'ocr_result') and m.ocr_result:
                                ocr_results_for_image.append(m.ocr_result)
                            else:
                                # Create OCR result from match data if not available
                                from ..core.models import OCRResult
                                ocr_result = OCRResult(
                                    text=m.original_text,
                                    confidence=getattr(m, 'confidence', 0.0),
                                    bounding_box=getattr(m, 'bounding_box', (0, 0, 0, 0))
                                )
                                ocr_results_for_image.append(ocr_result)
                        
                        reconstructed_path = self._append_replacer_for_reconstruction.replace_text_in_image(
                            Path(wiped_image_path), ocr_results_for_image, matches
                        )
                        image_to_insert = reconstructed_path or wiped_image_path
                    except Exception as e:
                        logger.error(f"Error in reconstruction for {image_info['location']}: {e}")
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
    
    def _apply_wipes_to_image(self, image_path: Path, matches: List[EnhancedImageMatch]) -> Optional[Path]:
        """
        Apply wipes to image (clear pattern areas without replacement text).
        
        Args:
            image_path: Path to original image
            matches: List of EnhancedImageMatch objects with pattern information
            
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
                    logger.error(f"Failed to apply wipe for match '{match.original_text}': {e}")
            
            # Save wiped image
            output_path = self._generate_output_path(image_path)
            image.save(output_path)
            
            logger.debug(f"Wipe application completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply wipes to image {image_path}: {e}")
            return None
    
    def _calculate_reconstruction_reasoning(self, image: Image.Image, match: EnhancedImageMatch) -> Dict[str, Any]:
        """
        Calculate reasoning for image reconstruction including dimensions, font size logic, and line reasoning.
        
        Args:
            image: PIL Image to analyze
            match: EnhancedImageMatch containing pattern information
            
        Returns:
            Dictionary containing reasoning information
        """
        try:
            # Get image dimensions
            img_width, img_height = image.size
            
            # Calculate text area dimensions based on bounding box
            bbox = getattr(match, 'bounding_box', (0, 0, 0, 0))
            if len(bbox) >= 4:
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Estimate text area if no bounding box
                text_width = img_width * 0.3  # Assume 30% of image width
                text_height = img_height * 0.1  # Assume 10% of image height
            
            # Font size logic based on text area
            if text_height < 20:
                font_size = 12
                font_reasoning = "Small text area - using 12pt font"
            elif text_height < 40:
                font_size = 16
                font_reasoning = "Medium text area - using 16pt font"
            else:
                font_size = 20
                font_reasoning = "Large text area - using 20pt font"
            
            # 1 line vs 2 line reasoning
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
            
            # Image reconstruction dimensions
            reconstruction_width = max(text_width + 20, 100)  # Add padding
            reconstruction_height = text_height * num_lines + 20  # Add padding for multiple lines
            
            reasoning = {
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height,
                    'text_area_width': text_width,
                    'text_area_height': text_height
                },
                'font_logic': {
                    'font_size': font_size,
                    'reasoning': font_reasoning
                },
                'line_reasoning': {
                    'num_lines': num_lines,
                    'text_length': text_length,
                    'reasoning': line_reasoning
                },
                'reconstruction_dimensions': {
                    'width': reconstruction_width,
                    'height': reconstruction_height
                }
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
        """
        Apply a single wipe to an image (clear pattern area without replacement text).
        
        Args:
            image: PIL Image to modify
            match: EnhancedImageMatch containing pattern information
            
        Returns:
            Modified PIL Image with pattern area wiped
        """
        try:
            # Calculate reconstruction reasoning
            reasoning = self._calculate_reconstruction_reasoning(image, match)
            
            # Get bounding box from match or use default
            bbox = getattr(match, 'bounding_box', (0, 0, 0, 0))
            original_full_text = match.original_text
            
            # Find the pattern in the original text
            pattern_matches = self.pattern_matcher.find_matches_universal(original_full_text)
            
            if not pattern_matches:
                logger.warning(f"No pattern matches found in text: '{original_full_text}'")
                return image
            
            # Find the specific match
            target_match = None
            for pm in pattern_matches:
                # Get replacement text (could be from mappings or default mapping)
                replacement = self.pattern_matcher.get_replacement(pm.matched_text)
                if not replacement:
                    # If no mapping found, use default mapping based on mode
                    if self.mode in ["append", "append-image"]:
                        replacement = self.default_mapping
                    else:
                        continue  # Skip in replace mode
                
                if replacement == match.replacement_text:
                    target_match = pm
                    break
            
            if not target_match:
                logger.warning(f"Could not find target match for replacement: '{match.replacement_text}'")
                return image
            
            # Convert PIL Image to cv2 format for precise wiping
            import numpy as np
            import cv2
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply wipe using precise replacer (wipe only, no replacement text)
            # Create OCR result object from EnhancedImageMatch data
            from ..core.models import OCRResult
            ocr_result = OCRResult(
                text=match.original_text,
                confidence=match.confidence or 0.0,
                bounding_box=bbox
            )
            
            # Apply wipe using text replacer
            if hasattr(self, 'text_replacer') and self.text_replacer:
                cv_image = self.text_replacer._apply_hybrid_replace(cv_image, match, [ocr_result])
            else:
                # Fallback: simple rectangle wipe
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
            
            # Convert back to PIL
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            wiped_image = Image.fromarray(cv_image_rgb)
            
            # Store reasoning in match for reporting
            match.reconstruction_reasoning = reasoning
            
            logger.debug(f"Applied wipe: '{target_match.matched_text}' in '{original_full_text}' at {bbox}")
            logger.debug(f"Reconstruction reasoning: {reasoning}")
            
            return wiped_image
            
        except Exception as e:
            logger.error(f"Failed to apply single wipe: {e}")
            return image
    
    def _extract_images_with_mapping(self, document: Document, **kwargs) -> List[Dict]:
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
            'ocr_engine': self.ocr_engine,
            'use_gpu': self.use_gpu,
            'confidence_threshold': self.confidence_min,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings),
            'preprocessing_enabled': True, # Preprocessing is handled by HybridOCRManager
            'debugging_enabled': self.enable_debugging # This attribute is not in the new_code, but the original had it. Assuming it's meant to be here.
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
                                            ocr_engine, confidence_threshold, use_gpu)
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
            # Initialize the underlying ImageProcessor
            success = self.image_processor.initialize(**kwargs)
            if success:
                self.initialized = True
                logger.info("Image processor wrapper initialized successfully")
                return success
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
            result = self.image_processor.process(document)
            image_matches = result.metadata.get('detailed_matches', []) if result.success else []
            
            # Get OCR comparison data for reporting
            ocr_comparison_data = self.image_processor.get_ocr_comparison_data()
            
            # Create detailed matches data for reporting with wipe boundaries
            detailed_matches = []
            for match in image_matches:
                # Handle both objects with to_dict() method and dictionaries
                if hasattr(match, 'to_dict'):
                    match_dict = match.to_dict()
                else:
                    match_dict = match.copy() if isinstance(match, dict) else {}
                
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
                'image_matches': [match.to_dict() if hasattr(match, 'to_dict') else match for match in image_matches],
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
                          mode: str = PROCESSING_MODES['REPLACE'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING,
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