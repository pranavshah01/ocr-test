"""
Image processor for OCR-based image text detection and replacement.
Handles image extraction, OCR processing, and text replacement with GPU support.
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

from ..core.models import OCRResult, OCRMatch, Match, create_ocr_result, create_ocr_match
from ..utils.docx_utils import PatternMatcher, create_pattern_matcher
from ..utils.shared_constants import DEFAULT_OCR_CONFIDENCE, FALLBACK_FONTS
from ..utils.platform_utils import PathManager

logger = logging.getLogger(__name__)

class OCREngine:
    """OCR engine interface with GPU support and fallback mechanisms."""
    
    def __init__(self, engine: str = "easyocr", use_gpu: bool = True, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE):
        """
        Initialize OCR engine.
        
        Args:
            engine: OCR engine to use ('easyocr' or 'tesseract')
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence threshold for OCR results
        """
        self.engine = engine
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.reader = None
        
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
                
            logger.info(f"OCR engine '{self.engine}' initialized successfully (GPU: {self.use_gpu})")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine '{self.engine}': {e}")
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
                    logger.info("Using CUDA GPU for EasyOCR")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    logger.info("Using MPS GPU for EasyOCR")
                else:
                    logger.info("GPU requested but not available, using CPU")
            
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(['en'], gpu=gpu_available)
            self.use_gpu = gpu_available
            
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            raise
    
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            logger.info(f"Using Tesseract version: {version}")
            
            # Tesseract doesn't use GPU, so set use_gpu to False
            self.use_gpu = False
            
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            raise
    
    def extract_text(self, image_path: Path) -> List[OCRResult]:
        """
        Extract text from image using the configured OCR engine.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of OCRResult objects
        """
        try:
            if self.engine == "easyocr":
                return self._extract_with_easyocr(image_path)
            elif self.engine == "tesseract":
                return self._extract_with_tesseract(image_path)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            # Try fallback engine
            return self._try_fallback_extraction(image_path)
    
    def _extract_with_easyocr(self, image_path: Path) -> List[OCRResult]:
        """Extract text using EasyOCR."""
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
            
            logger.debug(f"EasyOCR extracted {len(results)} text regions from {image_path}")
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            raise
        
        return results
    
    def _extract_with_tesseract(self, image_path: Path) -> List[OCRResult]:
        """Extract text using Tesseract OCR."""
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
            
            logger.debug(f"Tesseract extracted {len(results)} text regions from {image_path}")
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise
        
        return results
    
    def _try_fallback_extraction(self, image_path: Path) -> List[OCRResult]:
        """Try fallback OCR engine if primary fails."""
        try:
            fallback_engine = "tesseract" if self.engine == "easyocr" else "easyocr"
            logger.info(f"Trying fallback OCR engine: {fallback_engine}")
            
            # Create temporary fallback engine
            fallback = OCREngine(fallback_engine, use_gpu=False, confidence_threshold=self.confidence_threshold)
            return fallback.extract_text(image_path)
            
        except Exception as e:
            logger.error(f"Fallback OCR also failed: {e}")
            return []

class ImageTextReplacer:
    """Handles text replacement in images with precise positioning."""
    
    def __init__(self, mode: str = "replace"):
        """
        Initialize image text replacer.
        
        Args:
            mode: Processing mode ('replace' or 'append')
        """
        self.mode = mode
    
    def replace_text_in_image(self, image_path: Path, ocr_matches: List[OCRMatch]) -> Optional[Path]:
        """
        Replace text in image using white rectangle overlay and new text.
        
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
            draw = ImageDraw.Draw(image)
            
            # Process each match
            for match in ocr_matches:
                if match.processing_mode == "replace":
                    self._replace_text_region(draw, match)
                elif match.processing_mode == "append":
                    self._append_text_region(draw, match)
            
            # Save modified image
            output_path = self._generate_output_path(image_path)
            image.save(output_path)
            
            logger.debug(f"Image text replacement completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Image text replacement failed for {image_path}: {e}")
            return None
    
    def _replace_text_region(self, draw: ImageDraw.Draw, match: OCRMatch):
        """Replace text in a specific region."""
        bbox = match.ocr_result.bounding_box
        x, y, width, height = bbox
        
        # Draw white rectangle to cover original text
        draw.rectangle([x, y, x + width, y + height], fill='white')
        
        # Draw replacement text
        self._draw_text(draw, match.replacement_text, (x, y, width, height))
    
    def _append_text_region(self, draw: ImageDraw.Draw, match: OCRMatch):
        """Append text below the original text region."""
        bbox = match.ocr_result.bounding_box
        x, y, width, height = bbox
        
        # Create combined text
        combined_text = f"{match.ocr_result.text}\n{match.replacement_text}"
        
        # Calculate new region height (approximately double)
        new_height = height * 2
        
        # Draw white rectangle for combined text
        draw.rectangle([x, y, x + width, y + new_height], fill='white')
        
        # Draw combined text
        self._draw_text(draw, combined_text, (x, y, width, new_height))
    
    def _draw_text(self, draw: ImageDraw.Draw, text: str, bbox: Tuple[int, int, int, int]):
        """Draw text within the specified bounding box."""
        x, y, width, height = bbox
        
        # Try to load appropriate font
        font = self._get_font(height)
        
        # Calculate text position (centered)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x + (width - text_width) // 2
        text_y = y + (height - text_height) // 2
        
        # Draw text
        draw.text((text_x, text_y), text, fill='black', font=font)
    
    def _get_font(self, target_height: int) -> ImageFont.FreeTypeFont:
        """Get appropriate font for the target height."""
        # Estimate font size based on target height
        font_size = max(8, int(target_height * 0.7))
        
        # Try to load system fonts
        for font_name in FALLBACK_FONTS:
            try:
                return ImageFont.truetype(font_name, font_size)
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path for modified image."""
        output_dir = input_path.parent
        stem = input_path.stem
        suffix = input_path.suffix
        
        return output_dir / f"{stem}_modified{suffix}"

class ImageProcessor:
    """Main image processor for OCR and text replacement."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str], 
                 mode: str = "replace", ocr_engine: str = "easyocr", 
                 use_gpu: bool = True, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE):
        """
        Initialize image processor.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            mode: Processing mode ('replace' or 'append')
            ocr_engine: OCR engine to use
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum OCR confidence threshold
        """
        self.patterns = patterns
        self.mappings = mappings
        self.mode = mode
        
        self.pattern_matcher = create_pattern_matcher(patterns, mappings)
        self.ocr_engine = OCREngine(ocr_engine, use_gpu, confidence_threshold)
        self.text_replacer = ImageTextReplacer(mode)
        
        logger.info(f"Image processor initialized with {len(patterns)} patterns, {len(mappings)} mappings, mode: {mode}")
    
    def process_images(self, document: Document, media_dir: Optional[Path] = None) -> List[OCRMatch]:
        """
        Process all images in the document.
        
        Args:
            document: Document to process
            media_dir: Media directory path (optional)
            
        Returns:
            List of OCRMatch objects representing successful replacements
        """
        matches = []
        
        logger.info("Starting image processing...")
        
        try:
            # Extract images from document
            image_paths = self._extract_images_from_document(document, media_dir)
            
            for i, image_path in enumerate(image_paths):
                try:
                    image_matches = self._process_image(image_path, f"image_{i}")
                    matches.extend(image_matches)
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
        
        logger.info(f"Image processing completed: {len(matches)} matches found")
        return matches
    
    def _extract_images_from_document(self, document: Document, media_dir: Optional[Path] = None) -> List[Path]:
        """
        Extract images from DOCX document.
        
        Args:
            document: Document to extract images from
            media_dir: Optional media directory
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        try:
            # Create temporary directory for extracted images
            temp_dir = PathManager.get_temp_directory() / f"docx_images_{uuid.uuid4().hex[:8]}"
            PathManager.ensure_directory(temp_dir)
            
            # Access document parts to find images
            for rel in document.part.rels.values():
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
                        image_path = temp_dir / f"image_{len(image_paths)}{ext}"
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        image_paths.append(image_path)
                        
                    except Exception as e:
                        logger.error(f"Error extracting image: {e}")
            
            logger.debug(f"Extracted {len(image_paths)} images from document")
            
        except Exception as e:
            logger.error(f"Error extracting images from document: {e}")
        
        return image_paths
    
    def _process_image(self, image_path: Path, location: str) -> List[OCRMatch]:
        """
        Process individual image for OCR and text replacement.
        
        Args:
            image_path: Path to image file
            location: Location description
            
        Returns:
            List of OCRMatch objects
        """
        matches = []
        
        try:
            # Step 1: Extract text using OCR
            ocr_results = self.ocr_engine.extract_text(image_path)
            
            if not ocr_results:
                return matches
            
            # Step 2: Find pattern matches in OCR text
            ocr_matches = []
            for ocr_result in ocr_results:
                pattern_matches = self.pattern_matcher.find_matches(ocr_result.text)
                
                for pattern_name, matched_text, start_pos, end_pos in pattern_matches:
                    replacement_text = self.pattern_matcher.get_replacement(matched_text)
                    if replacement_text:
                        ocr_match = create_ocr_match(
                            ocr_result, pattern_name, replacement_text, 
                            image_path, self.mode
                        )
                        ocr_matches.append(ocr_match)
            
            # Step 3: Apply text replacements to image
            if ocr_matches:
                modified_image_path = self.text_replacer.replace_text_in_image(image_path, ocr_matches)
                
                if modified_image_path:
                    # Update image paths in matches
                    for match in ocr_matches:
                        match.image_path = modified_image_path
                    
                    matches.extend(ocr_matches)
                    logger.debug(f"Image processing: {len(ocr_matches)} replacements in {location}")
        
        except Exception as e:
            logger.error(f"Error processing image at {location}: {e}")
        
        return matches
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the image processor configuration."""
        return {
            'mode': self.mode,
            'ocr_engine': self.ocr_engine.engine,
            'use_gpu': self.ocr_engine.use_gpu,
            'confidence_threshold': self.ocr_engine.confidence_threshold,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings)
        }

def create_image_processor(patterns: Dict[str, str], mappings: Dict[str, str], 
                          mode: str = "replace", ocr_engine: str = "easyocr",
                          use_gpu: bool = True, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE) -> ImageProcessor:
    """
    Factory function to create an ImageProcessor instance.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        mode: Processing mode ('replace' or 'append')
        ocr_engine: OCR engine to use
        use_gpu: Whether to use GPU acceleration
        confidence_threshold: Minimum OCR confidence threshold
        
    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(patterns, mappings, mode, ocr_engine, use_gpu, confidence_threshold)