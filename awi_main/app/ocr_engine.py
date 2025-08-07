from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import pytesseract
import numpy as np
import logging
import math
from datetime import datetime
try:
    from .shared_constants import (DEFAULT_OCR_CONFIDENCE, DEFAULT_OCR_ENGINE, 
                                   OCR_SUPPORTED_ENGINES, XML_NAMESPACES, 
                                   SharedUtilities)
except ImportError:
    from shared_constants import (DEFAULT_OCR_CONFIDENCE, DEFAULT_OCR_ENGINE, 
                                  OCR_SUPPORTED_ENGINES, XML_NAMESPACES, 
                                  SharedUtilities)

logger = logging.getLogger(__name__)


class EnhancedOCREngine:
    """
    Enhanced OCR Engine implementing Task 4.1 requirements:
    - EasyOCR as primary engine with Tesseract fallback
    - Position, size, and orientation extraction
    - Confidence-based filtering
    - GPU acceleration support
    """
    
    def __init__(self, primary_engine: str = DEFAULT_OCR_ENGINE, 
                 fallback_engine: str = "tesseract", 
                 confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                 gpu_enabled: bool = True):
        """
        Initialize the enhanced OCR engine.
        
        Args:
            primary_engine: Primary OCR engine ("easyocr" or "tesseract")
            fallback_engine: Fallback OCR engine
            confidence_threshold: Minimum confidence threshold for results
            gpu_enabled: Whether to use GPU acceleration
        """
        self.primary_engine = primary_engine if primary_engine in OCR_SUPPORTED_ENGINES else DEFAULT_OCR_ENGINE
        self.fallback_engine = fallback_engine if fallback_engine in OCR_SUPPORTED_ENGINES else "tesseract"
        self.confidence_threshold = confidence_threshold
        self.gpu_enabled = gpu_enabled
        
        # Engine initialization flags
        self.easyocr_reader = None
        self.easyocr_initialized = False
        self.tesseract_available = False
        
        # Statistics
        self.primary_engine_success = 0
        self.fallback_engine_used = 0
        self.total_ocr_calls = 0
        
        logger.info(f"Enhanced OCR Engine initialized: primary={self.primary_engine}, "
                   f"fallback={self.fallback_engine}, confidence_threshold={self.confidence_threshold}")
    
    def _initialize_easyocr(self) -> bool:
        """
        Initialize EasyOCR reader with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        if self.easyocr_initialized:
            return self.easyocr_reader is not None
        
        try:
            import easyocr
            
            # Initialize with English only for compatibility
            self.easyocr_reader = easyocr.Reader(
                ['en'],  # English only support
                gpu=self.gpu_enabled
            )
            
            self.easyocr_initialized = True
            logger.info(f"EasyOCR initialized successfully (GPU: {self.gpu_enabled})")
            return True
            
        except ImportError:
            logger.warning("EasyOCR not available - install with: pip install easyocr")
            self.easyocr_initialized = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_initialized = True
            return False
    
    def _check_tesseract_availability(self) -> bool:
        """
        Check if Tesseract is available and functional.
        
        Returns:
            True if Tesseract is available, False otherwise
        """
        try:
            # Test Tesseract with a simple operation
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            self.tesseract_available = True
            logger.debug("Tesseract OCR is available")
            return True
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            self.tesseract_available = False
            return False
    
    def preprocess_image_for_ocr(self, image: Image.Image, include_rotations: bool = True) -> List[Tuple[str, Image.Image]]:
        """
        Apply multiple preprocessing techniques to improve OCR accuracy.
        
        Args:
            image: Input PIL Image
            include_rotations: Whether to include rotated versions for robust text detection
            
        Returns:
            List of (preprocessing_type, processed_image) tuples
        """
        from PIL import ImageEnhance, ImageFilter, ImageOps
        
        processed_images = []
        
        try:
            # Original image
            processed_images.append(("original", image))
            
            # Convert to grayscale for better OCR
            gray = image.convert('L')
            processed_images.append(("grayscale", gray))
            
            # High contrast version
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(2.5)
            processed_images.append(("high_contrast", high_contrast))
            
            # Sharp version for better edge detection
            enhancer = ImageEnhance.Sharpness(high_contrast)
            sharp = enhancer.enhance(2.0)
            processed_images.append(("sharp", sharp))
            
            # Inverted (for dark backgrounds with light text)
            inverted = ImageOps.invert(gray)
            processed_images.append(("inverted", inverted))
            
            # Auto-level (normalize brightness)
            autocontrast = ImageOps.autocontrast(gray)
            processed_images.append(("autocontrast", autocontrast))
            
            # Brightness enhanced
            enhancer = ImageEnhance.Brightness(gray)
            bright = enhancer.enhance(1.3)
            processed_images.append(("bright", bright))
            
            # Add rotation preprocessing for robust text detection
            if include_rotations:
                rotation_versions = self._add_rotation_preprocessing(sharp)  # Use sharp version as base
                processed_images.extend(rotation_versions)
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            # Return at least the original image
            processed_images = [("original", image)]
        
        return processed_images
    
    def _add_rotation_preprocessing(self, base_image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """
        Add rotated versions of the image for robust text detection.
        This is crucial for detecting text at various orientations.
        
        Args:
            base_image: Base processed image to rotate
            
        Returns:
            List of (rotation_type, rotated_image) tuples
        """
        rotated_images = []
        
        try:
            # Cardinal rotations (most common orientations)
            cardinal_angles = [90, 180, 270]
            for angle in cardinal_angles:
                rotated = base_image.rotate(angle, expand=True, fillcolor='white')
                rotated_images.append((f"rotated_{angle}", rotated))
            
            # Fine-grained rotations for skewed text (common in scanned documents)
            fine_angles = [-30, -15, 15, 30, 45, -45]
            for angle in fine_angles:
                rotated = base_image.rotate(angle, expand=True, fillcolor='white')
                rotated_images.append((f"rotated_{angle:+d}", rotated))
                
        except Exception as e:
            logger.warning(f"Error in rotation preprocessing: {e}")
        
        return rotated_images
    
    def detect_text_orientation(self, image: Image.Image) -> float:
        """
        Detect the primary text orientation in an image.
        This can be used to optimize rotation preprocessing.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Detected rotation angle in degrees (0 = horizontal text)
        """
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Use basic edge detection to find text orientation
            from PIL import ImageFilter
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Simple heuristic: try OCR on original and 90° rotation
            # Compare confidence scores to determine likely orientation
            original_results = self._quick_ocr_test(gray)
            rotated_90 = gray.rotate(90, expand=True)
            rotated_results = self._quick_ocr_test(rotated_90)
            
            # Return angle based on which orientation gives better results
            if rotated_results and len(rotated_results) > len(original_results):
                return 90.0
            
            return 0.0  # Default to horizontal
            
        except Exception as e:
            logger.warning(f"Error in orientation detection: {e}")
            return 0.0
    
    def _quick_ocr_test(self, image: Image.Image) -> List[str]:
        """
        Quick OCR test to determine text orientation quality.
        
        Args:
            image: PIL Image to test
            
        Returns:
            List of detected text strings
        """
        try:
            if self.tesseract_available:
                import pytesseract
                text = pytesseract.image_to_string(image, config='--psm 6')
                return [line.strip() for line in text.split('\n') if line.strip()]
        except Exception:
            pass
        return []
    
    def calculate_text_orientation(self, bbox: List[List[int]]) -> float:
        """
        Calculate text orientation angle from bounding box coordinates.
        
        Args:
            bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Orientation angle in degrees
        """
        try:
            if len(bbox) >= 2:
                # Calculate angle from first two points
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                
                # Calculate angle in degrees
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                return angle
        except Exception as e:
            logger.debug(f"Error calculating orientation: {e}")
        
        return 0.0
    
    def extract_text_dimensions(self, bbox: List[List[int]]) -> Tuple[int, int]:
        """
        Extract text width and height from bounding box.
        
        Args:
            bbox: Bounding box coordinates
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            if len(bbox) >= 4:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                return width, height
        except Exception as e:
            logger.debug(f"Error extracting dimensions: {e}")
        
        return 0, 0
    
    def run_easyocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run EasyOCR on an image with comprehensive result extraction.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of OCR results with enhanced metadata
        """
        results = []
        
        if not self._initialize_easyocr():
            return results
        
        try:
            # Get preprocessed versions of the image
            processed_images = self.preprocess_image_for_ocr(image)
            
            # Try OCR on each preprocessed version
            for preprocess_type, processed_img in processed_images:
                try:
                    img_array = np.array(processed_img)
                    
                    # Run EasyOCR with optimized parameters
                    ocr_results = self.easyocr_reader.readtext(
                        img_array,
                        detail=1,
                        paragraph=False,
                        width_ths=0.4,  # Lower threshold for narrow text
                        height_ths=0.4,  # Lower threshold for small text
                        decoder='beamsearch',  # Better for technical text
                        beamWidth=5,
                        batch_size=1
                    )
                    
                    for result in ocr_results:
                        bbox, text, confidence = result
                        text = text.strip()
                        
                        # Apply confidence filtering
                        if confidence >= self.confidence_threshold and text and len(text) > 0:
                            # Calculate additional metadata
                            orientation = self.calculate_text_orientation(bbox)
                            width, height = self.extract_text_dimensions(bbox)
                            
                            results.append({
                                "text": text,
                                "confidence": float(confidence),
                                "bbox": bbox,
                                "position": {
                                    "x": min([point[0] for point in bbox]),
                                    "y": min([point[1] for point in bbox]),
                                    "width": width,
                                    "height": height
                                },
                                "orientation": orientation,
                                "engine": "easyocr",
                                "preprocessing": preprocess_type,
                                "fallback_used": False,
                                "gpu_used": self.gpu_enabled
                            })
                            
                except Exception as e:
                    logger.debug(f"EasyOCR failed on {preprocess_type} preprocessing: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
        
        return results
    
    def run_tesseract(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run Tesseract OCR on an image with comprehensive result extraction.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of OCR results with enhanced metadata
        """
        results = []
        
        if not self._check_tesseract_availability():
            return results
        
        try:
            # Get preprocessed versions of the image
            processed_images = self.preprocess_image_for_ocr(image)
            
            for preprocess_type, processed_img in processed_images:
                try:
                    # Get detailed OCR data from Tesseract
                    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                    
                    n_boxes = len(data['level'])
                    for i in range(n_boxes):
                        text = data['text'][i].strip()
                        conf = float(data['conf'][i])
                        
                        # Apply confidence filtering and text validation
                        if conf >= (self.confidence_threshold * 100) and text and len(text) > 0:
                            # Extract position and size information
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            
                            # Create bounding box in EasyOCR format for consistency
                            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                            
                            results.append({
                                "text": text,
                                "confidence": conf / 100.0,  # Normalize to 0-1 range
                                "bbox": bbox,
                                "position": {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h
                                },
                                "orientation": 0.0,  # Tesseract doesn't provide orientation directly
                                "engine": "tesseract",
                                "preprocessing": preprocess_type,
                                "fallback_used": True,
                                "gpu_used": False
                            })
                            
                except Exception as e:
                    logger.debug(f"Tesseract failed on {preprocess_type} preprocessing: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
        
        return results
    
    def run_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run comprehensive OCR using primary engine with fallback support.
        Implements Task 4.1 requirements for enhanced OCR processing.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary containing OCR results and processing metadata
        """
        self.total_ocr_calls += 1
        start_time = datetime.now()
        
        results = {
            "text_results": [],
            "primary_engine": self.primary_engine,
            "fallback_engine": self.fallback_engine,
            "primary_success": False,
            "fallback_used": False,
            "confidence_threshold": self.confidence_threshold,
            "gpu_enabled": self.gpu_enabled,
            "processing_time": 0.0,
            "total_detections": 0,
            "high_confidence_detections": 0,
            "errors": []
        }
        
        try:
            # Try primary engine first
            if self.primary_engine == "easyocr":
                primary_results = self.run_easyocr(image)
            else:
                primary_results = self.run_tesseract(image)
            
            if primary_results:
                results["text_results"].extend(primary_results)
                results["primary_success"] = True
                self.primary_engine_success += 1
                logger.debug(f"Primary engine ({self.primary_engine}) succeeded with {len(primary_results)} results")
            else:
                logger.info(f"Primary engine ({self.primary_engine}) failed, trying fallback")
                
                # Try fallback engine
                if self.fallback_engine == "tesseract":
                    fallback_results = self.run_tesseract(image)
                else:
                    fallback_results = self.run_easyocr(image)
                
                if fallback_results:
                    results["text_results"].extend(fallback_results)
                    results["fallback_used"] = True
                    self.fallback_engine_used += 1
                    logger.debug(f"Fallback engine ({self.fallback_engine}) succeeded with {len(fallback_results)} results")
                else:
                    logger.warning("Both primary and fallback engines failed")
                    results["errors"].append("Both OCR engines failed to extract text")
            
            # Sort results by confidence (highest first)
            results["text_results"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Calculate statistics
            results["total_detections"] = len(results["text_results"])
            results["high_confidence_detections"] = len([
                r for r in results["text_results"] 
                if r.get("confidence", 0) >= (self.confidence_threshold + 0.1)
            ])
            
        except Exception as e:
            error_msg = f"OCR processing failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # Calculate processing time
        end_time = datetime.now()
        results["processing_time"] = (end_time - start_time).total_seconds()
        
        return results
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about OCR engine usage.
        
        Returns:
            Dictionary containing engine usage statistics
        """
        return {
            "total_ocr_calls": self.total_ocr_calls,
            "primary_engine_success": self.primary_engine_success,
            "fallback_engine_used": self.fallback_engine_used,
            "primary_success_rate": (self.primary_engine_success / max(self.total_ocr_calls, 1)) * 100,
            "fallback_usage_rate": (self.fallback_engine_used / max(self.total_ocr_calls, 1)) * 100,
            "primary_engine": self.primary_engine,
            "fallback_engine": self.fallback_engine,
            "confidence_threshold": self.confidence_threshold,
            "gpu_enabled": self.gpu_enabled,
            "easyocr_available": self.easyocr_reader is not None,
            "tesseract_available": self.tesseract_available
        }


# Legacy function for backward compatibility
def _run_tesseract_fallback(image: Image.Image, preprocess_type: str = "original") -> List[Dict[str, Any]]:
    """
    Legacy Tesseract OCR function for backward compatibility.
    
    Args:
        image: PIL Image to process
        preprocess_type: Type of preprocessing applied
        
    Returns:
        List of OCR results in legacy format
    """
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if conf > 30 and text:  # Legacy confidence threshold
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                results.append({
                    "text": text,
                    "conf": conf / 100.0,
                    "bbox": (x, y, x + w, y + h),
                    "engine": "tesseract",
                    "preprocessing": preprocess_type,
                    "fallback_used": True
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Legacy Tesseract OCR failed: {e}")
        return []


def run_ocr(image: Image.Image, use_gpu: bool = True, engine: str = DEFAULT_OCR_ENGINE, 
           confidence_threshold: float = DEFAULT_OCR_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Convenience function for running OCR with the enhanced engine.
    Maintains backward compatibility while providing access to new features.
    
    Args:
        image: PIL Image object
        use_gpu: Whether to use GPU acceleration
        engine: OCR engine to use ("easyocr" or "tesseract")
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        List of OCR results with enhanced metadata
    """
    ocr_engine = EnhancedOCREngine(
        primary_engine=engine,
        fallback_engine="tesseract" if engine == "easyocr" else "easyocr",
        confidence_threshold=confidence_threshold,
        gpu_enabled=use_gpu
    )
    
    results = ocr_engine.run_ocr(image)
    return results.get("text_results", [])


def create_ocr_engine(primary_engine: str = DEFAULT_OCR_ENGINE, 
                     fallback_engine: str = "tesseract",
                     confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                     gpu_enabled: bool = True) -> EnhancedOCREngine:
    """
    Factory function to create an enhanced OCR engine instance.
    
    Args:
        primary_engine: Primary OCR engine
        fallback_engine: Fallback OCR engine
        confidence_threshold: Minimum confidence threshold
        gpu_enabled: Whether to enable GPU acceleration
        
    Returns:
        Configured EnhancedOCREngine instance
    """
    return EnhancedOCREngine(
        primary_engine=primary_engine,
        fallback_engine=fallback_engine,
        confidence_threshold=confidence_threshold,
        gpu_enabled=gpu_enabled
    )