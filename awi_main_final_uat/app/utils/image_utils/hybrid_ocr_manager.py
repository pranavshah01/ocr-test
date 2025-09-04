"""
Hybrid OCR Engine Manager for combining EasyOCR and Tesseract results.
Provides parallel execution and intelligent result merging to maximize text pattern detection coverage.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from PIL import Image
import cv2
import easyocr
import pytesseract

from .ocr_models import OCRResult, HybridOCRResult, create_ocr_result, create_hybrid_ocr_result
from config import DEFAULT_OCR_CONFIDENCE

logger = logging.getLogger(__name__)

class HybridOCRManager:
    """
    Hybrid OCR Engine Manager that combines EasyOCR and Tesseract results
    to maximize text pattern detection coverage through parallel execution.
    """
    
    def __init__(self, max_workers: int = 2, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                 use_gpu: bool = True, overlap_threshold: float = 0.5):
        """
        Initialize hybrid OCR manager with parallel execution support.
        
        Args:
            max_workers: Maximum number of parallel workers (default: 2 for dual OCR engines)
            confidence_threshold: Minimum confidence threshold for OCR results
            use_gpu: Whether to use GPU acceleration for EasyOCR
            overlap_threshold: Threshold for considering bounding boxes as overlapping (0.0-1.0)
        """
        self.max_workers = max_workers
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.overlap_threshold = overlap_threshold
        
        # Initialize OCR engines
        self.easyocr_reader = None
        self.tesseract_initialized = False
        
        self._initialize_engines()
        
        logger.info(f"Hybrid OCR Manager initialized with {max_workers} workers, "
                   f"confidence threshold: {confidence_threshold}, GPU: {use_gpu}")
    
    def _initialize_engines(self):
        """Initialize both OCR engines with error handling."""
        # Initialize EasyOCR
        try:
            import torch
            
            # Detect GPU availability for EasyOCR
            gpu_available = False
            if self.use_gpu:
                if torch.cuda.is_available():
                    gpu_available = True
                    logger.info("Using CUDA GPU for Hybrid EasyOCR")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    logger.info("Using MPS GPU for Hybrid EasyOCR")
                else:
                    logger.info("GPU requested but not available for Hybrid EasyOCR, using CPU")
            
            self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
            logger.info("Hybrid EasyOCR engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hybrid EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Initialize Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            self.tesseract_initialized = True
            logger.info(f"Hybrid Tesseract engine initialized successfully (version: {version})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hybrid Tesseract: {e}")
            self.tesseract_initialized = False
    
    def process_hybrid(self, image: Union[np.ndarray, Path]) -> List[HybridOCRResult]:
        """
        Process image with both OCR engines in parallel and combine results.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            List of HybridOCRResult objects with merged detections
        """
        try:
            start_time = time.time()
            
            # Execute OCR engines in parallel
            easyocr_results, tesseract_results = self.execute_ocr_parallel(image)
            logger.debug(f"EasyOCR Results Count: {len(easyocr_results)}")
            logger.debug(f"Tesseract Results Count: {len(tesseract_results)}")
            
            # Merge results to maximize text detection coverage
            hybrid_results = self.merge_ocr_results(easyocr_results, tesseract_results)
            logger.debug(f"Merged Results Count: {len(hybrid_results)}")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Hybrid OCR processing completed in {processing_time:.2f}s: "
                       f"EasyOCR: {len(easyocr_results)}, Tesseract: {len(tesseract_results)}, "
                       f"Merged: {len(hybrid_results)} unique detections")
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid OCR processing failed: {e}")
            return []
    
    def execute_ocr_parallel(self, image: Union[np.ndarray, Path]) -> Tuple[List[OCRResult], List[OCRResult]]:
        """
        Execute EasyOCR and Tesseract in parallel using ThreadPoolExecutor.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            Tuple of (easyocr_results, tesseract_results)
        """
        easyocr_results = []
        tesseract_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit both OCR tasks
                futures = {}
                
                if self.easyocr_reader is not None:
                    future_easyocr = executor.submit(self.run_easyocr, image)
                    futures[future_easyocr] = 'easyocr'
                
                if self.tesseract_initialized:
                    future_tesseract = executor.submit(self.run_tesseract, image)
                    futures[future_tesseract] = 'tesseract'
                
                # Collect results as they complete
                for future in as_completed(futures):
                    engine_name = futures[future]
                    try:
                        results = future.result()
                        if engine_name == 'easyocr':
                            easyocr_results = results
                            logger.debug(f"EasyOCR completed: {len(results)} detections")
                        elif engine_name == 'tesseract':
                            tesseract_results = results
                            logger.debug(f"Tesseract completed: {len(results)} detections")
                    except Exception as e:
                        logger.error(f"Hybrid {engine_name} execution failed: {e}")
            
        except Exception as e:
            logger.error(f"Parallel OCR execution failed: {e}")
        
        return easyocr_results, tesseract_results
    
    def run_easyocr(self, image: Union[np.ndarray, Path]) -> List[OCRResult]:
        """
        Run EasyOCR engine on image.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            List of OCRResult objects
        """
        results = []
        
        try:
            if self.easyocr_reader is None:
                logger.warning("EasyOCR reader not initialized, attempting to reinitialize...")
                # Try to reinitialize EasyOCR
                try:
                    import torch
                    gpu_available = False
                    if self.use_gpu:
                        if torch.cuda.is_available():
                            gpu_available = True
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            gpu_available = True
                    
                    self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
                    logger.info("EasyOCR reader reinitialized successfully")
                except Exception as reinit_error:
                    logger.error(f"Failed to reinitialize EasyOCR: {reinit_error}")
                    return results
            
            if self.easyocr_reader is None:
                logger.error("EasyOCR reader still not available")
                return results
            
            # Handle different image input types
            if isinstance(image, Path):
                ocr_detections = self.easyocr_reader.readtext(str(image))
            else:
                ocr_detections = self.easyocr_reader.readtext(image)
            
            # Convert EasyOCR results to OCRResult objects
            for detection in ocr_detections:
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
            
            logger.debug(f"Hybrid EasyOCR extracted {len(results)} text regions")
            
        except Exception as e:
            logger.error(f"Hybrid EasyOCR execution failed: {e}")
        
        return results
    
    def run_tesseract(self, image: Union[np.ndarray, Path]) -> List[OCRResult]:
        """
        Run Tesseract engine on image.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            List of OCRResult objects
        """
        results = []
        
        try:
            if not self.tesseract_initialized:
                logger.warning("Tesseract not initialized")
                return results
            
            # Handle different image input types
            if isinstance(image, Path):
                pil_image = Image.open(image)
            else:
                # Convert numpy array to PIL Image
                if len(image.shape) == 3:
                    # BGR to RGB conversion for OpenCV arrays
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            
            # Get detailed OCR data from Tesseract
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
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
            
            logger.debug(f"Hybrid Tesseract extracted {len(results)} text regions")
            
        except Exception as e:
            logger.error(f"Hybrid Tesseract execution failed: {e}")
        
        return results
    
    def merge_ocr_results(self, easyocr_results: List[OCRResult], 
                         tesseract_results: List[OCRResult]) -> List[HybridOCRResult]:
        """
        Merge results from both engines to maximize text detection coverage.
        
        Args:
            easyocr_results: Results from EasyOCR engine
            tesseract_results: Results from Tesseract engine
            
        Returns:
            List of HybridOCRResult objects with merged detections
        """
        hybrid_results = []
        
        try:
            # First, add all unique detections from both engines
            unique_detections = self.combine_unique_detections(easyocr_results, tesseract_results)
            hybrid_results.extend(unique_detections)
            
            # Then, handle overlapping detections
            overlapping_groups = self._find_overlapping_groups(easyocr_results, tesseract_results)
            
            for group in overlapping_groups:
                resolved_result = self.handle_overlapping_detections(group)
                if resolved_result:
                    hybrid_results.append(resolved_result)
            
            # Sort results by confidence (highest first)
            hybrid_results.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.debug(f"Merged OCR results: {len(easyocr_results)} EasyOCR + "
                        f"{len(tesseract_results)} Tesseract -> {len(hybrid_results)} hybrid")
            
        except Exception as e:
            logger.error(f"OCR result merging failed: {e}")
        
        return hybrid_results
    
    def combine_unique_detections(self, easyocr_results: List[OCRResult], 
                                tesseract_results: List[OCRResult]) -> List[HybridOCRResult]:
        """
        Combine unique detections from both engines to ensure no text is missed.
        
        Args:
            easyocr_results: Results from EasyOCR engine
            tesseract_results: Results from Tesseract engine
            
        Returns:
            List of HybridOCRResult objects for unique detections
        """
        unique_results = []
        
        # Track which results have been processed to avoid duplicates
        processed_easyocr = set()
        processed_tesseract = set()
        
        # Find overlapping results first
        for i, easy_result in enumerate(easyocr_results):
            for j, tess_result in enumerate(tesseract_results):
                overlap = self.calculate_region_overlap(easy_result.bounding_box, tess_result.bounding_box)
                if overlap > self.overlap_threshold:
                    processed_easyocr.add(i)
                    processed_tesseract.add(j)
        
        # Add unique EasyOCR detections
        for i, result in enumerate(easyocr_results):
            if i not in processed_easyocr:
                hybrid_result = create_hybrid_ocr_result(
                    text=result.text,
                    confidence=result.confidence,
                    bbox=result.bounding_box,
                    source_engine='easyocr',
                    easyocr_result=result,
                    selection_reason='unique_easyocr_detection'
                )
                unique_results.append(hybrid_result)
        
        # Add unique Tesseract detections
        for j, result in enumerate(tesseract_results):
            if j not in processed_tesseract:
                hybrid_result = create_hybrid_ocr_result(
                    text=result.text,
                    confidence=result.confidence,
                    bbox=result.bounding_box,
                    source_engine='tesseract',
                    tesseract_result=result,
                    selection_reason='unique_tesseract_detection'
                )
                unique_results.append(hybrid_result)
        
        logger.debug(f"Combined unique detections: {len(unique_results)} "
                    f"({len([r for r in unique_results if r.source_engine == 'easyocr'])} EasyOCR, "
                    f"{len([r for r in unique_results if r.source_engine == 'tesseract'])} Tesseract)")
        
        return unique_results
    
    def _find_overlapping_groups(self, easyocr_results: List[OCRResult], 
                               tesseract_results: List[OCRResult]) -> List[List[OCRResult]]:
        """Find groups of overlapping OCR results from both engines."""
        overlapping_groups = []
        processed_easy = set()
        processed_tess = set()
        
        for i, easy_result in enumerate(easyocr_results):
            if i in processed_easy:
                continue
                
            group = [easy_result]
            processed_easy.add(i)
            
            # Find all Tesseract results that overlap with this EasyOCR result
            for j, tess_result in enumerate(tesseract_results):
                if j in processed_tess:
                    continue
                    
                overlap = self.calculate_region_overlap(easy_result.bounding_box, tess_result.bounding_box)
                if overlap > self.overlap_threshold:
                    group.append(tess_result)
                    processed_tess.add(j)
            
            # Only add groups that have overlapping results from both engines
            if len(group) > 1:
                overlapping_groups.append(group)
        
        return overlapping_groups
    
    def handle_overlapping_detections(self, overlapping_results: List[OCRResult]) -> Optional[HybridOCRResult]:
        """
        Handle overlapping detections by selecting best result or merging complementary info.
        
        Args:
            overlapping_results: List of overlapping OCR results from different engines
            
        Returns:
            HybridOCRResult with resolved conflict or None if resolution failed
        """
        if not overlapping_results:
            return None
        
        try:
            # Separate results by engine
            easyocr_results = [r for r in overlapping_results if hasattr(r, '_engine') and r._engine == 'easyocr']
            tesseract_results = [r for r in overlapping_results if hasattr(r, '_engine') and r._engine == 'tesseract']
            
            # If we can't determine engine, use confidence-based selection
            if not easyocr_results and not tesseract_results:
                best_result = max(overlapping_results, key=lambda x: x.confidence)
                return create_hybrid_ocr_result(
                    text=best_result.text,
                    confidence=best_result.confidence,
                    bbox=best_result.bounding_box,
                    source_engine='hybrid',
                    selection_reason='highest_confidence_unknown_engine',
                    conflict_resolved=True
                )
            
            # Select best result based on confidence
            all_results = overlapping_results
            best_result = max(all_results, key=lambda x: x.confidence)
            
            # Determine source engine
            source_engine = 'hybrid'
            easyocr_result = None
            tesseract_result = None
            
            if easyocr_results:
                easyocr_result = max(easyocr_results, key=lambda x: x.confidence)
                if best_result in easyocr_results:
                    source_engine = 'easyocr'
            
            if tesseract_results:
                tesseract_result = max(tesseract_results, key=lambda x: x.confidence)
                if best_result in tesseract_results:
                    source_engine = 'tesseract'
            
            # Create hybrid result
            hybrid_result = create_hybrid_ocr_result(
                text=best_result.text,
                confidence=best_result.confidence,
                bbox=best_result.bounding_box,
                source_engine=source_engine,
                easyocr_result=easyocr_result,
                tesseract_result=tesseract_result,
                selection_reason=f'best_confidence_{best_result.confidence:.3f}',
                conflict_resolved=len(overlapping_results) > 1
            )
            
            logger.debug(f"Resolved overlap: selected {source_engine} result "
                        f"'{best_result.text}' (conf: {best_result.confidence:.3f}) "
                        f"from {len(overlapping_results)} candidates")
            
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Failed to handle overlapping detections: {e}")
            return None
    
    def calculate_region_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap percentage between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)
            
        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        try:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection rectangle
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            # Check if there's an intersection
            if left >= right or top >= bottom:
                return 0.0
            
            # Calculate intersection area
            intersection_area = (right - left) * (bottom - top)
            
            # Calculate union area
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - intersection_area
            
            # Calculate overlap percentage (IoU - Intersection over Union)
            if union_area == 0:
                return 0.0
            
            overlap = intersection_area / union_area
            return overlap
            
        except Exception as e:
            logger.error(f"Failed to calculate region overlap: {e}")
            return 0.0

def create_hybrid_ocr_manager(max_workers: int = 2, confidence_threshold: float = DEFAULT_OCR_CONFIDENCE,
                            use_gpu: bool = True, overlap_threshold: float = 0.5) -> HybridOCRManager:
    """
    Factory function to create a HybridOCRManager instance.
    
    Args:
        max_workers: Maximum number of parallel workers
        confidence_threshold: Minimum confidence threshold for OCR results
        use_gpu: Whether to use GPU acceleration for EasyOCR
        overlap_threshold: Threshold for considering bounding boxes as overlapping
        
    Returns:
        HybridOCRManager instance
    """
    return HybridOCRManager(
        max_workers=max_workers,
        confidence_threshold=confidence_threshold,
        use_gpu=use_gpu,
        overlap_threshold=overlap_threshold
    )