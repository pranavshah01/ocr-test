"""
Image preprocessing pipeline for enhanced OCR text detection.
Provides adaptive image enhancement techniques to improve OCR accuracy.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing pipeline for OCR enhancement."""
    
    def __init__(self):
        """Initialize image preprocessor with default settings."""
        self.preprocessing_methods = {
            'contrast_enhancement': self.enhance_contrast,
            'noise_reduction': self.reduce_noise,
            'deskewing': self.deskew_text,
            'sharpening': self.sharpen_image,
            'binarization': self.binarize_image
        }
        
        logger.info("Image preprocessor initialized with methods: " + 
                   ", ".join(self.preprocessing_methods.keys()))
    
    def enhance_image(self, image: np.ndarray, methods: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Apply multiple enhancement techniques and return list of variants.
        
        Args:
            image: Input image as numpy array
            methods: List of preprocessing methods to apply (None for all)
            
        Returns:
            List of enhanced image variants
        """
        if methods is None:
            methods = ['contrast_enhancement', 'noise_reduction', 'sharpening']
        
        enhanced_images = [image.copy()]  # Include original
        
        logger.debug(f"Applying preprocessing methods: {methods}")
        
        # Apply individual methods
        for method_name in methods:
            if method_name in self.preprocessing_methods:
                try:
                    enhanced = self.preprocessing_methods[method_name](image.copy())
                    enhanced_images.append(enhanced)
                    logger.debug(f"Applied {method_name} successfully")
                except Exception as e:
                    logger.warning(f"Failed to apply {method_name}: {e}")
        
        # Apply combinations of methods for difficult cases
        if len(methods) > 1:
            try:
                # Contrast + noise reduction
                if 'contrast_enhancement' in methods and 'noise_reduction' in methods:
                    combined = self.enhance_contrast(image.copy())
                    combined = self.reduce_noise(combined)
                    enhanced_images.append(combined)
                
                # Full enhancement pipeline
                if len(methods) >= 3:
                    full_enhanced = self.apply_full_pipeline(image.copy())
                    enhanced_images.append(full_enhanced)
                    
            except Exception as e:
                logger.warning(f"Failed to apply combined methods: {e}")
        
        logger.debug(f"Generated {len(enhanced_images)} image variants")
        return enhanced_images
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast enhancement using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert back to original format if needed
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            logger.debug("Applied CLAHE contrast enhancement")
            return enhanced
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using bilateral filtering and morphological operations.
        
        Args:
            image: Input image
            
        Returns:
            Noise-reduced image
        """
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            if len(image.shape) == 3:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            else:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply morphological operations to clean up small noise
            kernel = np.ones((2, 2), np.uint8)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            logger.debug("Applied bilateral filtering and morphological noise reduction")
            return denoised
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return image
    
    def deskew_text(self, image: np.ndarray) -> np.ndarray:
        """
        Correct text rotation/skewing using Hough line detection.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle of detected lines
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    # Convert to rotation angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                # Use median angle to avoid outliers
                if angles:
                    rotation_angle = np.median(angles)
                    
                    # Only rotate if angle is significant (> 1 degree)
                    if abs(rotation_angle) > 1:
                        # Get image center and rotation matrix
                        height, width = image.shape[:2]
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        
                        # Apply rotation
                        deskewed = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        
                        logger.debug(f"Applied deskewing with rotation angle: {rotation_angle:.2f}°")
                        return deskewed
            
            logger.debug("No significant skew detected, returning original image")
            return image
            
        except Exception as e:
            logger.error(f"Deskewing failed: {e}")
            return image
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image sharpening to enhance text clarity.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            # Define sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # Apply sharpening filter
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            alpha = 0.7  # Weight for sharpened image
            beta = 0.3   # Weight for original image
            result = cv2.addWeighted(sharpened, alpha, image, beta, 0)
            
            logger.debug("Applied image sharpening")
            return result
            
        except Exception as e:
            logger.error(f"Image sharpening failed: {e}")
            return image
    
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization for better text contrast.
        
        Args:
            image: Input image
            
        Returns:
            Binarized image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to original format if needed
            if len(image.shape) == 3:
                binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            logger.debug("Applied adaptive binarization")
            return binary
            
        except Exception as e:
            logger.error(f"Binarization failed: {e}")
            return image
    
    def apply_full_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline for challenging images.
        
        Args:
            image: Input image
            
        Returns:
            Fully processed image
        """
        try:
            # Step 1: Noise reduction
            processed = self.reduce_noise(image)
            
            # Step 2: Contrast enhancement
            processed = self.enhance_contrast(processed)
            
            # Step 3: Deskewing
            processed = self.deskew_text(processed)
            
            # Step 4: Sharpening
            processed = self.sharpen_image(processed)
            
            logger.debug("Applied full preprocessing pipeline")
            return processed
            
        except Exception as e:
            logger.error(f"Full pipeline processing failed: {e}")
            return image
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image quality to determine optimal preprocessing methods.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with quality analysis results
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            analysis = {
                'mean_brightness': float(np.mean(gray)),
                'std_brightness': float(np.std(gray)),
                'contrast_ratio': 0.0,
                'noise_level': 0.0,
                'sharpness': 0.0,
                'recommended_methods': []
            }
            
            # Calculate contrast ratio
            min_val, max_val = np.min(gray), np.max(gray)
            if max_val > 0:
                analysis['contrast_ratio'] = float((max_val - min_val) / max_val)
            
            # Estimate noise level using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            analysis['noise_level'] = float(laplacian_var)
            
            # Estimate sharpness using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            analysis['sharpness'] = float(sharpness)
            
            # Recommend preprocessing methods based on analysis
            if analysis['contrast_ratio'] < 0.3:
                analysis['recommended_methods'].append('contrast_enhancement')
            
            if analysis['noise_level'] < 100:  # Low noise threshold
                analysis['recommended_methods'].append('noise_reduction')
            
            if analysis['sharpness'] < 50:  # Low sharpness threshold
                analysis['recommended_methods'].append('sharpening')
            
            # Always try deskewing for text images
            analysis['recommended_methods'].append('deskewing')
            
            logger.debug(f"Image quality analysis: brightness={analysis['mean_brightness']:.1f}, "
                        f"contrast={analysis['contrast_ratio']:.2f}, "
                        f"noise={analysis['noise_level']:.1f}, "
                        f"sharpness={analysis['sharpness']:.1f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image quality analysis failed: {e}")
            return {
                'mean_brightness': 128.0,
                'std_brightness': 64.0,
                'contrast_ratio': 0.5,
                'noise_level': 100.0,
                'sharpness': 50.0,
                'recommended_methods': ['contrast_enhancement', 'noise_reduction']
            }
    
    def preprocess_for_ocr(self, image_path: Path, save_variants: bool = False) -> List[np.ndarray]:
        """
        Preprocess image specifically for OCR with automatic method selection.
        
        Args:
            image_path: Path to input image
            save_variants: Whether to save preprocessing variants for debugging
            
        Returns:
            List of preprocessed image variants
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Analyze image quality
            quality_analysis = self.analyze_image_quality(image)
            recommended_methods = quality_analysis['recommended_methods']
            
            logger.info(f"Preprocessing image: {image_path.name}")
            logger.info(f"Recommended methods: {recommended_methods}")
            
            # Generate enhanced variants
            enhanced_images = self.enhance_image(image, recommended_methods)
            
            # Save variants for debugging if requested
            if save_variants:
                self._save_preprocessing_variants(image_path, enhanced_images, recommended_methods)
            
            return enhanced_images
            
        except Exception as e:
            logger.error(f"OCR preprocessing failed for {image_path}: {e}")
            # Return original image as fallback
            try:
                original = cv2.imread(str(image_path))
                return [original] if original is not None else []
            except:
                return []
    
    def _save_preprocessing_variants(self, original_path: Path, variants: List[np.ndarray], 
                                   methods: List[str]):
        """Save preprocessing variants for debugging."""
        try:
            debug_dir = original_path.parent / "preprocessing_debug"
            debug_dir.mkdir(exist_ok=True)
            
            base_name = original_path.stem
            
            for i, variant in enumerate(variants):
                if i == 0:
                    variant_name = f"{base_name}_original.png"
                elif i <= len(methods):
                    variant_name = f"{base_name}_{methods[i-1]}.png"
                else:
                    variant_name = f"{base_name}_combined_{i}.png"
                
                variant_path = debug_dir / variant_name
                cv2.imwrite(str(variant_path), variant)
            
            logger.debug(f"Saved {len(variants)} preprocessing variants to {debug_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save preprocessing variants: {e}")

def create_image_preprocessor() -> ImagePreprocessor:
    """
    Factory function to create an ImagePreprocessor instance.
    
    Returns:
        ImagePreprocessor instance
    """
    return ImagePreprocessor()