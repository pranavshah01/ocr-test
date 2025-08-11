"""
Multiple preprocessing strategies for different image conditions.
Provides automatic strategy selection and fallback mechanisms.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
from enum import Enum

from .image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

class ImageCondition(Enum):
    """Enumeration of different image conditions."""
    LOW_CONTRAST = "low_contrast"
    HIGH_NOISE = "high_noise"
    BLURRY = "blurry"
    ROTATED = "rotated"
    DARK = "dark"
    BRIGHT = "bright"
    NORMAL = "normal"

class PreprocessingStrategy:
    """Base class for preprocessing strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.preprocessor = ImagePreprocessor()
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply the preprocessing strategy to an image."""
        raise NotImplementedError
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if this strategy is suitable for the given image analysis."""
        raise NotImplementedError

class LowContrastStrategy(PreprocessingStrategy):
    """Strategy for low contrast images."""
    
    def __init__(self):
        super().__init__("low_contrast", "Enhanced contrast processing for low contrast images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply low contrast enhancement strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # CLAHE enhancement
        enhanced = self.preprocessor.enhance_contrast(image.copy())
        variants.append(enhanced)
        
        # CLAHE + noise reduction
        enhanced_clean = self.preprocessor.reduce_noise(enhanced.copy())
        variants.append(enhanced_clean)
        
        # Histogram equalization (alternative approach)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        equalized = cv2.equalizeHist(gray)
        if len(image.shape) == 3:
            equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        variants.append(equalized)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if suitable for low contrast images."""
        return image_analysis.get('contrast_ratio', 1.0) < 0.3

class HighNoiseStrategy(PreprocessingStrategy):
    """Strategy for high noise images."""
    
    def __init__(self):
        super().__init__("high_noise", "Aggressive noise reduction for noisy images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply high noise reduction strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Bilateral filtering
        denoised = self.preprocessor.reduce_noise(image.copy())
        variants.append(denoised)
        
        # Gaussian blur + sharpening
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharpened = self.preprocessor.sharpen_image(blurred)
        variants.append(sharpened)
        
        # Non-local means denoising (more aggressive)
        if len(image.shape) == 3:
            nlm_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            nlm_denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        variants.append(nlm_denoised)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if suitable for high noise images."""
        return image_analysis.get('noise_level', 0) > 200

class BlurryImageStrategy(PreprocessingStrategy):
    """Strategy for blurry images."""
    
    def __init__(self):
        super().__init__("blurry", "Sharpening and enhancement for blurry images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply blurry image enhancement strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Standard sharpening
        sharpened = self.preprocessor.sharpen_image(image.copy())
        variants.append(sharpened)
        
        # Unsharp masking
        unsharp_masked = self._apply_unsharp_mask(image.copy())
        variants.append(unsharp_masked)
        
        # Laplacian sharpening
        laplacian_sharpened = self._apply_laplacian_sharpening(image.copy())
        variants.append(laplacian_sharpened)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def _apply_unsharp_mask(self, image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking for sharpening."""
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            
            return unsharp_mask
        except Exception as e:
            logger.warning(f"Unsharp masking failed: {e}")
            return image
    
    def _apply_laplacian_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian-based sharpening."""
        try:
            # Convert to grayscale for Laplacian
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Sharpen by subtracting Laplacian
            sharpened = gray - 0.8 * laplacian
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Convert back to original format
            if len(image.shape) == 3:
                sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return sharpened
        except Exception as e:
            logger.warning(f"Laplacian sharpening failed: {e}")
            return image
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if suitable for blurry images."""
        return image_analysis.get('sharpness', 100) < 30

class RotatedTextStrategy(PreprocessingStrategy):
    """Strategy for rotated text images."""
    
    def __init__(self):
        super().__init__("rotated", "Deskewing and rotation correction for rotated text")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply rotation correction strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Automatic deskewing
        deskewed = self.preprocessor.deskew_text(image.copy())
        variants.append(deskewed)
        
        # Try multiple rotation angles
        for angle in [-2, -1, 1, 2]:  # Small angle corrections
            rotated = self._rotate_image(image.copy(), angle)
            variants.append(rotated)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            logger.warning(f"Image rotation failed: {e}")
            return image
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Always suitable as rotation detection is part of deskewing."""
        return True

class DarkImageStrategy(PreprocessingStrategy):
    """Strategy for dark images."""
    
    def __init__(self):
        super().__init__("dark", "Brightness and gamma correction for dark images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply dark image enhancement strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Gamma correction
        gamma_corrected = self._apply_gamma_correction(image.copy(), gamma=0.7)
        variants.append(gamma_corrected)
        
        # Brightness adjustment
        brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        variants.append(brightened)
        
        # Histogram equalization
        equalized = self.preprocessor.enhance_contrast(image.copy())
        variants.append(equalized)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to brighten dark images."""
        try:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            return cv2.LUT(image, table)
        except Exception as e:
            logger.warning(f"Gamma correction failed: {e}")
            return image
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if suitable for dark images."""
        return image_analysis.get('mean_brightness', 128) < 80

class BrightImageStrategy(PreprocessingStrategy):
    """Strategy for overly bright images."""
    
    def __init__(self):
        super().__init__("bright", "Brightness reduction and contrast enhancement for bright images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply bright image processing strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Reduce brightness
        darkened = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        variants.append(darkened)
        
        # Gamma correction for bright images
        gamma_corrected = self._apply_gamma_correction(image.copy(), gamma=1.3)
        variants.append(gamma_corrected)
        
        # Contrast enhancement
        enhanced = self.preprocessor.enhance_contrast(image.copy())
        variants.append(enhanced)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to darken bright images."""
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        except Exception as e:
            logger.warning(f"Gamma correction failed: {e}")
            return image
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Check if suitable for bright images."""
        return image_analysis.get('mean_brightness', 128) > 200

class NormalImageStrategy(PreprocessingStrategy):
    """Strategy for normal quality images."""
    
    def __init__(self):
        super().__init__("normal", "Standard processing for normal quality images")
    
    def apply(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply standard processing strategy."""
        variants = []
        
        # Original image
        variants.append(image.copy())
        
        # Light contrast enhancement
        enhanced = self.preprocessor.enhance_contrast(image.copy())
        variants.append(enhanced)
        
        # Light sharpening
        sharpened = self.preprocessor.sharpen_image(image.copy())
        variants.append(sharpened)
        
        # Deskewing
        deskewed = self.preprocessor.deskew_text(image.copy())
        variants.append(deskewed)
        
        logger.debug(f"Applied {self.name} strategy, generated {len(variants)} variants")
        return variants
    
    def is_suitable(self, image_analysis: Dict[str, Any]) -> bool:
        """Always suitable as fallback strategy."""
        return True

class PreprocessingStrategyManager:
    """Manages multiple preprocessing strategies and automatic selection."""
    
    def __init__(self):
        """Initialize strategy manager with all available strategies."""
        self.strategies = [
            LowContrastStrategy(),
            HighNoiseStrategy(),
            BlurryImageStrategy(),
            RotatedTextStrategy(),
            DarkImageStrategy(),
            BrightImageStrategy(),
            NormalImageStrategy()  # Always last as fallback
        ]
        
        self.preprocessor = ImagePreprocessor()
        
        logger.info(f"Initialized preprocessing strategy manager with {len(self.strategies)} strategies")
    
    def select_strategies(self, image: np.ndarray) -> List[PreprocessingStrategy]:
        """
        Automatically select suitable strategies based on image analysis.
        
        Args:
            image: Input image to analyze
            
        Returns:
            List of suitable preprocessing strategies
        """
        # Analyze image quality
        image_analysis = self.preprocessor.analyze_image_quality(image)
        
        # Find suitable strategies
        suitable_strategies = []
        for strategy in self.strategies:
            if strategy.is_suitable(image_analysis):
                suitable_strategies.append(strategy)
        
        # Ensure we have at least one strategy (normal as fallback)
        if not suitable_strategies:
            suitable_strategies = [self.strategies[-1]]  # Normal strategy
        
        logger.info(f"Selected {len(suitable_strategies)} strategies: " + 
                   ", ".join([s.name for s in suitable_strategies]))
        
        return suitable_strategies
    
    def apply_strategies(self, image: np.ndarray, strategies: Optional[List[PreprocessingStrategy]] = None) -> List[np.ndarray]:
        """
        Apply selected strategies to generate image variants.
        
        Args:
            image: Input image
            strategies: List of strategies to apply (None for automatic selection)
            
        Returns:
            List of processed image variants
        """
        if strategies is None:
            strategies = self.select_strategies(image)
        
        all_variants = []
        
        for strategy in strategies:
            try:
                variants = strategy.apply(image)
                all_variants.extend(variants)
                logger.debug(f"Strategy '{strategy.name}' generated {len(variants)} variants")
            except Exception as e:
                logger.error(f"Strategy '{strategy.name}' failed: {e}")
        
        # Remove duplicates (keep unique variants)
        unique_variants = self._remove_duplicate_variants(all_variants)
        
        logger.info(f"Generated {len(unique_variants)} unique image variants from {len(strategies)} strategies")
        return unique_variants
    
    def _remove_duplicate_variants(self, variants: List[np.ndarray]) -> List[np.ndarray]:
        """Remove duplicate image variants based on hash comparison."""
        if not variants:
            return variants
        
        unique_variants = []
        seen_hashes = set()
        
        for variant in variants:
            # Create a simple hash of the image
            variant_hash = hash(variant.tobytes())
            
            if variant_hash not in seen_hashes:
                unique_variants.append(variant)
                seen_hashes.add(variant_hash)
        
        logger.debug(f"Removed {len(variants) - len(unique_variants)} duplicate variants")
        return unique_variants
    
    def process_image_with_fallbacks(self, image_path: Path, save_debug: bool = False) -> List[np.ndarray]:
        """
        Process image with automatic strategy selection and fallback mechanisms.
        
        Args:
            image_path: Path to input image
            save_debug: Whether to save debug information
            
        Returns:
            List of processed image variants
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"Processing image with fallback strategies: {image_path.name}")
            
            # Apply automatic strategy selection
            variants = self.apply_strategies(image)
            
            # Save debug information if requested
            if save_debug:
                self._save_strategy_debug(image_path, image, variants)
            
            return variants
            
        except Exception as e:
            logger.error(f"Strategy processing failed for {image_path}: {e}")
            # Return original image as ultimate fallback
            try:
                original = cv2.imread(str(image_path))
                return [original] if original is not None else []
            except:
                return []
    
    def _save_strategy_debug(self, original_path: Path, original_image: np.ndarray, variants: List[np.ndarray]):
        """Save strategy debugging information."""
        try:
            debug_dir = original_path.parent / "strategy_debug"
            debug_dir.mkdir(exist_ok=True)
            
            base_name = original_path.stem
            
            # Save image analysis
            analysis = self.preprocessor.analyze_image_quality(original_image)
            analysis_path = debug_dir / f"{base_name}_analysis.json"
            
            import json
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Save variants
            for i, variant in enumerate(variants):
                variant_path = debug_dir / f"{base_name}_variant_{i:02d}.png"
                cv2.imwrite(str(variant_path), variant)
            
            logger.debug(f"Saved strategy debug information to {debug_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save strategy debug information: {e}")

def create_preprocessing_strategy_manager() -> PreprocessingStrategyManager:
    """
    Factory function to create a PreprocessingStrategyManager instance.
    
    Returns:
        PreprocessingStrategyManager instance
    """
    return PreprocessingStrategyManager()