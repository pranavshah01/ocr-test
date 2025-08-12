"""
Text analysis and orientation detection for enhanced OCR processing.
Provides comprehensive text property analysis including orientation, font size, and color detection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import math

logger = logging.getLogger(__name__)

@dataclass
class TextProperties:
    """Comprehensive text properties including orientation."""
    font_size: int
    font_family: str
    color: Tuple[int, int, int]
    baseline_y: int
    character_spacing: float
    line_height: int
    is_bold: bool
    is_italic: bool
    orientation: float  # Text rotation angle in degrees (0, 90, 180, 270)
    is_rotated: bool    # Whether text is rotated from horizontal
    confidence: float   # Confidence in the analysis

class TextAnalyzer:
    """Advanced text analysis including orientation detection."""
    
    def __init__(self):
        """Initialize text analyzer with default settings."""
        self.orientation_angles = [0, 90, 180, 270]  # Standard orientations
        self.font_size_ratios = {
            'height_ratio': 1.0,  # Font size should match bounding box height for better visibility
            'width_ratio': 0.6    # Character width is typically 60% of font size
        }
        
        logger.info("Text analyzer initialized with orientation detection")
    
    def analyze_text_properties(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> TextProperties:
        """
        Analyze comprehensive text properties within bounding box including orientation.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            TextProperties object with comprehensive analysis
        """
        try:
            x, y, width, height = bbox
            
            # Extract text region
            text_region = image[y:y+height, x:x+width]
            
            # Detect orientation
            orientation = self.detect_text_orientation(image, bbox)
            is_rotated = abs(orientation) > 5  # Consider rotated if > 5 degrees
            
            # Estimate font size considering orientation
            font_size = self.estimate_font_size(bbox, orientation)
            
            # Extract color
            color = self.extract_text_color(image, bbox)
            
            # Detect baseline considering orientation
            baseline_y = self.detect_baseline(image, bbox, orientation)
            
            # Analyze spacing and line height
            character_spacing = self._estimate_character_spacing(text_region, orientation)
            line_height = self._estimate_line_height(text_region, orientation)
            
            # Detect font style (basic detection)
            is_bold, is_italic = self._detect_font_style(text_region)
            
            # Calculate confidence based on analysis quality
            confidence = self._calculate_analysis_confidence(text_region, orientation)
            
            properties = TextProperties(
                font_size=font_size,
                font_family="Arial",  # Default, would need more sophisticated detection
                color=color,
                baseline_y=baseline_y,
                character_spacing=character_spacing,
                line_height=line_height,
                is_bold=is_bold,
                is_italic=is_italic,
                orientation=orientation,
                is_rotated=is_rotated,
                confidence=confidence
            )
            
            logger.debug(f"Analyzed text properties: size={font_size}, orientation={orientation:.1f}°, "
                        f"color={color}, confidence={confidence:.2f}")
            
            return properties
            
        except Exception as e:
            logger.error(f"Text property analysis failed: {e}")
            # Return default properties
            return TextProperties(
                font_size=12,
                font_family="Arial",
                color=(0, 0, 0),
                baseline_y=bbox[1] + bbox[3],
                character_spacing=1.0,
                line_height=int(bbox[3] * 1.2),
                is_bold=False,
                is_italic=False,
                orientation=0.0,
                is_rotated=False,
                confidence=0.5
            )
    
    def detect_text_orientation(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Detect text orientation angle in degrees (0, 90, 180, 270).
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Orientation angle in degrees
        """
        try:
            x, y, width, height = bbox
            
            # Extract text region with some padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + width + padding)
            y2 = min(image.shape[0], y + height + padding)
            
            text_region = image[y1:y2, x1:x2]
            
            # Convert to grayscale if needed
            if len(text_region.shape) == 3:
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = text_region.copy()
            
            # Method 1: Gradient-based orientation detection
            gradient_angle = self._detect_orientation_by_gradients(gray)
            
            # Method 2: Line detection based orientation
            line_angle = self._detect_orientation_by_lines(gray)
            
            # Method 3: Projection profile based orientation
            projection_angle = self._detect_orientation_by_projection(gray)
            
            # Combine results and choose the most consistent one
            angles = [gradient_angle, line_angle, projection_angle]
            angles = [angle for angle in angles if angle is not None]
            
            if angles:
                # Find the most common orientation (quantized to standard angles)
                quantized_angles = [self._quantize_to_standard_angle(angle) for angle in angles]
                orientation = max(set(quantized_angles), key=quantized_angles.count)
            else:
                orientation = 0.0  # Default to horizontal
            
            logger.debug(f"Detected text orientation: {orientation}° (methods: {angles})")
            return orientation
            
        except Exception as e:
            logger.error(f"Orientation detection failed: {e}")
            return 0.0
    
    def _detect_orientation_by_gradients(self, gray: np.ndarray) -> Optional[float]:
        """Detect orientation using image gradients."""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
            
            # Filter by magnitude threshold
            threshold = np.percentile(magnitude, 75)
            strong_gradients = direction[magnitude > threshold]
            
            if len(strong_gradients) > 10:
                # Find dominant direction
                hist, bins = np.histogram(strong_gradients, bins=36, range=(-180, 180))
                dominant_angle = bins[np.argmax(hist)]
                
                # Convert to text orientation (perpendicular to gradient)
                text_angle = dominant_angle + 90
                if text_angle > 180:
                    text_angle -= 360
                
                return text_angle
            
            return None
            
        except Exception as e:
            logger.debug(f"Gradient-based orientation detection failed: {e}")
            return None
    
    def _detect_orientation_by_lines(self, gray: np.ndarray) -> Optional[float]:
        """Detect orientation using Hough line detection."""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(10, min(gray.shape)//4))
            
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    
                    # Convert to standard angle range
                    if angle > 90:
                        angle = angle - 180
                    
                    angles.append(angle)
                
                if angles:
                    # Use median to avoid outliers
                    return np.median(angles)
            
            return None
            
        except Exception as e:
            logger.debug(f"Line-based orientation detection failed: {e}")
            return None
    
    def _detect_orientation_by_projection(self, gray: np.ndarray) -> Optional[float]:
        """Detect orientation using projection profiles."""
        try:
            best_angle = 0
            max_variance = 0
            
            # Test different angles
            for angle in range(-45, 46, 5):
                # Rotate image
                center = (gray.shape[1]//2, gray.shape[0]//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, rotation_matrix, (gray.shape[1], gray.shape[0]))
                
                # Calculate horizontal projection
                projection = np.sum(rotated, axis=1)
                
                # Calculate variance of projection (higher variance = better alignment)
                variance = np.var(projection)
                
                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle
            
            return best_angle if max_variance > 0 else None
            
        except Exception as e:
            logger.debug(f"Projection-based orientation detection failed: {e}")
            return None
    
    def _quantize_to_standard_angle(self, angle: float) -> float:
        """Quantize angle to nearest standard orientation."""
        # Find nearest standard angle
        distances = [abs(angle - std_angle) for std_angle in self.orientation_angles]
        min_distance_idx = distances.index(min(distances))
        
        # Also check wrapped angles (e.g., -90 vs 270)
        wrapped_angles = [angle + 360 if angle < 0 else angle - 360]
        for wrapped_angle in wrapped_angles:
            wrapped_distances = [abs(wrapped_angle - std_angle) for std_angle in self.orientation_angles]
            min_wrapped_distance = min(wrapped_distances)
            if min_wrapped_distance < distances[min_distance_idx]:
                min_distance_idx = wrapped_distances.index(min_wrapped_distance)
        
        return self.orientation_angles[min_distance_idx]
    
    def estimate_font_size(self, bbox: Tuple[int, int, int, int], orientation: float) -> int:
        """
        Estimate font size from bounding box dimensions considering orientation.
        
        Args:
            bbox: Bounding box (x, y, width, height)
            orientation: Text orientation in degrees
            
        Returns:
            Estimated font size in points
        """
        try:
            x, y, width, height = bbox
            
            # Determine text height based on orientation
            if abs(orientation) < 45 or abs(orientation - 180) < 45:
                # Horizontal text (0° or 180°)
                text_height = height
            else:
                # Vertical text (90° or 270°)
                text_height = width
            
            # Estimate font size (typically 70% of text height)
            font_size = int(text_height * self.font_size_ratios['height_ratio'])
            
            # Ensure reasonable bounds
            font_size = max(8, min(72, font_size))
            
            logger.debug(f"Estimated font size: {font_size}pt for bbox {bbox} at {orientation}°")
            return font_size
            
        except Exception as e:
            logger.error(f"Font size estimation failed: {e}")
            return 12  # Default font size
    
    def extract_text_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """
        Extract dominant text color from text region.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            RGB color tuple
        """
        try:
            x, y, width, height = bbox
            text_region = image[y:y+height, x:x+width]
            
            # Convert to RGB if needed
            if len(text_region.shape) == 3:
                if text_region.shape[2] == 3:  # BGR to RGB
                    text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale - convert to RGB
                text_region = cv2.cvtColor(text_region, cv2.COLOR_GRAY2RGB)
            
            # Find text pixels (assuming text is darker than background)
            gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's thresholding to separate text from background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Determine if text is dark on light or light on dark
            mean_gray = np.mean(gray)
            if mean_gray > 127:
                # Light background, dark text
                text_mask = binary == 0
            else:
                # Dark background, light text
                text_mask = binary == 255
            
            # Extract text pixels
            if np.any(text_mask):
                text_pixels = text_region[text_mask]
                
                # Calculate mean color of text pixels
                mean_color = np.mean(text_pixels, axis=0).astype(int)
                color = tuple(mean_color)
            else:
                # Fallback to overall mean color
                color = tuple(np.mean(text_region.reshape(-1, 3), axis=0).astype(int))
            
            logger.debug(f"Extracted text color: {color}")
            return color
            
        except Exception as e:
            logger.error(f"Text color extraction failed: {e}")
            return (0, 0, 0)  # Default to black
    
    def detect_baseline(self, image: np.ndarray, bbox: Tuple[int, int, int, int], orientation: float) -> int:
        """
        Detect text baseline for proper positioning considering orientation.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            orientation: Text orientation in degrees
            
        Returns:
            Baseline Y coordinate
        """
        try:
            x, y, width, height = bbox
            
            # For horizontal text, baseline is typically at 80% of height from top
            # For vertical text, this needs to be adjusted
            
            if abs(orientation) < 45 or abs(orientation - 180) < 45:
                # Horizontal text
                baseline_y = y + int(height * 0.8)
            else:
                # Vertical text - baseline concept is different
                baseline_y = y + height // 2
            
            logger.debug(f"Detected baseline at y={baseline_y} for orientation {orientation}°")
            return baseline_y
            
        except Exception as e:
            logger.error(f"Baseline detection failed: {e}")
            return bbox[1] + bbox[3]  # Default to bottom of bbox
    
    def _estimate_character_spacing(self, text_region: np.ndarray, orientation: float) -> float:
        """Estimate character spacing in the text region."""
        try:
            # This is a simplified estimation
            # In practice, would need more sophisticated analysis
            
            if abs(orientation) < 45 or abs(orientation - 180) < 45:
                # Horizontal text
                spacing = text_region.shape[1] / max(1, text_region.shape[1] // 10)  # Rough estimate
            else:
                # Vertical text
                spacing = text_region.shape[0] / max(1, text_region.shape[0] // 10)
            
            return max(1.0, spacing)
            
        except Exception as e:
            logger.debug(f"Character spacing estimation failed: {e}")
            return 1.0
    
    def _estimate_line_height(self, text_region: np.ndarray, orientation: float) -> int:
        """Estimate line height in the text region."""
        try:
            if abs(orientation) < 45 or abs(orientation - 180) < 45:
                # Horizontal text
                line_height = int(text_region.shape[0] * 1.2)
            else:
                # Vertical text
                line_height = int(text_region.shape[1] * 1.2)
            
            return max(10, line_height)
            
        except Exception as e:
            logger.debug(f"Line height estimation failed: {e}")
            return 15
    
    def _detect_font_style(self, text_region: np.ndarray) -> Tuple[bool, bool]:
        """Detect if text is bold or italic (basic detection)."""
        try:
            # Convert to grayscale if needed
            if len(text_region.shape) == 3:
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = text_region.copy()
            
            # Bold detection: thicker strokes (higher density of dark pixels)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_pixel_ratio = np.sum(binary == 0) / binary.size
            is_bold = dark_pixel_ratio > 0.3  # Threshold for bold text
            
            # Italic detection: skewed text (simplified)
            # This would need more sophisticated analysis in practice
            is_italic = False  # Placeholder
            
            return is_bold, is_italic
            
        except Exception as e:
            logger.debug(f"Font style detection failed: {e}")
            return False, False
    
    def _calculate_analysis_confidence(self, text_region: np.ndarray, orientation: float) -> float:
        """Calculate confidence score for the analysis."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on text region size
            region_size = text_region.shape[0] * text_region.shape[1]
            if region_size > 1000:
                confidence += 0.2
            
            # Increase confidence if orientation is a standard angle
            if orientation in self.orientation_angles:
                confidence += 0.2
            
            # Increase confidence based on contrast
            if len(text_region.shape) == 3:
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = text_region.copy()
            
            contrast = np.std(gray)
            if contrast > 50:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.debug(f"Confidence calculation failed: {e}")
            return 0.5
    
    def analyze_rotated_text_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analyze text region that may be rotated and return orientation-aware properties.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Dictionary with comprehensive analysis including orientation
        """
        try:
            properties = self.analyze_text_properties(image, bbox)
            
            return {
                'properties': properties,
                'bbox': bbox,
                'orientation_detected': properties.orientation != 0.0,
                'rotation_angle': properties.orientation,
                'analysis_confidence': properties.confidence,
                'recommended_processing': self._get_processing_recommendations(properties)
            }
            
        except Exception as e:
            logger.error(f"Rotated text region analysis failed: {e}")
            return {
                'properties': None,
                'bbox': bbox,
                'orientation_detected': False,
                'rotation_angle': 0.0,
                'analysis_confidence': 0.0,
                'recommended_processing': ['standard']
            }
    
    def _get_processing_recommendations(self, properties: TextProperties) -> List[str]:
        """Get processing recommendations based on text properties."""
        recommendations = []
        
        if properties.is_rotated:
            recommendations.append('rotation_correction')
        
        if properties.font_size < 10:
            recommendations.append('upscaling')
        
        if properties.confidence < 0.7:
            recommendations.append('preprocessing_enhancement')
        
        if not recommendations:
            recommendations.append('standard')
        
        return recommendations

def create_text_analyzer() -> TextAnalyzer:
    """
    Factory function to create a TextAnalyzer instance.
    
    Returns:
        TextAnalyzer instance
    """
    return TextAnalyzer()