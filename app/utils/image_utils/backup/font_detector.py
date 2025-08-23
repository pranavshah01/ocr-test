"""
Font Detection Engine for identifying and matching original text font characteristics.
Provides font family detection, system font matching, and fallback font selection.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

class FontDetector:
    """Font detection system for analyzing and matching text font characteristics."""
    
    def __init__(self):
        """Initialize font detection system."""
        self.system_fonts = self._discover_system_fonts()
        self.fallback_fonts = self._get_fallback_fonts()
        self.font_cache = {}
        
        logger.info(f"Font detector initialized with {len(self.system_fonts)} system fonts")
    
    def _discover_system_fonts(self) -> List[str]:
        """Discover available system fonts."""
        system_fonts = []
        
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                font_dirs = [
                    "/System/Library/Fonts",
                    "/Library/Fonts",
                    "~/Library/Fonts"
                ]
            elif system == "Windows":
                font_dirs = [
                    "C:/Windows/Fonts",
                    "C:/Windows/System32/Fonts"
                ]
            else:  # Linux and others
                font_dirs = [
                    "/usr/share/fonts",
                    "/usr/local/share/fonts",
                    "~/.fonts",
                    "~/.local/share/fonts"
                ]
            
            # Scan font directories
            for font_dir in font_dirs:
                expanded_dir = Path(font_dir).expanduser()
                if expanded_dir.exists():
                    for font_file in expanded_dir.rglob("*.ttf"):
                        system_fonts.append(str(font_file))
                    for font_file in expanded_dir.rglob("*.otf"):
                        system_fonts.append(str(font_file))
                    for font_file in expanded_dir.rglob("*.ttc"):
                        system_fonts.append(str(font_file))
            
            logger.debug(f"Discovered {len(system_fonts)} system fonts")
            
        except Exception as e:
            logger.warning(f"Font discovery failed: {e}")
        
        return system_fonts
    
    def _get_fallback_fonts(self) -> List[str]:
        """Get ordered list of fallback fonts."""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return [
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Times.ttc",
                "/System/Library/Fonts/Courier.ttc",
                "/System/Library/Fonts/Geneva.ttf"
            ]
        elif system == "Windows":
            return [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/times.ttf",
                "C:/Windows/Fonts/cour.ttf",
                "C:/Windows/Fonts/tahoma.ttf"
            ]
        else:  # Linux
            return [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
            ]
    
    def detect_font_family(self, image: np.ndarray, bbox: Tuple[int, int, int, int], text: str) -> str:
        """
        Detect font family from text region using image analysis.
        
        Args:
            image: Image array containing the text
            bbox: Bounding box (x, y, width, height) of the text region
            text: The text content for validation
            
        Returns:
            Best matching font family name
        """
        try:
            x, y, width, height = bbox
            
            # Extract text region
            text_region = image[y:y+height, x:x+width]
            
            # Analyze font characteristics
            characteristics = self._analyze_font_characteristics(text_region, text)
            
            # Find matching system font
            best_font = self.find_matching_system_font(characteristics)
            
            logger.debug(f"Detected font family: {best_font} for text '{text}'")
            return best_font
            
        except Exception as e:
            logger.error(f"Font family detection failed: {e}")
            return "Arial"  # Default fallback
    
    def _analyze_font_characteristics(self, text_region: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Analyze font characteristics from text region.
        
        Args:
            text_region: Image region containing the text
            text: The text content
            
        Returns:
            Dictionary of font characteristics
        """
        characteristics = {
            'is_serif': False,
            'is_bold': False,
            'is_italic': False,
            'stroke_width': 1.0,
            'character_width_ratio': 1.0,
            'x_height_ratio': 0.5,
            'ascender_ratio': 0.3,
            'descender_ratio': 0.2
        }
        
        try:
            # Convert to grayscale if needed
            if len(text_region.shape) == 3:
                gray_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = text_region.copy()
            
            # Analyze stroke characteristics
            characteristics.update(self._analyze_stroke_characteristics(gray_region))
            
            # Analyze character proportions
            characteristics.update(self._analyze_character_proportions(gray_region, text))
            
            # Detect serif characteristics
            characteristics['is_serif'] = self._detect_serif(gray_region)
            
            # Detect bold characteristics
            characteristics['is_bold'] = self._detect_bold(gray_region)
            
            # Detect italic characteristics
            characteristics['is_italic'] = self._detect_italic(gray_region)
            
            logger.debug(f"Font characteristics: {characteristics}")
            
        except Exception as e:
            logger.warning(f"Font characteristic analysis failed: {e}")
        
        return characteristics
    
    def _analyze_stroke_characteristics(self, gray_region: np.ndarray) -> Dict[str, float]:
        """Analyze stroke width and characteristics."""
        try:
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'stroke_width': 1.0}
            
            # Analyze stroke width using morphological operations
            kernel_sizes = [1, 2, 3, 4, 5]
            stroke_widths = []
            
            for kernel_size in kernel_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                eroded = cv2.erode(binary, kernel, iterations=1)
                
                # Count remaining pixels
                remaining_pixels = cv2.countNonZero(eroded)
                total_pixels = cv2.countNonZero(binary)
                
                if total_pixels > 0:
                    ratio = remaining_pixels / total_pixels
                    if ratio > 0.1:  # Still has significant content
                        stroke_widths.append(kernel_size)
            
            avg_stroke_width = np.mean(stroke_widths) if stroke_widths else 1.0
            
            return {'stroke_width': float(avg_stroke_width)}
            
        except Exception as e:
            logger.debug(f"Stroke analysis failed: {e}")
            return {'stroke_width': 1.0}
    
    def _analyze_character_proportions(self, gray_region: np.ndarray, text: str) -> Dict[str, float]:
        """Analyze character width and height proportions."""
        try:
            height, width = gray_region.shape
            
            # Estimate character width ratio (width/height)
            char_count = len(text.strip())
            if char_count > 0:
                avg_char_width = width / char_count
                character_width_ratio = avg_char_width / height
            else:
                character_width_ratio = 0.6  # Default ratio
            
            # Analyze vertical proportions using horizontal projection
            horizontal_projection = np.sum(gray_region < 128, axis=1)
            
            # Find text baseline and x-height
            non_zero_rows = np.where(horizontal_projection > 0)[0]
            if len(non_zero_rows) > 0:
                text_top = non_zero_rows[0]
                text_bottom = non_zero_rows[-1]
                text_height = text_bottom - text_top + 1
                
                # Estimate x-height (main body of lowercase letters)
                x_height_ratio = 0.6  # Typical ratio
                ascender_ratio = 0.25
                descender_ratio = 0.15
            else:
                x_height_ratio = 0.6
                ascender_ratio = 0.25
                descender_ratio = 0.15
            
            return {
                'character_width_ratio': character_width_ratio,
                'x_height_ratio': x_height_ratio,
                'ascender_ratio': ascender_ratio,
                'descender_ratio': descender_ratio
            }
            
        except Exception as e:
            logger.debug(f"Character proportion analysis failed: {e}")
            return {
                'character_width_ratio': 0.6,
                'x_height_ratio': 0.6,
                'ascender_ratio': 0.25,
                'descender_ratio': 0.15
            }
    
    def _detect_serif(self, gray_region: np.ndarray) -> bool:
        """Detect if the font has serif characteristics."""
        try:
            # Apply edge detection to find fine details
            edges = cv2.Canny(gray_region, 50, 150)
            
            # Count edge pixels
            edge_count = cv2.countNonZero(edges)
            total_pixels = gray_region.shape[0] * gray_region.shape[1]
            
            # Serif fonts typically have more edge details
            edge_ratio = edge_count / total_pixels
            
            # Threshold for serif detection (tunable)
            return edge_ratio > 0.05
            
        except Exception as e:
            logger.debug(f"Serif detection failed: {e}")
            return False
    
    def _detect_bold(self, gray_region: np.ndarray) -> bool:
        """Detect if the font appears bold."""
        try:
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate stroke thickness using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(binary, kernel, iterations=1)
            
            original_pixels = cv2.countNonZero(binary)
            eroded_pixels = cv2.countNonZero(eroded)
            
            if original_pixels > 0:
                thickness_ratio = eroded_pixels / original_pixels
                # Bold fonts retain more pixels after erosion
                return thickness_ratio > 0.6
            
            return False
            
        except Exception as e:
            logger.debug(f"Bold detection failed: {e}")
            return False
    
    def _detect_italic(self, gray_region: np.ndarray) -> bool:
        """Detect if the font appears italic."""
        try:
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # Analyze contour orientation
            italic_angles = []
            
            for contour in contours:
                if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        angle = ellipse[2]  # Angle of the ellipse
                        
                        # Normalize angle to 0-180 range
                        if angle > 90:
                            angle = 180 - angle
                        
                        italic_angles.append(angle)
                    except:
                        continue
            
            if italic_angles:
                avg_angle = np.mean(italic_angles)
                # Italic text typically has angles between 10-30 degrees
                return 10 <= avg_angle <= 30
            
            return False
            
        except Exception as e:
            logger.debug(f"Italic detection failed: {e}")
            return False
    
    def find_matching_system_font(self, characteristics: Dict[str, Any]) -> str:
        """
        Find best matching system font based on detected characteristics.
        
        Args:
            characteristics: Font characteristics dictionary
            
        Returns:
            Best matching font path or name
        """
        try:
            # Define font categories based on characteristics
            if characteristics.get('is_serif', False):
                preferred_fonts = self._get_serif_fonts()
            else:
                preferred_fonts = self._get_sans_serif_fonts()
            
            # Add weight preference
            if characteristics.get('is_bold', False):
                preferred_fonts = self._get_bold_variants(preferred_fonts)
            
            # Add style preference
            if characteristics.get('is_italic', False):
                preferred_fonts = self._get_italic_variants(preferred_fonts)
            
            # Find the first available font
            for font_name in preferred_fonts:
                font_path = self._find_font_path(font_name)
                if font_path and self.validate_font_rendering(font_path, 12, "Test", (100, 20)):
                    logger.debug(f"Selected font: {font_path}")
                    return font_path
            
            # Fallback to system fonts
            for font_path in self.fallback_fonts:
                if os.path.exists(font_path):
                    logger.debug(f"Using fallback font: {font_path}")
                    return font_path
            
            # Ultimate fallback
            logger.warning("No suitable font found, using Arial")
            return "Arial"
            
        except Exception as e:
            logger.error(f"Font matching failed: {e}")
            return "Arial"
    
    def _get_serif_fonts(self) -> List[str]:
        """Get list of serif font names."""
        return [
            "Times New Roman",
            "Times",
            "Georgia",
            "Garamond",
            "Book Antiqua",
            "Palatino",
            "Century",
            "Minion Pro"
        ]
    
    def _get_sans_serif_fonts(self) -> List[str]:
        """Get list of sans-serif font names."""
        return [
            "Arial",
            "Helvetica",
            "Calibri",
            "Verdana",
            "Tahoma",
            "Geneva",
            "Lucida Grande",
            "Segoe UI",
            "Ubuntu",
            "DejaVu Sans"
        ]
    
    def _get_bold_variants(self, font_names: List[str]) -> List[str]:
        """Get bold variants of font names."""
        bold_variants = []
        for font_name in font_names:
            bold_variants.extend([
                f"{font_name} Bold",
                f"{font_name}-Bold",
                f"{font_name}Bold",
                font_name  # Include regular as fallback
            ])
        return bold_variants
    
    def _get_italic_variants(self, font_names: List[str]) -> List[str]:
        """Get italic variants of font names."""
        italic_variants = []
        for font_name in font_names:
            italic_variants.extend([
                f"{font_name} Italic",
                f"{font_name}-Italic",
                f"{font_name}Italic",
                f"{font_name} Oblique",
                font_name  # Include regular as fallback
            ])
        return italic_variants
    
    def _find_font_path(self, font_name: str) -> Optional[str]:
        """Find the file path for a font name."""
        # Check if it's already a path
        if os.path.exists(font_name):
            return font_name
        
        # Search in system fonts
        for font_path in self.system_fonts:
            font_file = os.path.basename(font_path).lower()
            if font_name.lower().replace(" ", "").replace("-", "") in font_file.replace(" ", "").replace("-", ""):
                return font_path
        
        return None
    
    def get_fallback_fonts(self) -> List[str]:
        """Get ordered list of fallback fonts."""
        return self.fallback_fonts.copy()
    
    def validate_font_rendering(self, font_path: str, size: int, text: str, 
                              target_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validate that font renders within target dimensions.
        
        Args:
            font_path: Path to font file
            size: Font size in points
            text: Text to render
            target_bbox: Target bounding box (x, y, width, height)
            
        Returns:
            True if font renders within target dimensions
        """
        try:
            # Load font
            if font_path in self.font_cache:
                font = self.font_cache[font_path]
            else:
                font = ImageFont.truetype(font_path, size)
                self.font_cache[font_path] = font
            
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Get text dimensions
            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Check if it fits within target
            target_width = target_bbox[2]
            target_height = target_bbox[3]
            
            fits = text_width <= target_width and text_height <= target_height
            
            logger.debug(f"Font validation: {font_path} size {size} - "
                        f"text: {text_width}x{text_height}, target: {target_width}x{target_height}, "
                        f"fits: {fits}")
            
            return fits
            
        except Exception as e:
            logger.debug(f"Font validation failed for {font_path}: {e}")
            return False

def create_font_detector() -> FontDetector:
    """
    Factory function to create a FontDetector instance.
    
    Returns:
        FontDetector instance
    """
    return FontDetector()