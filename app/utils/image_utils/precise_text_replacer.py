"""
Precise text replacement with pixel-perfect positioning and orientation support.
Handles text replacement while maintaining original orientation and appearance.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple, Optional, List
import logging
import math
from pathlib import Path

from .text_analyzer import TextProperties, TextAnalyzer

logger = logging.getLogger(__name__)

class PreciseTextReplacer:
    """Precise text replacement system with orientation-aware capabilities."""
    
    def __init__(self):
        """Initialize precise text replacement system."""
        self.text_analyzer = TextAnalyzer()
        self.fallback_fonts = [
            "Arial.ttf",
            "arial.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/calibri.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]
        
        logger.info("Precise text replacer initialized with orientation support")
    
    def replace_text_precise(self, image: Image.Image, original_text: str, replacement_text: str,
                           bbox: Tuple[int, int, int, int], properties: Optional[TextProperties] = None) -> Image.Image:
        """
        Replace text with pixel-perfect positioning maintaining orientation.
        
        Args:
            image: PIL Image to modify
            original_text: Original text being replaced
            replacement_text: New text to render
            bbox: Bounding box (x, y, width, height)
            properties: Text properties (will analyze if not provided)
            
        Returns:
            Modified image with replaced text
        """
        try:
            # Analyze text properties if not provided
            if properties is None:
                image_array = np.array(image)
                properties = self.text_analyzer.analyze_text_properties(image_array, bbox)
            
            logger.debug(f"Replacing '{original_text}' with '{replacement_text}' at {bbox}")
            logger.debug(f"Text properties: size={properties.font_size}, orientation={properties.orientation}°")
            
            # Create a copy of the image to work with
            result_image = image.copy()
            
            # Draw precise white rectangle to cover original text
            self.draw_precise_white_rectangle(result_image, bbox, properties.orientation)
            
            # Render replacement text with matching properties and orientation
            result_image = self.render_replacement_text(result_image, replacement_text, bbox, properties)
            
            logger.debug("Text replacement completed successfully")
            return result_image
            
        except Exception as e:
            logger.error(f"Precise text replacement failed: {e}")
            return image
    
    def draw_precise_white_rectangle(self, image: Image.Image, bbox: Tuple[int, int, int, int], 
                                   orientation: float):
        """
        Draw white rectangle covering exact text area considering orientation.
        
        Args:
            image: PIL Image to modify
            bbox: Bounding box (x, y, width, height)
            orientation: Text orientation in degrees
        """
        try:
            draw = ImageDraw.Draw(image)
            x, y, width, height = bbox
            
            if abs(orientation) < 5:
                # Horizontal text - simple rectangle
                draw.rectangle([x, y, x + width, y + height], fill='white')
                logger.debug(f"Drew horizontal white rectangle at {bbox}")
                
            elif abs(orientation - 90) < 5 or abs(orientation - 270) < 5:
                # Vertical text - simple rectangle (bounding box already accounts for rotation)
                draw.rectangle([x, y, x + width, y + height], fill='white')
                logger.debug(f"Drew vertical white rectangle at {bbox}")
                
            else:
                # Rotated text - draw rotated rectangle
                self._draw_rotated_rectangle(draw, bbox, orientation, 'white')
                logger.debug(f"Drew rotated white rectangle at {bbox} with {orientation}° rotation")
                
        except Exception as e:
            logger.error(f"Failed to draw white rectangle: {e}")
    
    def _draw_rotated_rectangle(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], 
                              angle: float, fill_color: str):
        """Draw a rotated rectangle."""
        try:
            x, y, width, height = bbox
            
            # Calculate rectangle corners
            corners = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            
            # Rotate corners around center
            center_x, center_y = width / 2, height / 2
            angle_rad = math.radians(angle)
            
            rotated_corners = []
            for corner_x, corner_y in corners:
                # Translate to origin
                rel_x = corner_x - center_x
                rel_y = corner_y - center_y
                
                # Rotate
                rot_x = rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad)
                rot_y = rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad)
                
                # Translate back and add offset
                final_x = rot_x + center_x + x
                final_y = rot_y + center_y + y
                
                rotated_corners.append((final_x, final_y))
            
            # Draw polygon
            draw.polygon(rotated_corners, fill=fill_color)
            
        except Exception as e:
            logger.error(f"Failed to draw rotated rectangle: {e}")
    
    def render_replacement_text(self, image: Image.Image, text: str, bbox: Tuple[int, int, int, int], 
                              properties: TextProperties) -> Image.Image:
        """
        Render replacement text with matching properties and orientation.
        
        Args:
            image: PIL Image to modify
            text: Text to render
            bbox: Bounding box (x, y, width, height)
            properties: Text properties including orientation
            
        Returns:
            Image with rendered text
        """
        try:
            if abs(properties.orientation) < 5:
                # Horizontal text - direct rendering
                return self._render_horizontal_text(image, text, bbox, properties)
            else:
                # Rotated text - create rotated text image
                return self.render_rotated_text(image, text, bbox, properties)
                
        except Exception as e:
            logger.error(f"Text rendering failed: {e}")
            return image
    
    def _render_horizontal_text(self, image: Image.Image, text: str, bbox: Tuple[int, int, int, int], 
                              properties: TextProperties) -> Image.Image:
        """Render horizontal text directly on the image."""
        try:
            draw = ImageDraw.Draw(image)
            x, y, width, height = bbox
            
            # Get font
            font = self._get_font(properties.font_family, properties.font_size)
            
            # Calculate optimal font size to fit within bounding box
            optimal_font_size = self.calculate_optimal_font_size(text, bbox, properties.font_family, 0)
            if optimal_font_size != properties.font_size:
                font = self._get_font(properties.font_family, optimal_font_size)
                logger.debug(f"Adjusted font size from {properties.font_size} to {optimal_font_size}")
            
            # Calculate text position (start at same pixel as original)
            text_x = x
            text_y = y
            
            # Fine-tune vertical positioning based on font metrics
            try:
                # Get text bounding box to adjust positioning
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_height = text_bbox[3] - text_bbox[1]
                
                # Center vertically within the bounding box
                text_y = y + (height - text_height) // 2
                
            except Exception as e:
                logger.debug(f"Font metrics adjustment failed: {e}")
            
            # Render text with original color
            color = properties.color
            draw.text((text_x, text_y), text, fill=color, font=font)
            
            logger.debug(f"Rendered horizontal text '{text}' at ({text_x}, {text_y}) "
                        f"with font size {optimal_font_size}")
            
            return image
            
        except Exception as e:
            logger.error(f"Horizontal text rendering failed: {e}")
            return image
    
    def render_rotated_text(self, image: Image.Image, text: str, bbox: Tuple[int, int, int, int], 
                          properties: TextProperties) -> Image.Image:
        """
        Render text rotated to match original orientation.
        
        Args:
            image: PIL Image to modify
            text: Text to render
            bbox: Bounding box (x, y, width, height)
            properties: Text properties including orientation
            
        Returns:
            Image with rotated text
        """
        try:
            # Calculate optimal font size considering rotation
            optimal_font_size = self.calculate_optimal_font_size(text, bbox, properties.font_family, 
                                                               properties.orientation)
            
            # Create rotated text image
            text_image = self.create_rotated_text_image(text, properties.font_family, 
                                                      optimal_font_size, properties.color, 
                                                      properties.orientation)
            
            if text_image is None:
                logger.warning("Failed to create rotated text image, falling back to horizontal")
                return self._render_horizontal_text(image, text, bbox, properties)
            
            # Calculate position to paste the rotated text
            x, y, width, height = bbox
            
            # Center the rotated text within the bounding box
            text_width, text_height = text_image.size
            paste_x = x + (width - text_width) // 2
            paste_y = y + (height - text_height) // 2
            
            # Ensure we don't paste outside image bounds
            paste_x = max(0, min(paste_x, image.width - text_width))
            paste_y = max(0, min(paste_y, image.height - text_height))
            
            # Paste the rotated text onto the image
            if text_image.mode == 'RGBA':
                image.paste(text_image, (paste_x, paste_y), text_image)
            else:
                image.paste(text_image, (paste_x, paste_y))
            
            logger.debug(f"Rendered rotated text '{text}' at ({paste_x}, {paste_y}) "
                        f"with {properties.orientation}° rotation")
            
            return image
            
        except Exception as e:
            logger.error(f"Rotated text rendering failed: {e}")
            return image
    
    def create_rotated_text_image(self, text: str, font_family: str, font_size: int, 
                                color: Tuple[int, int, int], orientation: float) -> Optional[Image.Image]:
        """
        Create rotated text image for precise placement.
        
        Args:
            text: Text to render
            font_family: Font family name
            font_size: Font size in points
            color: RGB color tuple
            orientation: Rotation angle in degrees
            
        Returns:
            PIL Image with rotated text or None if failed
        """
        try:
            # Get font
            font = self._get_font(font_family, font_size)
            
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Get text dimensions
            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Create image large enough for rotated text
            # Use diagonal as safe size
            diagonal = int(math.sqrt(text_width**2 + text_height**2)) + 20
            text_img = Image.new('RGBA', (diagonal, diagonal), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            # Draw text at center
            text_x = (diagonal - text_width) // 2
            text_y = (diagonal - text_height) // 2
            text_draw.text((text_x, text_y), text, fill=color, font=font)
            
            # Rotate the image
            rotated_img = text_img.rotate(-orientation, expand=True, fillcolor=(255, 255, 255, 0))
            
            # Crop to remove excess transparent area
            bbox = rotated_img.getbbox()
            if bbox:
                rotated_img = rotated_img.crop(bbox)
            
            logger.debug(f"Created rotated text image: {rotated_img.size} at {orientation}°")
            return rotated_img
            
        except Exception as e:
            logger.error(f"Rotated text image creation failed: {e}")
            return None
    
    def calculate_optimal_font_size(self, text: str, bbox: Tuple[int, int, int, int], 
                                  font_family: str, orientation: float) -> int:
        """
        Calculate optimal font size to fit within bounding box considering orientation.
        
        Args:
            text: Text to fit
            bbox: Bounding box (x, y, width, height)
            font_family: Font family name
            orientation: Text orientation in degrees
            
        Returns:
            Optimal font size in points
        """
        try:
            x, y, width, height = bbox
            
            # Determine available space based on orientation
            if abs(orientation) < 45 or abs(orientation - 180) < 45:
                # Horizontal text
                available_width = width
                available_height = height
            else:
                # Vertical text - swap dimensions
                available_width = height
                available_height = width
            
            # Start with estimated font size
            estimated_size = int(available_height * 0.7)
            
            # Binary search for optimal size
            min_size = 6
            max_size = min(72, estimated_size * 2)
            optimal_size = estimated_size
            
            for _ in range(10):  # Limit iterations
                test_size = (min_size + max_size) // 2
                
                if self._text_fits_in_bounds(text, font_family, test_size, available_width, available_height):
                    optimal_size = test_size
                    min_size = test_size + 1
                else:
                    max_size = test_size - 1
                
                if min_size > max_size:
                    break
            
            # Ensure reasonable bounds
            optimal_size = max(6, min(72, optimal_size))
            
            logger.debug(f"Calculated optimal font size: {optimal_size}pt for text '{text}' "
                        f"in bounds {available_width}x{available_height}")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Font size calculation failed: {e}")
            return 12  # Default size
    
    def _text_fits_in_bounds(self, text: str, font_family: str, font_size: int, 
                           max_width: int, max_height: int) -> bool:
        """Check if text fits within given bounds."""
        try:
            font = self._get_font(font_family, font_size)
            
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Get text dimensions
            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            return text_width <= max_width and text_height <= max_height
            
        except Exception as e:
            logger.debug(f"Text bounds check failed: {e}")
            return False
    
    def _get_font(self, font_family: str, font_size: int) -> ImageFont.FreeTypeFont:
        """
        Get font object with fallback mechanism.
        
        Args:
            font_family: Preferred font family
            font_size: Font size in points
            
        Returns:
            ImageFont object
        """
        # Try to load the specified font family
        font_paths_to_try = []
        
        # Add specific font family paths
        if font_family and font_family.lower() != 'arial':
            font_paths_to_try.extend([
                f"{font_family}.ttf",
                f"{font_family.lower()}.ttf",
                f"/System/Library/Fonts/{font_family}.ttf",
                f"C:/Windows/Fonts/{font_family.lower()}.ttf"
            ])
        
        # Add fallback fonts
        font_paths_to_try.extend(self.fallback_fonts)
        
        # Try each font path
        for font_path in font_paths_to_try:
            try:
                return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue
        
        # Ultimate fallback to default font
        try:
            return ImageFont.load_default()
        except:
            # Create a minimal font object if all else fails
            logger.warning("All font loading attempts failed, using minimal fallback")
            return ImageFont.load_default()

def create_precise_text_replacer() -> PreciseTextReplacer:
    """
    Factory function to create a PreciseTextReplacer instance.
    
    Returns:
        PreciseTextReplacer instance
    """
    return PreciseTextReplacer()