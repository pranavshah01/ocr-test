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
import os
from pathlib import Path

from .text_analyzer import TextProperties, TextAnalyzer
from .font_detector import FontDetector, create_font_detector

logger = logging.getLogger(__name__)

class PreciseTextReplacer:
    """Precise text replacement system with orientation-aware capabilities."""
    
    def __init__(self):
        """Initialize precise text replacement system."""
        self.text_analyzer = TextAnalyzer()
        self.font_detector = create_font_detector()
        self.fallback_fonts = self.font_detector.get_fallback_fonts()
        
        logger.info("Precise text replacer initialized with orientation support and font detection")
    
    def replace_text_precise(self, image: Image.Image, original_text: str, replacement_text: str,
                           bbox: Tuple[int, int, int, int], properties: Optional[TextProperties] = None) -> Image.Image:
        """
        Replace text with pixel-perfect positioning maintaining orientation.
        
        Args:
            image: PIL Image to modify
            original_text: Original text being replaced
            replacement_text: New text to render
            bbox: Bounding box (x, y, width, height) of the entire original text
            properties: Text properties (will analyze if not provided)
            
        Returns:
            Modified image with replaced text
        """
        try:
            # Analyze text properties if not provided
            if properties is None:
                image_array = np.array(image)
                properties = self.text_analyzer.analyze_text_properties(image_array, bbox)
                
                # Enhance font detection using the font detector
                detected_font = self.font_detector.detect_font_family(image_array, bbox, original_text)
                properties.font_family = detected_font
            
            logger.info(f"Replacing '{original_text}' with '{replacement_text}' at {bbox}")
            logger.info(f"Text properties: size={properties.font_size}, orientation={properties.orientation}°")
            logger.info(f"Original text length: {len(original_text)}, Replacement text length: {len(replacement_text)}")
            logger.info(f"Bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
            
            # Create a copy of the image to work with
            result_image = image.copy()
            
            # Check if we actually need to do any replacement
            if original_text == replacement_text:
                # No change needed
                return result_image
            
            # Calculate a more precise white rectangle based on the actual text being replaced
            precise_bbox = self.calculate_precise_replacement_bbox(original_text, replacement_text, bbox, properties)
            
            # Draw white rectangle only over the area that needs to be replaced
            self.draw_precise_white_rectangle(result_image, precise_bbox, properties.orientation)
            
            # Render the complete replacement text using the precise bounding box
            result_image = self.render_replacement_text(result_image, replacement_text, precise_bbox, properties)
            
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
                logger.info(f"Drew white rectangle: x={x}, y={y}, width={width}, height={height}")
                
            elif abs(orientation - 90) < 5 or abs(orientation - 270) < 5:
                # Vertical text - simple rectangle (bounding box already accounts for rotation)
                draw.rectangle([x, y, x + width, y + height], fill='white')
                logger.info(f"Drew vertical white rectangle at {bbox}")
                
            else:
                # Rotated text - draw rotated rectangle
                self._draw_rotated_rectangle(draw, bbox, orientation, 'white')
                logger.info(f"Drew rotated white rectangle at {bbox} with {orientation}° rotation")
                
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
            
            # Use the original font size as starting point
            optimal_font_size = properties.font_size
            
            # Apply bounds checking
            if optimal_font_size <= 0:
                # Fallback to bounding box height if no size detected
                optimal_font_size = max(8, height)
            elif optimal_font_size < 6:
                optimal_font_size = 6
            elif optimal_font_size > 72:
                optimal_font_size = 72
            
            # Get font and check if text fits
            font = self._get_font(properties.font_family, optimal_font_size)
            
            # Quick check if text fits in the bounding box
            temp_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = temp_bbox[2] - temp_bbox[0]
            text_height = temp_bbox[3] - temp_bbox[1]
            
            # If text is too wide, reduce font size
            if text_width > width:
                scale_factor = width / text_width * 0.95  # 95% of available width
                adjusted_size = max(6, int(optimal_font_size * scale_factor))
                font = self._get_font(properties.font_family, adjusted_size)
                optimal_font_size = adjusted_size
                logger.debug(f"Reduced font size from {properties.font_size} to {optimal_font_size} "
                           f"to fit text width ({text_width} -> {width})")
            
            # If text is too tall, reduce font size
            elif text_height > height:
                scale_factor = height / text_height * 0.95  # 95% of available height
                adjusted_size = max(6, int(optimal_font_size * scale_factor))
                font = self._get_font(properties.font_family, adjusted_size)
                optimal_font_size = adjusted_size
                logger.debug(f"Reduced font size from {properties.font_size} to {optimal_font_size} "
                           f"to fit text height ({text_height} -> {height})")
            else:
                logger.debug(f"Using font size: {optimal_font_size} (text fits: {text_width}x{text_height} in {width}x{height})")
            
            # Calculate text position - start at the exact same position as original
            text_x = x
            text_y = y
            
            # Simple positioning - just use the original bounding box position
            # This ensures the replacement text starts exactly where the original text was
            logger.info(f"Text positioning: ({text_x}, {text_y}) with font size {optimal_font_size}")
            logger.info(f"Rendering text: '{text}' in bbox {width}x{height}")
            
            # Get actual text dimensions for debugging
            actual_bbox = draw.textbbox((text_x, text_y), text, font=font)
            actual_width = actual_bbox[2] - actual_bbox[0]
            actual_height = actual_bbox[3] - actual_bbox[1]
            
            # Render text with original color at the exact position
            color = properties.color
            draw.text((text_x, text_y), text, fill=color, font=font)
            
            logger.info(f"Rendered text '{text}' at ({text_x}, {text_y}) with font size {optimal_font_size}")
            logger.info(f"Text dimensions: {actual_width}x{actual_height} vs bbox: {width}x{height}")
            
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
        Get font object with enhanced fallback mechanism using font detector.
        
        Args:
            font_family: Preferred font family
            font_size: Font size in points
            
        Returns:
            ImageFont object
        """
        # Try to load the specified font family first
        if font_family and os.path.exists(font_family):
            try:
                return ImageFont.truetype(font_family, font_size)
            except (OSError, IOError):
                pass
        
        # Use font detector to find the best matching font
        try:
            # Create dummy characteristics for font matching
            characteristics = {
                'is_serif': 'serif' in font_family.lower() if font_family else False,
                'is_bold': 'bold' in font_family.lower() if font_family else False,
                'is_italic': 'italic' in font_family.lower() if font_family else False
            }
            
            best_font_path = self.font_detector.find_matching_system_font(characteristics)
            
            if best_font_path and os.path.exists(best_font_path):
                return ImageFont.truetype(best_font_path, font_size)
                
        except Exception as e:
            logger.debug(f"Font detector matching failed: {e}")
        
        # Try fallback fonts from font detector
        for font_path in self.fallback_fonts:
            try:
                if os.path.exists(font_path):
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
    
    def calculate_optimal_font_size_with_validation(self, text: str, bbox: Tuple[int, int, int, int], 
                                                  font_family: str, orientation: float, 
                                                  original_size: int) -> int:
        """
        Calculate optimal font size with enhanced validation and conservative sizing.
        
        Args:
            text: Text to fit
            bbox: Bounding box (x, y, width, height)
            font_family: Font family name
            orientation: Text orientation in degrees
            original_size: Original font size for reference
            
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
            
            # Start with the original size if available, otherwise estimate
            if original_size > 0:
                start_size = original_size
            else:
                start_size = int(available_height * 0.6)  # More conservative estimate
            
            # Use binary search with validation
            min_size = 6
            max_size = min(48, start_size * 2)  # Limit maximum size
            optimal_size = start_size
            
            # First, check if the original size works
            if self._text_fits_in_bounds_with_margin(text, font_family, start_size, 
                                                   available_width, available_height):
                optimal_size = start_size
            else:
                # Binary search for optimal size
                for _ in range(15):  # More iterations for precision
                    test_size = (min_size + max_size) // 2
                    
                    if self._text_fits_in_bounds_with_margin(text, font_family, test_size, 
                                                           available_width, available_height):
                        optimal_size = test_size
                        min_size = test_size + 1
                    else:
                        max_size = test_size - 1
                    
                    if min_size > max_size:
                        break
            
            # Ensure reasonable bounds and conservative sizing
            optimal_size = max(6, min(48, optimal_size))
            
            # If we have an original size, don't deviate too much from it
            if original_size > 0:
                size_ratio = optimal_size / original_size
                if size_ratio > 1.3:  # Don't make text more than 30% larger
                    optimal_size = int(original_size * 1.3)
                elif size_ratio < 0.7:  # Don't make text more than 30% smaller
                    optimal_size = int(original_size * 0.7)
            
            logger.debug(f"Calculated optimal font size: {optimal_size}pt for text '{text}' "
                        f"in bounds {available_width}x{available_height} "
                        f"(original: {original_size}pt)")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Font size calculation failed: {e}")
            return max(6, original_size if original_size > 0 else 12)
    
    def _text_fits_in_bounds_with_margin(self, text: str, font_family: str, font_size: int, 
                                       max_width: int, max_height: int, margin: int = 2) -> bool:
        """
        Check if text fits within given bounds with a safety margin.
        
        Args:
            text: Text to check
            font_family: Font family name
            font_size: Font size in points
            max_width: Maximum width
            max_height: Maximum height
            margin: Safety margin in pixels
            
        Returns:
            True if text fits within bounds with margin
        """
        try:
            font = self._get_font(font_family, font_size)
            
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Get text dimensions
            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Check if it fits within bounds with margin
            fits = (text_width <= max_width - margin and 
                   text_height <= max_height - margin)
            
            logger.debug(f"Text bounds check: {text_width}x{text_height} vs "
                        f"{max_width-margin}x{max_height-margin} (with {margin}px margin): {fits}")
            
            return fits
            
        except Exception as e:
            logger.debug(f"Text bounds check failed: {e}")
            return False
    
    def validate_font_rendering_in_bbox(self, font: ImageFont.FreeTypeFont, text: str, 
                                      bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validate that font renders properly within the given bounding box.
        
        Args:
            font: Font object to validate
            text: Text to render
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            True if font renders properly within the bounding box
        """
        try:
            x, y, width, height = bbox
            
            # Create temporary image to test rendering
            temp_img = Image.new('RGB', (width + 10, height + 10), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Try to render the text
            temp_draw.text((5, 5), text, fill='black', font=font)
            
            # Get actual text dimensions
            text_bbox = temp_draw.textbbox((5, 5), text, font=font)
            actual_width = text_bbox[2] - text_bbox[0]
            actual_height = text_bbox[3] - text_bbox[1]
            
            # Check if it fits with some tolerance
            width_ok = actual_width <= width * 1.1  # Allow 10% overflow
            height_ok = actual_height <= height * 1.1  # Allow 10% overflow
            
            validation_result = width_ok and height_ok
            
            logger.debug(f"Font rendering validation: actual {actual_width}x{actual_height} "
                        f"vs target {width}x{height}: {validation_result}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Font rendering validation failed: {e}")
            return False
    
    def create_fallback_rendering_strategy(self, text: str, bbox: Tuple[int, int, int, int], 
                                         properties: TextProperties) -> Dict[str, Any]:
        """
        Create fallback rendering strategy when standard rendering fails.
        
        Args:
            text: Text to render
            bbox: Bounding box (x, y, width, height)
            properties: Text properties
            
        Returns:
            Dictionary with fallback rendering parameters
        """
        try:
            x, y, width, height = bbox
            
            # Calculate conservative font size
            conservative_size = max(6, int(height * 0.5))  # Very conservative
            
            # Use most reliable font
            reliable_fonts = ["Arial", "Helvetica", "DejaVu Sans"]
            fallback_font = None
            
            for font_name in reliable_fonts:
                try:
                    fallback_font = self._get_font(font_name, conservative_size)
                    if self.validate_font_rendering_in_bbox(fallback_font, text, bbox):
                        break
                except:
                    continue
            
            if fallback_font is None:
                fallback_font = ImageFont.load_default()
            
            return {
                'font': fallback_font,
                'font_size': conservative_size,
                'position': (x, y + height // 4),  # Conservative positioning
                'color': properties.color,
                'strategy': 'conservative_fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback strategy creation failed: {e}")
            return {
                'font': ImageFont.load_default(),
                'font_size': 12,
                'position': (x, y),
                'color': (0, 0, 0),
                'strategy': 'emergency_fallback'
            }
    
    def calculate_changed_text_bbox(self, original_text: str, replacement_text: str, 
                                  full_bbox: Tuple[int, int, int, int], 
                                  properties: TextProperties) -> Tuple[int, int, int, int]:
        """
        Calculate precise bounding box for the changed portion of text.
        
        Args:
            original_text: Original full text
            replacement_text: New full text (with pattern replaced)
            full_bbox: Bounding box of the entire original text
            properties: Text properties
            
        Returns:
            Precise bounding box (x, y, width, height) for the changed portion
        """
        try:
            x, y, width, height = full_bbox
            
            # If the replacement text is the same length or longer, use the full bbox
            # This ensures we don't cut off any text
            if len(replacement_text) >= len(original_text):
                # Calculate width based on text length ratio
                length_ratio = len(replacement_text) / len(original_text) if len(original_text) > 0 else 1
                new_width = int(width * length_ratio)
                return (x, y, new_width, height)
            
            # For shorter replacement text, we still need to cover the full area
            # to avoid leaving artifacts from the original text
            return full_bbox
            
        except Exception as e:
            logger.error(f"Failed to calculate changed text bbox: {e}")
            return full_bbox
    
    def calculate_precise_replacement_bbox(self, original_text: str, replacement_text: str,
                                         full_bbox: Tuple[int, int, int, int],
                                         properties: TextProperties) -> Tuple[int, int, int, int]:
        """
        Calculate a more precise bounding box for just the replacement text.
        This prevents the white rectangle from covering adjacent text.
        
        Args:
            original_text: Original full text
            replacement_text: New full text (with pattern replaced)
            full_bbox: Full OCR bounding box
            properties: Text properties
            
        Returns:
            More precise bounding box for the replacement
        """
        try:
            x, y, width, height = full_bbox
            
            # Get font for measurements
            font = self._get_font(properties.font_family, properties.font_size)
            
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Measure the replacement text
            replacement_bbox = temp_draw.textbbox((0, 0), replacement_text, font=font)
            replacement_width = replacement_bbox[2] - replacement_bbox[0]
            replacement_height = replacement_bbox[3] - replacement_bbox[1]
            
            # Use the smaller of the measured width or the original bbox width
            # This prevents the white rectangle from being too wide
            # But ensure we have enough space to fully cover the original text
            precise_width = min(replacement_width + 20, width)  # Increased padding to 20px
            precise_height = min(replacement_height + 8, height)  # Increased padding to 8px
            # Ensure minimum coverage - if replacement is much shorter, still cover reasonable area
            min_coverage_width = min(width * 0.8, replacement_width + 40)  # At least 80% of original or replacement + 40px
            precise_width = max(precise_width, min_coverage_width)
            
            # Keep the same position but adjust size
            precise_bbox = (x, y, precise_width, precise_height)
            
            logger.info(f"Precise bbox: original {width}x{height} -> precise {precise_width}x{precise_height}")
            logger.info(f"Replacement text width: {replacement_width}px, coverage: {precise_width/width*100:.1f}% of original")
            
            return precise_bbox
            
        except Exception as e:
            logger.error(f"Failed to calculate precise replacement bbox: {e}")
            return full_bbox

def create_precise_text_replacer() -> PreciseTextReplacer:
    """
    Factory function to create a PreciseTextReplacer instance.
    
    Returns:
        PreciseTextReplacer instance
    """
    return PreciseTextReplacer()