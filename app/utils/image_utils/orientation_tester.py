"""
Systematic orientation testing for OCR text detection.
Tests all four cardinal orientations (0°, 90°, 180°, 270°) to find the best one for OCR.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import tempfile

from ...core.models import OCRResult, create_ocr_result

logger = logging.getLogger(__name__)

class OrientationTester:
    """Systematic orientation testing for OCR optimization."""
    
    def __init__(self):
        """Initialize orientation tester."""
        self.orientations = [0, 90, 180, 270]  # Cardinal orientations to test
        self.temp_dir = Path(tempfile.gettempdir())
        
        logger.info("Orientation tester initialized for systematic rotation testing")
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle (0, 90, 180, 270)
            
        Returns:
            Rotated image
        """
        try:
            if angle == 0:
                return image.copy()
            elif angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # For arbitrary angles, use affine transformation
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new dimensions
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_width = int((height * sin_val) + (width * cos_val))
                new_height = int((height * cos_val) + (width * sin_val))
                
                # Adjust translation
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]
                
                return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        except Exception as e:
            logger.error(f"Failed to rotate image by {angle}°: {e}")
            return image.copy()
    
    def test_orientation_with_ocr(self, image_path: Path, ocr_engine, angle: int) -> Tuple[List[OCRResult], Dict]:
        """
        Test a specific orientation with OCR engine.
        
        Args:
            image_path: Path to original image
            ocr_engine: OCR engine instance
            angle: Rotation angle to test
            
        Returns:
            Tuple of (OCR results, metadata)
        """
        try:
            # Load and rotate image
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return [], {"error": "Failed to load image"}
            
            rotated_image = self.rotate_image(original_image, angle)
            
            # Save rotated image temporarily
            temp_path = self.temp_dir / f"rotated_{angle}_{image_path.name}"
            cv2.imwrite(str(temp_path), rotated_image)
            
            # Run OCR on rotated image
            ocr_results = ocr_engine.extract_text(temp_path)
            
            # Calculate quality metrics
            total_confidence = sum(result.confidence for result in ocr_results)
            avg_confidence = total_confidence / len(ocr_results) if ocr_results else 0
            total_text_length = sum(len(result.text.strip()) for result in ocr_results)
            
            metadata = {
                "angle": angle,
                "text_regions": len(ocr_results),
                "avg_confidence": avg_confidence,
                "total_confidence": total_confidence,
                "total_text_length": total_text_length,
                "temp_image_path": str(temp_path)
            }
            
            # Clean up temp file
            try:
                temp_path.unlink()
            except:
                pass
            
            logger.debug(f"Orientation {angle}°: {len(ocr_results)} regions, "
                        f"avg confidence: {avg_confidence:.3f}, "
                        f"total text: {total_text_length} chars")
            
            return ocr_results, metadata
        
        except Exception as e:
            logger.error(f"Failed to test orientation {angle}°: {e}")
            return [], {"error": str(e), "angle": angle}
    
    def find_best_orientation(self, image_path: Path, ocr_engine, 
                            test_angles: List[int] = None) -> Tuple[int, List[OCRResult], Dict]:
        """
        Find the best orientation for OCR by testing multiple angles.
        
        Args:
            image_path: Path to image file
            ocr_engine: OCR engine instance
            test_angles: List of angles to test (default: [0, 90, 180, 270])
            
        Returns:
            Tuple of (best_angle, best_ocr_results, all_results_metadata)
        """
        if test_angles is None:
            test_angles = self.orientations
        
        logger.info(f"Testing orientations {test_angles} for {image_path.name}")
        
        all_results = {}
        best_angle = 0
        best_results = []
        best_score = -1
        
        for angle in test_angles:
            ocr_results, metadata = self.test_orientation_with_ocr(image_path, ocr_engine, angle)
            all_results[angle] = {
                "ocr_results": ocr_results,
                "metadata": metadata
            }
            
            # Calculate score for this orientation
            score = self._calculate_orientation_score(ocr_results, metadata)
            
            logger.info(f"Orientation {angle}°: score={score:.3f}, "
                       f"regions={len(ocr_results)}, "
                       f"avg_conf={metadata.get('avg_confidence', 0):.3f}")
            
            if score > best_score:
                best_score = score
                best_angle = angle
                best_results = ocr_results
        
        summary = {
            "best_angle": best_angle,
            "best_score": best_score,
            "all_results": all_results,
            "angles_tested": test_angles
        }
        
        logger.info(f"Best orientation for {image_path.name}: {best_angle}° "
                   f"(score: {best_score:.3f}, {len(best_results)} text regions)")
        
        return best_angle, best_results, summary
    
    def _calculate_orientation_score(self, ocr_results: List[OCRResult], metadata: Dict) -> float:
        """
        Calculate a score for an orientation based on OCR results quality.
        
        Args:
            ocr_results: OCR results for this orientation
            metadata: Metadata about the OCR results
            
        Returns:
            Score (higher is better)
        """
        if not ocr_results:
            return 0.0
        
        # Base score from number of text regions
        region_score = len(ocr_results) * 10
        
        # Confidence score
        avg_confidence = metadata.get('avg_confidence', 0)
        confidence_score = avg_confidence * 50
        
        # Text length score (more text is generally better)
        text_length = metadata.get('total_text_length', 0)
        length_score = min(text_length * 0.1, 20)  # Cap at 20 points
        
        # Bonus for high-confidence results
        high_conf_bonus = sum(5 for result in ocr_results if result.confidence > 0.8)
        
        # Penalty for very low confidence results
        low_conf_penalty = sum(2 for result in ocr_results if result.confidence < 0.3)
        
        total_score = region_score + confidence_score + length_score + high_conf_bonus - low_conf_penalty
        
        return max(0.0, total_score)
    
    def create_orientation_comparison_image(self, image_path: Path, all_results: Dict, 
                                          output_path: Path = None) -> Path:
        """
        Create a visual comparison of all tested orientations.
        
        Args:
            image_path: Original image path
            all_results: Results from find_best_orientation
            output_path: Where to save the comparison image
            
        Returns:
            Path to saved comparison image
        """
        try:
            # Load original image
            original = cv2.imread(str(image_path))
            if original is None:
                logger.error(f"Failed to load image for comparison: {image_path}")
                return None
            
            # Create comparison grid
            comparison_images = []
            labels = []
            
            for angle in sorted(all_results.keys()):
                result_data = all_results[angle]
                metadata = result_data['metadata']
                
                # Rotate image for display
                rotated = self.rotate_image(original, angle)
                
                # Resize for grid display
                display_size = (300, 200)
                rotated_resized = cv2.resize(rotated, display_size)
                
                # Add text overlay with results
                overlay = rotated_resized.copy()
                text_info = [
                    f"Angle: {angle}°",
                    f"Regions: {metadata.get('text_regions', 0)}",
                    f"Avg Conf: {metadata.get('avg_confidence', 0):.2f}",
                    f"Text Len: {metadata.get('total_text_length', 0)}"
                ]
                
                y_offset = 20
                for text in text_info:
                    cv2.putText(overlay, text, (5, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                
                comparison_images.append(overlay)
                labels.append(f"{angle}°")
            
            # Arrange in 2x2 grid
            if len(comparison_images) >= 4:
                top_row = np.hstack([comparison_images[0], comparison_images[1]])
                bottom_row = np.hstack([comparison_images[2], comparison_images[3]])
                grid = np.vstack([top_row, bottom_row])
            elif len(comparison_images) == 2:
                grid = np.hstack(comparison_images)
            else:
                grid = comparison_images[0] if comparison_images else original
            
            # Save comparison
            if output_path is None:
                output_path = Path(f"debug_output/orientation_comparison_{image_path.stem}.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_path), grid)
            logger.info(f"Orientation comparison saved: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to create orientation comparison: {e}")
            return None
    
    def apply_best_orientation(self, image_path: Path, best_angle: int, 
                             output_path: Path = None) -> Path:
        """
        Apply the best orientation to an image and save it.
        
        Args:
            image_path: Original image path
            best_angle: Best rotation angle found
            output_path: Where to save the corrected image
            
        Returns:
            Path to corrected image
        """
        try:
            # Load and rotate image
            original = cv2.imread(str(image_path))
            if original is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            corrected = self.rotate_image(original, best_angle)
            
            # Save corrected image
            if output_path is None:
                output_path = Path(f"debug_output/corrected_{image_path.stem}_rot{best_angle}.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_path), corrected)
            logger.info(f"Orientation-corrected image saved: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to apply orientation correction: {e}")
            return None

def create_orientation_tester() -> OrientationTester:
    """
    Factory function to create an OrientationTester instance.
    
    Returns:
        OrientationTester instance
    """
    return OrientationTester()