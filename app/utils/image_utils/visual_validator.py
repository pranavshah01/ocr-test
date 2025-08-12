"""
Visual validation system for enhanced OCR text detection.
Provides comprehensive visual debugging and validation capabilities for OCR processing.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

from ..shared_constants import DEFAULT_OCR_CONFIDENCE
from ...core.models import OCRResult

logger = logging.getLogger(__name__)

class VisualValidator:
    """Comprehensive visual validation system for OCR processing."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize visual validator.
        
        Args:
            output_dir: Directory to save validation images and reports
        """
        self.output_dir = output_dir or Path("./debug_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "before_after").mkdir(exist_ok=True)
        (self.output_dir / "bounding_boxes").mkdir(exist_ok=True)
        (self.output_dir / "preprocessing").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        logger.info(f"Visual validator initialized with output directory: {self.output_dir}")
    
    def create_ocr_visualization(self, image_path: Path, ocr_results: List[OCRResult], 
                                title: str = "OCR Detection Results") -> Path:
        """
        Create visualization of OCR detection results with bounding boxes.
        
        Args:
            image_path: Path to original image
            ocr_results: List of OCR results to visualize
            title: Title for the visualization
            
        Returns:
            Path to saved visualization image
        """
        try:
            # Load original image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Create a copy for visualization
            vis_image = image.copy()
            
            # Draw bounding boxes and text
            for i, result in enumerate(ocr_results):
                x, y, w, h = result.bounding_box
                confidence = result.confidence
                text = result.text
                
                # Choose color based on confidence
                if confidence >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence >= 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence and text label
                label = f"{i}: {confidence:.2f} - {text[:20]}..."
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background for label
                cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                
                # Text label
                cv2.putText(vis_image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add title and summary
            summary = f"Detected: {len(ocr_results)} text regions"
            cv2.putText(vis_image, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_image, summary, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / "bounding_boxes" / f"{image_path.stem}_ocr_viz_{timestamp}.png"
            cv2.imwrite(str(output_path), vis_image)
            
            logger.info(f"OCR visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create OCR visualization: {e}")
            return None
    
    def create_before_after_comparison(self, original_path: Path, processed_path: Path, 
                                     ocr_results: List[OCRResult] = None, 
                                     matches: List[Dict] = None) -> Path:
        """
        Create before/after comparison image showing original and processed versions.
        
        Args:
            original_path: Path to original image
            processed_path: Path to processed image
            ocr_results: OCR results for annotation
            matches: Pattern matches for highlighting
            
        Returns:
            Path to saved comparison image
        """
        try:
            # Load images
            original = cv2.imread(str(original_path))
            processed = cv2.imread(str(processed_path))
            
            if original is None or processed is None:
                logger.error(f"Failed to load images: {original_path}, {processed_path}")
                return None
            
            # Resize images to same height for comparison
            height = max(original.shape[0], processed.shape[0])
            original_resized = cv2.resize(original, 
                                        (int(original.shape[1] * height / original.shape[0]), height))
            processed_resized = cv2.resize(processed, 
                                         (int(processed.shape[1] * height / processed.shape[0]), height))
            
            # Create side-by-side comparison
            comparison = np.hstack([original_resized, processed_resized])
            
            # Add labels
            cv2.putText(comparison, "BEFORE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "AFTER", (original_resized.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Highlight matches if provided
            if matches:
                for match in matches:
                    # Highlight on original (left side)
                    if 'bounding_box' in match:
                        x, y, w, h = match['bounding_box']
                        cv2.rectangle(comparison, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(comparison, f"Match: {match.get('matched_text', '')[:10]}", 
                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save comparison
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / "before_after" / f"{original_path.stem}_comparison_{timestamp}.png"
            cv2.imwrite(str(output_path), comparison)
            
            logger.info(f"Before/after comparison saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create before/after comparison: {e}")
            return None
    
    def create_preprocessing_visualization(self, original_path: Path, 
                                        preprocessed_variants: List[np.ndarray],
                                        strategy_names: List[str] = None) -> Path:
        """
        Create visualization showing original image and all preprocessing variants.
        
        Args:
            original_path: Path to original image
            preprocessed_variants: List of preprocessed image arrays
            strategy_names: Names of preprocessing strategies used
            
        Returns:
            Path to saved visualization
        """
        try:
            # Load original image
            original = cv2.imread(str(original_path))
            if original is None:
                logger.error(f"Failed to load original image: {original_path}")
                return None
            
            # Prepare images for grid layout
            images = [original] + preprocessed_variants
            labels = ["Original"] + (strategy_names or [f"Variant {i}" for i in range(len(preprocessed_variants))])
            
            # Calculate grid dimensions
            num_images = len(images)
            cols = min(4, num_images)  # Max 4 columns
            rows = (num_images + cols - 1) // cols
            
            # Resize all images to same size for grid
            target_size = (300, 200)  # Standard size for grid
            resized_images = []
            
            for img in images:
                if isinstance(img, np.ndarray):
                    resized = cv2.resize(img, target_size)
                    resized_images.append(resized)
                else:
                    logger.warning("Skipping invalid image in preprocessing variants")
            
            # Create grid
            grid_height = rows * (target_size[1] + 40)  # Extra space for labels
            grid_width = cols * target_size[0]
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, (img, label) in enumerate(zip(resized_images, labels)):
                row = i // cols
                col = i % cols
                
                y_start = row * (target_size[1] + 40)
                y_end = y_start + target_size[1]
                x_start = col * target_size[0]
                x_end = x_start + target_size[0]
                
                # Place image
                grid[y_start:y_end, x_start:x_end] = img
                
                # Add label
                cv2.putText(grid, label, (x_start + 5, y_start + target_size[1] + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save preprocessing visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / "preprocessing" / f"{original_path.stem}_preprocessing_{timestamp}.png"
            cv2.imwrite(str(output_path), grid)
            
            logger.info(f"Preprocessing visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create preprocessing visualization: {e}")
            return None
    
    def create_detailed_analysis_report(self, image_path: Path, ocr_results: List[OCRResult],
                                      matches: List[Dict], processing_info: Dict) -> Path:
        """
        Create detailed HTML analysis report with all validation information.
        
        Args:
            image_path: Path to analyzed image
            ocr_results: OCR detection results
            matches: Pattern matching results
            processing_info: Additional processing information
            
        Returns:
            Path to saved HTML report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / "reports" / f"{image_path.stem}_analysis_{timestamp}.html"
            
            # Prepare data for report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'image_path': str(image_path),
                'image_name': image_path.name,
                'ocr_results_count': len(ocr_results),
                'matches_count': len(matches),
                'processing_info': processing_info,
                'ocr_results': [
                    {
                        'text': result.text,
                        'confidence': result.confidence,
                        'bounding_box': result.bounding_box,
                        'bbox_area': result.bounding_box[2] * result.bounding_box[3]
                    }
                    for result in ocr_results
                ],
                'matches': matches
            }
            
            # Generate HTML report
            html_content = self._generate_html_report(report_data)
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Detailed analysis report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to create analysis report: {e}")
            return None
    
    def _generate_html_report(self, data: Dict) -> str:
        """Generate HTML content for analysis report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OCR Analysis Report - {data['image_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .bbox {{ font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OCR Analysis Report</h1>
        <p><strong>Image:</strong> {data['image_name']}</p>
        <p><strong>Generated:</strong> {data['timestamp']}</p>
        <p><strong>OCR Results:</strong> {data['ocr_results_count']} text regions detected</p>
        <p><strong>Pattern Matches:</strong> {data['matches_count']} matches found</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p class="{'success' if data['ocr_results_count'] > 0 else 'error'}">
            OCR Detection: {data['ocr_results_count']} text regions found
        </p>
        <p class="{'success' if data['matches_count'] > 0 else 'warning'}">
            Pattern Matching: {data['matches_count']} matches found
        </p>
    </div>
    
    <div class="section">
        <h2>OCR Detection Results</h2>
        {self._generate_ocr_table(data['ocr_results'])}
    </div>
    
    <div class="section">
        <h2>Pattern Matches</h2>
        {self._generate_matches_table(data['matches'])}
    </div>
    
    <div class="section">
        <h2>Processing Information</h2>
        <pre>{json.dumps(data['processing_info'], indent=2)}</pre>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {self._generate_recommendations(data)}
    </div>
</body>
</html>
"""
        return html
    
    def _generate_ocr_table(self, ocr_results: List[Dict]) -> str:
        """Generate HTML table for OCR results."""
        if not ocr_results:
            return "<p class='error'>No OCR results found. This indicates the OCR engine could not detect any text in the image.</p>"
        
        html = "<table><tr><th>Text</th><th>Confidence</th><th>Bounding Box</th><th>Area</th></tr>"
        for result in ocr_results:
            confidence_class = "success" if result['confidence'] >= 0.8 else "warning" if result['confidence'] >= 0.5 else "error"
            html += f"""
            <tr>
                <td>{result['text']}</td>
                <td class="{confidence_class}">{result['confidence']:.3f}</td>
                <td class="bbox">{result['bounding_box']}</td>
                <td>{result['bbox_area']}</td>
            </tr>
            """
        html += "</table>"
        return html
    
    def _generate_matches_table(self, matches: List[Dict]) -> str:
        """Generate HTML table for pattern matches."""
        if not matches:
            return "<p class='warning'>No pattern matches found.</p>"
        
        html = "<table><tr><th>Pattern</th><th>Matched Text</th><th>Position</th><th>Context</th></tr>"
        for match in matches:
            html += f"""
            <tr>
                <td>{match.get('pattern_name', 'Unknown')}</td>
                <td><strong>{match.get('matched_text', '')}</strong></td>
                <td>{match.get('start_pos', '')}-{match.get('end_pos', '')}</td>
                <td>{match.get('preceding_context', '')}|<strong>{match.get('matched_text', '')}</strong>|{match.get('following_context', '')}</td>
            </tr>
            """
        html += "</table>"
        return html
    
    def _generate_recommendations(self, data: Dict) -> str:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if data['ocr_results_count'] == 0:
            recommendations.extend([
                "❌ <strong>Critical Issue:</strong> No text detected by OCR engine",
                "🔍 <strong>Possible Causes:</strong>",
                "   • Image resolution too low for text recognition",
                "   • Poor contrast between text and background",
                "   • Text embedded as graphics rather than selectable text",
                "   • Image preprocessing needed (contrast, noise reduction, etc.)",
                "   • Wrong OCR engine for this type of content",
                "🛠️ <strong>Recommended Actions:</strong>",
                "   • Try different OCR engines (EasyOCR vs Tesseract vs Hybrid)",
                "   • Apply image preprocessing (contrast enhancement, noise reduction)",
                "   • Check if image contains actual text or is purely graphical",
                "   • Increase image resolution if possible",
                "   • Try manual text extraction to verify content"
            ])
        elif data['ocr_results_count'] > 0 and data['matches_count'] == 0:
            recommendations.extend([
                "⚠️ <strong>OCR Working but No Pattern Matches:</strong>",
                "🔍 <strong>Possible Causes:</strong>",
                "   • OCR text doesn't match expected patterns",
                "   • Pattern definitions need adjustment",
                "   • OCR confidence too low",
                "   • Text formatting issues in OCR output",
                "🛠️ <strong>Recommended Actions:</strong>",
                "   • Review OCR text output for accuracy",
                "   • Adjust pattern matching rules",
                "   • Lower OCR confidence threshold",
                "   • Check mapping file for missing entries"
            ])
        elif data['matches_count'] > 0:
            recommendations.append("✅ <strong>System Working:</strong> OCR detection and pattern matching successful")
        
        return "<ul><li>" + "</li><li>".join(recommendations) + "</li></ul>"
    
    def validate_replacement_accuracy(self, original_path: Path, processed_path: Path,
                                    expected_replacements: List[Dict]) -> Dict[str, Any]:
        """
        Validate the accuracy of text replacements by comparing before/after images.
        
        Args:
            original_path: Path to original image
            processed_path: Path to processed image
            expected_replacements: List of expected replacement operations
            
        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'original_path': str(original_path),
                'processed_path': str(processed_path),
                'expected_replacements': len(expected_replacements),
                'validation_passed': False,
                'issues_found': [],
                'recommendations': []
            }
            
            # Load images for comparison
            original = cv2.imread(str(original_path))
            processed = cv2.imread(str(processed_path))
            
            if original is None or processed is None:
                validation_results['issues_found'].append("Failed to load images for comparison")
                return validation_results
            
            # Basic image comparison
            if original.shape != processed.shape:
                validation_results['issues_found'].append(
                    f"Image dimensions changed: {original.shape} -> {processed.shape}"
                )
            
            # Calculate difference between images
            diff = cv2.absdiff(original, processed)
            diff_percentage = (np.sum(diff) / (original.shape[0] * original.shape[1] * original.shape[2] * 255)) * 100
            
            validation_results['difference_percentage'] = diff_percentage
            
            if diff_percentage < 0.1:
                validation_results['issues_found'].append("Images are nearly identical - no replacements detected")
            elif diff_percentage > 50:
                validation_results['issues_found'].append("Images are very different - possible processing error")
            
            # Check if validation passed
            validation_results['validation_passed'] = len(validation_results['issues_found']) == 0
            
            # Generate recommendations
            if not validation_results['validation_passed']:
                validation_results['recommendations'].extend([
                    "Review OCR detection accuracy",
                    "Check pattern matching configuration",
                    "Verify image preprocessing settings",
                    "Test with different OCR engines"
                ])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate replacement accuracy: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'issues_found': [f"Validation error: {e}"]
            }

def create_visual_validator(output_dir: Path = None) -> VisualValidator:
    """
    Factory function to create a VisualValidator instance.
    
    Args:
        output_dir: Directory to save validation images and reports
        
    Returns:
        VisualValidator instance
    """
    return VisualValidator(output_dir)