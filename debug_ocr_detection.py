#!/usr/bin/env python3
"""
Comprehensive OCR detection debugging script with visual validation.
This will help identify why the "77-" parts aren't being detected in your document.
"""

import json
import logging
from pathlib import Path
import sys
import cv2
import numpy as np

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.processors.image_processor import OCREngine
from app.utils.pattern_matcher import create_pattern_matcher
from app.utils.image_utils.pattern_debugger import create_pattern_debugger
from app.utils.image_utils.visual_validator import create_visual_validator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_image_quality(image_path: Path) -> dict:
    """Analyze image quality metrics that affect OCR performance."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Failed to load image"}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        height, width = gray.shape
        
        # Brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Noise estimation (using high-frequency components)
        noise = np.mean(cv2.GaussianBlur(gray, (5, 5), 0) - gray) ** 2
        
        # Text-like edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        return {
            "dimensions": f"{width}x{height}",
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1),
            "noise": round(noise, 3),
            "edge_density": round(edge_density, 3),
            "quality_assessment": assess_quality(brightness, contrast, sharpness, edge_density)
        }
    
    except Exception as e:
        return {"error": str(e)}

def assess_quality(brightness, contrast, sharpness, edge_density):
    """Assess overall image quality for OCR."""
    issues = []
    
    if brightness < 50:
        issues.append("Too dark")
    elif brightness > 200:
        issues.append("Too bright")
    
    if contrast < 30:
        issues.append("Low contrast")
    
    if sharpness < 100:
        issues.append("Blurry/Low sharpness")
    
    if edge_density < 0.01:
        issues.append("Very few edges (may not contain text)")
    
    if not issues:
        return "Good quality for OCR"
    else:
        return "Issues: " + ", ".join(issues)

def test_ocr_with_different_settings(image_path: Path, visual_validator):
    """Test OCR with different engines and settings."""
    logger.info(f"\n=== Testing OCR on {image_path.name} ===")
    
    # Analyze image quality first
    quality_info = analyze_image_quality(image_path)
    logger.info(f"Image Quality Analysis: {quality_info}")
    
    # Test different OCR configurations
    test_configs = [
        {"engine": "easyocr", "confidence": 0.1, "preprocessing": True, "gpu": True},
        {"engine": "easyocr", "confidence": 0.1, "preprocessing": False, "gpu": True},
        {"engine": "tesseract", "confidence": 0.1, "preprocessing": True, "gpu": False},
        {"engine": "tesseract", "confidence": 0.1, "preprocessing": False, "gpu": False},
        {"engine": "hybrid", "confidence": 0.1, "preprocessing": True, "gpu": True},
    ]
    
    best_results = []
    best_count = 0
    
    for config in test_configs:
        logger.info(f"\nTesting {config['engine']} (confidence: {config['confidence']}, "
                   f"preprocessing: {config['preprocessing']}, gpu: {config['gpu']})")
        
        try:
            # Create OCR engine with current config
            ocr_engine = OCREngine(
                engine=config['engine'],
                use_gpu=config['gpu'],
                confidence_threshold=config['confidence'],
                enable_preprocessing=config['preprocessing']
            )
            
            # Extract text
            ocr_results = ocr_engine.extract_text(image_path)
            result_count = len(ocr_results)
            
            logger.info(f"  Results: {result_count} text regions detected")
            
            # Log detected text
            for i, result in enumerate(ocr_results):
                logger.info(f"    {i}: '{result.text}' (confidence: {result.confidence:.3f}, "
                           f"bbox: {result.bounding_box})")
            
            # Create visualization
            if result_count > 0:
                viz_path = visual_validator.create_ocr_visualization(
                    image_path, ocr_results, 
                    f"{config['engine'].upper()} Detection (conf: {config['confidence']})"
                )
                logger.info(f"  Visualization saved: {viz_path}")
            
            # Track best results
            if result_count > best_count:
                best_results = ocr_results
                best_count = result_count
        
        except Exception as e:
            logger.error(f"  Failed with {config['engine']}: {e}")
    
    return best_results, quality_info

def test_manual_text_extraction(image_path: Path):
    """Test if we can manually extract text using different approaches."""
    logger.info(f"\n=== Manual Text Extraction Test for {image_path.name} ===")
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error("Failed to load image")
            return
        
        # Try different preprocessing approaches
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Approach 1: Threshold
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Approach 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Approach 3: OTSU threshold
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save preprocessed versions for manual inspection
        debug_dir = Path("debug_output/manual_preprocessing")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_original.png"), image)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_gray.png"), gray)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_thresh_binary.png"), thresh1)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_thresh_adaptive.png"), thresh2)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_thresh_otsu.png"), thresh3)
        
        logger.info(f"Manual preprocessing images saved to: {debug_dir}")
        logger.info("Please manually inspect these images to see if text is visible")
        
        # Try to find contours that might be text
        contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (potential text regions)
        text_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Heuristics for text-like regions
            if (10 < area < 10000 and  # Reasonable size
                0.1 < aspect_ratio < 10 and  # Not too thin or wide
                w > 5 and h > 5):  # Minimum dimensions
                text_contours.append((x, y, w, h))
        
        logger.info(f"Found {len(text_contours)} potential text regions using contour analysis")
        
        # Draw potential text regions
        contour_image = image.copy()
        for i, (x, y, w, h) in enumerate(text_contours):
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(contour_image, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_potential_text_regions.png"), contour_image)
        logger.info(f"Potential text regions visualization saved")
        
    except Exception as e:
        logger.error(f"Manual text extraction failed: {e}")

def comprehensive_ocr_debug():
    """Run comprehensive OCR debugging on document images."""
    logger.info("=== Comprehensive OCR Detection Debugging ===")
    
    # Create visual validator
    visual_validator = create_visual_validator(Path("debug_output"))
    
    # Load patterns and mappings for context
    patterns_path = Path("patterns.json")
    mapping_path = Path("mapping.json")
    
    if not patterns_path.exists() or not mapping_path.exists():
        logger.error("Patterns or mapping file not found")
        return
    
    with open(patterns_path) as f:
        patterns = json.load(f)
    with open(mapping_path) as f:
        mappings = json.load(f)
    
    logger.info(f"Loaded {len(patterns)} patterns and {len(mappings)} mappings")
    
    # Test images from the document
    test_images = [
        Path("source_documents/test_file2_unzip/word/media/image1.jpeg"),
        Path("source_documents/test_file2_unzip/word/media/image2.jpeg"),
        Path("source_documents/test_file2_unzip/word/media/image3.png"),
        Path("source_documents/test_file2_unzip/word/media/image4.png"),
        Path("source_documents/test_file2_unzip/word/media/image5.jpeg"),
    ]
    
    # Find all available images
    media_dir = Path("source_documents/test_file2_unzip/word/media")
    if media_dir.exists():
        available_images = list(media_dir.glob("*.jpeg")) + list(media_dir.glob("*.png"))
        logger.info(f"Found {len(available_images)} images in media directory")
        test_images = available_images[:10]  # Test first 10 images
    
    all_results = []
    
    for image_path in test_images:
        if image_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"ANALYZING: {image_path.name}")
            logger.info(f"{'='*60}")
            
            # Test OCR with different settings
            best_ocr_results, quality_info = test_ocr_with_different_settings(image_path, visual_validator)
            
            # Test manual text extraction
            test_manual_text_extraction(image_path)
            
            # Create detailed analysis report
            processing_info = {
                "image_quality": quality_info,
                "ocr_engines_tested": ["easyocr", "tesseract", "hybrid"],
                "best_ocr_result_count": len(best_ocr_results),
                "preprocessing_attempted": True
            }
            
            # Test pattern matching on any detected text
            matches = []
            if best_ocr_results:
                pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
                for result in best_ocr_results:
                    text_matches = pattern_matcher.find_matches_universal(result.text)
                    matches.extend([match.to_dict() for match in text_matches])
            
            # Create comprehensive report
            report_path = visual_validator.create_detailed_analysis_report(
                image_path, best_ocr_results, matches, processing_info
            )
            
            all_results.append({
                "image": image_path.name,
                "ocr_results": len(best_ocr_results),
                "matches": len(matches),
                "quality": quality_info,
                "report": str(report_path) if report_path else None
            })
        
        else:
            logger.warning(f"Image not found: {image_path}")
    
    # Summary report
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY REPORT")
    logger.info(f"{'='*60}")
    
    total_images = len(all_results)
    images_with_text = sum(1 for r in all_results if r["ocr_results"] > 0)
    images_with_matches = sum(1 for r in all_results if r["matches"] > 0)
    
    logger.info(f"Total images analyzed: {total_images}")
    logger.info(f"Images with OCR text detected: {images_with_text}")
    logger.info(f"Images with pattern matches: {images_with_matches}")
    
    if images_with_text == 0:
        logger.error("\n❌ CRITICAL ISSUE: No text detected in any images!")
        logger.info("\n🔍 POSSIBLE CAUSES:")
        logger.info("   • Images contain graphics/charts rather than readable text")
        logger.info("   • Text is embedded as vector graphics, not raster text")
        logger.info("   • Image resolution too low for OCR")
        logger.info("   • Poor contrast between text and background")
        logger.info("   • Text is part of complex layouts (tables, forms)")
        
        logger.info("\n🛠️ RECOMMENDED ACTIONS:")
        logger.info("   1. Check debug_output/manual_preprocessing/ for preprocessed images")
        logger.info("   2. Manually verify if text is visible in the images")
        logger.info("   3. Try extracting images at higher resolution from the DOCX")
        logger.info("   4. Consider using different OCR tools or manual text extraction")
        logger.info("   5. Check if the table data is available in the DOCX XML structure")
    
    # Save summary
    summary_path = Path("debug_output/reports/summary_report.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nDetailed summary saved to: {summary_path}")
    logger.info(f"Visual reports and images saved to: debug_output/")

if __name__ == "__main__":
    comprehensive_ocr_debug()