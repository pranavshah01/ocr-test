#!/usr/bin/env python3
"""
Comprehensive search for the table image containing 77- parts.
Tests all images and looks for table-like patterns and artifacts.
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
from app.utils.image_utils.orientation_tester import create_orientation_tester
from app.utils.image_utils.visual_validator import create_visual_validator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_image_for_table_characteristics(image_path: Path) -> dict:
    """Analyze image to determine if it looks like a table."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Failed to load image"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect horizontal and vertical lines (table structure)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//30, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//30))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / (width * height)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        vertical_score = np.sum(vertical_lines > 0) / (width * height)
        
        # Overall table-like score
        table_score = horizontal_score + vertical_score
        
        # Check for regular patterns (grid-like structure)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular contours (potential table cells)
        rectangular_contours = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Rectangle
                rectangular_contours += 1
        
        return {
            "dimensions": f"{width}x{height}",
            "horizontal_line_score": round(horizontal_score, 4),
            "vertical_line_score": round(vertical_score, 4),
            "table_score": round(table_score, 4),
            "rectangular_contours": rectangular_contours,
            "likely_table": table_score > 0.01 or rectangular_contours > 10
        }
    
    except Exception as e:
        return {"error": str(e)}

def find_table_image():
    """Comprehensive search for the table image."""
    logger.info("=== Comprehensive Table Image Search ===")
    
    # Load patterns and mappings
    patterns_path = Path("patterns.json")
    mapping_path = Path("mapping.json")
    
    with open(patterns_path) as f:
        patterns = json.load(f)
    with open(mapping_path) as f:
        mappings = json.load(f)
    
    # Create tools
    orientation_tester = create_orientation_tester()
    pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
    visual_validator = create_visual_validator(Path("debug_output"))
    
    # Get all images
    media_dir = Path("source_documents/test_file2_unzip/word/media")
    all_images = list(media_dir.glob("*.jpeg")) + list(media_dir.glob("*.png"))
    
    logger.info(f"Scanning {len(all_images)} images for table characteristics...")
    
    # Create OCR engines
    tesseract_engine = OCREngine("tesseract", use_gpu=False, confidence_threshold=0.05, enable_preprocessing=False)
    easyocr_engine = OCREngine("easyocr", use_gpu=True, confidence_threshold=0.05, enable_preprocessing=False)
    
    candidates = []
    
    # First pass: identify table-like images
    for image_path in all_images:
        table_analysis = analyze_image_for_table_characteristics(image_path)
        
        if table_analysis.get("likely_table", False):
            logger.info(f"📋 Table candidate: {image_path.name}")
            logger.info(f"   Table score: {table_analysis['table_score']}")
            logger.info(f"   Rectangular contours: {table_analysis['rectangular_contours']}")
            candidates.append((image_path, table_analysis))
    
    if not candidates:
        logger.warning("No obvious table candidates found. Testing all images with text...")
        # Fallback: test all images
        candidates = [(img, {"table_score": 0}) for img in all_images[:20]]  # Test first 20
    
    logger.info(f"\nTesting {len(candidates)} candidate images with OCR...")
    
    results = []
    
    for image_path, table_info in candidates:
        logger.info(f"\n{'='*50}")
        logger.info(f"TESTING: {image_path.name}")
        logger.info(f"{'='*50}")
        
        best_results = []
        best_engine = None
        best_angle = 0
        best_score = 0
        
        # Test with both engines
        for engine_name, engine in [("tesseract", tesseract_engine), ("easyocr", easyocr_engine)]:
            try:
                best_angle_engine, ocr_results, orientation_summary = orientation_tester.find_best_orientation(
                    image_path, engine
                )
                
                if len(ocr_results) > len(best_results):
                    best_results = ocr_results
                    best_engine = engine_name
                    best_angle = best_angle_engine
                    best_score = orientation_summary.get('best_score', 0)
                
                logger.info(f"{engine_name}: {len(ocr_results)} regions at {best_angle_engine}°")
                
            except Exception as e:
                logger.error(f"Error with {engine_name}: {e}")
        
        # Analyze results
        if best_results:
            logger.info(f"\nBest: {best_engine} at {best_angle}° ({len(best_results)} regions)")
            
            # Look for patterns and artifacts
            seventy_seven_parts = []
            table_artifacts = []
            all_text = []
            
            for result in best_results:
                text = result.text.strip()
                all_text.append(text)
                
                # Check for 77- patterns
                matches = pattern_matcher.find_matches_universal(text)
                for match in matches:
                    if '77-' in match.matched_text:
                        seventy_seven_parts.append(match.matched_text)
                        logger.info(f"  🎯 FOUND: {match.matched_text}")
                
                # Check for table artifacts
                if text == "1" and result.bounding_box[2] < 30:  # Width < 30 pixels
                    table_artifacts.append(result)
                    logger.info(f"  📏 Table line: '{text}' {result.bounding_box}")
                
                # Check for any "77" occurrences
                if '77' in text:
                    logger.info(f"  📋 Contains 77: '{text}'")
            
            # Score this image
            image_score = (
                len(seventy_seven_parts) * 100 +  # 77- parts are most important
                len([t for t in all_text if '77' in t]) * 50 +  # Any 77 occurrences
                len(table_artifacts) * 10 +  # Table artifacts suggest table structure
                table_info.get('table_score', 0) * 1000  # Visual table characteristics
            )
            
            result_data = {
                "image": image_path.name,
                "engine": best_engine,
                "angle": best_angle,
                "score": image_score,
                "ocr_regions": len(best_results),
                "seventy_seven_parts": seventy_seven_parts,
                "table_artifacts": len(table_artifacts),
                "all_text": all_text,
                "table_analysis": table_info
            }
            
            results.append(result_data)
            
            # Create visualizations for promising candidates
            if image_score > 50 or seventy_seven_parts:
                logger.info(f"Creating visualizations for {image_path.name} (score: {image_score})")
                
                viz_path = visual_validator.create_ocr_visualization(
                    image_path, best_results, 
                    f"{image_path.name} - {best_engine} at {best_angle}° (Score: {image_score})"
                )
                
                corrected_path = orientation_tester.apply_best_orientation(
                    image_path, best_angle,
                    Path(f"debug_output/{image_path.name}_corrected.png")
                )
                
                logger.info(f"  Saved: {viz_path}")
                logger.info(f"  Corrected: {corrected_path}")
        
        else:
            logger.info("No text detected")
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    
    if results:
        logger.info("Top candidates (by score):")
        for i, result in enumerate(results[:10]):  # Top 10
            logger.info(f"{i+1}. {result['image']} (score: {result['score']})")
            logger.info(f"   Engine: {result['engine']} at {result['angle']}°")
            logger.info(f"   OCR regions: {result['ocr_regions']}")
            logger.info(f"   77- parts: {len(result['seventy_seven_parts'])}")
            logger.info(f"   Table artifacts: {result['table_artifacts']}")
            
            if result['seventy_seven_parts']:
                logger.info(f"   🎉 FOUND 77- PARTS: {result['seventy_seven_parts']}")
                for part in result['seventy_seven_parts']:
                    replacement = mappings.get(part, "No mapping")
                    logger.info(f"      {part} -> {replacement}")
            
            if result['all_text']:
                logger.info(f"   Sample text: {result['all_text'][:5]}")
            
            logger.info("")
    
    # Save detailed results
    output_path = Path("debug_output/table_search_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {output_path}")
    
    # Recommendations
    if any(r['seventy_seven_parts'] for r in results):
        logger.info("✅ SUCCESS: Found images with 77- parts!")
    elif any(r['table_artifacts'] > 5 for r in results):
        logger.info("📋 Found images with table structure but no 77- parts detected")
        logger.info("   This suggests the table lines are interfering with OCR")
        logger.info("   Consider image preprocessing to remove table lines")
    else:
        logger.info("❌ No clear table images found")
        logger.info("   The table might be in a different format or embedded differently")

if __name__ == "__main__":
    find_table_image()