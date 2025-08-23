#!/usr/bin/env python3
"""
Debug script to test "77-" pattern detection on the specific image.
This will help identify why the "77-" parts aren't being processed.
"""

import json
import logging
from pathlib import Path
import sys

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.processors.image_processor import OCREngine
from app.utils.pattern_matcher import create_pattern_matcher
from app.utils.image_utils.pattern_debugger import create_pattern_debugger

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_77_detection():
    """Debug "77-" pattern detection on the document image."""
    
    # Load patterns and mappings
    patterns_path = Path("patterns.json")
    mapping_path = Path("mapping.json")
    
    if not patterns_path.exists():
        logger.error(f"Patterns file not found: {patterns_path}")
        return
    
    if not mapping_path.exists():
        logger.error(f"Mapping file not found: {mapping_path}")
        return
    
    with open(patterns_path) as f:
        patterns = json.load(f)
    
    with open(mapping_path) as f:
        mappings = json.load(f)
    
    logger.info(f"Loaded {len(patterns)} patterns and {len(mappings)} mappings")
    
    # Create pattern matcher and debugger
    pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
    debugger = create_pattern_debugger(pattern_matcher)
    
    # Test pattern matching on known "77-" parts from mapping
    test_cases = [
        "77-110-810096-000",  # Direct from mapping
        ":77-110-810096-000",  # With colon prefix
        "(77-110-810096-000)",  # With parentheses
        "Part 77-110-810096-000 needed",  # With context
        "Heat:77-531-116BLK-245",  # Real-world example
        "22 AWG(77-525-551WH1-000)",  # Another real-world example
    ]
    
    logger.info("=== Testing Pattern Matching ===")
    for test_text in test_cases:
        logger.info(f"\nTesting: '{test_text}'")
        debug_info = debugger.debug_pattern_matching(test_text, verbose=True)
        
        if debug_info['final_matches']:
            logger.info(f"✓ Found {len(debug_info['final_matches'])} matches")
            for match in debug_info['final_matches']:
                logger.info(f"  - '{match['matched_text']}' at {match['start_pos']}-{match['end_pos']}")
        else:
            logger.warning("✗ No matches found")
            for reason in debug_info['no_matches_reasons']:
                logger.warning(f"  - {reason['pattern_name']}: {reason['reason']}")
    
    # Test OCR on actual images if they exist
    image_paths = [
        Path("source_documents/test_file2_unzip/word/media/image1.jpeg"),
        Path("source_documents/test_file2_unzip/word/media/image2.jpeg"),
        Path("source_documents/test_file2_unzip/word/media/image3.png"),
    ]
    
    logger.info("\n=== Testing OCR on Sample Images ===")
    
    # Test different OCR engines
    for engine in ["easyocr", "tesseract", "hybrid"]:
        logger.info(f"\n--- Testing {engine.upper()} OCR Engine ---")
        
        try:
            ocr_engine = OCREngine(
                engine=engine,
                use_gpu=True,
                confidence_threshold=0.3,  # Lower threshold for debugging
                enable_preprocessing=True
            )
            
            for image_path in image_paths:
                if image_path.exists():
                    logger.info(f"\nProcessing: {image_path.name}")
                    
                    try:
                        # Extract text with OCR
                        ocr_results = ocr_engine.extract_text(image_path)
                        logger.info(f"OCR found {len(ocr_results)} text regions")
                        
                        # Check each OCR result for "77-" patterns
                        for i, result in enumerate(ocr_results):
                            logger.info(f"  Region {i}: '{result.text}' (confidence: {result.confidence:.2f})")
                            
                            # Test pattern matching on this OCR text
                            matches = pattern_matcher.find_matches_universal(result.text)
                            if matches:
                                logger.info(f"    ✓ Found {len(matches)} pattern matches:")
                                for match in matches:
                                    logger.info(f"      - '{match.matched_text}' -> replacement available")
                            else:
                                logger.info(f"    ✗ No pattern matches in this text")
                    
                    except Exception as e:
                        logger.error(f"Error processing {image_path.name}: {e}")
                else:
                    logger.warning(f"Image not found: {image_path}")
        
        except Exception as e:
            logger.error(f"Failed to initialize {engine} OCR engine: {e}")
    
    # Run comprehensive pattern tests
    logger.info("\n=== Running Comprehensive Pattern Tests ===")
    test_results = debugger.run_comprehensive_tests()
    
    logger.info(f"Test Results: {test_results['passed_tests']}/{test_results['total_tests']} passed "
               f"({test_results['success_rate']:.1f}% success rate)")
    
    if test_results['failed_tests'] > 0:
        logger.warning("Failed tests:")
        for result in test_results['test_results']:
            if not result['passed']:
                logger.warning(f"  - {result['description']}: '{result['text']}'")
                logger.warning(f"    Expected: {result['expected_matches']}")
                logger.warning(f"    Actual: {result['actual_matches']}")

if __name__ == "__main__":
    debug_77_detection()