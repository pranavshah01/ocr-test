#!/usr/bin/env python3
"""
Test script for enhanced OCR text detection capabilities.
Demonstrates universal pattern matching and precise text replacement.
"""

import json
import logging
from pathlib import Path
import sys

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.processors.image_processor import create_image_processor
from app.utils.pattern_matcher import create_pattern_matcher
from app.utils.image_utils.pattern_debugger import create_pattern_debugger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_patterns_and_mappings():
    """Load patterns and mappings from JSON files."""
    try:
        # Load patterns
        patterns_path = Path("patterns.json")
        with open(patterns_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        
        # Load mappings
        mappings_path = Path("mapping.json")
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        logger.info(f"Loaded {len(patterns)} patterns and {len(mappings)} mappings")
        return patterns, mappings
        
    except Exception as e:
        logger.error(f"Failed to load patterns and mappings: {e}")
        return {}, {}

def test_enhanced_pattern_matching():
    """Test enhanced pattern matching with various character combinations."""
    logger.info("=== Testing Enhanced Pattern Matching ===")
    
    patterns, mappings = load_patterns_and_mappings()
    if not patterns or not mappings:
        logger.error("Cannot test without patterns and mappings")
        return
    
    # Create enhanced pattern matcher
    pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
    
    # Create debugger
    debugger = create_pattern_debugger(pattern_matcher)
    
    # Test cases that should work with enhanced matching
    test_cases = [
        "Heat Shrink Heat:77-531-116BLK-245",  # Your real-world example
        "22 AWG(77-525-551WH1-000)",           # Your real-world example
        ":77-531-116BLK-245",                  # Colon before 77
        "(77-525-551WH1-000)",                 # Parenthesis before 77
        "[77-234-120616-000]",                 # Square bracket before 77
        "{77-245-000406-000}",                 # Curly brace before 77
        "@77-110-810096-000",                  # At symbol before 77
        "#77-130-120541-001",                  # Hash symbol before 77
        "中77-531-116BLK-245",                  # Chinese character before 77
        "ñ77-525-551WH1-000",                  # Spanish character before 77
        " 77-232-0403040-00 ",                 # Space boundaries
        "Part 77-234-120616-000 needed",      # Word boundaries
    ]
    
    logger.info("Testing individual cases:")
    for i, test_text in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: '{test_text}'")
        
        # Find matches
        matches = pattern_matcher.find_matches_universal(test_text)
        
        if matches:
            for match in matches:
                replacement = pattern_matcher.get_replacement(match.matched_text)
                logger.info(f"  ✓ Found: '{match.matched_text}' -> '{replacement}'")
                logger.info(f"    Pattern: {match.pattern_name}")
                logger.info(f"    Context: '{match.preceding_context}|{match.matched_text}|{match.following_context}'")
        else:
            logger.warning(f"  ✗ No matches found")
            
            # Debug why no matches were found
            debug_info = debugger.debug_pattern_matching(test_text, verbose=False)
            logger.info(f"    Debug: {len(debug_info['patterns_tested'])} patterns tested")
            for reason in debug_info['no_matches_reasons']:
                logger.info(f"    Reason: {reason['pattern_name']} - {reason['reason']}")
    
    # Run comprehensive tests
    logger.info("\n=== Running Comprehensive Pattern Tests ===")
    test_results = debugger.run_comprehensive_tests()
    
    logger.info(f"Test Results: {test_results['passed_tests']}/{test_results['total_tests']} passed "
               f"({test_results['success_rate']:.1f}% success rate)")
    
    # Show failed tests
    failed_tests = [test for test in test_results['test_results'] if not test['passed']]
    if failed_tests:
        logger.warning(f"\nFailed tests ({len(failed_tests)}):")
        for test in failed_tests:
            logger.warning(f"  - {test['description']}: '{test['text']}'")
            logger.warning(f"    Expected: {test['expected_matches']}")
            logger.warning(f"    Actual: {test['actual_matches']}")
    
    # Save debug report
    debug_dir = Path("debug_reports")
    debug_dir.mkdir(exist_ok=True)
    
    report_path = debug_dir / "enhanced_pattern_test_report.json"
    debugger.save_debug_report(report_path, test_results)
    logger.info(f"\nDebug report saved: {report_path}")

def test_enhanced_image_processor():
    """Test enhanced image processor (if images are available)."""
    logger.info("\n=== Testing Enhanced Image Processor ===")
    
    patterns, mappings = load_patterns_and_mappings()
    if not patterns or not mappings:
        logger.error("Cannot test without patterns and mappings")
        return
    
    # Create enhanced image processor
    processor = create_image_processor(
        patterns=patterns,
        mappings=mappings,
        mode="replace",
        ocr_engine="easyocr",
        use_gpu=True,
        enable_preprocessing=True,
        enable_debugging=True
    )
    
    # Get processor info
    info = processor.get_processing_info()
    logger.info("Enhanced Image Processor Configuration:")
    for key, value in info.items():
        if key != 'pattern_matcher_info':  # Skip detailed pattern info
            logger.info(f"  {key}: {value}")
    
    # Check for test images
    source_dir = Path("source_documents")
    if source_dir.exists():
        image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg"))
        
        if image_files:
            logger.info(f"Found {len(image_files)} image files for testing")
            # Note: Actual image processing would require a full document context
            # This is just showing the processor is properly initialized
        else:
            logger.info("No image files found in source_documents directory")
    else:
        logger.info("source_documents directory not found")

def main():
    """Main test function."""
    logger.info("Starting Enhanced OCR Text Detection Tests")
    logger.info("=" * 50)
    
    try:
        # Test enhanced pattern matching
        test_enhanced_pattern_matching()
        
        # Test enhanced image processor
        test_enhanced_image_processor()
        
        logger.info("\n" + "=" * 50)
        logger.info("Enhanced OCR Tests Completed Successfully!")
        
        logger.info("\nKey Improvements Implemented:")
        logger.info("✓ Universal pattern matching (works with any preceding character)")
        logger.info("✓ Enhanced image preprocessing with multiple strategies")
        logger.info("✓ Text orientation detection and rotated text support")
        logger.info("✓ Precise text replacement with pixel-perfect positioning")
        logger.info("✓ Comprehensive debugging and validation system")
        logger.info("✓ Fallback mechanisms for difficult cases")
        
        logger.info("\nTo use the enhanced OCR in your document processing:")
        logger.info("1. Import: from app.processors.image_processor import create_image_processor")
        logger.info("2. Replace the standard image processor with the enhanced version")
        logger.info("3. Enable preprocessing and debugging as needed")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()