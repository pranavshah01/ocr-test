#!/usr/bin/env python3
"""
Analyze the detected text to understand why 77- patterns aren't matching.
"""

import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_detected_text():
    """Analyze the OCR results to understand pattern matching issues."""
    
    # Load the results from the previous test
    results_path = Path("debug_output/test_image_results.json")
    if not results_path.exists():
        logger.error("Results file not found. Run test_single_image.py first.")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    detected_text = results['all_detected_text']
    
    logger.info("=== Analyzing Detected Text for 77- Patterns ===")
    logger.info(f"Total text regions: {len(detected_text)}")
    
    # Load mappings to see what we're looking for
    with open("mapping.json") as f:
        mappings = json.load(f)
    
    expected_parts = list(mappings.keys())
    logger.info(f"Expected 77- parts: {len(expected_parts)}")
    
    # Analyze detected text for potential matches
    potential_matches = []
    partial_matches = []
    
    for i, text in enumerate(detected_text):
        text_clean = text.strip()
        
        # Look for anything that might be a 77- part
        if '77' in text_clean or '-' in text_clean:
            logger.info(f"  {i+1}: '{text_clean}' - Contains 77 or dash")
            
            # Check if it's close to any expected part
            for expected in expected_parts:
                if expected in text_clean:
                    potential_matches.append((text_clean, expected, "exact_substring"))
                elif text_clean in expected:
                    potential_matches.append((text_clean, expected, "partial_match"))
                elif len(text_clean) > 5 and any(part in text_clean for part in expected.split('-')):
                    partial_matches.append((text_clean, expected, "fragment_match"))
    
    # Look for patterns that might be OCR errors
    logger.info(f"\n=== Potential OCR Issues ===")
    
    for text in detected_text:
        text_clean = text.strip()
        
        # Common OCR errors for "77-"
        if text_clean.startswith('(0-') or text_clean.startswith('17-') or text_clean.startswith('177-'):
            logger.info(f"Possible OCR error: '{text_clean}' might be '77-...'")
        
        # Look for fragments that might combine to form 77- parts
        if len(text_clean) > 8 and ('-' in text_clean or any(c.isdigit() for c in text_clean)):
            # Check if this could be part of a 77- pattern
            for expected in expected_parts[:5]:  # Check first 5 for brevity
                expected_parts_list = expected.split('-')
                if len(expected_parts_list) > 1:
                    # Check if any part of the expected pattern is in the detected text
                    for part in expected_parts_list[1:]:  # Skip the "77" part
                        if part in text_clean and len(part) > 2:
                            logger.info(f"Fragment match: '{text_clean}' contains '{part}' from '{expected}'")
                            break
    
    # Try to reconstruct potential 77- parts from fragments
    logger.info(f"\n=== Attempting Pattern Reconstruction ===")
    
    # Look for number sequences that might be part numbers
    number_sequences = []
    for text in detected_text:
        text_clean = text.strip()
        if len(text_clean) > 5 and any(c.isdigit() for c in text_clean) and '-' in text_clean:
            number_sequences.append(text_clean)
    
    logger.info(f"Found {len(number_sequences)} potential part number sequences:")
    for seq in number_sequences:
        logger.info(f"  '{seq}'")
        
        # Try to match with expected patterns by adding "77-" prefix
        reconstructed = f"77-{seq.lstrip('(0-').lstrip('17-').lstrip('177-')}"
        if reconstructed in mappings:
            logger.info(f"    ✅ RECONSTRUCTED MATCH: '{seq}' -> '{reconstructed}' -> '{mappings[reconstructed]}'")
        else:
            # Try other common OCR corrections
            for prefix_error in ['(0-', '17-', '177-', '7-', '(7-']:
                if seq.startswith(prefix_error):
                    corrected = f"77-{seq[len(prefix_error):]}"
                    if corrected in mappings:
                        logger.info(f"    ✅ CORRECTED MATCH: '{seq}' -> '{corrected}' -> '{mappings[corrected]}'")
                        break
    
    # Summary
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Potential exact matches: {len(potential_matches)}")
    logger.info(f"Partial matches: {len(partial_matches)}")
    
    if potential_matches:
        logger.info("Exact matches found:")
        for detected, expected, match_type in potential_matches:
            logger.info(f"  '{detected}' matches '{expected}' ({match_type})")
    
    # Recommendations
    logger.info(f"\n=== RECOMMENDATIONS ===")
    logger.info("1. OCR is detecting text but with errors in the '77-' prefix")
    logger.info("2. Common OCR errors: '77-' -> '(0-', '17-', '177-'")
    logger.info("3. Need to implement OCR error correction in pattern matching")
    logger.info("4. Consider preprocessing to improve OCR accuracy on the '77' digits")
    
    # Create enhanced pattern matching suggestions
    logger.info(f"\n=== ENHANCED PATTERN SUGGESTIONS ===")
    logger.info("Add these OCR error patterns to catch common misreads:")
    logger.info("  - Pattern for '(0-' -> '77-'")
    logger.info("  - Pattern for '17-' -> '77-'") 
    logger.info("  - Pattern for '177-' -> '77-'")
    logger.info("  - Pattern for '7-' -> '77-'")

if __name__ == "__main__":
    analyze_detected_text()