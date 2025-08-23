#!/usr/bin/env python3
"""
Debug script to test why specific part numbers are not being detected.
"""

import json
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from utils.pattern_matcher import PatternMatcher
from utils.image_utils.pattern_debugger import PatternDebugger

def load_patterns_and_mappings():
    """Load patterns and mappings from JSON files."""
    with open('patterns.json', 'r') as f:
        patterns = json.load(f)
    
    with open('mapping.json', 'r') as f:
        mappings = json.load(f)
    
    return patterns, mappings

def test_specific_part_numbers():
    """Test the specific part numbers that are not being detected."""
    patterns, mappings = load_patterns_and_mappings()
    
    # Create pattern matcher
    pattern_matcher = PatternMatcher(patterns, mappings, enhanced_mode=True)
    
    # Create debugger
    debugger = PatternDebugger(pattern_matcher)
    
    # Test the specific part numbers
    test_numbers = ['77-141-810054-000', '77-141-810035-000']
    
    print("=== Testing Specific Part Numbers ===")
    print(f"Patterns loaded: {len(patterns)}")
    print(f"Mappings loaded: {len(mappings)}")
    print()
    
    for part_number in test_numbers:
        print(f"Testing: {part_number}")
        print(f"In mappings: {part_number in mappings}")
        if part_number in mappings:
            print(f"Maps to: {mappings[part_number]}")
        
        # Test with various contexts that might appear in the document
        test_contexts = [
            part_number,  # Standalone
            f"RING, ID .674\n{part_number}",  # With description above
            f"O-ring #{part_number}",  # With prefix
            f"Part: {part_number}",  # With label
            f"({part_number})",  # In parentheses
            f"[{part_number}]",  # In brackets
            f"ID .674 {part_number}",  # With ID prefix
            f"O-ring #2-126\n{part_number}",  # Multi-line context
        ]
        
        print(f"\nTesting contexts for {part_number}:")
        for i, context in enumerate(test_contexts):
            matches = pattern_matcher.find_matches_universal(context)
            found_matches = [match.matched_text for match in matches]
            status = "✓ FOUND" if part_number in found_matches else "✗ NOT FOUND"
            print(f"  Context {i+1}: {status}")
            context_display = context.replace('\n', '\\n')
            print(f"    Text: '{context_display}'")
            if found_matches:
                print(f"    Matches: {found_matches}")
            print()
        
        # Debug pattern matching in detail
        print(f"Detailed debug for standalone '{part_number}':")
        debug_info = debugger.debug_pattern_matching(part_number, verbose=False)
        
        if debug_info['final_matches']:
            print("  ✓ Pattern matching successful")
            for match in debug_info['final_matches']:
                print(f"    Match: '{match['matched_text']}' -> Pattern: {match['pattern_name']}")
        else:
            print("  ✗ Pattern matching failed")
            print("  Reasons:")
            for reason in debug_info['no_matches_reasons']:
                print(f"    - {reason['pattern_name']}: {reason['reason']}")
        
        print("-" * 60)

def test_image_context():
    """Test contexts that might appear in images based on the provided image."""
    patterns, mappings = load_patterns_and_mappings()
    pattern_matcher = PatternMatcher(patterns, mappings, enhanced_mode=True)
    
    print("\n=== Testing Image-like Contexts ===")
    
    # Based on the image provided, test contexts that might appear in OCR
    image_contexts = [
        # Context from the image showing the ring part
        "RING, ID .674\n77-141-810054-000",
        "RING ID .674 77-141-810054-000",
        "RING,ID.674 77-141-810054-000",
        
        # Context from the O-ring part
        "O-ring #2-126\n77-141-810035-000",
        "O-ring #2-126 77-141-810035-000",
        "O-ring#2-126 77-141-810035-000",
        
        # Possible OCR variations
        "77-141-810054-000 RING",
        "77-141-810035-000 O-ring",
        
        # With surrounding text that might be OCR'd
        "Chamber外部parts組装\n77-141-810054-000",
        "Vacuum\n77-141-810035-000",
    ]
    
    for context in image_contexts:
        context_display = context.replace('\n', '\\n')
        print(f"Testing image context: '{context_display}'")
        matches = pattern_matcher.find_matches_universal(context)
        found_matches = [match.matched_text for match in matches]
        
        if found_matches:
            print(f"  ✓ Found matches: {found_matches}")
            for match in matches:
                print(f"    '{match.matched_text}' -> '{pattern_matcher.get_replacement(match.matched_text)}'")
        else:
            print(f"  ✗ No matches found")
        print()

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    test_specific_part_numbers()
    test_image_context()