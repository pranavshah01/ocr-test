#!/usr/bin/env python3
"""
Test script to verify the enhanced bounding box calculation improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.image_utils.precise_text_replacer import PreciseTextReplacer
from app.utils.image_utils.text_analyzer import TextProperties
from PIL import Image, ImageDraw, ImageFont

def test_bbox_calculation():
    """Test the enhanced bounding box calculation."""
    
    # Create a test replacer
    replacer = PreciseTextReplacer()
    
    # Create test properties
    properties = TextProperties(
        font_size=12,
        font_family="Arial",
        color=(0, 0, 0),
        orientation=0.0
    )
    
    # Test cases
    test_cases = [
        {
            'original': '77-618-0100361',
            'replacement': '4022-618-0100361',
            'bbox': (100, 50, 120, 20),  # x, y, width, height
            'description': 'Two-dash pattern replacement'
        },
        {
            'original': '77-110-817895-000',
            'replacement': '4022-817823A-HG2',
            'bbox': (100, 50, 150, 20),
            'description': 'Three-dash pattern replacement'
        },
        {
            'original': 'Short',
            'replacement': 'Much Longer Replacement Text',
            'bbox': (100, 50, 50, 20),
            'description': 'Short to long replacement'
        }
    ]
    
    print("Testing Enhanced Bounding Box Calculation")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['description']}")
        print(f"Original: '{case['original']}'")
        print(f"Replacement: '{case['replacement']}'")
        print(f"Original bbox: {case['bbox']}")
        
        try:
            # Calculate precise bbox
            precise_bbox = replacer.calculate_precise_replacement_bbox(
                case['original'],
                case['replacement'],
                case['bbox'],
                properties
            )
            
            original_width = case['bbox'][2]
            precise_width = precise_bbox[2]
            coverage_percent = (precise_width / original_width) * 100
            
            print(f"Precise bbox: {precise_bbox}")
            print(f"Width change: {original_width} -> {precise_width} ({coverage_percent:.1f}% coverage)")
            
            # Check if coverage is reasonable
            if coverage_percent >= 80:
                print("✓ Good coverage - should fully wipe original text")
            elif coverage_percent >= 60:
                print("⚠ Moderate coverage - may leave some text remnants")
            else:
                print("✗ Low coverage - likely to leave text remnants")
                
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_bbox_calculation()