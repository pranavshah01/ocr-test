#!/usr/bin/env python3
"""
Test to isolate image processor and check for conflicts with text processor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
from app.processors.image_processor import ImageProcessor
from app.config import OCRConfig
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s')

def test_image_processor_conflict():
    """Test if image processor is processing the same header content as text processor."""
    
    config = OCRConfig()
    
    # Load patterns and mappings
    with open(config.patterns_path, 'r') as f:
        patterns = json.load(f)
    with open(config.mapping_path, 'r') as f:
        mappings = json.load(f)
    
    # Initialize image processor
    image_processor = ImageProcessor(
        patterns=patterns,
        mappings=mappings,
        mode=config.ocr_mode,
        ocr_engine=config.ocr_engine,
        confidence_min=config.confidence_min,
        gpu=config.gpu
    )
    
    # Load the document
    docx_path = "source_documents/test_file2.docx"
    document = Document(docx_path)
    
    print("=== TESTING IMAGE PROCESSOR FOR CONFLICTS ===")
    
    # Process only image content
    matches = image_processor.process_images(document)
    
    print(f"Image processor found {len(matches)} matches")
    
    # Check if any matches are for our target part number
    header_matches = []
    for match in matches:
        # Check different possible attribute names for the match data
        original_text = ""
        replacement_text = ""
        location = ""
        
        if hasattr(match, 'original_text'):
            original_text = match.original_text
        elif hasattr(match, 'original'):
            original_text = match.original
        
        if hasattr(match, 'replacement_text'):
            replacement_text = match.replacement_text
        elif hasattr(match, 'replacement'):
            replacement_text = match.replacement
            
        if hasattr(match, 'location'):
            location = match.location
        
        if "77-620-1908713-03" in original_text:
            header_matches.append(match)
            print(f"CONFLICT DETECTED: Image processor found header match: '{original_text}' -> '{replacement_text}' at {location}")
    
    if not header_matches:
        print("✅ No conflict detected - Image processor is not processing the header part number")
    else:
        print(f"❌ CONFLICT DETECTED - Image processor found {len(header_matches)} matches for the same header part number")
        print("This explains why the text processor's changes are being overwritten!")
    
    return len(header_matches) > 0

if __name__ == "__main__":
    conflict_detected = test_image_processor_conflict()
    if conflict_detected:
        print("\n🔧 SOLUTION: Need to prevent image processor from overwriting text processor changes")
    else:
        print("\n🤔 No conflict detected with image processor either - need to investigate document saving process")
