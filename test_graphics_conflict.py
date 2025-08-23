#!/usr/bin/env python3
"""
Test to isolate graphics processor and check for conflicts with text processor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
from app.processors.graphics_processor import GraphicsProcessor
from app.config import OCRConfig
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s')

def test_graphics_processor_conflict():
    """Test if graphics processor is processing the same header content as text processor."""
    
    config = OCRConfig()
    
    # Load patterns and mappings
    with open(config.patterns_path, 'r') as f:
        patterns = json.load(f)
    with open(config.mapping_path, 'r') as f:
        mappings = json.load(f)
    
    # Initialize graphics processor
    graphics_processor = GraphicsProcessor(
        patterns=patterns,
        mappings=mappings,
        mode=config.text_mode
    )
    
    # Load the document
    docx_path = "source_documents/test_file2.docx"
    document = Document(docx_path)
    
    print("=== TESTING GRAPHICS PROCESSOR ON HEADERS ===")
    
    # Process only graphics content
    matches = graphics_processor.process_graphics(document)
    
    print(f"Graphics processor found {len(matches)} matches")
    
    # Check if any matches are for our target part number
    header_matches = []
    for match in matches:
        if hasattr(match, 'original_text') and "77-620-1908713-03" in match.original_text:
            header_matches.append(match)
            print(f"CONFLICT DETECTED: Graphics processor found header match: '{match.original_text}' -> '{match.replacement_text}' at {match.location}")
        elif hasattr(match, 'original') and "77-620-1908713-03" in match.original:
            header_matches.append(match)
            print(f"CONFLICT DETECTED: Graphics processor found header match: '{match.original}' -> '{match.replacement}' at {match.location}")
    
    if not header_matches:
        print("✅ No conflict detected - Graphics processor is not processing the header part number")
    else:
        print(f"❌ CONFLICT DETECTED - Graphics processor found {len(header_matches)} matches for the same header part number")
        print("This explains why the text processor's changes are being overwritten!")
    
    return len(header_matches) > 0

if __name__ == "__main__":
    conflict_detected = test_graphics_processor_conflict()
    if conflict_detected:
        print("\n🔧 SOLUTION: Need to prevent graphics processor from overwriting text processor changes")
    else:
        print("\n🤔 No conflict detected - need to investigate other causes")
