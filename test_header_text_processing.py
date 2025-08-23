#!/usr/bin/env python3
"""
Test script to specifically test header text processing.
"""

import sys
import logging
from pathlib import Path
import json

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
from app.processors.text_processor import TextProcessor

def setup_logging():
    """Set up detailed logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

def test_header_text_processing(docx_file_path: str):
    """
    Test header text processing specifically.
    """
    logger = setup_logging()
    
    try:
        logger.info(f"Testing header text processing on: {docx_file_path}")
        
        # Load the document
        document = Document(docx_file_path)
        logger.info("Document loaded successfully")
        
        # Load patterns and mappings
        with open("patterns.json", "r") as f:
            patterns = json.load(f)
        with open("mapping.json", "r") as f:
            mappings = json.load(f)
        
        logger.info(f"Loaded {len(patterns)} patterns and {len(mappings)} mappings")
        
        # Create text processor
        text_processor = TextProcessor(patterns, mappings, "append")
        
        # Test 1: Process entire document
        logger.info("\n=== TEST 1: Full Document Processing ===")
        all_matches = text_processor.process_document_text(document)
        logger.info(f"Full document processing found {len(all_matches)} matches")
        
        for match in all_matches:
            logger.info(f"Match: '{match.original_text}' -> '{match.replacement_text}' at {match.location}")
        
        # Test 2: Process headers only
        logger.info("\n=== TEST 2: Headers Only Processing ===")
        header_matches = text_processor._process_headers_footers(document)
        logger.info(f"Header-only processing found {len(header_matches)} matches")
        
        for match in header_matches:
            logger.info(f"Header match: '{match.original_text}' -> '{match.replacement_text}' at {match.location}")
        
        # Test 3: Manual header paragraph inspection
        logger.info("\n=== TEST 3: Manual Header Inspection ===")
        for section_idx, section in enumerate(document.sections):
            if section.header:
                logger.info(f"Inspecting header section {section_idx}...")
                
                for para_idx, paragraph in enumerate(section.header.paragraphs):
                    para_text = paragraph.text
                    logger.info(f"  Paragraph {para_idx}: '{para_text}'")
                    
                    # Check if target text is in this paragraph
                    if "77-620-1908713-03" in para_text:
                        logger.info(f"  🎯 FOUND target text in paragraph {para_idx}!")
                        
                        # Test pattern matching on this specific text
                        from app.utils.pattern_matcher import create_pattern_matcher
                        pattern_matcher = create_pattern_matcher(patterns, mappings)
                        matches = pattern_matcher.find_matches(para_text)
                        
                        logger.info(f"  Pattern matches in this paragraph: {len(matches)}")
                        for match in matches:
                            logger.info(f"    Match: '{match['matched_text']}' -> '{match['replacement_text']}'")
        
        return len(all_matches), len(header_matches)
        
    except Exception as e:
        logger.error(f"Error testing header text processing: {e}")
        return 0, 0

def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_header_text_processing.py <path_to_docx_file>")
        print("Example: python test_header_text_processing.py source_documents/test_file2.docx")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print("=== Header Text Processing Test ===")
    total_matches, header_matches = test_header_text_processing(file_path)
    
    print(f"\n=== Results ===")
    print(f"Total matches: {total_matches}")
    print(f"Header matches: {header_matches}")
    
    if header_matches > 0:
        print("✅ Header text processing is working!")
    else:
        print("❌ Header text processing found no matches")
        print("This suggests an issue with the header text processing logic")

if __name__ == "__main__":
    main()
