#!/usr/bin/env python3
"""
Detailed diagnostic to examine textbox structure in headers and understand why 77-620-1908713-03 is not detected.
"""

import sys
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
from app.processors.graphics_processor import TextboxParser

def setup_logging():
    """Set up detailed logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

def analyze_textbox_structure(docx_file_path: str, target_text: str = "77-620-1908713-03"):
    """
    Detailed analysis of textbox structure to find why target text is not detected.
    """
    logger = setup_logging()
    
    try:
        logger.info(f"Analyzing textbox structure in: {docx_file_path}")
        logger.info(f"Looking for: {target_text}")
        
        # Load the document
        document = Document(docx_file_path)
        logger.info("Document loaded successfully")
        
        # Test 1: Check main document textboxes using current method
        logger.info("\n=== TEST 1: Main Document Textboxes ===")
        main_textboxes = TextboxParser.find_textboxes(document)
        logger.info(f"Found {len(main_textboxes)} textboxes in main document")
        
        for i, textbox in enumerate(main_textboxes):
            try:
                combined_text, wt_elements = TextboxParser.extract_text_from_textbox(textbox)
                logger.info(f"Textbox {i}: '{combined_text.strip()}'")
                if target_text in combined_text:
                    logger.info(f"🎯 FOUND target text in main textbox {i}!")
            except Exception as e:
                logger.error(f"Error processing main textbox {i}: {e}")
        
        # Test 2: Check headers manually
        logger.info("\n=== TEST 2: Header Analysis ===")
        for section_idx, section in enumerate(document.sections):
            if section.header:
                logger.info(f"Analyzing header in section {section_idx}...")
                analyze_header_xml(section.header, f"section_{section_idx}_header", target_text)
        
        # Test 3: Check if textbox detection works in headers
        logger.info("\n=== TEST 3: Header Textbox Detection ===")
        for section_idx, section in enumerate(document.sections):
            if section.header:
                logger.info(f"Testing textbox detection in header section {section_idx}...")
                test_header_textbox_detection(section.header, f"section_{section_idx}_header", target_text)
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")

def analyze_header_xml(header, location_name: str, target_text: str):
    """Analyze the raw XML structure of a header."""
    logger = logging.getLogger(__name__)
    
    try:
        header_xml = header._element
        xml_string = ET.tostring(header_xml, encoding='unicode')
        
        logger.info(f"Header XML length: {len(xml_string)} characters")
        
        # Check if target text exists in XML
        if target_text in xml_string:
            logger.info(f"🎯 Target text '{target_text}' FOUND in {location_name} XML!")
            
            # Find the context around the target text
            start_pos = xml_string.find(target_text)
            context_start = max(0, start_pos - 100)
            context_end = min(len(xml_string), start_pos + len(target_text) + 100)
            context = xml_string[context_start:context_end]
            
            logger.info(f"Context around target text:")
            logger.info(f"'{context}'")
            
        else:
            logger.info(f"Target text '{target_text}' NOT found in {location_name} XML")
        
        # Count different types of elements
        element_counts = {}
        for element in header_xml.iter():
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            element_counts[tag_name] = element_counts.get(tag_name, 0) + 1
        
        logger.info(f"Element counts in {location_name}:")
        for tag, count in sorted(element_counts.items()):
            logger.info(f"  {tag}: {count}")
            
        # Look specifically for textbox-related elements
        textbox_elements = []
        for element in header_xml.iter():
            if ('textbox' in element.tag.lower() or 
                element.tag.endswith('}txbxContent') or
                element.tag.endswith('}txbx') or
                'pict' in element.tag.lower() or
                'shape' in element.tag.lower()):
                textbox_elements.append(element.tag)
        
        if textbox_elements:
            logger.info(f"Textbox-related elements found: {textbox_elements}")
        else:
            logger.info("No textbox-related elements found")
            
    except Exception as e:
        logger.error(f"Error analyzing header XML: {e}")

def test_header_textbox_detection(header, location_name: str, target_text: str):
    """Test textbox detection specifically in a header."""
    logger = logging.getLogger(__name__)
    
    try:
        header_xml = header._element
        
        # Method 1: Direct iteration (current approach)
        logger.info("Method 1: Direct iteration")
        textbox_count = 0
        for element in header_xml.iter():
            if (element.tag.endswith('}txbxContent') or 
                'textbox' in element.tag.lower() or
                element.tag.endswith('}txbx')):
                
                textbox_count += 1
                logger.info(f"Found textbox element: {element.tag}")
                
                # Try to extract text
                try:
                    combined_text, wt_elements = TextboxParser.extract_text_from_textbox(element)
                    logger.info(f"  Text: '{combined_text.strip()}'")
                    if target_text in combined_text:
                        logger.info(f"  🎯 FOUND target text!")
                except Exception as e:
                    logger.error(f"  Error extracting text: {e}")
        
        logger.info(f"Method 1 found {textbox_count} textboxes")
        
        # Method 2: Look for any element containing target text
        logger.info("Method 2: Search all elements for target text")
        found_elements = []
        for element in header_xml.iter():
            element_text = element.text or ""
            if target_text in element_text:
                found_elements.append((element.tag, element_text))
                logger.info(f"Found target text in element {element.tag}: '{element_text}'")
        
        if not found_elements:
            logger.info("Target text not found in any individual element text")
            
            # Method 3: Check if text spans multiple elements
            logger.info("Method 3: Checking if text spans multiple elements")
            all_text = ""
            for element in header_xml.iter():
                if element.text:
                    all_text += element.text + " "
            
            if target_text in all_text:
                logger.info(f"🎯 Target text found spanning multiple elements!")
                logger.info(f"Combined text: '{all_text.strip()}'")
            else:
                logger.info("Target text not found even when combining all element text")
        
    except Exception as e:
        logger.error(f"Error testing header textbox detection: {e}")

def main():
    """Main diagnostic function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_textbox_structure.py <path_to_docx_file> [target_text]")
        print("Example: python debug_textbox_structure.py source_documents/test_file2.docx '77-620-1908713-03'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    target_text = sys.argv[2] if len(sys.argv) > 2 else "77-620-1908713-03"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print("=== Detailed Textbox Structure Analysis ===")
    analyze_textbox_structure(file_path, target_text)

if __name__ == "__main__":
    main()
