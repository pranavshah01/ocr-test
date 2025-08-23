#!/usr/bin/env python3
"""
Diagnostic script to examine header structure and find missing part numbers.
"""

import sys
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
import xml.etree.ElementTree as ET

def setup_logging():
    """Set up logging for the diagnostic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

def analyze_header_structure(docx_file_path: str, target_text: str = "77-620-1908713-03"):
    """
    Analyze the header structure to find where the target text is located.
    
    Args:
        docx_file_path: Path to the DOCX file
        target_text: Text to search for
    """
    logger = setup_logging()
    
    try:
        logger.info(f"Analyzing header structure in: {docx_file_path}")
        logger.info(f"Looking for: {target_text}")
        
        # Load the document
        document = Document(docx_file_path)
        logger.info("Document loaded successfully")
        
        found_locations = []
        
        # Check each section's headers
        for section_idx, section in enumerate(document.sections):
            logger.info(f"\n--- Section {section_idx} ---")
            
            # Check primary header
            if section.header:
                logger.info("Checking primary header...")
                locations = check_header_content(section.header, f"section_{section_idx}_header", target_text)
                found_locations.extend(locations)
            
            # Check first page header
            if hasattr(section, 'first_page_header') and section.first_page_header:
                logger.info("Checking first page header...")
                locations = check_header_content(section.first_page_header, f"section_{section_idx}_first_page_header", target_text)
                found_locations.extend(locations)
            
            # Check even page header
            if hasattr(section, 'even_page_header') and section.even_page_header:
                logger.info("Checking even page header...")
                locations = check_header_content(section.even_page_header, f"section_{section_idx}_even_page_header", target_text)
                found_locations.extend(locations)
        
        # Summary
        logger.info(f"\n=== SUMMARY ===")
        if found_locations:
            logger.info(f"✅ Found '{target_text}' in {len(found_locations)} locations:")
            for location in found_locations:
                logger.info(f"  - {location}")
        else:
            logger.warning(f"❌ '{target_text}' not found in any headers")
            logger.info("This could mean:")
            logger.info("  1. The text is in the document body, not headers")
            logger.info("  2. The text is in an image within the header")
            logger.info("  3. The text is in a textbox within the header")
            logger.info("  4. The text has different formatting/spacing")
        
        return found_locations
        
    except Exception as e:
        logger.error(f"Error analyzing header structure: {e}")
        return []

def check_header_content(header, location_name: str, target_text: str):
    """
    Check header content for target text in various formats.
    
    Args:
        header: Header object to check
        location_name: Name of the location for logging
        target_text: Text to search for
        
    Returns:
        List of locations where text was found
    """
    logger = logging.getLogger(__name__)
    found_locations = []
    
    try:
        # Check paragraph text
        paragraph_text = ""
        for para_idx, paragraph in enumerate(header.paragraphs):
            para_text = paragraph.text
            paragraph_text += para_text + " "
            
            if target_text in para_text:
                location = f"{location_name}_paragraph_{para_idx}"
                logger.info(f"🎯 FOUND in paragraph text: {location}")
                logger.info(f"   Full paragraph: '{para_text}'")
                found_locations.append(location)
        
        # Check if target text spans multiple paragraphs
        if target_text in paragraph_text:
            logger.info(f"🎯 FOUND spanning paragraphs in {location_name}")
            logger.info(f"   Combined text: '{paragraph_text.strip()}'")
            found_locations.append(f"{location_name}_combined_paragraphs")
        
        # Check XML structure for textboxes and other elements
        try:
            header_xml = header._element
            xml_text = ET.tostring(header_xml, encoding='unicode')
            
            if target_text in xml_text:
                logger.info(f"🎯 FOUND in XML structure: {location_name}")
                
                # Look for textboxes
                for element in header_xml.iter():
                    if ('textbox' in element.tag.lower() or 
                        element.tag.endswith('}txbxContent') or
                        element.tag.endswith('}txbx')):
                        logger.info(f"   Found textbox element: {element.tag}")
                        
                        # Check if textbox contains target text
                        textbox_text = ET.tostring(element, encoding='unicode')
                        if target_text in textbox_text:
                            logger.info(f"   🎯 Target text found in textbox!")
                            found_locations.append(f"{location_name}_textbox")
                
                # Look for images
                for element in header_xml.iter():
                    if ('image' in element.tag.lower() or 
                        'pic' in element.tag.lower() or
                        element.tag.endswith('}pic')):
                        logger.info(f"   Found image element: {element.tag}")
                        found_locations.append(f"{location_name}_image_element")
        
        except Exception as xml_error:
            logger.debug(f"Error checking XML structure: {xml_error}")
    
    except Exception as e:
        logger.error(f"Error checking header content in {location_name}: {e}")
    
    return found_locations

def main():
    """Main diagnostic function."""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_header.py <path_to_docx_file> [target_text]")
        print("Example: python diagnose_header.py source_documents/test_file2.docx '77-620-1908713-03'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    target_text = sys.argv[2] if len(sys.argv) > 2 else "77-620-1908713-03"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print("=== Header Structure Diagnostic ===")
    locations = analyze_header_structure(file_path, target_text)
    
    if locations:
        print(f"\n✅ Found '{target_text}' in headers!")
        print("This suggests the text processor should have detected it.")
        print("Check if section processing is enabled in your configuration.")
    else:
        print(f"\n❌ '{target_text}' not found in headers.")
        print("The text might be:")
        print("1. In the document body instead of headers")
        print("2. In an image that requires OCR processing")
        print("3. In a textbox that requires graphics processing")

if __name__ == "__main__":
    main()
