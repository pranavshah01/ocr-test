#!/usr/bin/env python3
"""
Debug specific textboxes to see why they're not being processed.
"""

import sys
import os
from pathlib import Path
from docx import Document
import xml.etree.ElementTree as ET
import json

def debug_specific_textboxes():
    """Debug the specific textboxes that should contain our target numbers."""
    
    doc_path = "source_documents/test_file2.docx"
    document = Document(doc_path)
    doc_xml = document._element
    
    # Find all textboxes
    textboxes = []
    for txbx_content in doc_xml.iter():
        if txbx_content.tag.endswith('}txbxContent'):
            textboxes.append(txbx_content)
    
    # Target textboxes based on our previous debug
    target_indices = [58, 72]  # The ones that should contain our numbers
    target_numbers = ['77-141-810054-000', '77-141-810035-000']
    
    print("=== Debugging Specific Textboxes ===")
    
    for idx in target_indices:
        if idx < len(textboxes):
            textbox = textboxes[idx]
            print(f"\n--- Textbox {idx} ---")
            
            # Extract text
            combined_text = ""
            wt_elements = []
            
            for wt_element in textbox.iter():
                if wt_element.tag.endswith('}t'):
                    text_content = wt_element.text or ""
                    combined_text += text_content
                    wt_elements.append(wt_element)
            
            print(f"Text: '{combined_text}'")
            print(f"Contains target: {any(num in combined_text for num in target_numbers)}")
            
            # Check textbox structure
            print(f"Number of w:t elements: {len(wt_elements)}")
            
            # Check for any unusual attributes or structure
            print("Textbox attributes:")
            for attr, value in textbox.attrib.items():
                print(f"  {attr}: {value}")
            
            # Check parent elements
            parent = textbox.getparent()
            if parent is not None:
                print(f"Parent tag: {parent.tag}")
                print("Parent attributes:")
                for attr, value in parent.attrib.items():
                    print(f"  {attr}: {value}")
            
            # Try to get dimensions (simplified version)
            try:
                # Look for dimension information in parent elements
                current = textbox
                dimensions_found = False
                for _ in range(5):  # Check up to 5 levels up
                    if current is None:
                        break
                    
                    # Look for common dimension attributes
                    for attr in current.attrib:
                        if any(dim_word in attr.lower() for dim_word in ['width', 'height', 'cx', 'cy']):
                            print(f"Dimension attribute found: {attr} = {current.attrib[attr]}")
                            dimensions_found = True
                    
                    current = current.getparent()
                
                if not dimensions_found:
                    print("No dimension attributes found")
                    
            except Exception as e:
                print(f"Error checking dimensions: {e}")
            
            # Check if there are any problematic characters or encoding issues
            try:
                combined_text.encode('utf-8')
                print("Text encoding: OK")
            except UnicodeEncodeError as e:
                print(f"Text encoding issue: {e}")
            
            print("-" * 50)
        else:
            print(f"Textbox {idx} not found (only {len(textboxes)} textboxes total)")

if __name__ == "__main__":
    debug_specific_textboxes()