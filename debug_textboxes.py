#!/usr/bin/env python3
"""
Debug script to check what textboxes are being found and their content.
"""

import sys
import os
from pathlib import Path
from docx import Document
import xml.etree.ElementTree as ET

def find_textboxes_debug(doc_path):
    """Debug textbox finding and content extraction."""
    print(f"=== Debugging Textboxes in {doc_path} ===")
    
    try:
        document = Document(doc_path)
        doc_xml = document._element
        
        textboxes = []
        
        # Find all textbox elements
        for txbx_content in doc_xml.iter():
            if txbx_content.tag.endswith('}txbxContent'):
                textboxes.append(('w:txbxContent', txbx_content))
        
        # Also look for VML textboxes
        for element in doc_xml.iter():
            if 'textbox' in element.tag.lower():
                textboxes.append(('VML textbox', element))
        
        # Look for drawing elements that might contain textboxes
        for drawing in doc_xml.iter():
            if drawing.tag.endswith('}drawing'):
                for child in drawing.iter():
                    if 'textbox' in child.tag.lower() or child.tag.endswith('}txbxContent'):
                        found = False
                        for _, existing in textboxes:
                            if existing == child:
                                found = True
                                break
                        if not found:
                            textboxes.append(('drawing textbox', child))
        
        print(f"Found {len(textboxes)} textboxes")
        
        # Extract text from each textbox
        target_numbers = ['77-141-810054-000', '77-141-810035-000']
        
        for i, (textbox_type, textbox_element) in enumerate(textboxes):
            print(f"\n--- Textbox {i} ({textbox_type}) ---")
            
            # Extract text
            combined_text = ""
            wt_elements = []
            
            for wt_element in textbox_element.iter():
                if wt_element.tag.endswith('}t'):
                    text_content = wt_element.text or ""
                    combined_text += text_content
                    wt_elements.append(wt_element)
            
            # Also check hyperlinks
            for hyperlink in textbox_element.iter():
                if hyperlink.tag.endswith('}hyperlink'):
                    for wt_element in hyperlink.iter():
                        if wt_element.tag.endswith('}t'):
                            text_content = wt_element.text or ""
                            if text_content not in combined_text:  # Avoid duplicates
                                combined_text += text_content
                                wt_elements.append(wt_element)
            
            print(f"Text content: '{combined_text}'")
            print(f"Text elements found: {len(wt_elements)}")
            
            # Check if our target numbers are in this textbox
            for target in target_numbers:
                if target in combined_text:
                    print(f"🎯 FOUND TARGET: {target}")
                    
                    # Show surrounding context
                    start_idx = combined_text.find(target)
                    context_start = max(0, start_idx - 20)
                    context_end = min(len(combined_text), start_idx + len(target) + 20)
                    context = combined_text[context_start:context_end]
                    print(f"Context: '...{context}...'")
            
            # Show first few characters if text is long
            if len(combined_text) > 100:
                print(f"Preview: '{combined_text[:100]}...'")
            
            # Check if this textbox contains any part numbers at all
            import re
            part_pattern = r'77-\d{3}-[A-Za-z0-9]+-\d{2,3}'
            matches = re.findall(part_pattern, combined_text)
            if matches:
                print(f"Part numbers found: {matches}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    doc_path = "source_documents/test_file2.docx"
    find_textboxes_debug(doc_path)