#!/usr/bin/env python3
"""
Script to fix all the remaining method calls in text_processor.py to include processed_patterns parameter.
"""

import re

def fix_text_processor():
    """Fix all method calls to include processed_patterns parameter."""
    
    with open('awi_main/app/processors/text_processor.py', 'r') as f:
        content = f.read()
    
    # Fix all _process_paragraphs calls that don't have processed_patterns
    content = re.sub(
        r'self\._process_paragraphs\(\s*([^,]+),\s*([^)]+)\s*\)',
        r'self._process_paragraphs(\1, \2, processed_patterns)',
        content
    )
    
    # Fix all _process_tables calls that don't have processed_patterns
    content = re.sub(
        r'self\._process_tables\(\s*([^,]+),\s*([^)]+)\s*\)',
        r'self._process_tables(\1, \2, processed_patterns)',
        content
    )
    
    # Fix cases where we already added processed_patterns but it got duplicated
    content = re.sub(
        r', processed_patterns, processed_patterns',
        r', processed_patterns',
        content
    )
    
    with open('awi_main/app/processors/text_processor.py', 'w') as f:
        f.write(content)
    
    print("Fixed text processor method calls")

if __name__ == "__main__":
    fix_text_processor()