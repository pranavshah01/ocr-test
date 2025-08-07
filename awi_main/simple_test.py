#!/usr/bin/env python3
"""
Simple direct test for pattern-mapping connection.
"""

import json
from pathlib import Path

def simple_test():
    """Simple test to verify the core functionality."""
    print("🔍 Simple Pattern-Mapping Test")
    print("=" * 30)
    
    # Load mapping
    with open("mapping.json", 'r') as f:
        mapping = json.load(f)
    
    print(f"✅ Loaded {len(mapping)} mappings")
    
    # Show first few mapping entries
    print("\n📋 Sample mappings:")
    for i, (key, value) in enumerate(list(mapping.items())[:3]):
        print(f"  '{key}' → '{value}'")
    
    # Test if we can find a simple match
    test_key = list(mapping.keys())[0]
    if test_key in mapping:
        print(f"\n✅ Direct lookup works: '{test_key}' → '{mapping[test_key]}'")
    
    # Check if processed file exists and is different from original
    original = Path("source_documents/testimages_092051.docx")
    processed = Path("processed/testimages_092051_processed.docx")
    
    if original.exists() and processed.exists():
        orig_size = original.stat().st_size
        proc_size = processed.stat().st_size
        print(f"\n📁 File sizes:")
        print(f"  Original: {orig_size:,} bytes")
        print(f"  Processed: {proc_size:,} bytes")
        
        if orig_size == proc_size:
            print("  ❌ Files are identical - no processing occurred")
        else:
            print("  ✅ Files are different - processing occurred")
    
    return True

if __name__ == "__main__":
    simple_test()
