#!/usr/bin/env python3
"""
Validation test to confirm the fixes work and replacements are applied.
"""

import sys
import os
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.docx_processor import process_docx

def main():
    print("🔍 Validating OCR DOCX Text Replacement Fixes...")
    
    # Test file paths
    source_file = Path("source_documents/testimages_092051.docx")
    
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        return False
    
    print(f"✅ Source file found: {source_file}")
    
    # Load mapping to see what we're working with
    try:
        with open("mapping.json", 'r') as f:
            mapping = json.load(f)
        print(f"✅ Loaded {len(mapping)} mappings")
        
        # Show first few mappings
        print("📋 Sample mappings:")
        for i, (key, value) in enumerate(list(mapping.items())[:3]):
            print(f"   '{key}' → '{value}'")
        
    except Exception as e:
        print(f"❌ Error loading mapping: {e}")
        return False
    
    # Test processing
    try:
        print("\n🚀 Starting processing test...")
        
        result = process_docx(
            docx_path=source_file,
            text_mode="replace",
            ocr_mode="replace",
            confidence_threshold=0.4
        )
        
        print(f"✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   - Success: {result.get('success', False)}")
        print(f"   - Total matches: {result.get('total_matches', 0)}")
        print(f"   - Text replacements: {result.get('body_replacements', 0)}")
        print(f"   - OCR replacements: {result.get('ocr_replacements', 0)}")
        print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Check if output file was created
        output_path = result.get('output_path')
        if output_path and Path(output_path).exists():
            print(f"✅ Output file created: {output_path}")
            
            # Check file size difference
            original_size = source_file.stat().st_size
            processed_size = Path(output_path).stat().st_size
            size_diff = processed_size - original_size
            
            print(f"📏 File sizes:")
            print(f"   - Original: {original_size:,} bytes")
            print(f"   - Processed: {processed_size:,} bytes")
            print(f"   - Difference: {size_diff:+,} bytes")
            
            if abs(size_diff) > 100:  # Significant change
                print("✅ File size changed significantly - likely has replacements!")
            else:
                print("⚠️  File size barely changed - may not have replacements")
                
        else:
            print("❌ No output file created")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
