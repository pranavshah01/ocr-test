#!/usr/bin/env python3
"""
Simple test to verify pattern-mapping connection.
"""

import sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_pattern_mapping():
    """Test if patterns connect to mappings."""
    print("🔍 Testing Pattern-Mapping Connection")
    print("=" * 40)
    
    try:
        # Load mapping
        with open("mapping.json", 'r') as f:
            mapping_dict = json.load(f)
        
        # Load patterns  
        with open("patterns.json", 'r') as f:
            patterns_data = json.load(f)
            patterns_list = [v for k, v in patterns_data.items() if not k.startswith('_')]
        
        print(f"✅ Loaded {len(mapping_dict)} mappings, {len(patterns_list)} patterns")
        
        # Test pattern matching
        from app.text_extractor import find_patterns_across_split_tags
        test_docx = Path("source_documents/testimages_092051.docx")
        
        if not test_docx.exists():
            print("❌ Test file not found")
            return False
        
        pattern_results = find_patterns_across_split_tags(test_docx, patterns_list)
        print(f"✅ Found {pattern_results['total_matches']} pattern matches")
        
        # Check first few matches
        matches_found = 0
        for i, match in enumerate(pattern_results['matches'][:10]):
            match_text = match.get('match_text', '')
            print(f"  {i+1}. Match: '{match_text}'")
            
            if match_text in mapping_dict:
                print(f"     ✅ Found in mapping: '{mapping_dict[match_text]}'")
                matches_found += 1
            else:
                print(f"     ❌ Not found in mapping")
        
        print(f"\n📊 Results: {matches_found}/{min(10, len(pattern_results['matches']))} matches found in mapping")
        
        # Show sample mapping keys
        print(f"\n📋 Sample mapping keys:")
        for i, key in enumerate(list(mapping_dict.keys())[:5]):
            print(f"  {i+1}. '{key}'")
        
        return matches_found > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pattern_mapping()
    print(f"\n{'🎉 SUCCESS' if success else '💥 FAILED'}: Pattern-mapping test")
