#!/usr/bin/env python3
"""
Diagnostic script to debug the pattern-mapping disconnect.
"""

import sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def debug_pattern_mapping():
    """Debug why pattern matches aren't connecting to mappings."""
    print("🔍 Debugging Pattern-Mapping Disconnect")
    print("=" * 50)
    
    try:
        # Import required modules
        from app.text_extractor import find_patterns_across_split_tags
        
        # Load test file
        test_docx = Path("source_documents/testimages_092051.docx")
        mapping_file = Path("mapping.json")
        patterns_file = Path("patterns.json")
        
        if not test_docx.exists():
            print(f"❌ Test file not found: {test_docx}")
            return False
        
        # Load mapping
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        
        # Load patterns
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
            patterns_list = [
                value for key, value in patterns_data.items() 
                if not key.startswith('_') and isinstance(value, str)
            ]
        
        print(f"✅ Loaded {len(mapping_dict)} mappings and {len(patterns_list)} patterns")
        
        # Run pattern matching
        print("\n🔍 Running pattern detection...")
        pattern_results = find_patterns_across_split_tags(test_docx, patterns_list)
        
        print(f"✅ Found {pattern_results['total_matches']} pattern matches")
        
        # Analyze first 10 matches
        if pattern_results['matches']:
            print("\n📋 First 10 Pattern Matches:")
            for i, match in enumerate(pattern_results['matches'][:10]):
                match_text = match.get('match_text', 'NO_TEXT')
                print(f"  {i+1:2d}. '{match_text}'")
                
                # Check if this text exists in mapping
                if match_text in mapping_dict:
                    print(f"      ✅ FOUND in mapping: '{match_text}' → '{mapping_dict[match_text]}'")
                else:
                    print(f"      ❌ NOT FOUND in mapping")
                    
                    # Check for close matches
                    close_matches = [key for key in mapping_dict.keys() if key.lower() == match_text.lower()]
                    if close_matches:
                        print(f"      🔍 Close match (case): {close_matches[0]}")
        
        # Show sample mapping entries
        print(f"\n📋 Sample Mapping Entries:")
        sample_mappings = list(mapping_dict.items())[:5]
        for key, value in sample_mappings:
            print(f"  '{key}' → '{value}'")
        
        # Test specific mapping lookup logic
        print(f"\n🧪 Testing Mapping Lookup Logic:")
        if pattern_results['matches']:
            from app.docx_processor import EnhancedMappingProcessor
            processor = EnhancedMappingProcessor("replace")
            
            for i, match in enumerate(pattern_results['matches'][:5]):
                match_text = match.get('match_text', '')
                result = processor.find_mapping_match(match_text, mapping_dict)
                print(f"  {i+1}. '{match_text}' → {result if result else 'NO MATCH'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_pattern_mapping()
    if success:
        print("\n🎯 Diagnostic completed successfully!")
    else:
        print("\n💥 Diagnostic failed!")
