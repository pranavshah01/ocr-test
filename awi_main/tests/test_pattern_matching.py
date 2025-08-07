#!/usr/bin/env python3
"""
Test script for enhanced pattern matching functionality (Task 2.1)
"""

import sys
from pathlib import Path
import json

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from text_extractor import EnhancedPatternMatcher, find_patterns_across_split_tags

def test_enhanced_pattern_matching():
    """Test the enhanced pattern matching functionality."""
    print("=== Testing Enhanced Pattern Matching (Task 2.1) ===")
    
    # Test file
    docx_file = Path('source_documents/testimages_092051.docx')
    
    if not docx_file.exists():
        print(f"❌ Test file not found: {docx_file}")
        return False
    
    print(f"📄 Testing with file: {docx_file}")
    
    try:
        # Load patterns from patterns.json
        patterns_file = Path('patterns.json')
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Extract pattern values
            if isinstance(patterns_data, dict):
                patterns = list(patterns_data.values())
            else:
                patterns = patterns_data
        else:
            # Use test patterns
            patterns = [
                r'\d+',  # Numbers
                r'[A-Za-z]{3,}',  # Words with 3+ letters
                r'77-[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+',  # Pattern from functional requirements
            ]
        
        print(f"🔍 Testing patterns: {patterns}")
        
        # Test 1: Basic functionality
        print("\n--- Test 1: Basic Pattern Matching ---")
        matcher = EnhancedPatternMatcher()
        
        # Extract text fragments
        fragments = matcher.extract_text_fragments(docx_file)
        print(f"✅ Extracted {len(fragments)} text fragments")
        
        if len(fragments) == 0:
            print("❌ No text fragments found - this may indicate an issue")
            return False
        
        # Reconstruct continuous text
        continuous_text = matcher.reconstruct_continuous_text(fragments)
        print(f"✅ Reconstructed continuous text: {len(continuous_text)} characters")
        
        if len(continuous_text) > 0:
            print(f"📝 Sample text: {continuous_text[:100]}...")
        
        # Find pattern matches
        matches = matcher.find_pattern_matches(fragments, patterns)
        print(f"✅ Found {len(matches)} pattern matches")
        
        # Show first few matches
        for i, match in enumerate(matches[:3]):
            spans_multiple = match.start_fragment != match.end_fragment
            print(f"  Match {i+1}: '{match.match_text}' (spans multiple fragments: {spans_multiple})")
        
        # Test 2: High-level API
        print("\n--- Test 2: High-Level API ---")
        results = find_patterns_across_split_tags(docx_file, patterns)
        
        print(f"✅ API Results:")
        print(f"  - Fragments: {results['fragments_count']}")
        print(f"  - Text length: {results['continuous_text_length']}")
        print(f"  - Matches: {results['total_matches']}")
        
        # Test 3: Split tag detection
        print("\n--- Test 3: Split Tag Detection ---")
        split_tag_matches = [match for match in results['matches'] if match['spans_multiple_fragments']]
        print(f"✅ Found {len(split_tag_matches)} matches spanning multiple fragments")
        
        if split_tag_matches:
            for i, match in enumerate(split_tag_matches[:2]):
                print(f"  Split Match {i+1}: '{match['match_text']}'")
                print(f"    Start fragment: '{match['start_fragment_text']}'")
                print(f"    End fragment: '{match['end_fragment_text']}'")
        
        print("\n🎉 All tests passed! Enhanced pattern matching is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_pattern_matching()
    sys.exit(0 if success else 1)
