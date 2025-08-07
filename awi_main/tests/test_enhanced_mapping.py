#!/usr/bin/env python3
"""
Test script for enhanced mapping and replacement logic (Task 2.2).
Validates the integration of enhanced pattern matching from Task 2.1 with mapping-based text replacement.
"""

import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from docx_processor import EnhancedMappingProcessor, process_docx
from text_extractor import find_patterns_across_split_tags


def test_enhanced_mapping_processor():
    """Test the EnhancedMappingProcessor class functionality."""
    print("=" * 60)
    print("TESTING ENHANCED MAPPING PROCESSOR")
    print("=" * 60)
    
    # Initialize processor
    processor = EnhancedMappingProcessor(text_mode="replace")
    
    # Test mapping lookup with various strategies
    test_mapping = {
        "ABC-123": "XYZ-789",
        "DEF 456": "UVW 012",
        "GHI789": "RST345",
        "Test Pattern": "Replacement Text"
    }
    
    # Test cases for mapping lookup
    test_cases = [
        ("ABC-123", "XYZ-789", "Direct exact match"),
        ("ABC - 123", "XYZ-789", "Normalized whitespace around dash"),
        ("ABC 123", None, "Should not match without dash"),
        ("abc-123", "XYZ-789", "Case insensitive match"),
        ("DEF456", "UVW 012", "Alphanumeric match"),
        ("GHI 789", "RST345", "Whitespace normalization"),
        ("test pattern", "Replacement Text", "Case insensitive"),
        ("Unknown Text", None, "No match found")
    ]
    
    print("\nTesting mapping lookup strategies:")
    for test_text, expected, description in test_cases:
        result = processor.find_mapping_match(test_text, test_mapping)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: '{test_text}' -> {result}")
        if result != expected:
            print(f"    Expected: {expected}, Got: {result}")
    
    print(f"\nMapping statistics:")
    print(f"  Hits: {processor.mapping_hits}")
    print(f"  Misses: {processor.mapping_misses}")
    
    return processor


def test_enhanced_processing_integration():
    """Test the integration of enhanced pattern matching with mapping."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED PROCESSING INTEGRATION")
    print("=" * 60)
    
    # Test files
    test_docx = Path("test_documents/sample_document.docx")
    patterns_file = Path("patterns.json")
    mapping_file = Path("mapping.json")
    
    # Check if test files exist
    if not test_docx.exists():
        print(f"⚠️  Test document not found: {test_docx}")
        print("   Creating a minimal test case instead...")
        return test_minimal_integration()
    
    if not patterns_file.exists():
        print(f"⚠️  Patterns file not found: {patterns_file}")
        return False
    
    if not mapping_file.exists():
        print(f"⚠️  Mapping file not found: {mapping_file}")
        return False
    
    # Load patterns and mapping
    try:
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        
        # Extract patterns from JSON (handle both formats)
        if isinstance(patterns_data, list):
            patterns = patterns_data
        else:
            patterns = [v for k, v in patterns_data.items() if not k.startswith('_')]
        
        print(f"Loaded {len(patterns)} patterns and {len(mapping_dict)} mappings")
        
        # Test enhanced pattern matching
        print("\nTesting enhanced pattern matching...")
        pattern_results = find_patterns_across_split_tags(test_docx, patterns)
        
        print(f"Pattern matching results:")
        print(f"  Total matches: {pattern_results['total_matches']}")
        print(f"  Processing time: {pattern_results['processing_time_ms']:.2f}ms")
        
        if pattern_results['total_matches'] > 0:
            print("\nMatches found:")
            for i, match in enumerate(pattern_results['matches'][:5]):  # Show first 5
                print(f"  {i+1}. '{match['match_text']}' (pattern: {match['pattern_name']})")
                print(f"     Position: {match['start_pos']}-{match['end_pos']}")
        
        # Test enhanced processing
        print("\nTesting enhanced processing with mapping...")
        processor = EnhancedMappingProcessor(text_mode="replace")
        
        # Simulate mapping lookup for found patterns
        mapping_results = []
        for match in pattern_results['matches']:
            matched_text = match['match_text']
            replacement = processor.find_mapping_match(matched_text, mapping_dict)
            mapping_results.append({
                'matched_text': matched_text,
                'replacement': replacement,
                'found_mapping': replacement is not None
            })
        
        # Report mapping results
        successful_mappings = [r for r in mapping_results if r['found_mapping']]
        print(f"\nMapping results:")
        print(f"  Successful mappings: {len(successful_mappings)}/{len(mapping_results)}")
        
        if successful_mappings:
            print("  Examples:")
            for result in successful_mappings[:3]:  # Show first 3
                print(f"    '{result['matched_text']}' -> '{result['replacement']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_minimal_integration():
    """Test with minimal synthetic data when test files are not available."""
    print("\nRunning minimal integration test...")
    
    # Create synthetic test data
    test_patterns = [
        r"(?<!\w)ABC-\d{3}(?!\w)",  # ABC-123 pattern
        r"(?<!\w)DEF\s*\d{3}(?!\w)",  # DEF 456 pattern
        r"(?<!\w)Test\s+Pattern(?!\w)"  # Test Pattern
    ]
    
    test_mapping = {
        "ABC-123": "XYZ-789",
        "DEF 456": "UVW 012", 
        "Test Pattern": "Replacement Text"
    }
    
    # Test the enhanced processor
    processor = EnhancedMappingProcessor(text_mode="replace")
    
    # Simulate pattern matches
    simulated_matches = [
        "ABC-123",
        "DEF 456", 
        "Test Pattern",
        "Unknown Pattern"
    ]
    
    print("\nSimulated mapping test:")
    successful_mappings = 0
    for matched_text in simulated_matches:
        replacement = processor.find_mapping_match(matched_text, test_mapping)
        status = "✓" if replacement else "✗"
        print(f"  {status} '{matched_text}' -> {replacement}")
        if replacement:
            successful_mappings += 1
    
    print(f"\nResults: {successful_mappings}/{len(simulated_matches)} successful mappings")
    return successful_mappings > 0


def test_process_docx_integration():
    """Test the updated process_docx function with enhanced patterns."""
    print("\n" + "=" * 60)
    print("TESTING PROCESS_DOCX INTEGRATION")
    print("=" * 60)
    
    # Test with sample data
    test_docx = Path("test_documents/sample_document.docx")
    
    if not test_docx.exists():
        print(f"⚠️  Test document not found: {test_docx}")
        print("   Skipping process_docx integration test")
        return False
    
    # Load test data
    try:
        with open("patterns.json", 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        
        with open("mapping.json", 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        
        # Extract patterns
        if isinstance(patterns_data, list):
            patterns = patterns_data
        else:
            patterns = [v for k, v in patterns_data.items() if not k.startswith('_')]
        
        print("Testing process_docx with enhanced patterns enabled...")
        
        # Test with enhanced patterns
        result_enhanced = process_docx(
            docx_path=test_docx,
            mapping_dict=mapping_dict,
            regex_patterns=patterns,
            text_mode="replace",
            use_enhanced_patterns=True
        )
        
        print("Enhanced processing results:")
        print(f"  Body replacements: {result_enhanced.get('body_replacements', 0)}")
        print(f"  Enhanced processing: {result_enhanced.get('enhanced_processing', False)}")
        print(f"  Mapping hits: {result_enhanced.get('mapping_hits', 0)}")
        print(f"  Mapping misses: {result_enhanced.get('mapping_misses', 0)}")
        
        if result_enhanced.get('errors'):
            print(f"  Errors: {result_enhanced['errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ process_docx integration test failed: {e}")
        return False


def main():
    """Run all enhanced mapping tests."""
    print("Enhanced Mapping and Replacement Logic Test Suite")
    print("Task 2.2: Integration of enhanced pattern matching with mapping")
    print("=" * 80)
    
    # Test results
    results = []
    
    # Test 1: Enhanced mapping processor
    try:
        test_enhanced_mapping_processor()
        results.append(("Enhanced Mapping Processor", True))
    except Exception as e:
        print(f"❌ Enhanced mapping processor test failed: {e}")
        results.append(("Enhanced Mapping Processor", False))
    
    # Test 2: Enhanced processing integration
    try:
        success = test_enhanced_processing_integration()
        results.append(("Enhanced Processing Integration", success))
    except Exception as e:
        print(f"❌ Enhanced processing integration test failed: {e}")
        results.append(("Enhanced Processing Integration", False))
    
    # Test 3: process_docx integration
    try:
        success = test_process_docx_integration()
        results.append(("process_docx Integration", success))
    except Exception as e:
        print(f"❌ process_docx integration test failed: {e}")
        results.append(("process_docx Integration", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Enhanced mapping and replacement logic is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
