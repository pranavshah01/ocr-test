"""
Pattern debugging and validation system for enhanced OCR text detection.
Provides comprehensive testing and debugging capabilities for pattern matching.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import json

from ..pattern_matcher import PatternMatcher, UniversalMatch

logger = logging.getLogger(__name__)

class PatternDebugger:
    """Comprehensive pattern debugging and validation system."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        """
        Initialize pattern debugger.
        
        Args:
            pattern_matcher: Enhanced pattern matcher to debug
        """
        self.pattern_matcher = pattern_matcher
        self.test_cases = []
        self.debug_results = []
    
    def add_test_case(self, text: str, expected_matches: List[str], description: str = ""):
        """
        Add a test case for pattern validation.
        
        Args:
            text: Text to test
            expected_matches: List of expected matched text
            description: Description of the test case
        """
        self.test_cases.append({
            'text': text,
            'expected_matches': expected_matches,
            'description': description
        })
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive pattern tests with various character combinations.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Running comprehensive pattern tests...")
        
        # Add standard test cases
        self._add_standard_test_cases()
        
        # Add edge case test cases
        self._add_edge_case_test_cases()
        
        # Run all test cases
        results = {
            'total_tests': len(self.test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': [],
            'pattern_analysis': self.pattern_matcher.get_debug_info()
        }
        
        for i, test_case in enumerate(self.test_cases):
            test_result = self._run_single_test(test_case, i)
            results['test_results'].append(test_result)
            
            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        results['success_rate'] = (results['passed_tests'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
        
        logger.info(f"Pattern tests completed: {results['passed_tests']}/{results['total_tests']} passed "
                   f"({results['success_rate']:.1f}% success rate)")
        
        return results
    
    def _add_standard_test_cases(self):
        """Add standard test cases for common scenarios."""
        # Test cases with punctuation before "77"
        self.add_test_case(":77-531-116BLK-245", ["77-531-116BLK-245"], "Colon before 77")
        self.add_test_case("(77-525-551WH1-000)", ["77-525-551WH1-000"], "Parenthesis before 77")
        self.add_test_case("[77-234-120616-000]", ["77-234-120616-000"], "Square bracket before 77")
        self.add_test_case("{77-245-000406-000}", ["77-245-000406-000"], "Curly brace before 77")
        
        # Test cases with symbols before "77"
        self.add_test_case("@77-110-810096-000", ["77-110-810096-000"], "At symbol before 77")
        self.add_test_case("#77-130-120541-001", ["77-130-120541-001"], "Hash symbol before 77")
        self.add_test_case("%77-151-0301701-00", ["77-151-0301701-00"], "Percent symbol before 77")
        self.add_test_case("&77-160-810105-000", ["77-160-810105-000"], "Ampersand before 77")
        
        # Test cases with spaces and normal boundaries
        self.add_test_case(" 77-232-0403040-00 ", ["77-232-0403040-00"], "Space boundaries")
        self.add_test_case("Part 77-234-120616-000 needed", ["77-234-120616-000"], "Word boundaries")
        
        # Test cases that should NOT match (within numbers)
        self.add_test_case("177-234-120616-000", [], "Should not match within number")
        self.add_test_case("77-234-120616-0001", [], "Should not match with trailing digit")
    
    def _add_edge_case_test_cases(self):
        """Add edge case test cases including non-English characters."""
        # Test cases with non-English characters
        self.add_test_case("中77-531-116BLK-245", ["77-531-116BLK-245"], "Chinese character before 77")
        self.add_test_case("ñ77-525-551WH1-000", ["77-525-551WH1-000"], "Spanish character before 77")
        self.add_test_case("ü77-234-120616-000", ["77-234-120616-000"], "German character before 77")
        self.add_test_case("α77-245-000406-000", ["77-245-000406-000"], "Greek character before 77")
        
        # Test cases with multiple patterns in same text
        self.add_test_case("Use :77-531-116BLK-245 and (77-525-551WH1-000)", 
                          ["77-531-116BLK-245", "77-525-551WH1-000"], "Multiple patterns")
        
        # Test cases at text boundaries
        self.add_test_case("77-110-810096-000", ["77-110-810096-000"], "Start of text")
        self.add_test_case("Part: 77-130-120541-001", ["77-130-120541-001"], "End of text")
        
        # Test cases with various separators
        self.add_test_case("Heat Shrink Heat:77-531-116BLK-245", ["77-531-116BLK-245"], "Real world example 1")
        self.add_test_case("22 AWG(77-525-551WH1-000)", ["77-525-551WH1-000"], "Real world example 2")
    
    def _run_single_test(self, test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_case: Test case dictionary
            test_index: Index of the test case
            
        Returns:
            Test result dictionary
        """
        text = test_case['text']
        expected_matches = test_case['expected_matches']
        description = test_case['description']
        
        try:
            # Find matches using enhanced pattern matcher
            universal_matches = self.pattern_matcher.find_matches_universal(text)
            actual_matches = [match.matched_text for match in universal_matches]
            
            # Check if matches are as expected
            passed = set(actual_matches) == set(expected_matches)
            
            result = {
                'test_index': test_index,
                'description': description,
                'text': text,
                'expected_matches': expected_matches,
                'actual_matches': actual_matches,
                'passed': passed,
                'match_details': [match.to_dict() for match in universal_matches]
            }
            
            if passed:
                logger.debug(f"Test {test_index} PASSED: {description}")
            else:
                logger.warning(f"Test {test_index} FAILED: {description}")
                logger.warning(f"  Text: '{text}'")
                logger.warning(f"  Expected: {expected_matches}")
                logger.warning(f"  Actual: {actual_matches}")
            
            return result
            
        except Exception as e:
            logger.error(f"Test {test_index} ERROR: {e}")
            return {
                'test_index': test_index,
                'description': description,
                'text': text,
                'expected_matches': expected_matches,
                'actual_matches': [],
                'passed': False,
                'error': str(e),
                'match_details': []
            }
    
    def debug_pattern_matching(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Debug pattern matching for a specific text with detailed analysis.
        
        Args:
            text: Text to debug
            verbose: Whether to include verbose debugging information
            
        Returns:
            Debugging information dictionary
        """
        logger.info(f"Debugging pattern matching for text: '{text}'")
        
        debug_info = {
            'input_text': text,
            'text_length': len(text),
            'patterns_tested': [],
            'matches_found': [],
            'no_matches_reasons': [],
            'character_analysis': self._analyze_characters(text)
        }
        
        # Test each pattern individually
        for pattern_name, compiled_pattern in self.pattern_matcher.compiled_patterns.items():
            pattern_debug = self._debug_single_pattern(text, pattern_name, compiled_pattern, verbose)
            debug_info['patterns_tested'].append(pattern_debug)
            
            if pattern_debug['matches']:
                debug_info['matches_found'].extend(pattern_debug['matches'])
            else:
                debug_info['no_matches_reasons'].append({
                    'pattern_name': pattern_name,
                    'reason': pattern_debug['no_match_reason']
                })
        
        # Get overall matches using the enhanced matcher
        universal_matches = self.pattern_matcher.find_matches_universal(text)
        debug_info['final_matches'] = [match.to_dict() for match in universal_matches]
        debug_info['final_match_count'] = len(universal_matches)
        
        if verbose:
            self._log_debug_info(debug_info)
        
        return debug_info
    
    def _debug_single_pattern(self, text: str, pattern_name: str, compiled_pattern: re.Pattern, 
                             verbose: bool) -> Dict[str, Any]:
        """
        Debug a single pattern against text.
        
        Args:
            text: Text to test
            pattern_name: Name of the pattern
            compiled_pattern: Compiled regex pattern
            verbose: Whether to include verbose information
            
        Returns:
            Pattern debugging information
        """
        pattern_debug = {
            'pattern_name': pattern_name,
            'pattern_string': compiled_pattern.pattern,
            'matches': [],
            'match_count': 0,
            'no_match_reason': None
        }
        
        try:
            matches = list(compiled_pattern.finditer(text))
            pattern_debug['match_count'] = len(matches)
            
            for match in matches:
                matched_text = match.group()
                replacement = self.pattern_matcher.get_replacement(matched_text)
                
                match_info = {
                    'matched_text': matched_text,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'has_replacement': replacement is not None,
                    'replacement_text': replacement,
                    'preceding_context': self.pattern_matcher.get_match_context(text, match.start(), 3, True),
                    'following_context': self.pattern_matcher.get_match_context(text, match.end(), 3, False)
                }
                
                if replacement:
                    pattern_debug['matches'].append(match_info)
            
            if not pattern_debug['matches']:
                if matches:
                    pattern_debug['no_match_reason'] = "Pattern matched but no replacement mapping found"
                else:
                    pattern_debug['no_match_reason'] = "Pattern did not match text"
        
        except Exception as e:
            pattern_debug['no_match_reason'] = f"Pattern matching error: {e}"
        
        return pattern_debug
    
    def _analyze_characters(self, text: str) -> Dict[str, Any]:
        """
        Analyze characters in text for debugging purposes.
        
        Args:
            text: Text to analyze
            
        Returns:
            Character analysis information
        """
        analysis = {
            'total_chars': len(text),
            'digit_chars': sum(1 for c in text if c.isdigit()),
            'alpha_chars': sum(1 for c in text if c.isalpha()),
            'space_chars': sum(1 for c in text if c.isspace()),
            'punctuation_chars': sum(1 for c in text if not c.isalnum() and not c.isspace()),
            'unicode_chars': sum(1 for c in text if ord(c) > 127),
            'character_breakdown': {}
        }
        
        # Count each unique character
        for char in text:
            char_type = self._classify_character(char)
            if char_type not in analysis['character_breakdown']:
                analysis['character_breakdown'][char_type] = []
            if char not in [item['char'] for item in analysis['character_breakdown'][char_type]]:
                analysis['character_breakdown'][char_type].append({
                    'char': char,
                    'unicode': ord(char),
                    'count': text.count(char)
                })
        
        return analysis
    
    def _classify_character(self, char: str) -> str:
        """Classify a character for analysis."""
        if char.isdigit():
            return 'digits'
        elif char.isalpha() and ord(char) <= 127:
            return 'ascii_letters'
        elif char.isalpha() and ord(char) > 127:
            return 'unicode_letters'
        elif char.isspace():
            return 'whitespace'
        elif char in '.,;:!?':
            return 'punctuation'
        elif char in '()[]{}':
            return 'brackets'
        elif char in '@#$%&*':
            return 'symbols'
        elif char in '-_=+':
            return 'operators'
        else:
            return 'other'
    
    def _log_debug_info(self, debug_info: Dict[str, Any]):
        """Log detailed debugging information."""
        logger.info("=== Pattern Matching Debug Information ===")
        logger.info(f"Input text: '{debug_info['input_text']}'")
        logger.info(f"Text length: {debug_info['text_length']}")
        
        logger.info("Character analysis:")
        char_analysis = debug_info['character_analysis']
        logger.info(f"  Total: {char_analysis['total_chars']}, "
                   f"Digits: {char_analysis['digit_chars']}, "
                   f"Alpha: {char_analysis['alpha_chars']}, "
                   f"Unicode: {char_analysis['unicode_chars']}")
        
        logger.info(f"Patterns tested: {len(debug_info['patterns_tested'])}")
        for pattern_debug in debug_info['patterns_tested']:
            logger.info(f"  {pattern_debug['pattern_name']}: {pattern_debug['match_count']} matches")
            if pattern_debug['no_match_reason']:
                logger.info(f"    Reason: {pattern_debug['no_match_reason']}")
        
        logger.info(f"Final matches found: {debug_info['final_match_count']}")
        for match in debug_info['final_matches']:
            logger.info(f"  '{match['matched_text']}' at {match['start_pos']}-{match['end_pos']}")
    
    def save_debug_report(self, output_path: Path, test_results: Optional[Dict] = None, 
                         debug_info: Optional[Dict] = None):
        """
        Save comprehensive debug report to file.
        
        Args:
            output_path: Path to save the report
            test_results: Test results from run_comprehensive_tests
            debug_info: Debug info from debug_pattern_matching
        """
        report = {
            'timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord(
                'debug', logging.INFO, '', 0, '', (), None)) if logger.handlers else 'unknown',
            'pattern_matcher_info': self.pattern_matcher.get_debug_info()
        }
        
        if test_results:
            report['test_results'] = test_results
        
        if debug_info:
            report['debug_info'] = debug_info
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save debug report: {e}")

def create_pattern_debugger(pattern_matcher: PatternMatcher) -> PatternDebugger:
    """
    Factory function to create a PatternDebugger instance.
    
    Args:
        pattern_matcher: Enhanced pattern matcher to debug
        
    Returns:
        PatternDebugger instance
    """
    return PatternDebugger(pattern_matcher)