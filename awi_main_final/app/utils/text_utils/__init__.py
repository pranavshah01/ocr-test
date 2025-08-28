"""
Text utilities module for DOCX document processing.

This module provides utilities specifically for text processing including:
- Text reconstruction from paragraph runs
- Font information extraction and management
- Pattern matching for text content
"""

from .text_docx_utils import (
    TextReconstructor,
    FontManager,
    PatternMatcher,
    create_pattern_matcher,
    load_patterns_and_mappings
)

__all__ = [
    'TextReconstructor',
    'FontManager', 
    'PatternMatcher',
    'create_pattern_matcher',
    'load_patterns_and_mappings'
]
