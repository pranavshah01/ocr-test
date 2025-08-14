"""
Image processing utilities for OCR-based image text detection and replacement.
Contains all utilities specific to image processor functionality.
"""

# PatternMatcher moved to shared utils/pattern_matcher.py
from .image_preprocessor import ImagePreprocessor, create_image_preprocessor
from .pattern_debugger import PatternDebugger, create_pattern_debugger
from .precise_replace import PreciseTextReplacer, create_precise_text_replacer
from .preprocessing_strategies import PreprocessingStrategyManager, create_preprocessing_strategy_manager
from .text_analyzer import TextAnalyzer, TextProperties, create_text_analyzer

__all__ = [
    'ImagePreprocessor',
    'create_image_preprocessor',
    'PatternDebugger',
    'create_pattern_debugger',
    'PreciseTextReplacer',
    'create_precise_text_replacer',
    'PreprocessingStrategyManager',
    'create_preprocessing_strategy_manager',
    'TextAnalyzer',
    'TextProperties',
    'create_text_analyzer'
]