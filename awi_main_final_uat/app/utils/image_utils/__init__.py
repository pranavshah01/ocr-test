"""
Image processing utilities for OCR-based image text detection and replacement.
Contains all utilities specific to image processor functionality.
"""

from .hybrid_ocr_manager import HybridOCRManager
from .image_preprocessor import ImagePreprocessor, create_image_preprocessor
from .pattern_debugger import PatternDebugger, create_pattern_debugger
from .preprocessing_strategies import PreprocessingStrategyManager, create_preprocessing_strategy_manager
from .text_analyzer import TextAnalyzer, TextProperties, create_text_analyzer

__all__ = [
    'HybridOCRManager',
    'ImagePreprocessor',
    'create_image_preprocessor',
    'PatternDebugger',
    'create_pattern_debugger',
    'PreprocessingStrategyManager',
    'create_preprocessing_strategy_manager',
    'TextAnalyzer',
    'TextProperties',
    'create_text_analyzer'
]

