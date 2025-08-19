"""
Utility modules for the document processing pipeline.
Contains shared utilities, constants, and processor-specific helper functions.

Organization:
- image_utils/: Utilities specific to image processor (OCR, pattern matching, text replacement)
- text_utils/: Utilities specific to text processor (DOCX text processing, font management)
- graphics_utils/: Utilities specific to graphics processor (textbox processing)
- Shared utilities: platform_utils, report_generator, shared_constants
"""

# Import commonly used shared utilities (non-editing related)
from .shared_constants import *
# from .platform_utils import PathManager  # Commented out - file is empty
# from .report_generator import ReportGenerator  # Commented out - not needed for minimal version
from .pattern_matcher import PatternMatcher, create_pattern_matcher

__all__ = [
    # Shared platform utilities
    # 'PathManager',  # Commented out - not available
    
    # Shared report generation
    # 'ReportGenerator',  # Commented out - not needed for minimal version
    
    # Shared pattern matching (patterns are same across all processors)
    'PatternMatcher',
    'create_pattern_matcher',
    
    # Processor-specific utilities are imported from their respective subdirectories:
    # - image_utils: for image processor utilities
    # - text_utils: for text processor utilities  
    # - graphics_utils: for graphics processor utilities
]