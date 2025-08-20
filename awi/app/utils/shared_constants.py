"""
Shared constants used across the awi application.
"""

# XML namespaces for DOCX processing
XML_NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
}

# Default font settings
DEFAULT_FONT_SIZE = 12.0
DEFAULT_FONT_FAMILY = "Arial"

# Processing modes
PROCESSING_MODES = {
    'APPEND': 'append',
    'REPLACE': 'replace'
}

# Content types
CONTENT_TYPES = {
    'PARAGRAPH': 'Paragraph',
    'TABLE': 'Table',
    'HEADER': 'Header',
    'FOOTER': 'Footer',
    'TEXTBOX': 'Textbox'
}

# Default processing settings
DEFAULT_SEPARATOR = ";"
DEFAULT_MAPPING = "4022-NA"

# OCR settings
DEFAULT_OCR_CONFIDENCE = 0.7
FALLBACK_FONTS = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]

# File Processing Constants
MAX_FILE_SIZE_MB = 300  # Maximum file size for processing (increased from 100MB)
LARGE_FILE_THRESHOLD_MB = 100  # Threshold for large file optimizations
VERY_LARGE_FILE_THRESHOLD_MB = 200  # Threshold for very large file optimizations

# Large File Processing Constants
LXML_HUGE_TREE_ENABLED = True
LXML_ATTRIBUTE_VALUE_LIMIT = 500000000  # 500MB for very large files
LXML_TEXT_VALUE_LIMIT = 500000000  # 500MB for very large text blocks

# Memory Management Constants
MIN_MEMORY_REQUIRED_MB = 500  # Minimum memory required for large file processing
GARBAGE_COLLECTION_THRESHOLD = (700, 10, 10)  # More aggressive GC for large files

# Error Codes
ERROR_CODES = {
    "SUCCESS": 0,
    "FILE_NOT_FOUND": 1,
    "INVALID_FORMAT": 2,
    "CONVERSION_FAILED": 3,
    "PROCESSING_FAILED": 4,
    "FILE_TOO_LARGE": 5,
    "MEMORY_ERROR": 6,
    "XML_PARSING_ERROR": 7
}

# Supported File Formats
SUPPORTED_FORMATS = [".doc", ".docx"]

# Common Error Messages
ERROR_FILE_NOT_FOUND = "File not found"
ERROR_INVALID_DOCX = "Invalid DOCX file format"
ERROR_XML_PARSING = "XML parsing error"
ERROR_FILE_TOO_LARGE = "File size exceeds maximum supported size"
ERROR_MEMORY_INSUFFICIENT = "Insufficient memory for large file processing"
