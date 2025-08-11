"""
Shared constants and utilities for the Document Processing Pipeline.
Eliminates code duplication across modules by centralizing common constants, namespaces, and utility functions.
"""

from typing import Dict, Any, List
import logging
import platform

# XML Namespaces used across DOCX processing
XML_NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'v': 'urn:schemas-microsoft-com:vml',
    'o': 'urn:schemas-microsoft-com:office:office'
}

# Common conversion constants
EMU_TO_PIXELS = 9525  # English Metric Units to pixels conversion
POINTS_TO_HALF_POINTS = 2  # Font size conversion
DEFAULT_FONT_SIZE = 12  # Default font size in points
DEFAULT_BOX_WIDTH = 300  # Default textbox width in pixels
DEFAULT_BOX_HEIGHT = 100  # Default textbox height in pixels

# OCR Configuration Constants
DEFAULT_OCR_CONFIDENCE = 0.7
DEFAULT_OCR_ENGINE = "easyocr"
OCR_ENGINES = ["easyocr", "tesseract"]
GPU_DEVICES = ["cuda", "mps", "cpu"]

# Processing Mode Constants
PROCESSING_MODES = {
    "text": ["append", "replace"],
    "ocr": ["replace", "append"]
}

# Supported File Formats
SUPPORTED_FORMATS = [".doc", ".docx"]
SUPPORTED_PLATFORMS = ["darwin", "win32", "linux"]

# Platform-specific Constants
PLATFORM_COMMANDS = {
    "darwin": {
        "libreoffice": "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "libreoffice_alt": "soffice"
    },
    "win32": {
        "libreoffice": "soffice.exe",
        "word_com": "Word.Application"
    },
    "linux": {
        "libreoffice": "soffice",
        "libreoffice_alt": "libreoffice"
    }
}

# Conversion Tools
CONVERSION_TOOLS = ["libreoffice", "word_com"]
CONVERSION_TIMEOUT = 60  # seconds

# Error Codes
ERROR_CODES = {
    "SUCCESS": 0,
    "FILE_NOT_FOUND": 1,
    "INVALID_FORMAT": 2,
    "CONVERSION_FAILED": 3,
    "PROCESSING_FAILED": 4,
    "OCR_FAILED": 5,
    "GPU_ERROR": 6,
    "TIMEOUT": 7
}

# File Processing Constants
MAX_FILE_SIZE_MB = 100  # Maximum file size for processing
TEMP_DIR_PREFIX = "docx_processor_"
OUTPUT_DIR_NAME = "processed"

# Formatting Constants
DEFAULT_FONT_FAMILY = "Arial"
FALLBACK_FONTS = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
MAX_SHAPE_SCALE_FACTOR = 2.0  # Maximum shape resize factor
TEXT_PADDING_FACTOR = 0.2  # Padding factor for text in shapes

# Common Error Messages
ERROR_FILE_NOT_FOUND = "File not found"
ERROR_INVALID_DOCX = "Invalid DOCX file format"
ERROR_XML_PARSING = "XML parsing error"
ERROR_FONT_LOADING = "Font loading error"
ERROR_OCR_ENGINE = "OCR engine error"

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DETAILED_LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s"

# Enhanced Logging Constants for Task 5.1
LOG_MATCH_TEMPLATE = "MATCH: '{original}' -> '{replacement}' | Mode: {mode} | Location: {location} | Confidence: {confidence}"
LOG_FONT_TEMPLATE = "FONT: Family={family}, Size={size}pt, Color={color}, Bold={bold}, Italic={italic}"
LOG_BBOX_TEMPLATE = "BBOX: x={x}, y={y}, w={width}, h={height}, rotation={rotation}°, confidence={confidence}"
LOG_PROCESSING_TEMPLATE = "PROCESSING: {phase} | Files: {files} | Matches: {matches} | Time: {time}s"


class SharedUtilities:
    """
    Shared utility functions used across multiple modules.
    """
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Set up a logger with consistent formatting.
        
        Args:
            name: Logger name
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        return logger
    
    @staticmethod
    def emu_to_pixels(emu_value: int) -> int:
        """
        Convert English Metric Units to pixels.
        
        Args:
            emu_value: Value in EMUs
            
        Returns:
            Value in pixels
        """
        return emu_value // EMU_TO_PIXELS
    
    @staticmethod
    def pixels_to_emu(pixel_value: int) -> int:
        """
        Convert pixels to English Metric Units.
        
        Args:
            pixel_value: Value in pixels
            
        Returns:
            Value in EMUs
        """
        return pixel_value * EMU_TO_PIXELS
    
    @staticmethod
    def points_to_half_points(points: int) -> int:
        """
        Convert font size from points to half-points (used in DOCX).
        
        Args:
            points: Font size in points
            
        Returns:
            Font size in half-points
        """
        return points * POINTS_TO_HALF_POINTS
    
    @staticmethod
    def half_points_to_points(half_points: int) -> int:
        """
        Convert font size from half-points to points.
        
        Args:
            half_points: Font size in half-points
            
        Returns:
            Font size in points
        """
        return half_points // POINTS_TO_HALF_POINTS
    
    @staticmethod
    def validate_file_size(file_path: str) -> bool:
        """
        Validate that file size is within processing limits.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file size is acceptable
        """
        try:
            from pathlib import Path
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return file_size_mb <= MAX_FILE_SIZE_MB
        except Exception:
            return False
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """
        Get a safe filename by removing/replacing problematic characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Safe filename
        """
        import re
        # Remove or replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(safe_name) > 200:
            name_part, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name_part[:190] + ('.' + ext if ext else '')
        return safe_name
    
    @staticmethod
    def create_processing_summary_template() -> Dict[str, Any]:
        """
        Create a template for processing summary results.
        
        Returns:
            Dictionary template for processing results
        """
        from datetime import datetime
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_replacements': 0,
            'text_replacements': 0,
            'ocr_replacements': 0,
            'textbox_replacements': 0,
            'section_replacements': 0,
            'formatting_preserved': 0,
            'errors': [],
            'warnings': [],
            'processing_time': 0.0,
            'file_size_mb': 0.0,
            'success': False
        }
    
    # Enhanced Logging Methods for Task 5.1
    @staticmethod
    def setup_detailed_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """
        Set up a detailed logger with enhanced formatting for comprehensive logging.
        
        Args:
            name: Logger name
            log_file: Optional log file path for file output
            level: Logging level
            
        Returns:
            Configured logger with detailed formatting
        """
        logger = logging.getLogger(name)
        
        # Clear existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with detailed format
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(DETAILED_LOG_FORMAT, LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            from pathlib import Path
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_formatter = logging.Formatter(DETAILED_LOG_FORMAT, LOG_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets all details
            logger.addHandler(file_handler)
        
        logger.setLevel(level)
        return logger
    
    @staticmethod
    def log_match_replacement(logger: logging.Logger, original: str, replacement: str, 
                            mode: str, location: str = "unknown", confidence: float = None):
        """
        Log a match and replacement with structured format.
        
        Args:
            logger: Logger instance
            original: Original text that was matched
            replacement: Replacement text
            mode: Processing mode used
            location: Location in document
            confidence: Confidence score (for OCR)
        """
        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        message = LOG_MATCH_TEMPLATE.format(
            original=original[:100] + "..." if len(original) > 100 else original,
            replacement=replacement[:100] + "..." if len(replacement) > 100 else replacement,
            mode=mode,
            location=location,
            confidence=confidence_str
        )
        logger.info(message)
    
    @staticmethod
    def log_font_info(logger: logging.Logger, family: str = None, size: float = None, 
                     color: str = None, bold: bool = False, italic: bool = False):
        """
        Log font information with structured format.
        
        Args:
            logger: Logger instance
            family: Font family name
            size: Font size in points
            color: Font color
            bold: Bold formatting
            italic: Italic formatting
        """
        message = LOG_FONT_TEMPLATE.format(
            family=family or "default",
            size=size or "default",
            color=color or "default",
            bold=bold,
            italic=italic
        )
        logger.debug(message)
    
    @staticmethod
    def log_bounding_box(logger: logging.Logger, x: float, y: float, width: float, height: float,
                        rotation: float = 0.0, confidence: float = None):
        """
        Log bounding box information with structured format.
        
        Args:
            logger: Logger instance
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
            rotation: Rotation angle in degrees
            confidence: Confidence score
        """
        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        message = LOG_BBOX_TEMPLATE.format(
            x=x, y=y, width=width, height=height,
            rotation=rotation, confidence=confidence_str
        )
        logger.debug(message)
    
    @staticmethod
    def log_processing_phase(logger: logging.Logger, phase: str, files: int = 0, 
                           matches: int = 0, time: float = 0.0):
        """
        Log processing phase information with structured format.
        
        Args:
            logger: Logger instance
            phase: Processing phase name
            files: Number of files processed
            matches: Number of matches found
            time: Processing time in seconds
        """
        message = LOG_PROCESSING_TEMPLATE.format(
            phase=phase, files=files, matches=matches, time=time
        )
        logger.info(message)
    
    @staticmethod
    def create_detailed_processing_log(file_path: str, matches: list, statistics: dict) -> dict:
        """
        Create a detailed processing log structure for JSON export.
        
        Args:
            file_path: Path to processed file
            matches: List of match/replacement dictionaries
            statistics: Processing statistics
            
        Returns:
            Detailed log structure
        """
        from datetime import datetime
        
        return {
            'file_info': {
                'path': file_path,
                'processed_at': datetime.now().isoformat(),
                'size_mb': statistics.get('file_size_mb', 0.0)
            },
            'processing_summary': {
                'total_matches': len(matches),
                'text_replacements': statistics.get('text_replacements', 0),
                'ocr_replacements': statistics.get('ocr_replacements', 0),
                'textbox_replacements': statistics.get('textbox_replacements', 0),
                'section_replacements': statistics.get('section_replacements', 0),
                'processing_time': statistics.get('processing_time', 0.0),
                'formatting_preserved': statistics.get('formatting_preserved', 0)
            },
            'matches_detail': matches,
            'errors': statistics.get('errors', []),
            'warnings': statistics.get('warnings', []),
            'success': statistics.get('success', False)
        }


# Platform Detection Utilities
def get_current_platform() -> str:
    """Get current platform identifier."""
    return platform.system().lower()

def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system().lower() == "darwin"

def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == "windows"

def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system().lower() == "linux"

# Export commonly used items for easy importing
__all__ = [
    'XML_NAMESPACES',
    'EMU_TO_PIXELS',
    'POINTS_TO_HALF_POINTS',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_BOX_WIDTH',
    'DEFAULT_BOX_HEIGHT',
    'DEFAULT_OCR_CONFIDENCE',
    'DEFAULT_OCR_ENGINE',
    'OCR_ENGINES',
    'GPU_DEVICES',
    'PROCESSING_MODES',
    'SUPPORTED_FORMATS',
    'SUPPORTED_PLATFORMS',
    'PLATFORM_COMMANDS',
    'CONVERSION_TOOLS',
    'CONVERSION_TIMEOUT',
    'ERROR_CODES',
    'DEFAULT_FONT_FAMILY',
    'FALLBACK_FONTS',
    'MAX_SHAPE_SCALE_FACTOR',
    'TEXT_PADDING_FACTOR',
    'SharedUtilities',
    'get_current_platform',
    'is_macos',
    'is_windows',
    'is_linux'
]
