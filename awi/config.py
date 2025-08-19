"""
Minimal Configuration module for the Document Processing Pipeline.
Centralizes essential configuration constants and CLI argument definitions.

This module provides:
- Default configuration constants for processing modes and settings
- CLI argument definitions for command-line interface
- A unified ProcessingConfig dataclass for configuration management
- GPU detection and validation logic
- Path validation and directory creation
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import torch

# =============================================================================
# PROCESSING CONFIGURATION CONSTANTS
# =============================================================================

# Text processing mode determines how mapped text is inserted into documents
TEXT_MODE = "append"  # Default text processing mode: "append" or "replace"

# Text processing separator for append mode
TEXT_SEPARATOR = ";"  # Default separator between original and appended text in append mode

# Default mapping for patterns not found in mapping.json
DEFAULT_MAPPING = "4022-NA"  # Default text to append when no mapping is found

# OCR processing mode determines how OCR text is handled in images
OCR_MODE = "append"  # Default OCR processing mode: "replace", "append", or "append-image"

# GPU acceleration setting for OCR processing
USE_GPU = True  # Default GPU usage setting for better performance

# Maximum number of parallel workers for processing multiple documents
# Calculated as CPU count minus 2, with a maximum of 8 workers
MAX_WORKERS = max(1, min((__import__('psutil').cpu_count(logical=True) or 8) - 2, 8))

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Default directory names for the pipeline
SOURCE_DIR = "source_documents"  # Directory containing input documents
OUTPUT_DIR = "processed"         # Directory for processed output documents
REPORTS_DIR = "reports"          # Directory for processing reports and logs
LOGS_DIR = "logs"               # Directory for detailed log files

# =============================================================================
# FILE CONFIGURATION
# =============================================================================

# Configuration files for pattern matching and text mapping
PATTERNS_FILE = "patterns.json"  # JSON file containing regex patterns to match
MAPPINGS_FILE = "mapping.json"   # JSON file containing text mappings and replacements

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Timeout settings for document processing
PROCESSING_TIMEOUT = 300  # Maximum time (seconds) to process a single document

# Retry settings for failed operations
MAX_RETRIES = 3  # Maximum number of retry attempts for failed operations

# OCR confidence threshold for text detection
CONFIDENCE_MIN = 0.4  # Minimum confidence score (0.0-1.0) for OCR text detection

# File size limits for processing
MIN_FILE_SIZE_MB = 0.1  # Minimum file size in MB for processing
MAX_FILE_SIZE_MB = 300.0  # Maximum file size in MB for processing (increased from 100MB to support large files)

# =============================================================================
# CLI FLAG DEFINITIONS - Essential only
# =============================================================================

# Dictionary defining all command-line interface arguments
# Each flag contains type, choices, default value, and help text
CLI_FLAGS = {
    # Text processing configuration
    '--text-mode': {
        'type': str,
        'choices': ['append', 'replace'],
        'default': TEXT_MODE,
        'help': 'Text processing mode: append mapped text after matched text or replace matched text'
    },
    '--separator': {
        'type': str,
        'default': TEXT_SEPARATOR,
        'help': f'Separator between original and appended text in append mode (default: {TEXT_SEPARATOR})'
    },
    '--default-mapping': {
        'type': str,
        'default': DEFAULT_MAPPING,
        'help': f'Default text to append when no mapping is found in mapping.json (default: {DEFAULT_MAPPING})'
    },
    
    # OCR processing configuration
    '--ocr-mode': {
        'type': str,
        'choices': ['replace', 'append', 'append-image'],
        'default': OCR_MODE,
        'help': 'OCR processing mode: replace (exact position), append (top-bottom text), append-image (new image after original)'
    },
    
    # OCR engine selection
    '--ocr-engine': {
        'type': str,
        'choices': ['easyocr', 'tesseract', 'hybrid'],
        'default': 'easyocr',
        'help': 'OCR engine to use: easyocr, tesseract, or hybrid (combines both engines)'
    },
    
    # GPU acceleration flags
    '--gpu': {
        'action': 'store_true',
        'help': 'Enable GPU acceleration for OCR processing'
    },
    '--no-gpu': {
        'action': 'store_true',
        'help': 'Disable GPU acceleration for OCR processing'
    },
    
    # Performance configuration
    '--workers': {
        'type': int,
        'default': MAX_WORKERS,
        'help': f'Number of parallel workers (default: {MAX_WORKERS})'
    },
    
    # OCR quality settings
    '--confidence-min': {
        'type': float,
        'default': CONFIDENCE_MIN,
        'help': f'Minimum OCR confidence threshold (default: {CONFIDENCE_MIN})'
    },
    
    # Logging configuration
    '--verbose': {
        'action': 'store_true',
        'help': 'Enable verbose logging output'
    },
    
    # File path configuration
    '--patterns': {
        'type': str,
        'default': PATTERNS_FILE,
        'help': f'Path to patterns JSON file (default: {PATTERNS_FILE})'
    },
    '--mapping': {
        'type': str,
        'default': MAPPINGS_FILE,
        'help': f'Path to mapping JSON file (default: {MAPPINGS_FILE})'
    },
    '--source-dir': {
        'type': str,
        'default': SOURCE_DIR,
        'help': f'Source documents directory (default: {SOURCE_DIR})'
    },
    '--output-dir': {
        'type': str,
        'default': OUTPUT_DIR,
        'help': f'Output directory for processed documents (default: {OUTPUT_DIR})'
    },
    '--reports-dir': {
        'type': str,
        'default': REPORTS_DIR,
        'help': f'Reports directory (default: {REPORTS_DIR})'
    },
    '--logs-dir': {
        'type': str,
        'default': LOGS_DIR,
        'help': f'Logs directory (default: {LOGS_DIR})'
    }
}

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class ProcessingConfig:
    """
    Unified configuration class for document processing pipeline.
    
    This dataclass centralizes all configuration settings and provides:
    - Default values for all processing parameters
    - Automatic validation of configuration values
    - GPU detection and configuration
    - Path validation and directory creation
    
    Attributes:
        text_mode: How to handle text replacements ("append" or "replace")
        text_separator: Separator between original and appended text in append mode
        default_mapping: Default text to append when no mapping is found in mapping.json
        ocr_mode: How to handle OCR text in images ("replace", "append", "append-image")
        ocr_engine: Which OCR engine to use ("easyocr", "tesseract", "hybrid")
        use_gpu: Whether to enable GPU acceleration
        max_workers: Number of parallel processing workers
        confidence_min: Minimum OCR confidence threshold
        verbose: Enable verbose logging output
        patterns_file: Path to patterns JSON file
        mappings_file: Path to mappings JSON file
        source_dir: Directory containing input documents
        output_dir: Directory for processed documents
        reports_dir: Directory for processing reports
        logs_dir: Directory for log files
        processing_timeout: Maximum time to process a document
        max_retries: Maximum retry attempts for failed operations
    """
    
    # Processing mode settings
    text_mode: str = TEXT_MODE
    text_separator: str = TEXT_SEPARATOR
    default_mapping: str = DEFAULT_MAPPING
    ocr_mode: str = OCR_MODE
    ocr_engine: str = "easyocr"
    
    # Performance settings
    use_gpu: bool = USE_GPU
    max_workers: int = MAX_WORKERS
    confidence_min: float = CONFIDENCE_MIN
    verbose: bool = False
    
    # GPU settings (will be set by _detect_gpu_availability)
    gpu_device: str = "cpu"
    gpu_available: bool = False
    
    # File paths - using Path objects for better path handling
    patterns_file: Path = Path(PATTERNS_FILE)
    mappings_file: Path = Path(MAPPINGS_FILE)
    source_dir: Path = Path(SOURCE_DIR)
    output_dir: Path = Path(OUTPUT_DIR)
    reports_dir: Path = Path(REPORTS_DIR)
    logs_dir: Path = Path(LOGS_DIR)
    
    # Processing settings
    processing_timeout: int = PROCESSING_TIMEOUT
    max_retries: int = MAX_RETRIES
    
    # File size limits
    min_file_size: float = MIN_FILE_SIZE_MB
    max_file_size: float = MAX_FILE_SIZE_MB
    
    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        This method is automatically called after the dataclass is initialized.
        It performs:
        - Mode validation (text_mode, ocr_mode)
        - Path validation and directory creation
        - GPU availability detection
        """
        self._validate_modes()
        self._validate_paths()
        self._detect_gpu_availability()
    
    def _validate_modes(self):
        """
        Validate processing modes to ensure they are valid.
        
        Raises:
            ValueError: If text_mode or ocr_mode contains invalid values
        """
        # Define valid options for each mode
        valid_text_modes = ["append", "replace"]
        valid_ocr_modes = ["replace", "append", "append-image"]
        
        # Validate text processing mode
        if self.text_mode not in valid_text_modes:
            raise ValueError(f"Invalid text_mode '{self.text_mode}'. Must be one of {valid_text_modes}")
        
        # Validate OCR processing mode
        if self.ocr_mode not in valid_ocr_modes:
            raise ValueError(f"Invalid ocr_mode '{self.ocr_mode}'. Must be one of {valid_ocr_modes}")
    
    def _validate_paths(self):
        """
        Validate and create necessary directories and files.
        
        This method:
        - Creates output directories if they don't exist
        - Validates that required input files exist
        - Ensures source directory exists
        
        Raises:
            FileNotFoundError: If required files or directories are missing
        """
        # Create output directories (processed, reports, logs)
        for directory in [self.output_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate that required configuration files exist
        if not self.patterns_file.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_file}")
        
        if not self.mappings_file.exists():
            raise FileNotFoundError(f"Mappings file not found: {self.mappings_file}")
        
        # Validate that source directory exists
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
    
    def _detect_gpu_availability(self):
        """
        Detect and configure GPU availability for OCR processing.
        
        This method:
        - Checks for CUDA (NVIDIA GPUs)
        - Checks for MPS (Apple Silicon GPUs)
        - Falls back to CPU if GPU is not available
        - Sets gpu_device and gpu_available attributes
        """
        if self.use_gpu:
            try:
                # Check for CUDA (NVIDIA GPUs)
                if torch.cuda.is_available():
                    self.gpu_device = "cuda"
                    self.gpu_available = True
                # Check for MPS (Apple Silicon GPUs)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.gpu_device = "mps"
                    self.gpu_available = True
                else:
                    # No GPU available, fall back to CPU
                    self.gpu_device = "cpu"
                    self.gpu_available = False
                    if self.verbose:
                        print("GPU requested but not available, falling back to CPU")
            except Exception as e:
                # GPU detection failed, fall back to CPU
                self.gpu_device = "cpu"
                self.gpu_available = False
                if self.verbose:
                    print(f"GPU detection failed: {e}, falling back to CPU")
        else:
            # GPU disabled by user
            self.gpu_device = "cpu"
            self.gpu_available = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with essential CLI flags from configuration.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all CLI flags
    """
    parser = argparse.ArgumentParser(
        description="Minimal Document Processing Pipeline - Process Word documents with text replacements and OCR-based image text replacement"
    )
    
    # Add all CLI flags from the configuration dictionary
    for flag, config in CLI_FLAGS.items():
        parser.add_argument(flag, **config)
    
    return parser

def load_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """
    Load configuration from parsed command line arguments.
    
    This function:
    - Handles GPU flag conflicts (--gpu vs --no-gpu)
    - Creates a ProcessingConfig instance with argument values
    - Performs validation and exits on errors
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        ProcessingConfig: Configured instance with argument values
        
    Raises:
        SystemExit: If configuration validation fails
    """
    # Handle GPU flags - --gpu takes precedence over --no-gpu
    use_gpu = USE_GPU
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    try:
        # Create ProcessingConfig with argument values
        return ProcessingConfig(
            text_mode=args.text_mode,
            text_separator=args.separator,
            default_mapping=args.default_mapping,
            ocr_mode=args.ocr_mode,
            ocr_engine=args.ocr_engine,
            use_gpu=use_gpu,
            max_workers=args.workers,
            confidence_min=args.confidence_min,
            verbose=args.verbose,
            patterns_file=Path(args.patterns),
            mappings_file=Path(args.mapping),
            source_dir=Path(args.source_dir),
            output_dir=Path(args.output_dir),
            reports_dir=Path(args.reports_dir),
            logs_dir=Path(args.logs_dir)
        )
    except (ValueError, FileNotFoundError) as e:
        # Print error and exit if configuration is invalid
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

def get_default_config() -> ProcessingConfig:
    """
    Get default configuration without CLI arguments.
    
    Returns:
        ProcessingConfig: Default configuration instance with all default values
    """
    return ProcessingConfig()