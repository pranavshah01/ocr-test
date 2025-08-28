"""
Unified Configuration Management for the Document Processing Pipeline.
Centralizes all configuration settings with YAML support and CLI argument definitions.

This module provides:
- YAML-based configuration loading and validation
- Default configuration constants for processing modes and settings
- CLI argument definitions for command-line interface
- A unified ProcessingConfig dataclass for configuration management
- GPU detection and validation logic
- Path validation and directory creation
- Centralized logging configuration
- Eliminates scattered configuration across multiple files
"""

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sys
import torch
from enum import Enum
import yaml
import logging
from typing import Dict, Any, Optional, Union
import os

# Core constants (migrated from app.utils.shared_constants)
DEFAULT_FONT_SIZE = 12.0
DEFAULT_FONT_FAMILY = "Arial"

PROCESSING_MODES = {
    'APPEND': 'append',
    'REPLACE': 'replace'
}

DEFAULT_SEPARATOR = ";"
DEFAULT_MAPPING = "4022-NA"
DEFAULT_OCR_CONFIDENCE = 0.7

# =============================================================================
# PROCESSING MODES ENUM
# =============================================================================

class ProcessingMode(Enum):
    """Enum for processing modes to reduce stringly-typed errors."""
    APPEND = "append"
    REPLACE = "replace"

class OCRMode(Enum):
    """Enum for OCR processing modes."""
    REPLACE = "replace"
    APPEND = "append"
    APPEND_IMAGE = "append-image"

# =============================================================================
# PROCESSING CONFIGURATION CONSTANTS
# =============================================================================

# Text processing mode determines how mapped text is inserted into documents
TEXT_MODE = PROCESSING_MODES['APPEND']  # Default text processing mode: "append" or "replace"

# Text processing separator for append mode
TEXT_SEPARATOR = DEFAULT_SEPARATOR  # Default separator between original and appended text in append mode

# Default mapping for patterns not found in mapping.json
# Use imported constant instead of redefining
# DEFAULT_MAPPING is imported from shared_constants

# OCR processing mode determines how OCR text is handled in images
OCR_MODE = PROCESSING_MODES['APPEND']  # Default OCR processing mode: "replace", "append", or "append-image"

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

# Default suffix for processed files
DEFAULT_SUFFIX = "_12NC_processed"  # Default suffix to append to processed file names

# =============================================================================
# YAML CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigManager:
    """
    Centralized configuration management with YAML support.
    
    This class provides:
    - YAML configuration loading and validation
    - Default configuration fallbacks
    - Configuration merging and overrides
    - Environment variable support
    - Configuration validation and error handling
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file. If None, uses default.
        """
        self.config_file = Path(config_file) if config_file else Path("config/default.yaml")
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file with fallbacks."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logging.info(f"Loaded configuration from {self.config_file}")
            else:
                logging.warning(f"Configuration file {self.config_file} not found, using defaults")
                self._config = {}
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self._config = {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_prefix = "DOC_PROCESSING_"
        
        # Common environment variable mappings
        env_mappings = {
            f"{env_prefix}LOG_LEVEL": ("logging", "level"),
            f"{env_prefix}MAX_FILE_SIZE": ("processing", "max_file_size_mb"),
            f"{env_prefix}TIMEOUT": ("processing", "timeout_seconds"),
            f"{env_prefix}MAX_WORKERS": ("performance", "max_workers"),
            f"{env_prefix}USE_GPU": ("ocr", "use_gpu"),
            f"{env_prefix}SOURCE_DIR": ("paths", "source_dir"),
            f"{env_prefix}OUTPUT_DIR": ("paths", "output_dir"),
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert string values to appropriate types
                if config_path[1] in ["max_file_size_mb", "timeout_seconds", "max_workers"]:
                    value = float(value) if "." in value else int(value)
                elif config_path[1] == "use_gpu":
                    value = value.lower() in ("true", "1", "yes", "on")
                
                self._set_nested_value(config_path, value)
                logging.info(f"Applied environment override: {env_var}={value}")
    
    def _set_nested_value(self, path: tuple, value: Any):
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self):
        """Validate configuration values."""
        validations = [
            (("logging", "level"), str, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            (("processing", "max_file_size_mb"), (int, float), lambda x: x > 0),
            (("processing", "timeout_seconds"), int, lambda x: x > 0),
            (("performance", "max_workers"), int, lambda x: 1 <= x <= 32),
            (("ocr", "confidence_min"), float, lambda x: 0.0 <= x <= 1.0),
        ]
        
        for path, expected_type, validator in validations:
            try:
                value = self.get_nested_value(path)
                if value is not None:
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Invalid type for {'.'.join(path)}: expected {expected_type}")
                    if callable(validator) and not validator(value):
                        raise ValueError(f"Invalid value for {'.'.join(path)}: {value}")
                    elif isinstance(validator, list) and value not in validator:
                        raise ValueError(f"Invalid value for {'.'.join(path)}: {value} (must be one of {validator})")
            except Exception as e:
                logging.warning(f"Configuration validation warning: {e}")
    
    def get_nested_value(self, path: tuple, default: Any = None) -> Any:
        """Get a nested configuration value."""
        current = self._config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get("logging", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self._config.get("processing", {})
    
    def get_text_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration."""
        return self._config.get("text_processing", {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration."""
        return self._config.get("ocr", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self._config.get("performance", {})
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self._config.get("error_handling", {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get("paths", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self._config.get("system", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self._config.get("monitoring", {})
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self._config.copy()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Timeout settings for document processing
PROCESSING_TIMEOUT = 300  # Maximum time (seconds) to process a single document

# Retry settings for failed operations
MAX_RETRIES = 3  # Maximum number of retry attempts for failed operations

# OCR confidence threshold for text detection (source of truth from shared constants; overridden by CLI if provided)
CONFIDENCE_MIN = DEFAULT_OCR_CONFIDENCE  # Minimum confidence score (0.0-1.0) for OCR text detection

# File size limits for processing
MIN_FILE_SIZE_MB = 0.1  # Minimum file size in MB for processing
MAX_FILE_SIZE_MB = 300.0  # Maximum file size in MB for processing (increased from 100MB to support large files)

# File size thresholds for parser selection
LARGE_FILE_THRESHOLD_MB = 100.0  # Threshold for large file optimizations
VERY_LARGE_FILE_THRESHOLD_MB = 200.0  # Threshold for very large file optimizations

# =============================================================================
# CLI FLAG DEFINITIONS - Essential only
# =============================================================================

# Dictionary defining all command-line interface arguments
# Each flag contains type, choices, default value, and help text
CLI_FLAGS = {
    # Text processing configuration
    '--text-mode': {
        'type': str,
        'choices': [mode.value for mode in ProcessingMode],
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
        'choices': [mode.value for mode in OCRMode],
        'default': OCR_MODE,
        'help': 'OCR processing mode: replace (exact position), append (top-bottom text), append-image (new image after original)'
    },
    
    # OCR engine selection
    '--ocr-engine': {
        'type': str,
        'choices': ['easyocr', 'tesseract', 'hybrid'],
        'default': 'hybrid',
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
    },
    '--suffix': {
        'type': str,
        'default': DEFAULT_SUFFIX,
        'help': f'Suffix to append to processed file names (default: {DEFAULT_SUFFIX})'
    }
}

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class ProcessingConfig:
    """
    Unified configuration class for document processing pipeline with YAML support.
    
    This dataclass centralizes all configuration settings and provides:
    - YAML configuration loading and integration
    - Default values for all processing parameters
    - Automatic validation of configuration values
    - GPU detection and configuration
    - Path validation and directory creation
    - Centralized logging configuration
    - Error handling configuration integration
    
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
        suffix: Suffix to append to processed file names
        processing_timeout: Maximum time to process a document
        max_retries: Maximum retry attempts for failed operations
        config_manager: YAML configuration manager instance
    """
    
    # Processing mode settings
    text_mode: str = TEXT_MODE
    text_separator: str = TEXT_SEPARATOR
    default_mapping: str = DEFAULT_MAPPING
    ocr_mode: str = OCR_MODE
    ocr_engine: str = "hybrid"
    
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
    
    # File naming configuration
    suffix: str = DEFAULT_SUFFIX  # Suffix to append to processed file names
    
    # Processing settings
    processing_timeout: int = PROCESSING_TIMEOUT
    max_retries: int = MAX_RETRIES
    
    # File size limits
    min_file_size: float = MIN_FILE_SIZE_MB
    max_file_size: float = MAX_FILE_SIZE_MB
    
    # YAML configuration manager
    config_manager: Optional[ConfigManager] = field(default_factory=get_config_manager)
    
    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        This method is automatically called after the dataclass is initialized.
        It performs:
        - YAML configuration loading and integration
        - Mode validation (text_mode, ocr_mode)
        - Path validation and directory creation
        - GPU availability detection
        """
        # Load YAML configuration values
        self._load_yaml_config()
        
        self._validate_modes()
        self._validate_paths()
        self._detect_gpu_availability()
    
    def _load_yaml_config(self):
        """Load configuration values from YAML file."""
        if not self.config_manager:
            return
        
        # Load text processing configuration
        text_config = self.config_manager.get_text_processing_config()
        if text_config:
            self.text_mode = text_config.get("mode", self.text_mode)
            self.text_separator = text_config.get("separator", self.text_separator)
            self.default_mapping = text_config.get("default_mapping", self.default_mapping)
        
        # Load OCR configuration
        ocr_config = self.config_manager.get_ocr_config()
        if ocr_config:
            self.ocr_mode = ocr_config.get("mode", self.ocr_mode)
            self.ocr_engine = ocr_config.get("engine", self.ocr_engine)
            self.confidence_min = ocr_config.get("confidence_min", self.confidence_min)
            self.use_gpu = ocr_config.get("use_gpu", self.use_gpu)
        
        # Load processing configuration
        processing_config = self.config_manager.get_processing_config()
        if processing_config:
            self.processing_timeout = processing_config.get("timeout_seconds", self.processing_timeout)
            self.max_retries = processing_config.get("retry_attempts", self.max_retries)
            self.max_file_size = processing_config.get("max_file_size_mb", self.max_file_size)
            self.min_file_size = processing_config.get("min_file_size_mb", self.min_file_size)
        
        # Load performance configuration
        performance_config = self.config_manager.get_performance_config()
        if performance_config:
            self.max_workers = performance_config.get("max_workers", self.max_workers)
        
        # Load paths configuration
        paths_config = self.config_manager.get_paths_config()
        if paths_config:
            self.source_dir = Path(paths_config.get("source_dir", self.source_dir))
            self.output_dir = Path(paths_config.get("output_dir", self.output_dir))
            self.reports_dir = Path(paths_config.get("reports_dir", self.reports_dir))
            self.logs_dir = Path(paths_config.get("logs_dir", self.logs_dir))
            self.patterns_file = Path(paths_config.get("patterns_file", self.patterns_file))
            self.mappings_file = Path(paths_config.get("mappings_file", self.mappings_file))
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration from YAML."""
        if self.config_manager:
            return self.config_manager.get_logging_config()
        return {}
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration from YAML."""
        if self.config_manager:
            return self.config_manager.get_error_handling_config()
        return {}
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration from YAML."""
        if self.config_manager:
            return self.config_manager.get_monitoring_config()
        return {}
    
    def _validate_modes(self):
        """
        Validate processing modes to ensure they are valid.
        
        Raises:
            ValueError: If text_mode or ocr_mode contains invalid values
        """
        # Use enum values for validation
        valid_text_modes = [mode.value for mode in ProcessingMode]
        valid_ocr_modes = [mode.value for mode in OCRMode]
        
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
            logs_dir=Path(args.logs_dir),
            suffix=args.suffix
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


def setup_logging_from_config(config: ProcessingConfig):
    """
    Setup logging using YAML configuration.
    
    Args:
        config: ProcessingConfig instance with YAML configuration
    """
    logging_config = config.get_logging_config()
    if not logging_config:
        return
    
    # Configure root logger
    log_level = getattr(logging, logging_config.get("level", "INFO").upper())
    logging.basicConfig(
        level=log_level,
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    )
    
    # Configure file handler if specified
    log_file = logging_config.get("file")
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=logging_config.get("max_size_mb", 10) * 1024 * 1024,
            backupCount=logging_config.get("backup_count", 5)
        )
        file_handler.setFormatter(logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Configure console handler if enabled
    if logging_config.get("console_enabled", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, logging_config.get("console_level", "INFO").upper()))
        console_handler.setFormatter(logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ))
        logging.getLogger().addHandler(console_handler)
    
    # Configure performance logging if enabled
    if logging_config.get("performance_logging", False):
        perf_log_file = logging_config.get("performance_log_file")
        if perf_log_file:
            perf_handler = logging.FileHandler(perf_log_file)
            perf_handler.setLevel(logging.INFO)
            perf_logger = logging.getLogger("performance")
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
    
    logging.info("Logging configured from YAML configuration")