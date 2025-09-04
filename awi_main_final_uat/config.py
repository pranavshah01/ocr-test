
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sys
import torch
from enum import Enum
import logging
from typing import Dict, Any, Optional, Union
import os

DEFAULT_FONT_SIZE = 12.0
DEFAULT_FONT_FAMILY = "Arial"

PROCESSING_MODES = {
    'APPEND': 'append',
    'REPLACE': 'replace'
}

DEFAULT_SEPARATOR = ";"
DEFAULT_MAPPING = "4022-NA"
DEFAULT_OCR_CONFIDENCE = 0.4

# OCR Language Configuration
# EasyOCR has language compatibility restrictions - Chinese languages must be paired with English
EASYOCR_LANGUAGES = ['en', 'ch_sim']  # English + Simplified Chinese (compatible pair)
EASYOCR_LANGUAGES_ALT = ['en', 'ch_tra']  # English + Traditional Chinese (compatible pair)
TESSERACT_LANGUAGES = ['eng', 'chi_sim', 'chi_tra']  # English, Simplified Chinese, Traditional Chinese

DEFAULT_FONT_WIDTH_MULTIPLIER = 0.6
DEFAULT_LINE_HEIGHT_MULTIPLIER = 1.25
DEFAULT_CONSERVATIVE_HEADROOM = 0.96
DEFAULT_SAFETY_MARGIN = 0.15
MIN_FONT_SIZE = 6.0

POINTS_PER_INCH = 72.0
CM_PER_INCH = 2.54

# Font metrics moved to font_guidelines.json; defaults kept above for fallback

class ProcessingMode(Enum):
    APPEND = "append"
    REPLACE = "replace"

class OCRMode(Enum):
    REPLACE = "replace"
    APPEND = "append"
    APPEND_IMAGE = "append-image"

TEXT_MODE = PROCESSING_MODES['APPEND']
TEXT_SEPARATOR = DEFAULT_SEPARATOR
OCR_MODE = PROCESSING_MODES['APPEND']
USE_GPU = True
MAX_WORKERS = max(1, min((__import__('psutil').cpu_count(logical=True) or 8) - 2, 8))

SOURCE_DIR = "source_documents"
OUTPUT_DIR = "processed"
COMPLETE_DIR = "../complete"
REPORTS_DIR = "reports"
LOGS_DIR = "logs"

PATTERNS_FILE = "patterns.json"
MAPPINGS_FILE = "mapping.json"
DEFAULT_SUFFIX = "_12NC_processed"

PROCESSING_TIMEOUT = 300
MAX_RETRIES = 3
CONFIDENCE_MIN = DEFAULT_OCR_CONFIDENCE
MIN_FILE_SIZE_MB = 0.0
MAX_FILE_SIZE_MB = 300.0
LARGE_FILE_THRESHOLD_MB = 100.0
VERY_LARGE_FILE_THRESHOLD_MB = 200.0

CLI_FLAGS = {
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
    '--ocr-mode': {
        'type': str,
        'choices': [mode.value for mode in OCRMode],
        'default': OCR_MODE,
        'help': 'OCR processing mode: replace (exact position), append (top-bottom text), append-image (new image after original)'
    },
    '--ocr-engine': {
        'type': str,
        'choices': ['easyocr', 'tesseract', 'hybrid'],
        'default': 'hybrid',
        'help': 'OCR engine to use: easyocr, tesseract, or hybrid (combines both engines)'
    },
    '--gpu': {
        'action': 'store_true',
        'help': 'Enable GPU acceleration for OCR processing'
    },
    '--no-gpu': {
        'action': 'store_true',
        'help': 'Disable GPU acceleration for OCR processing'
    },
    '--workers': {
        'type': int,
        'default': MAX_WORKERS,
        'help': f'Number of parallel workers (default: {MAX_WORKERS})'
    },
    '--confidence-min': {
        'type': float,
        'default': CONFIDENCE_MIN,
        'help': f'Minimum OCR confidence threshold (default: {CONFIDENCE_MIN})'
    },

    '--max-file-size': {
        'type': float,
        'default': MAX_FILE_SIZE_MB,
        'help': f'Maximum file size in MB (default: {MAX_FILE_SIZE_MB})'
    },
    '--large-file-threshold': {
        'type': float,
        'default': LARGE_FILE_THRESHOLD_MB,
        'help': f'Large file threshold in MB (default: {LARGE_FILE_THRESHOLD_MB})'
    },
    '--very-large-file-threshold': {
        'type': float,
        'default': VERY_LARGE_FILE_THRESHOLD_MB,
        'help': f'Very large file threshold in MB (default: {VERY_LARGE_FILE_THRESHOLD_MB})'
    },
    '--verbose': {
        'action': 'store_true',
        'help': 'Enable verbose logging output'
    },
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
    
    # Font calculation system selection removed - using new system only
    '--suffix': {
        'type': str,
        'default': DEFAULT_SUFFIX,
        'help': f'Suffix to append to processed file names (default: {DEFAULT_SUFFIX})'
    }
}

@dataclass
class ProcessingConfig:
    text_mode: str = TEXT_MODE
    text_separator: str = TEXT_SEPARATOR
    default_mapping: str = DEFAULT_MAPPING
    ocr_mode: str = OCR_MODE
    ocr_engine: str = "hybrid"
    use_gpu: bool = USE_GPU
    max_workers: int = MAX_WORKERS
    confidence_min: float = CONFIDENCE_MIN
    verbose: bool = False
    gpu_device: str = "cpu"
    
    # Font calculation system - using new system only
    gpu_available: bool = False
    patterns_file: Path = Path(PATTERNS_FILE)
    mappings_file: Path = Path(MAPPINGS_FILE)
    source_dir: Path = Path(SOURCE_DIR)
    output_dir: Path = Path(OUTPUT_DIR)
    complete_dir: Path = Path(COMPLETE_DIR)
    reports_dir: Path = Path(REPORTS_DIR)
    logs_dir: Path = Path(LOGS_DIR)
    suffix: str = DEFAULT_SUFFIX
    processing_timeout: int = PROCESSING_TIMEOUT
    max_retries: int = MAX_RETRIES
    min_file_size: float = MIN_FILE_SIZE_MB
    max_file_size: float = MAX_FILE_SIZE_MB
    large_file_threshold: float = LARGE_FILE_THRESHOLD_MB
    very_large_file_threshold: float = VERY_LARGE_FILE_THRESHOLD_MB

    def __post_init__(self):
        self._validate_modes()
        self._validate_paths()
        self._detect_gpu_availability()

    def _load_yaml_config(self):
        return

    def get_logging_config(self) -> Dict[str, Any]:
        return {}

    def get_error_handling_config(self) -> Dict[str, Any]:
        return {}

    def get_monitoring_config(self) -> Dict[str, Any]:
        return {}

    def _validate_modes(self):
        valid_text_modes = [mode.value for mode in ProcessingMode]
        valid_ocr_modes = [mode.value for mode in OCRMode]

        if self.text_mode not in valid_text_modes:
            raise ValueError(f"Invalid text_mode '{self.text_mode}'. Must be one of {valid_text_modes}")

        if self.ocr_mode not in valid_ocr_modes:
            raise ValueError(f"Invalid ocr_mode '{self.ocr_mode}'. Must be one of {valid_ocr_modes}")

    def _validate_paths(self):
        for directory in [self.output_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        if not self.patterns_file.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_file}")

        if not self.mappings_file.exists():
            raise FileNotFoundError(f"Mappings file not found: {self.mappings_file}")

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

    def _detect_gpu_availability(self):
        if self.use_gpu:
            try:
                if torch.cuda.is_available():
                    self.gpu_device = "cuda"
                    self.gpu_available = True
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.gpu_device = "mps"
                    self.gpu_available = True
                else:
                    self.gpu_device = "cpu"
                    self.gpu_available = False
                    if self.verbose:
                        print("GPU requested but not available, falling back to CPU")
            except Exception as e:
                self.gpu_device = "cpu"
                self.gpu_available = False
                if self.verbose:
                    print(f"GPU detection failed: {e}, falling back to CPU")
        else:
            self.gpu_device = "cpu"
            self.gpu_available = False

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal Document Processing Pipeline - Process Word documents with text replacements and OCR-based image text replacement"
    )

    for flag, config in CLI_FLAGS.items():
        parser.add_argument(flag, **config)

    return parser

def load_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    use_gpu = USE_GPU
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False

    try:
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


            max_file_size=args.max_file_size,
            large_file_threshold=args.large_file_threshold,
            very_large_file_threshold=args.very_large_file_threshold,
            patterns_file=Path(args.patterns),
            mappings_file=Path(args.mapping),
            source_dir=Path(args.source_dir),
            output_dir=Path(args.output_dir),
            reports_dir=Path(args.reports_dir),
            logs_dir=Path(args.logs_dir),
            suffix=args.suffix
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

def get_default_config() -> ProcessingConfig:
    return ProcessingConfig()

def setup_logging_from_config(config: ProcessingConfig):
    logging_config = {}

    log_level = getattr(logging, logging_config.get("level", "INFO").upper())
    logging.basicConfig(
        level=log_level,
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    )

    log_file = logging_config.get("file")
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=logging_config.get("max_size_mb", 10) * 1024 * 1024,
            backupCount=logging_config.get("backup_count", 5),
            encoding='utf-8'  # Fix for Windows charset issues
        )
        file_handler.setFormatter(logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ))
        logging.getLogger().addHandler(file_handler)

    if logging_config.get("console_enabled", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, logging_config.get("console_level", "INFO").upper()))
        console_handler.setFormatter(logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ))
        logging.getLogger().addHandler(console_handler)

    if logging_config.get("performance_logging", False):
        perf_log_file = logging_config.get("performance_log_file")
        if perf_log_file:
            perf_handler = logging.FileHandler(perf_log_file, encoding='utf-8')  # Fix for Windows charset issues
            perf_handler.setLevel(logging.INFO)
            perf_logger = logging.getLogger("performance")
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)

    logging.info("Logging configured from YAML configuration")