"""
Configuration module for the Document Processing Pipeline.
Centralizes all configuration constants and CLI argument definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import argparse
import sys
import torch

# Processing Configuration Constants
TEXT_MODE = "append"  # Default text processing mode
OCR_MODE = "replace"  # Default OCR processing mode
USE_GPU = True  # Default GPU usage setting
MAX_WORKERS = max(1, min((__import__('psutil').cpu_count(logical=True) or 8) - 2, 8))

# Directory Configuration
SOURCE_DIR = "source_documents"
OUTPUT_DIR = "processed"
REPORTS_DIR = "reports"
LOGS_DIR = "logs"

# File Configuration
PATTERNS_FILE = "patterns.json"
MAPPINGS_FILE = "mapping.json"

# Processing Configuration
PROCESSING_TIMEOUT = 300  # 5 minutes per document
MAX_RETRIES = 3
CONFIDENCE_MIN = 0.7

# CLI Flag Definitions
CLI_FLAGS = {
    '--text-mode': {
        'type': str,
        'choices': ['append', 'replace'],
        'default': TEXT_MODE,
        'help': 'Text processing mode: append mapped text after matched text or replace matched text'
    },
    '--ocr-mode': {
        'type': str,
        'choices': ['replace', 'append'],
        'default': OCR_MODE,
        'help': 'OCR processing mode: replace text in-place or append both original and mapped text'
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
    }
}

@dataclass
class ProcessingConfig:
    """Configuration class for document processing pipeline."""
    text_mode: str = TEXT_MODE
    ocr_mode: str = OCR_MODE
    use_gpu: bool = USE_GPU
    max_workers: int = MAX_WORKERS
    confidence_min: float = CONFIDENCE_MIN
    verbose: bool = False
    
    # File paths
    patterns_file: Path = Path(PATTERNS_FILE)
    mappings_file: Path = Path(MAPPINGS_FILE)
    source_dir: Path = Path(SOURCE_DIR)
    output_dir: Path = Path(OUTPUT_DIR)
    reports_dir: Path = Path(REPORTS_DIR)
    logs_dir: Path = Path(LOGS_DIR)
    
    # Processing settings
    processing_timeout: int = PROCESSING_TIMEOUT
    max_retries: int = MAX_RETRIES
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_modes()
        self._validate_paths()
        self._detect_gpu_availability()
    
    def _validate_modes(self):
        """Validate processing modes."""
        valid_text_modes = ["append", "replace"]
        valid_ocr_modes = ["replace", "append"]
        
        if self.text_mode not in valid_text_modes:
            raise ValueError(f"Invalid text_mode '{self.text_mode}'. Must be one of {valid_text_modes}")
        
        if self.ocr_mode not in valid_ocr_modes:
            raise ValueError(f"Invalid ocr_mode '{self.ocr_mode}'. Must be one of {valid_ocr_modes}")
    
    def _validate_paths(self):
        """Validate and create necessary paths."""
        # Create output directories
        for directory in [self.output_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate input files exist
        if not self.patterns_file.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_file}")
        
        if not self.mappings_file.exists():
            raise FileNotFoundError(f"Mappings file not found: {self.mappings_file}")
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
    
    def _detect_gpu_availability(self):
        """Detect and configure GPU availability."""
        if self.use_gpu:
            try:
                # Check for CUDA
                if torch.cuda.is_available():
                    self.gpu_device = "cuda"
                    self.gpu_available = True
                # Check for MPS (Apple Silicon)
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
    """Create argument parser with all CLI flags from configuration."""
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline - Process Word documents with text replacements, graphics processing, and OCR-based image text replacement"
    )
    
    # Add all CLI flags from configuration
    for flag, config in CLI_FLAGS.items():
        parser.add_argument(flag, **config)
    
    return parser

def load_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """Load configuration from parsed command line arguments."""
    # Handle GPU flags
    use_gpu = USE_GPU
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    try:
        return ProcessingConfig(
            text_mode=args.text_mode,
            ocr_mode=args.ocr_mode,
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
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

def get_default_config() -> ProcessingConfig:
    """Get default configuration without CLI arguments."""
    return ProcessingConfig()