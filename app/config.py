from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import argparse
import json
import sys

@dataclass
class OCRConfig:
    """
    Dataclass for storing configuration options for the OCR DOCX Text Replacer pipeline.
    Only supports text-mode and ocr-mode with append/replace values as per functional requirements.
    """
    patterns_path: Path = Path("patterns.json")
    mapping_path: Path = Path("mapping.json")
    processed_dir: Path = Path("./processed")
    reports_dir: Path = Path("./reports")
    logs_dir: Path = Path("./logs")

    ocr_mode: str = "replace"  # "replace", "append", or "append-image" allowed
    text_mode: str = "append"  # Only "append" or "replace" allowed
    ocr_engine: str = "easyocr"  # "easyocr", "tesseract", or "hybrid"
    workers: int = max(1, min((__import__('psutil').cpu_count(logical=True) or 8) - 2, 8))
    gpu: bool = True
    verbose: bool = False
    confidence_min: float = 0.4
    
    # Phase control flags
    process_text_replacements: bool = True
    process_image_replacements: bool = True
    process_textbox_replacements: bool = True
    process_section_replacements: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_modes()
        self._validate_paths()
    
    def _validate_modes(self):
        """Validate that only supported modes are used."""
        valid_text_modes = ["append", "replace"]
        valid_ocr_modes = ["replace", "append", "append-image"]
        
        if self.text_mode not in valid_text_modes:
            raise ValueError(
                f"Invalid text_mode '{self.text_mode}'. "
                f"Only {valid_text_modes} are supported as per functional requirements."
            )
        
        if self.ocr_mode not in valid_ocr_modes:
            raise ValueError(
                f"Invalid ocr_mode '{self.ocr_mode}'. "
                f"Only {valid_ocr_modes} are supported. "
                f"Modes: 'replace' (exact position replacement), "
                f"'append' (two-line text), 'append-image' (new image after original)."
            )
    
    def _validate_paths(self):
        """Validate that required paths exist or can be created."""
        # Validate input files exist
        if not self.patterns_path.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_path}")
        
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
        
        # Create output directories if they don't exist
        for directory in [self.processed_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_cli_args() -> "OCRConfig":
        """
        Parse command-line arguments and return an OCRConfig instance.
        Also validates the resulting paths.
        """
        parser = argparse.ArgumentParser(description="OCR DOCX Text Replacer")
        parser.add_argument("--patterns", type=str, default="patterns.json", help="Path to regex patterns list JSON")
        parser.add_argument("--mapping", type=str, default="mapping.json", help="Path to mapping JSON (old->new replacements)")
        parser.add_argument("--processed-dir", type=str, default="./processed", help="Directory for processed files")
        parser.add_argument("--reports-dir", type=str, default="./reports", help="Directory for reports")
        parser.add_argument("--logs-dir", type=str, default="./logs", help="Directory for logs")

        parser.add_argument("--ocr-mode", type=str, choices=["replace", "append", "append-image"], default="replace", help="OCR processing mode: 'replace' (exact position), 'append' (two-line text), 'append-image' (new image after original)")
        parser.add_argument("--text-mode", type=str, choices=["replace", "append"], default="append", help="Text processing mode: replace original text or append to it")
        parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract", "hybrid"], default="easyocr")
        parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
        parser.add_argument("--gpu", action="store_true", help="Enable GPU for OCR if available")
        parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for OCR")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        parser.add_argument("--confidence-min", type=float, default=0.7, help="Minimum OCR confidence for replacement")
        
        # Phase control arguments
        parser.add_argument("--text-replacements", action="store_true", default=True, help="Enable text replacements (body text)")
        parser.add_argument("--no-text-replacements", action="store_true", help="Disable text replacements")
        parser.add_argument("--image-replacements", action="store_true", default=True, help="Enable image OCR replacements")
        parser.add_argument("--no-image-replacements", action="store_true", help="Disable image OCR replacements")
        parser.add_argument("--textbox-replacements", action="store_true", default=True, help="Enable textbox/callout replacements")
        parser.add_argument("--no-textbox-replacements", action="store_true", help="Disable textbox/callout replacements")
        parser.add_argument("--section-replacements", action="store_true", default=True, help="Enable header/footer/section replacements")
        parser.add_argument("--no-section-replacements", action="store_true", help="Disable header/footer/section replacements")
        
        args = parser.parse_args()

        # Workers logic
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True) or 8
        except ImportError:
            cpu_count = 8
        workers = args.workers if args.workers is not None else max(1, min(cpu_count - 2, 8))
        gpu = args.gpu or not args.no_gpu

        # Create OCRConfig instance - validation will be handled by __post_init__
        try:
            # Handle phase control flags
            process_text_replacements = args.text_replacements and not args.no_text_replacements
            process_image_replacements = args.image_replacements and not args.no_image_replacements
            process_textbox_replacements = args.textbox_replacements and not args.no_textbox_replacements
            process_section_replacements = args.section_replacements and not args.no_section_replacements
            
            return OCRConfig(
                patterns_path=Path(args.patterns),
                mapping_path=Path(args.mapping),
                processed_dir=Path(args.processed_dir),
                reports_dir=Path(args.reports_dir),
                logs_dir=Path(args.logs_dir),

                ocr_mode=args.ocr_mode,
                text_mode=args.text_mode,
                ocr_engine=args.ocr_engine,
                workers=workers,
                gpu=gpu,
                verbose=args.verbose,
                confidence_min=args.confidence_min,
                process_text_replacements=process_text_replacements,
                process_image_replacements=process_image_replacements,
                process_textbox_replacements=process_textbox_replacements,
                process_section_replacements=process_section_replacements
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(2)