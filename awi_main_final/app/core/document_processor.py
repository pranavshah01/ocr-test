"""
Minimal DocumentProcessor initialization for awi_main_final.

This implementation intentionally does NOT perform text/graphics/image processing yet.
It only exposes a conversion stub so the pipeline can be wired in phases.
"""

from pathlib import Path
from typing import Optional, Tuple
import subprocess
import shutil
import logging
from config import (
    MAX_FILE_SIZE_MB, 
    LARGE_FILE_THRESHOLD_MB, 
    VERY_LARGE_FILE_THRESHOLD_MB
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Lightweight document processor scaffold.

    Responsibilities (current phase):
    - Initialize processor context based on config
    - Provide a conversion stub (doc -> docx) API; not implemented yet
    - Do NOT do any parsing/ocr; reserved for future phases
    - Detect file size and select appropriate parser
    - Update ProcessingResult.parser field
    - Handle "AttValue too large" errors with parser fallback
    """

    def __init__(self, config) -> None:
        self.config = config
        self.initialized = False
        self.text_processor = None

    def initialize(self) -> None:
        """Initialize the processor context. Safe to call multiple times."""
        # Initialize text processor
        try:
            from app.processors.text_processor import create_text_processor
            self.text_processor = create_text_processor(self.config)
            self.text_processor.initialize()
            logger.info("Text processor initialized in document processor")
        except Exception as e:
            logger.error(f"Failed to initialize text processor: {e}")
            self.text_processor = None
        
        self.initialized = True

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.text_processor:
            try:
                self.text_processor.cleanup()
                logger.info("Text processor cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up text processor: {e}")
        self.text_processor = None
        self.initialized = False

    def detect_and_set_parser(self, file_path: Path) -> str:
        """
        Detect file size and determine appropriate parser for the document.
        
        This method implements a tiered parser selection approach:
        1. Standard parser for files < 100MB
        2. Enhanced parser for files 100MB - 200MB  
        3. Custom parser for files > 200MB
        4. Fallback mechanisms for each tier
        
        Args:
            file_path: Path to the document file
            
        Returns:
            String indicating the parser type to be used
        """
        try:
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Detecting parser for document: {size_mb:.1f}MB - {file_path.name}")
            
            # Check if file size exceeds maximum supported size
            if size_mb > MAX_FILE_SIZE_MB:
                error_msg = f"File size {size_mb:.1f}MB exceeds maximum supported size of {MAX_FILE_SIZE_MB}MB"
                logger.error(error_msg)
                return "unsupported_size"
            
            # Tier 1: Standard parser for files < 100MB
            if size_mb < LARGE_FILE_THRESHOLD_MB:
                logger.info(f"Selected standard parser for {size_mb:.1f}MB file ({file_path.name})")
                return "standard"
            
            # Tier 2: Enhanced parser for files 100MB - 200MB
            if size_mb < VERY_LARGE_FILE_THRESHOLD_MB:
                logger.info(f"Selected enhanced parser for {size_mb:.1f}MB file ({file_path.name})")
                return "enhanced"
            
            # Tier 3: Custom parser for files > 200MB
            if size_mb >= VERY_LARGE_FILE_THRESHOLD_MB:
                logger.info(f"Selected custom parser for {size_mb:.1f}MB file ({file_path.name})")
                return "custom"
            
            # Fallback: Standard parser
            logger.info(f"Using standard parser as fallback for {size_mb:.1f}MB file ({file_path.name})")
            return "standard"
            
        except Exception as e:
            logger.error(f"Error detecting parser for {file_path}: {e}")
            return "error"

    def is_attvalue_too_large_error(self, error_message: str) -> bool:
        """
        Check if the error message indicates an "AttValue too large" error.
        
        Args:
            error_message: The error message to check
            
        Returns:
            True if it's an AttValue too large error, False otherwise
        """
        error_lower = error_message.lower()
        attvalue_indicators = [
            "attvalue too large",
            "attribute value too large",
            "xml attribute too large",
            "lxml attribute value limit",
            "huge_tree",
            "xml parsing error"
        ]
        
        return any(indicator in error_lower for indicator in attvalue_indicators)

    def get_next_parser(self, current_parser: str) -> Optional[str]:
        """
        Get the next parser to try in the fallback sequence.
        
        Args:
            current_parser: Current parser that failed
            
        Returns:
            Next parser to try, or None if no more parsers available
        """
        parser_sequence = ["standard", "enhanced", "custom"]
        
        try:
            current_index = parser_sequence.index(current_parser)
            if current_index + 1 < len(parser_sequence):
                next_parser = parser_sequence[current_index + 1]
                logger.info(f"Falling back from {current_parser} to {next_parser} parser")
                return next_parser
        except ValueError:
            logger.warning(f"Unknown parser type: {current_parser}")
        
        return None

    def try_parser_with_fallback(self, file_path: Path, initial_parser: str = None) -> Tuple[str, Optional[str]]:
        """
        Try parsing with fallback logic for "AttValue too large" errors.
        
        This method implements reactive fallback:
        1. Start with the detected parser (or provided initial parser)
        2. If "AttValue too large" error occurs, try the next parser
        3. Continue until success or all parsers exhausted
        
        Args:
            file_path: Path to the document file
            initial_parser: Optional initial parser to start with
            
        Returns:
            Tuple of (final_parser, error_message)
        """
        if initial_parser is None:
            initial_parser = self.detect_and_set_parser(file_path)
        
        current_parser = initial_parser
        last_error = None
        
        logger.info(f"Starting parser fallback sequence with {current_parser} parser for {file_path.name}")
        
        while current_parser:
            try:
                # Future: Actually try to load the document with the current parser
                # For now, we'll simulate the behavior
                logger.debug(f"Attempting to load with {current_parser} parser")
                
                # Simulate potential "AttValue too large" error for testing
                # In real implementation, this would be the actual document loading
                if current_parser == "standard" and self._should_simulate_attvalue_error(file_path):
                    raise Exception("AttValue too large: XML attribute exceeds size limit")
                
                # If we get here, the parser worked
                logger.info(f"Successfully loaded document with {current_parser} parser")
                return current_parser, None
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"{current_parser} parser failed: {last_error}")
                
                # Check if this is an "AttValue too large" error
                if self.is_attvalue_too_large_error(last_error):
                    logger.info(f"Detected AttValue too large error, attempting fallback")
                    current_parser = self.get_next_parser(current_parser)
                else:
                    # Not an AttValue error, don't fallback
                    logger.error(f"Non-AttValue error encountered, stopping fallback: {last_error}")
                    break
        
        # All parsers failed
        logger.error(f"All parsers failed for {file_path.name}. Last error: {last_error}")
        return "failed", last_error

    def _should_simulate_attvalue_error(self, file_path: Path) -> bool:
        """
        Simulate "AttValue too large" error for testing purposes.
        In real implementation, this would be removed.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if we should simulate the error (for testing)
        """
        # Simulate error for files between 50-80MB to test fallback
        try:
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            return 50 <= size_mb <= 80
        except:
            return False

    def prepare_for_large_file(self, size_mb: float) -> None:
        """
        Prepare system for processing large files.
        
        Args:
            size_mb: File size in MB
        """
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            logger.info(f"Preparing system for large file processing: {size_mb:.1f}MB")
            # Future: Implement memory management, garbage collection, etc.
            # For now, just log the preparation

    def prepare_for_processing(self, file_path: Path) -> None:
        """Prepare the processor for handling a specific file."""
        # For now, just log the preparation
        logger.info(f"Preparing to process: {file_path}")

    def process_document_text(self, document, processing_result) -> bool:
        """
        Process document text using the text processor if available.
        
        Args:
            document: The document to process
            processing_result: ProcessingResult object to update
            
        Returns:
            True if processing was successful, False otherwise
        """
        if not self.text_processor:
            logger.warning("Text processor not available, skipping text processing")
            return False
        
        try:
            logger.info("Processing document text with text processor...")
            
            # Process the document with text processor
            updated_result = self.text_processor.process_document(document, processing_result)
            
            if updated_result:
                logger.info(f"Text processing completed: {updated_result.matches_found} matches found")
                return True
            else:
                logger.error("Text processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during text processing: {e}")
            return False

    def convert_if_needed(self, src_path: Path) -> Tuple[bool, Optional[Path], str]:
        """Convert legacy .doc to .docx in-place (keeps .docx in same dir).

        Also moves the original .doc to an 'orig_doc_files' folder under the
        source directory after successful conversion.

        Returns:
            (success, output_path, message)
        """
        suffix = src_path.suffix.lower()
        if suffix == ".docx":
            return True, src_path, "Source is already .docx"
        if suffix != ".doc":
            return False, None, f"Unsupported input type: {suffix}"

        parent = src_path.parent
        target = parent / f"{src_path.stem}.docx"
        try:
            # Resolve LibreOffice binary
            soffice = shutil.which("soffice")
            if not soffice:
                mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
                if Path(mac_path).exists():
                    soffice = mac_path
            if not soffice:
                return False, target, "LibreOffice (soffice) not found; cannot convert .doc"

            # Use LibreOffice for conversion
            cmd = [
                soffice,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(parent),
                str(src_path)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                return False, target, f"Conversion failed: {proc.stderr.strip() or proc.stdout.strip()}"
            if not target.exists():
                return False, target, "Conversion reported success but target .docx not found"

            # Move the original .doc into a top-level 'orig_doc_files' under the source root
            # Determine source root as the configured source_dir (keeps folder from nesting)
            try:
                source_root = Path(self.config.source_dir)
            except Exception:
                source_root = parent
            archive_dir = source_root / "orig_doc_files"
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / src_path.name
            try:
                shutil.move(str(src_path), str(dest))
            except Exception as move_err:
                return True, target, f"Converted, but failed to move original: {move_err}"

            return True, target, "Converted .doc to .docx and archived original"
        except FileNotFoundError:
            return False, target, "LibreOffice (soffice) not found; cannot convert .doc"
        except Exception as e:
            return False, target, f"Unexpected conversion error: {e}"
