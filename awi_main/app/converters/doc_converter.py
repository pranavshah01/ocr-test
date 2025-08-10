"""
Document format converter for .doc to .docx conversion.
Supports cross-platform conversion using LibreOffice CLI and Windows COM interface.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ..utils.platform_utils import ConversionToolDetector, PathManager, is_windows
from ..utils.shared_constants import CONVERSION_TIMEOUT, ERROR_CODES

logger = logging.getLogger(__name__)

class ConversionError(Exception):
    """Exception raised when document conversion fails."""
    pass

class DocConverter:
    """Converts .doc files to .docx format using available conversion tools."""
    
    def __init__(self, timeout: int = CONVERSION_TIMEOUT):
        """
        Initialize converter with platform detection.
        
        Args:
            timeout: Conversion timeout in seconds
        """
        self.timeout = timeout
        self.detector = ConversionToolDetector()
        self.preferred_tool = self.detector.get_preferred_tool()
        
        if not self.preferred_tool:
            logger.warning("No conversion tools detected. Conversion may fail.")
        else:
            logger.info(f"Using conversion tool: {self.preferred_tool}")
    
    def convert_to_docx(self, doc_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Convert .doc file to .docx format.
        
        Args:
            doc_path: Path to .doc file
            output_dir: Output directory (defaults to same directory as input)
            
        Returns:
            Path to converted .docx file
            
        Raises:
            ConversionError: If conversion fails
        """
        if not doc_path.exists():
            raise ConversionError(f"Input file not found: {doc_path}")
        
        if doc_path.suffix.lower() != '.doc':
            raise ConversionError(f"Input file is not a .doc file: {doc_path}")
        
        # Determine output path
        if output_dir is None:
            output_dir = doc_path.parent
        else:
            PathManager.ensure_directory(output_dir)
        
        output_path = output_dir / (doc_path.stem + '.docx')
        
        # Check if output already exists
        if output_path.exists():
            logger.info(f"Output file already exists: {output_path}")
            return output_path
        
        logger.info(f"Converting {doc_path} to {output_path}")
        
        # Try conversion with available tools
        if self.preferred_tool == 'libreoffice':
            return self._convert_with_libreoffice(doc_path, output_dir)
        elif self.preferred_tool == 'word_com':
            return self._convert_with_word_com(doc_path, output_dir)
        else:
            raise ConversionError("No conversion tools available")
    
    def _convert_with_libreoffice(self, doc_path: Path, output_dir: Path) -> Path:
        """
        Convert using LibreOffice CLI.
        
        Args:
            doc_path: Path to .doc file
            output_dir: Output directory
            
        Returns:
            Path to converted .docx file
        """
        libreoffice_cmd = self.detector.get_tool_path('libreoffice')
        if not libreoffice_cmd:
            raise ConversionError("LibreOffice not found")
        
        # Prepare command
        cmd = [
            libreoffice_cmd,
            '--headless',
            '--convert-to', 'docx',
            '--outdir', str(output_dir),
            str(doc_path)
        ]
        
        logger.debug(f"Running LibreOffice command: {' '.join(cmd)}")
        
        try:
            # Run conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"LibreOffice conversion failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise ConversionError(error_msg)
            
            # Check if output file was created
            output_path = output_dir / (doc_path.stem + '.docx')
            if not output_path.exists():
                raise ConversionError(f"Conversion completed but output file not found: {output_path}")
            
            logger.info(f"Successfully converted to: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise ConversionError(f"LibreOffice conversion timed out after {self.timeout} seconds")
        except Exception as e:
            raise ConversionError(f"LibreOffice conversion failed: {e}")
    
    def _convert_with_word_com(self, doc_path: Path, output_dir: Path) -> Path:
        """
        Convert using Microsoft Word COM interface (Windows only).
        
        Args:
            doc_path: Path to .doc file
            output_dir: Output directory
            
        Returns:
            Path to converted .docx file
        """
        if not is_windows():
            raise ConversionError("Word COM interface is only available on Windows")
        
        try:
            import win32com.client
        except ImportError:
            raise ConversionError("pywin32 package required for Word COM interface")
        
        output_path = output_dir / (doc_path.stem + '.docx')
        
        try:
            # Initialize Word application
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False
            
            # Open document
            doc = word.Documents.Open(str(doc_path.absolute()))
            
            # Save as DOCX (format code 16 for .docx)
            doc.SaveAs2(str(output_path.absolute()), FileFormat=16)
            
            # Close document and quit Word
            doc.Close()
            word.Quit()
            
            # Verify output file was created
            if not output_path.exists():
                raise ConversionError(f"Conversion completed but output file not found: {output_path}")
            
            logger.info(f"Successfully converted to: {output_path}")
            return output_path
            
        except Exception as e:
            # Ensure Word is closed even if error occurs
            try:
                if 'word' in locals():
                    word.Quit()
            except:
                pass
            raise ConversionError(f"Word COM conversion failed: {e}")
    
    def batch_convert(self, doc_files: list[Path], output_dir: Path) -> Dict[Path, Any]:
        """
        Convert multiple .doc files to .docx format.
        
        Args:
            doc_files: List of .doc file paths
            output_dir: Output directory
            
        Returns:
            Dictionary mapping input paths to results (Path or Exception)
        """
        PathManager.ensure_directory(output_dir)
        results = {}
        
        logger.info(f"Starting batch conversion of {len(doc_files)} files")
        
        for doc_file in doc_files:
            try:
                converted_path = self.convert_to_docx(doc_file, output_dir)
                results[doc_file] = converted_path
                logger.info(f"✓ Converted: {doc_file.name}")
            except Exception as e:
                results[doc_file] = e
                logger.error(f"✗ Failed to convert {doc_file.name}: {e}")
        
        successful = sum(1 for result in results.values() if isinstance(result, Path))
        failed = len(results) - successful
        
        logger.info(f"Batch conversion completed: {successful} successful, {failed} failed")
        
        return results
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about available conversion tools."""
        return {
            'available_tools': self.detector.get_available_tools(),
            'preferred_tool': self.preferred_tool,
            'tool_paths': {
                tool: self.detector.get_tool_path(tool) 
                for tool in self.detector.get_available_tools()
            },
            'timeout': self.timeout
        }
    
    def validate_conversion(self, original_path: Path, converted_path: Path) -> bool:
        """
        Validate that conversion was successful.
        
        Args:
            original_path: Path to original .doc file
            converted_path: Path to converted .docx file
            
        Returns:
            True if conversion appears successful
        """
        if not converted_path.exists():
            logger.error(f"Converted file does not exist: {converted_path}")
            return False
        
        # Check file size (converted file should not be empty)
        if converted_path.stat().st_size == 0:
            logger.error(f"Converted file is empty: {converted_path}")
            return False
        
        # Try to open with python-docx to verify it's a valid DOCX
        try:
            from docx import Document
            doc = Document(converted_path)
            # Just accessing the document should be enough to validate
            _ = len(doc.paragraphs)
            logger.debug(f"Conversion validation successful: {converted_path}")
            return True
        except Exception as e:
            logger.error(f"Converted file validation failed: {e}")
            return False

def create_converter(timeout: int = CONVERSION_TIMEOUT) -> DocConverter:
    """
    Factory function to create a DocConverter instance.
    
    Args:
        timeout: Conversion timeout in seconds
        
    Returns:
        Configured DocConverter instance
    """
    return DocConverter(timeout=timeout)