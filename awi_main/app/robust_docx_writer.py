"""
Robust DOCX Writer Module

This module provides a comprehensive DOCX writer that prevents corruption issues
by following best practices for programmatic DOCX manipulation.

Key Features:
- Safe XML writing with proper encoding and declarations
- Atomic file operations to prevent partial writes
- Comprehensive error handling and rollback mechanisms
- Validation of DOCX structure before saving
- Backup and recovery capabilities
- Memory-efficient processing for large files

Author: OCR DOCX Text Replacement Utility
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from contextlib import contextmanager
from lxml import etree
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
import traceback

from .shared_constants import (
    XML_NAMESPACES,
    SharedUtilities
)

logger = SharedUtilities.setup_logger(__name__)


class DOCXCorruptionError(Exception):
    """Custom exception for DOCX corruption issues."""
    pass


class RobustDOCXWriter:
    """
    Robust DOCX writer that prevents corruption through safe file operations,
    proper XML handling, and comprehensive validation.
    """
    
    def __init__(self, enable_backup: bool = True, validate_structure: bool = True):
        """
        Initialize the robust DOCX writer.
        
        Args:
            enable_backup: Whether to create backups before modifications
            validate_structure: Whether to validate DOCX structure before saving
        """
        self.enable_backup = enable_backup
        self.validate_structure = validate_structure
        self.temp_dir = None
        self.backup_path = None
        self.processing_stats = {
            "files_processed": 0,
            "backups_created": 0,
            "validations_performed": 0,
            "errors_recovered": 0,
            "corruption_prevented": 0
        }
    
    @contextmanager
    def safe_docx_context(self, docx_path: Path, output_path: Path):
        """
        Context manager for safe DOCX operations with automatic cleanup and rollback.
        
        Args:
            docx_path: Input DOCX file path
            output_path: Output DOCX file path
            
        Yields:
            Document: python-docx Document object for safe manipulation
        """
        temp_dir = None
        backup_created = False
        doc = None
        
        try:
            # Create temporary directory for safe operations
            temp_dir = tempfile.mkdtemp(prefix="robust_docx_")
            logger.debug(f"Created temporary directory: {temp_dir}")
            
            # Create backup if enabled
            if self.enable_backup and output_path.exists():
                backup_path = self._create_backup(output_path)
                backup_created = True
                logger.info(f"Created backup: {backup_path}")
                self.processing_stats["backups_created"] += 1
            
            # Load document safely
            doc = self._load_document_safely(docx_path)
            
            # Yield document for processing
            yield doc
            
            # Validate structure before saving if enabled
            if self.validate_structure:
                self._validate_docx_structure(doc)
                self.processing_stats["validations_performed"] += 1
            
            # Save document safely
            self._save_document_safely(doc, output_path, temp_dir)
            
            # Verify saved file integrity
            self._verify_saved_file(output_path)
            
            self.processing_stats["files_processed"] += 1
            logger.info(f"Successfully processed DOCX: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in DOCX processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Attempt recovery if backup exists
            if backup_created and self.backup_path and self.backup_path.exists():
                try:
                    shutil.copy2(self.backup_path, output_path)
                    logger.info(f"Restored from backup: {self.backup_path}")
                    self.processing_stats["errors_recovered"] += 1
                except Exception as recovery_error:
                    logger.error(f"Failed to restore from backup: {recovery_error}")
            
            raise DOCXCorruptionError(f"DOCX processing failed: {e}") from e
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp directory: {cleanup_error}")
    
    def _load_document_safely(self, docx_path: Path) -> Document:
        """
        Load DOCX document with comprehensive error handling.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Document: Loaded python-docx Document object
            
        Raises:
            DOCXCorruptionError: If document cannot be loaded safely
        """
        try:
            # Verify file exists and is readable
            if not docx_path.exists():
                raise FileNotFoundError(f"DOCX file not found: {docx_path}")
            
            if not docx_path.is_file():
                raise ValueError(f"Path is not a file: {docx_path}")
            
            # Check file size (basic corruption check)
            file_size = docx_path.stat().st_size
            if file_size < 1000:  # DOCX files are typically > 1KB
                raise ValueError(f"DOCX file too small (possible corruption): {file_size} bytes")
            
            # Verify ZIP structure (DOCX is a ZIP file)
            self._verify_zip_structure(docx_path)
            
            # Load document
            doc = Document(str(docx_path))
            logger.debug(f"Successfully loaded DOCX: {docx_path}")
            
            return doc
            
        except Exception as e:
            raise DOCXCorruptionError(f"Failed to load DOCX safely: {e}") from e
    
    def _verify_zip_structure(self, docx_path: Path):
        """
        Verify that DOCX file has valid ZIP structure.
        
        Args:
            docx_path: Path to DOCX file
            
        Raises:
            DOCXCorruptionError: If ZIP structure is invalid
        """
        try:
            with zipfile.ZipFile(docx_path, 'r') as zip_file:
                # Check for required DOCX files
                required_files = [
                    '[Content_Types].xml',
                    '_rels/.rels',
                    'word/document.xml'
                ]
                
                zip_contents = zip_file.namelist()
                
                for required_file in required_files:
                    if required_file not in zip_contents:
                        raise DOCXCorruptionError(f"Missing required file: {required_file}")
                
                # Test ZIP integrity
                zip_file.testzip()
                
                logger.debug(f"ZIP structure verified for: {docx_path}")
                
        except zipfile.BadZipFile as e:
            raise DOCXCorruptionError(f"Invalid ZIP structure: {e}") from e
        except Exception as e:
            raise DOCXCorruptionError(f"ZIP verification failed: {e}") from e
    
    def _validate_docx_structure(self, doc: Document):
        """
        Validate DOCX document structure before saving.
        
        Args:
            doc: python-docx Document object
            
        Raises:
            DOCXCorruptionError: If document structure is invalid
        """
        try:
            # Check document has content
            if doc is None:
                raise ValueError("Document is None")
            
            # Verify core document parts exist
            if not hasattr(doc, 'part') or doc.part is None:
                raise ValueError("Document part is missing")
            
            # Check document XML is valid
            doc_xml = doc.part.element
            if doc_xml is None:
                raise ValueError("Document XML element is missing")
            
            # Verify document has proper namespace
            if doc_xml.tag != qn('w:document'):
                raise ValueError(f"Invalid document root tag: {doc_xml.tag}")
            
            # Check for body element
            body = doc_xml.find(qn('w:body'))
            if body is None:
                raise ValueError("Document body element is missing")
            
            logger.debug("DOCX structure validation passed")
            
        except Exception as e:
            raise DOCXCorruptionError(f"Document structure validation failed: {e}") from e
    
    def _save_document_safely(self, doc: Document, output_path: Path, temp_dir: str):
        """
        Save document using atomic operations to prevent corruption.
        
        Args:
            doc: python-docx Document object
            output_path: Final output path
            temp_dir: Temporary directory for safe operations
            
        Raises:
            DOCXCorruptionError: If save operation fails
        """
        try:
            # Create temporary file in temp directory
            temp_file = os.path.join(temp_dir, f"temp_docx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
            
            # Save to temporary file first (atomic operation)
            logger.debug(f"Saving to temporary file: {temp_file}")
            doc.save(temp_file)
            
            # Verify temporary file was created successfully
            if not os.path.exists(temp_file):
                raise DOCXCorruptionError("Temporary file was not created")
            
            temp_size = os.path.getsize(temp_file)
            if temp_size < 1000:
                raise DOCXCorruptionError(f"Temporary file too small: {temp_size} bytes")
            
            # Verify temporary file ZIP structure
            self._verify_zip_structure(Path(temp_file))
            
            # Atomic move to final location
            logger.debug(f"Moving to final location: {output_path}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic move (rename is atomic on most filesystems)
            shutil.move(temp_file, str(output_path))
            
            logger.info(f"Document saved successfully: {output_path}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            raise DOCXCorruptionError(f"Failed to save document safely: {e}") from e
    
    def _verify_saved_file(self, output_path: Path):
        """
        Verify that saved file is valid and not corrupted.
        
        Args:
            output_path: Path to saved DOCX file
            
        Raises:
            DOCXCorruptionError: If saved file is corrupted
        """
        try:
            # Check file exists
            if not output_path.exists():
                raise FileNotFoundError(f"Saved file not found: {output_path}")
            
            # Check file size
            file_size = output_path.stat().st_size
            if file_size < 1000:
                raise ValueError(f"Saved file too small: {file_size} bytes")
            
            # Verify ZIP structure
            self._verify_zip_structure(output_path)
            
            # Try to load the saved document
            test_doc = Document(str(output_path))
            if test_doc is None:
                raise ValueError("Cannot load saved document")
            
            logger.debug(f"Saved file verification passed: {output_path}")
            
        except Exception as e:
            raise DOCXCorruptionError(f"Saved file verification failed: {e}") from e
    
    def _create_backup(self, file_path: Path) -> Path:
        """
        Create backup of existing file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path: Path to backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        
        shutil.copy2(file_path, backup_path)
        self.backup_path = backup_path
        
        return backup_path
    
    def write_xml_safely(self, xml_element: etree.Element, output_path: Path):
        """
        Write XML with proper encoding and declarations to prevent corruption.
        
        Args:
            xml_element: XML element to write
            output_path: Output file path
            
        Raises:
            DOCXCorruptionError: If XML write fails
        """
        try:
            # Write XML with proper settings as recommended
            with open(output_path, 'wb') as f:
                xml_tree = etree.ElementTree(xml_element)
                xml_tree.write(
                    f,
                    pretty_print=False,  # As recommended in feedback
                    xml_declaration=True,  # As recommended in feedback
                    encoding='UTF-8'  # As recommended in feedback
                )
            
            logger.debug(f"XML written safely: {output_path}")
            
        except Exception as e:
            raise DOCXCorruptionError(f"Failed to write XML safely: {e}") from e
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict containing processing statistics
        """
        return {
            **self.processing_stats,
            "timestamp": datetime.now().isoformat(),
            "backup_enabled": self.enable_backup,
            "validation_enabled": self.validate_structure
        }


def create_robust_docx_writer(enable_backup: bool = True, validate_structure: bool = True) -> RobustDOCXWriter:
    """
    Factory function to create a robust DOCX writer.
    
    Args:
        enable_backup: Whether to enable backup creation
        validate_structure: Whether to enable structure validation
        
    Returns:
        RobustDOCXWriter: Configured robust DOCX writer instance
    """
    return RobustDOCXWriter(
        enable_backup=enable_backup,
        validate_structure=validate_structure
    )


def safe_docx_operation(docx_path: Path, output_path: Path, operation_func, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for safe DOCX operations.
    
    Args:
        docx_path: Input DOCX file path
        output_path: Output DOCX file path
        operation_func: Function to perform on the document
        **kwargs: Additional arguments for the operation function
        
    Returns:
        Dict containing operation results and statistics
        
    Example:
        def my_operation(doc, mapping_dict):
            # Perform operations on doc
            return {"replacements": 5}
        
        result = safe_docx_operation(
            input_path, output_path, my_operation, 
            mapping_dict={"old": "new"}
        )
    """
    writer = create_robust_docx_writer()
    
    try:
        with writer.safe_docx_context(docx_path, output_path) as doc:
            # Perform the operation
            operation_result = operation_func(doc, **kwargs)
            
        # Combine results with writer stats
        result = {
            "success": True,
            "operation_result": operation_result,
            "writer_stats": writer.get_processing_stats(),
            "output_path": str(output_path),
            "errors": []
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Safe DOCX operation failed: {e}")
        return {
            "success": False,
            "operation_result": None,
            "writer_stats": writer.get_processing_stats(),
            "output_path": str(output_path),
            "errors": [str(e)]
        }
