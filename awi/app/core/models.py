"""
Data Models for Document Processing Pipeline.

This module defines the data structures used throughout the document processing
pipeline for consistent data handling and reporting.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .processor_interface import ProcessingResult

@dataclass
class Match:
    """Represents a text match and replacement."""
    pattern: str
    original_text: str
    replacement_text: str
    position: int
    font_info: Dict[str, Any] = field(default_factory=dict)
    location: str = "unknown"
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern': self.pattern,
            'original_text': self.original_text,
            'replacement_text': self.replacement_text,
            'position': self.position,
            'font_info': self.font_info,
            'location': self.location,
            'confidence': self.confidence
        }

@dataclass
class ProcessingLog:
    """
    Comprehensive log of processing operations for a document.
    
    This dataclass tracks all processing activities, errors, and results
    for a single document, providing detailed information for reporting
    and debugging purposes.
    
    Attributes:
        document_path: Path to the original document
        start_time: When processing started
        end_time: When processing completed
        processing_time: Total processing time in seconds
        text_matches: Results from text processing
        graphics_matches: Results from graphics processing
        image_matches: Results from image processing
        errors: List of errors encountered during processing
        warnings: List of warnings generated during processing
        info_messages: List of informational messages
        ocr_comparison_data: OCR comparison data for images
        layout_impact_data: Layout impact analysis results
    """
    document_path: Path
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    text_matches: List[Dict[str, Any]] = field(default_factory=list)
    graphics_matches: List[Dict[str, Any]] = field(default_factory=list)
    image_matches: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    ocr_comparison_data: List[Dict[str, Any]] = field(default_factory=list)
    layout_impact_data: Optional[Dict[str, Any]] = None
    
    def add_error(self, error_message: str):
        """Add an error message to the log."""
        self.errors.append(f"{datetime.now().strftime('%H:%M:%S')}: {error_message}")
    
    def add_warning(self, warning_message: str):
        """Add a warning message to the log."""
        self.warnings.append(f"{datetime.now().strftime('%H:%M:%S')}: {warning_message}")
    
    def add_info(self, info_message: str):
        """Add an informational message to the log."""
        self.info_messages.append(f"{datetime.now().strftime('%H:%M:%S')}: {info_message}")
    
    def complete(self):
        """Mark processing as complete and calculate total time."""
        self.end_time = datetime.now()
        self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary for serialization."""
        return {
            'document_path': str(self.document_path),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'processing_time': self.processing_time,
            'text_matches_count': len(self.text_matches),
            'graphics_matches_count': len(self.graphics_matches),
            'image_matches_count': len(self.image_matches),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'info_messages_count': len(self.info_messages),
            'errors': self.errors,
            'warnings': self.warnings,
            'info_messages': self.info_messages,
            'layout_impact_data': self.layout_impact_data
        }

@dataclass
class DocumentInfo:
    """
    Information about a document being processed.
    
    This dataclass stores metadata about a document, including its
    file properties, processing status, and results.
    
    Attributes:
        file_path: Path to the document file
        file_size: Size of the file in bytes
        file_extension: File extension (e.g., '.docx')
        processing_status: Current processing status
        processing_time: Time taken for processing
        matches_found: Total number of matches found
        output_path: Path to processed output file
        error_message: Error message if processing failed
    """
    file_path: Path
    file_size: int = 0
    file_extension: str = ""
    processing_status: str = "pending"
    processing_time: float = 0.0
    matches_found: int = 0
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields after object creation."""
        if self.file_path.exists():
            self.file_size = self.file_path.stat().st_size
        self.file_extension = self.file_path.suffix.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document info to dictionary for serialization."""
        return {
            'file_path': str(self.file_path),
            'file_name': self.file_path.name,
            'file_size': self.file_size,
            'file_size_mb': round(self.file_size / (1024 * 1024), 2),
            'file_extension': self.file_extension,
            'processing_status': self.processing_status,
            'processing_time': self.processing_time,
            'matches_found': self.matches_found,
            'output_path': str(self.output_path) if self.output_path else None,
            'error_message': self.error_message
        }

@dataclass
class BatchProcessingResult:
    """
    Results from processing a batch of documents.
    
    This dataclass aggregates results from processing multiple documents,
    providing summary statistics and detailed information for reporting.
    
    Attributes:
        total_documents: Total number of documents processed
        successful_documents: Number of successfully processed documents
        failed_documents: Number of documents that failed processing
        total_processing_time: Total time spent processing all documents
        total_matches_found: Total matches found across all documents
        documents: List of individual document results
        errors: List of errors encountered during batch processing
        warnings: List of warnings generated during batch processing
        start_time: When batch processing started
        end_time: When batch processing completed
    """
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_processing_time: float = 0.0
    total_matches_found: int = 0
    documents: List[DocumentInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_document_result(self, doc_info: DocumentInfo):
        """Add a document result to the batch."""
        self.documents.append(doc_info)
        # Don't increment counters here since they're set externally in run.py
        
        # Only accumulate processing time and matches
        self.total_processing_time += doc_info.processing_time
        self.total_matches_found += doc_info.matches_found
    
    def add_error(self, error_message: str):
        """Add an error message to the batch."""
        self.errors.append(f"{datetime.now().strftime('%H:%M:%S')}: {error_message}")
    
    def add_warning(self, warning_message: str):
        """Add a warning message to the batch."""
        self.warnings.append(f"{datetime.now().strftime('%H:%M:%S')}: {warning_message}")
    
    def complete(self):
        """Mark batch processing as complete."""
        self.end_time = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful_documents / self.total_documents) * 100
    
    def get_average_processing_time(self) -> float:
        """Calculate average processing time per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_processing_time / self.total_documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary for serialization."""
        return {
            'total_documents': self.total_documents,
            'successful_documents': self.successful_documents,
            'failed_documents': self.failed_documents,
            'success_rate': round(self.get_success_rate(), 1),
            'total_processing_time_seconds': round(self.total_processing_time, 3),
            'average_processing_time_seconds': round(self.get_average_processing_time(), 3),
            'total_matches_found': self.total_matches_found,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'documents': [doc.to_dict() for doc in self.documents]
        }


@dataclass
class ProcessingStatistics:
    """Processing statistics for comprehensive reporting."""
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    total_text_matches: int = 0
    total_graphics_matches: int = 0
    total_image_matches: int = 0
    total_processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.files_successful / self.files_processed * 100) if self.files_processed > 0 else 0.0
    
    @property
    def total_matches(self) -> int:
        """Get total matches across all categories."""
        return self.total_text_matches + self.total_graphics_matches + self.total_image_matches
    
    @property
    def average_processing_time(self) -> float:
        """Get average processing time per file."""
        return self.total_processing_time / self.files_processed if self.files_processed > 0 else 0.0
    
    def add_result(self, result: ProcessingResult):
        """Add a processing result to statistics."""
        self.files_processed += 1
        if result.success:
            self.files_successful += 1
        else:
            self.files_failed += 1
            if result.error_message:
                self.errors.append(f"{result.input_path.name if result.input_path else 'Unknown'}: {result.error_message}")
        
        self.total_text_matches += result.text_matches
        self.total_graphics_matches += result.graphics_matches
        self.total_image_matches += result.image_matches
        self.total_processing_time += result.processing_time
        
        # Add any errors/warnings from metadata if available
        if result.metadata:
            if 'errors' in result.metadata:
                self.errors.extend(result.metadata['errors'])
            if 'warnings' in result.metadata:
                self.warnings.extend(result.metadata['warnings'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'files_processed': self.files_processed,
            'files_successful': self.files_successful,
            'files_failed': self.files_failed,
            'success_rate': round(self.success_rate, 1),
            'total_text_matches': self.total_text_matches,
            'total_graphics_matches': self.total_graphics_matches,
            'total_image_matches': self.total_image_matches,
            'total_matches': self.total_matches,
            'total_processing_time_seconds': round(self.total_processing_time, 3),
            'average_processing_time_seconds': round(self.average_processing_time, 3),
            'errors': self.errors,
            'warnings': self.warnings
        }

# --- OCR data models (needed for image processor) ---

@dataclass
class OCRResult:
    """OCR detection result with bounding box information."""
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bounding_box': {
                'x': self.bounding_box[0],
                'y': self.bounding_box[1],
                'width': self.bounding_box[2],
                'height': self.bounding_box[3]
            }
        }

@dataclass
class HybridOCRResult:
    """Result from hybrid OCR processing combining multiple engines."""
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    source_engine: str  # 'easyocr', 'tesseract', or 'hybrid'
    easyocr_result: Optional['OCRResult'] = None
    tesseract_result: Optional['OCRResult'] = None
    selection_reason: str = ""  # Why this result was chosen
    conflict_resolved: bool = False  # Whether there was a conflict between engines
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bounding_box': {
                'x': self.bounding_box[0],
                'y': self.bounding_box[1],
                'width': self.bounding_box[2],
                'height': self.bounding_box[3]
            },
            'source_engine': self.source_engine,
            'easyocr_result': self.easyocr_result.to_dict() if self.easyocr_result else None,
            'tesseract_result': self.tesseract_result.to_dict() if self.tesseract_result else None,
            'selection_reason': self.selection_reason,
            'conflict_resolved': self.conflict_resolved
        }

@dataclass
class OCRMatch:
    """OCR match with replacement information and wipe boundary details."""
    ocr_result: OCRResult
    pattern: str
    replacement_text: str
    image_path: Path
    processing_mode: str  # 'replace' or 'append'
    extracted_pattern_text: str = ""  # The actual pattern text that was extracted (e.g., "77-531-116BLK-245")
    wipe_boundaries: Optional[Tuple[int, int]] = None  # Character start/end positions for wipe
    calculated_text_boundary: Optional[Tuple[int, int, int, int]] = None  # Precise pixel coordinates for wipe area
    wipe_area_info: Optional[Dict[str, Any]] = None  # Additional wipe metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ocr_result': self.ocr_result.to_dict(),
            'pattern': self.pattern,
            'replacement_text': self.replacement_text,
            'image_path': str(self.image_path),
            'processing_mode': self.processing_mode,
            'extracted_pattern_text': self.extracted_pattern_text,
            'wipe_boundaries': self.wipe_boundaries,
            'calculated_text_boundary': self.calculated_text_boundary,
            'wipe_area_info': self.wipe_area_info
        }

# Factory functions for creating OCR objects

def create_ocr_result(text: str, confidence: float, bbox: Tuple[int, int, int, int]) -> OCRResult:
    """
    Factory function to create an OCRResult instance.
    
    Args:
        text: Detected text
        confidence: OCR confidence score
        bbox: Bounding box (x, y, width, height)
        
    Returns:
        OCRResult instance
    """
    return OCRResult(
        text=text,
        confidence=confidence,
        bounding_box=bbox
    )

def create_hybrid_ocr_result(text: str, confidence: float, bbox: Tuple[int, int, int, int],
                           source_engine: str, easyocr_result: Optional[OCRResult] = None,
                           tesseract_result: Optional[OCRResult] = None, 
                           selection_reason: str = "", conflict_resolved: bool = False) -> HybridOCRResult:
    """
    Factory function to create a HybridOCRResult instance.
    
    Args:
        text: Final selected text
        confidence: Final confidence score
        bbox: Bounding box (x, y, width, height)
        source_engine: Engine that provided the final result
        easyocr_result: Original EasyOCR result (if any)
        tesseract_result: Original Tesseract result (if any)
        selection_reason: Reason for selection
        conflict_resolved: Whether there was a conflict between engines
        
    Returns:
        HybridOCRResult instance
    """
    return HybridOCRResult(
        text=text,
        confidence=confidence,
        bounding_box=bbox,
        source_engine=source_engine,
        easyocr_result=easyocr_result,
        tesseract_result=tesseract_result,
        selection_reason=selection_reason,
        conflict_resolved=conflict_resolved
    )

def create_ocr_match(ocr_result: OCRResult, pattern: str, replacement: str, 
                    image_path: Path, mode: str = "replace", extracted_pattern_text: str = "") -> OCRMatch:
    """
    Factory function to create an OCRMatch instance.
    
    Args:
        ocr_result: OCR detection result
        pattern: Regex pattern that matched
        replacement: Replacement text
        image_path: Path to image file
        mode: Processing mode ('replace' or 'append')
        extracted_pattern_text: The actual pattern text that was extracted (e.g., "77-531-116BLK-245")
        
    Returns:
        OCRMatch instance
    """
    return OCRMatch(
        ocr_result=ocr_result,
        pattern=pattern,
        replacement_text=replacement,
        image_path=image_path,
        processing_mode=mode,
        extracted_pattern_text=extracted_pattern_text
    )
