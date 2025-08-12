"""
Data models for the document processing pipeline.
Defines structures for processing results, logs, and match information.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

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
    """OCR match with replacement information."""
    ocr_result: OCRResult
    pattern: str
    replacement_text: str
    image_path: Path
    processing_mode: str  # 'replace' or 'append'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ocr_result': self.ocr_result.to_dict(),
            'pattern': self.pattern,
            'replacement_text': self.replacement_text,
            'image_path': str(self.image_path),
            'processing_mode': self.processing_mode
        }

@dataclass
class ProcessingLog:
    """Detailed processing log for a document."""
    document_path: Path
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    text_matches: List[Match] = field(default_factory=list)
    graphics_matches: List[Match] = field(default_factory=list)
    image_matches: List[OCRMatch] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ocr_comparison_data: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_matches(self) -> int:
        """Get total number of matches across all processors."""
        return len(self.text_matches) + len(self.graphics_matches) + len(self.image_matches)
    
    @property
    def success(self) -> bool:
        """Check if processing was successful (no errors)."""
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_path': str(self.document_path),
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'text_matches': [match.to_dict() for match in self.text_matches],
            'graphics_matches': [match.to_dict() for match in self.graphics_matches],
            'image_matches': [match.to_dict() for match in self.image_matches],
            'errors': self.errors,
            'warnings': self.warnings,
            'total_matches': self.total_matches,
            'success': self.success,
            'ocr_comparison_data': self.ocr_comparison_data
        }

@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    success: bool
    input_path: Path
    output_path: Optional[Path] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    processing_log: Optional[ProcessingLog] = None
    
    @property
    def text_matches(self) -> int:
        """Get number of text matches."""
        return len(self.processing_log.text_matches) if self.processing_log else 0
    
    @property
    def graphics_matches(self) -> int:
        """Get number of graphics matches."""
        return len(self.processing_log.graphics_matches) if self.processing_log else 0
    
    @property
    def image_matches(self) -> int:
        """Get number of image matches."""
        return len(self.processing_log.image_matches) if self.processing_log else 0
    
    @property
    def total_matches(self) -> int:
        """Get total number of matches."""
        return self.text_matches + self.graphics_matches + self.image_matches
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path) if self.output_path else None,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'text_matches': self.text_matches,
            'graphics_matches': self.graphics_matches,
            'image_matches': self.image_matches,
            'total_matches': self.total_matches,
            'processing_log': self.processing_log.to_dict() if self.processing_log else None
        }

@dataclass
class BatchProcessingResult:
    """Result of batch processing operation."""
    total_files: int
    successful_files: int
    failed_files: int
    processing_time: float
    results: List[ProcessingResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0.0
    
    @property
    def total_matches(self) -> int:
        """Get total matches across all files."""
        return sum(result.total_matches for result in self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'success_rate': self.success_rate,
            'processing_time': self.processing_time,
            'total_matches': self.total_matches,
            'results': [result.to_dict() for result in self.results]
        }

@dataclass
class ProcessingStatistics:
    """Processing statistics for reporting."""
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
                self.errors.append(f"{result.input_path.name}: {result.error_message}")
        
        self.total_text_matches += result.text_matches
        self.total_graphics_matches += result.graphics_matches
        self.total_image_matches += result.image_matches
        self.total_processing_time += result.processing_time
        
        if result.processing_log:
            self.errors.extend(result.processing_log.errors)
            self.warnings.extend(result.processing_log.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'files_processed': self.files_processed,
            'files_successful': self.files_successful,
            'files_failed': self.files_failed,
            'success_rate': self.success_rate,
            'total_text_matches': self.total_text_matches,
            'total_graphics_matches': self.total_graphics_matches,
            'total_image_matches': self.total_image_matches,
            'total_matches': self.total_matches,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.average_processing_time,
            'errors': self.errors,
            'warnings': self.warnings
        }

def create_match(pattern: str, original: str, replacement: str, position: int = 0, 
                location: str = "unknown", font_info: Optional[Dict] = None, 
                confidence: Optional[float] = None) -> Match:
    """
    Factory function to create a Match instance.
    
    Args:
        pattern: Regex pattern that matched
        original: Original text that was matched
        replacement: Replacement text
        position: Position in document
        location: Location description
        font_info: Font information dictionary
        confidence: Confidence score (for OCR)
        
    Returns:
        Match instance
    """
    return Match(
        pattern=pattern,
        original_text=original,
        replacement_text=replacement,
        position=position,
        location=location,
        font_info=font_info or {},
        confidence=confidence
    )

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
                    image_path: Path, mode: str = "replace") -> OCRMatch:
    """
    Factory function to create an OCRMatch instance.
    
    Args:
        ocr_result: OCR detection result
        pattern: Regex pattern that matched
        replacement: Replacement text
        image_path: Path to image file
        mode: Processing mode ('replace' or 'append')
        
    Returns:
        OCRMatch instance
    """
    return OCRMatch(
        ocr_result=ocr_result,
        pattern=pattern,
        replacement_text=replacement,
        image_path=image_path,
        processing_mode=mode
    )