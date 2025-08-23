"""
Processor Interface for Document Processing Pipeline.

This module defines the common interface that all document processors must implement.
This ensures consistent behavior across different processor types and makes it easy
to add new processors to the pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """
    Standard result structure for all processor operations.
    
    This dataclass provides a consistent way to return processing results
    from any processor, making it easy to aggregate and report on results
    across different processor types.
    
    Attributes:
        success: Whether the processing operation was successful
        processor_type: Type of processor that generated this result
        matches_found: Number of matches or items processed
        output_path: Path to the processed output file (if applicable)
        processing_time: Time taken for processing in seconds
        error_message: Error message if processing failed
        metadata: Additional processor-specific data
        layout_impact: Layout impact analysis results (if applicable)
    """
    success: bool
    processor_type: str
    matches_found: int = 0
    output_path: Optional[Path] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    layout_impact: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the processing result to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the processing result
        """
        return {
            'success': self.success,
            'processor_type': self.processor_type,
            'matches_found': self.matches_found,
            'output_path': str(self.output_path) if self.output_path else None,
            'processing_time_seconds': round(self.processing_time, 3),
            'error_message': self.error_message,
            'metadata': self.metadata,
            'layout_impact': self.layout_impact
        }
    
    @property
    def input_path(self) -> Optional[Path]:
        """
        Get the input path from metadata if available.
        
        Returns:
            Input path or None if not available
        """
        if self.metadata and 'input_path' in self.metadata:
            return Path(self.metadata['input_path'])
        return None
    
    @property
    def total_matches(self) -> int:
        """
        Get total matches (alias for matches_found for compatibility).
        
        Returns:
            Number of matches found
        """
        return self.matches_found
    
    @property
    def text_matches(self) -> int:
        """
        Get text matches from metadata or based on processor type.
        
        Returns:
            Number of text matches found
        """
        # For document processor, extract from metadata
        if self.processor_type == "document_processor" and self.metadata:
            return self.metadata.get('text_matches', 0)
        # For individual processors, use matches_found if it's a text processor
        return self.matches_found if self.processor_type == "text" else 0
    
    @property
    def graphics_matches(self) -> int:
        """
        Get graphics matches from metadata or based on processor type.
        
        Returns:
            Number of graphics matches found
        """
        # For document processor, extract from metadata
        if self.processor_type == "document_processor" and self.metadata:
            return self.metadata.get('graphics_matches', 0)
        # For individual processors, use matches_found if it's a graphics processor
        return self.matches_found if self.processor_type == "graphics" else 0
    
    @property
    def image_matches(self) -> int:
        """
        Get image matches from metadata or based on processor type.
        
        Returns:
            Number of image matches found
        """
        # For document processor, extract from metadata
        if self.processor_type == "document_processor" and self.metadata:
            return self.metadata.get('image_matches', 0)
        # For individual processors, use matches_found if it's an image processor
        return self.matches_found if self.processor_type == "image" else 0

class BaseProcessor(ABC):
    """
    Abstract base class for all document processors.
    
    This class defines the common interface and behavior that all processors
    must implement. It provides a consistent way to initialize, configure,
    and process documents regardless of the processor type.
    
    All processors should inherit from this class and implement the required
    abstract methods. This ensures they can be used interchangeably in the
    document processing pipeline.
    """
    
    def __init__(self, processor_type: str, config: Dict[str, Any]):
        """
        Initialize the processor with type and configuration.
        
        Args:
            processor_type: Type identifier for this processor
            config: Configuration dictionary for this processor
        """
        self.processor_type = processor_type
        self.config = config
        self.initialized = False
        logger.debug(f"Initialized {processor_type} processor")
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the processor with required resources.
        
        This method should perform any necessary setup, such as loading
        models, patterns, or other resources required for processing.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, document_path: Path, **kwargs) -> ProcessingResult:
        """
        Process a document and return results.
        
        This is the main processing method that all processors must implement.
        It should handle the document processing logic and return a standardized
        ProcessingResult object.
        
        Args:
            document_path: Path to the document to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult containing processing results and metadata
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of file formats supported by this processor.
        
        Returns:
            List of supported file extensions (e.g., ['.docx', '.pdf'])
        """
        pass
    
    def cleanup(self):
        """
        Clean up resources used by the processor.
        
        This method should release any resources, close files, or perform
        other cleanup operations. It's called when the processor is no
        longer needed.
        """
        logger.debug(f"Cleaning up {self.processor_type} processor")
        self.initialized = False
    
    def is_initialized(self) -> bool:
        """
        Check if the processor has been successfully initialized.
        
        Returns:
            True if the processor is ready for processing
        """
        return self.initialized
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about this processor.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            'type': self.processor_type,
            'initialized': self.initialized,
            'supported_formats': self.get_supported_formats(),
            'config': self.config
        }
