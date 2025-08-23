"""
Core Infrastructure for Document Processing Pipeline.

This package provides the core infrastructure components for the document
processing pipeline, including parallel processing, GPU management, memory
management, performance monitoring, processor factory, and the main document
processor orchestrator.
"""

# Core infrastructure components
from .parallel_manager import ParallelManager, ProcessingTask, ProcessingResult as ParallelResult
from .processor_factory import ProcessorFactory, ProcessorConfig, ProcessorInfo
from .gpu_manager import GPUManager, GPUConfig, GPUInfo
from .memory_manager import MemoryManager, MemoryStats
from .performance_monitor import PerformanceMonitor, PerformanceMetrics

# Error handling
from .error_handler import ErrorHandler, ProcessingError, ErrorSeverity, ErrorCategory

# Processor interface and models
from .processor_interface import BaseProcessor, ProcessingResult
from .models import ProcessingLog, DocumentInfo, BatchProcessingResult

# Main document processor
from .document_processor import DocumentProcessor, create_document_processor

__all__ = [
    # Core infrastructure
    'ParallelManager', 'ProcessingTask', 'ParallelResult',
    'ProcessorFactory', 'ProcessorConfig', 'ProcessorInfo',
    'GPUManager', 'GPUConfig', 'GPUInfo',
    'MemoryManager', 'MemoryStats',
    'PerformanceMonitor', 'PerformanceMetrics',
    
    # Error handling
    'ErrorHandler', 'ProcessingError', 'ErrorSeverity', 'ErrorCategory',
    
    # Processor interface and models
    'BaseProcessor', 'ProcessingResult',
    'ProcessingLog', 'DocumentInfo', 'BatchProcessingResult',
    
    # Main document processor
    'DocumentProcessor', 'create_document_processor'
]
