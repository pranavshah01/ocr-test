"""
Unified Report Models for Document Processing Pipeline.

This module defines a clean, unified data model structure:
- ProcessingResult: Single file-level model with performance metrics
- BatchReport: Aggregated batch report that uses ProcessingResult instances
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
import time


class ProcessingStatus(Enum):
    """Enum for processing status."""
    PENDING = "Pending"
    CONVERTED = "Converted"
    PROCESSED = "Processed"
    ERROR = "Error"
    SUCCESS = "Success"
    FAIL = "Fail"


class ProcessorType(Enum):
    """Enum for processor types."""
    TEXT = "Text"
    GRAPHICS = "Graphics"
    IMAGE = "Image"


class MatchFlag(Enum):
    """Enum for match flags."""
    YES = "Y"
    NO = "N"


class FallbackFlag(Enum):
    """Enum for fallback flags."""
    YES = "Y"
    NO = "N"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for processing with detailed monitoring."""
    # Basic timing metrics
    processing_time_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    initial_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    memory_increase_mb: float = 0.0
    current_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    total_memory_usage_mb: float = 0.0
    
    # CPU metrics
    peak_cpu_percent: float = 0.0
    initial_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    current_cpu_percent: float = 0.0
    cpu_samples_count: int = 0
    
    # GPU metrics (if available)
    gpu_available: bool = False
    gpu_utilization_samples: int = 0
    gpu_memory_samples: int = 0
    
    # File I/O metrics
    file_operations_count: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    
    # System metrics
    system_memory_total_gb: float = 0.0
    system_memory_available_gb: float = 0.0
    system_cpu_count: int = 0
    
    # Processing metrics
    total_files_processed: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_matches_found: int = 0
    average_processing_time_per_file: float = 0.0
    max_processing_time_seconds: float = 0.0
    total_file_size_bytes: int = 0
    average_file_size_mb: float = 0.0
    
    # Performance indicators
    success_rate_percent: float = 0.0
    memory_efficiency_mb_per_file: float = 0.0
    io_intensity_bytes_per_file: float = 0.0
    memory_usage_peak_percent: float = 0.0
    cpu_utilization_peak_percent: float = 0.0
    processing_throughput_files_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timing": {
                "processing_time_seconds": round(self.processing_time_seconds, 3),
                "start_time": self.start_time,
                "end_time": self.end_time
            },
            "memory": {
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "initial_memory_mb": round(self.initial_memory_mb, 2),
                "final_memory_mb": round(self.final_memory_mb, 2),
                "memory_increase_mb": round(self.memory_increase_mb, 2),
                "current_memory_mb": round(self.current_memory_mb, 2),
                "average_memory_mb": round(self.average_memory_mb, 2),
                "total_memory_usage_mb": round(self.total_memory_usage_mb, 2)
            },
            "cpu": {
                "peak_cpu_percent": round(self.peak_cpu_percent, 2),
                "initial_cpu_percent": round(self.initial_cpu_percent, 2),
                "average_cpu_percent": round(self.average_cpu_percent, 2),
                "current_cpu_percent": round(self.current_cpu_percent, 2),
                "cpu_samples_count": self.cpu_samples_count
            },
            "gpu": {
                "available": self.gpu_available,
                "utilization_samples": self.gpu_utilization_samples,
                "memory_samples": self.gpu_memory_samples
            },
            "file_io": {
                "operations_count": self.file_operations_count,
                "total_bytes_read": self.total_bytes_read,
                "total_bytes_written": self.total_bytes_written
            },
            "system": {
                "memory_total_gb": round(self.system_memory_total_gb, 2),
                "memory_available_gb": round(self.system_memory_available_gb, 2),
                "cpu_count": self.system_cpu_count
            },
            "processing": {
                "total_files_processed": self.total_files_processed,
                "successful_files": self.successful_files,
                "failed_files": self.failed_files,
                "skipped_files": self.skipped_files,
                "total_matches_found": self.total_matches_found,
                "average_processing_time_per_file": round(self.average_processing_time_per_file, 3),
                "max_processing_time_seconds": round(self.max_processing_time_seconds, 3),
                "total_file_size_bytes": self.total_file_size_bytes,
                "average_file_size_mb": round(self.average_file_size_mb, 2)
            },
            "performance_indicators": {
                "success_rate_percent": round(self.success_rate_percent, 2),
                "memory_efficiency_mb_per_file": round(self.memory_efficiency_mb_per_file, 2),
                "io_intensity_bytes_per_file": round(self.io_intensity_bytes_per_file, 2),
                "memory_usage_peak_percent": round(self.memory_usage_peak_percent, 2),
                "cpu_utilization_peak_percent": round(self.cpu_utilization_peak_percent, 2),
                "processing_throughput_files_per_second": round(self.processing_throughput_files_per_second, 3)
            }
        }


@dataclass
class CLIParameters:
    """CLI parameters used for processing."""
    # Parameter values (current values)
    text_mode: str = ""
    text_separator: str = ""
    default_mapping: str = ""
    ocr_mode: str = ""
    ocr_engine: str = ""
    use_gpu: bool = False
    gpu_device: str = ""
    gpu_available: bool = False
    max_workers: int = 0
    confidence_min: float = 0.0
    verbose: bool = False
    patterns_file: str = ""
    mappings_file: str = ""
    source_dir: str = ""
    output_dir: str = ""
    reports_dir: str = ""
    logs_dir: str = ""
    suffix: str = ""
    processing_timeout: int = 0
    max_retries: int = 0
    min_file_size: float = 0.0
    max_file_size: float = 0.0
    
    # Track which parameters were user-provided (not defaults)
    user_provided: set = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text_mode": self.text_mode,
            "text_separator": self.text_separator,
            "default_mapping": self.default_mapping,
            "ocr_mode": self.ocr_mode,
            "ocr_engine": self.ocr_engine,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
            "gpu_available": self.gpu_available,
            "max_workers": self.max_workers,
            "confidence_min": self.confidence_min,
            "verbose": self.verbose,
            "patterns_file": self.patterns_file,
            "mappings_file": self.mappings_file,
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "reports_dir": self.reports_dir,
            "logs_dir": self.logs_dir,
            "suffix": self.suffix,
            "processing_timeout": self.processing_timeout,
            "max_retries": self.max_retries,
            "min_file_size": self.min_file_size,
            "max_file_size": self.max_file_size,
            "user_provided": list(self.user_provided)
        }


@dataclass
class PatternInfo:
    """Information about patterns used in processing."""
    patterns_file: str = ""
    total_patterns: int = 0
    patterns: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "patterns_file": self.patterns_file,
            "total_patterns": self.total_patterns,
            "patterns": self.patterns
        }


@dataclass
class GraphicsReasoning:
    """Reasoning details for graphics processing."""
    orig_total_char: int = 0
    orig_total_lines: int = 0
    new_total_char: int = 0
    new_total_lines: int = 0
    new_size_fit_percent: str = ""  # e.g., "30% Over"
    reduction_percent: str = ""     # e.g., "40% Font Reduction"
    new_size_change: str = ""       # e.g., "10.0 -> 6.5"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "orig_total_char": self.orig_total_char,
            "orig_total_lines": self.orig_total_lines,
            "new_total_char": self.new_total_char,
            "new_total_lines": self.new_total_lines,
            "new_size_fit_percent": self.new_size_fit_percent,
            "reduction_percent": self.reduction_percent,
            "new_size_change": self.new_size_change
        }


@dataclass
class ImageReasoning:
    """Reasoning details for image processing."""
    available_size_for_new_text: str = ""
    total_characters: int = 0
    line_reasoning: str = ""  # e.g., "1 Line vs 2 Line Reasoning"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "available_size_for_new_text": self.available_size_for_new_text,
            "total_characters": self.total_characters,
            "line_reasoning": self.line_reasoning
        }


@dataclass
class MatchDetail:
    """Detailed information about a single match."""
    sr_no: int = 0
    type: ProcessorType = ProcessorType.TEXT
    orig_id_name: str = ""  # Paragraph ID / Graphics Element ID / Image Name
    src_text: str = ""      # e.g., "77-531-116BLK-245"
    src_text_font: str = "" # e.g., "Arial/Default"
    src_text_color: str = "" # e.g., "Black/Default"
    src_text_size: str = ""  # e.g., "8.0" or "8.0,9.0"
    src_dimension: str = ""  # Text: NA, Graphics: Element Dimension, Image: BBox
    mapped_text: str = ""
    mapped_text_font: str = ""
    mapped_text_color: str = ""
    mapped_text_size: str = ""
    match_flag: MatchFlag = MatchFlag.NO
    is_fallback: FallbackFlag = FallbackFlag.NO
    reasoning: Union[GraphicsReasoning, ImageReasoning, None] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "sr_no": self.sr_no,
            "type": self.type.value,
            "orig_id_name": self.orig_id_name,
            "src_text": self.src_text,
            "src_text_font": self.src_text_font,
            "src_text_color": self.src_text_color,
            "src_text_size": self.src_text_size,
            "src_dimension": self.src_dimension,
            "mapped_text": self.mapped_text,
            "mapped_text_font": self.mapped_text_font,
            "mapped_text_color": self.mapped_text_color,
            "mapped_text_size": self.mapped_text_size,
            "match_flag": self.match_flag.value,
            "is_fallback": self.is_fallback.value,
            "reasoning": self.reasoning.to_dict() if self.reasoning else None
        }
        return result


@dataclass
class ProcessingResult:
    """
    Unified processing result for a single document with all metadata and performance metrics.
    
    This is the single source of truth for file-level processing data, including:
    - Processing metadata (timestamp, CLI params, patterns)
    - Processing results (success, timing, matches)
    - File information (paths, sizes, status)
    - Performance metrics (memory, CPU, I/O)
    - Match details
    """
    # Processing metadata
    timestamp: str = ""
    cli_parameters: CLIParameters = field(default_factory=CLIParameters)
    patterns: PatternInfo = field(default_factory=PatternInfo)
    
    # Processing results
    success: bool = False
    processor_type: str = ""
    parser: str = ""  # Parser used for processing (e.g., "text", "graphics", "image", "hybrid")
    processing_time: float = 0.0
    error_message: str = ""
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    matches_found: int = 0
    total_matches: int = 0
    
    # File information
    file_name: str = ""  # Primary file name (doc or docx)
    doc_file_name: str = ""
    doc_file_size: int = 0
    docx_file_name: str = ""
    docx_file_size: int = 0
    processed_file_name: str = ""
    processed_file_size: int = 0
    processing_time_minutes: float = 0.0
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    failure_reason: str = ""
    
    # Match counts
    total_text_matches: int = 0
    total_graphics_matches: int = 0
    total_image_matches: int = 0
    total_graphics_no_match: int = 0
    total_image_no_match: int = 0
    
    # Performance metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Match details
    match_details: List[MatchDetail] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "cli_parameters": self.cli_parameters.to_dict(),
            "patterns": self.patterns.to_dict(),
            "success": self.success,
            "processor_type": self.processor_type,
            "parser": self.parser,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "input_path": str(self.input_path) if self.input_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "matches_found": self.matches_found,
            "total_matches": self.total_matches,
            "file_name": self.file_name,
            "doc_file_name": self.doc_file_name,
            "doc_file_size": self.doc_file_size,
            "docx_file_name": self.docx_file_name,
            "docx_file_size": self.docx_file_size,
            "processed_file_name": self.processed_file_name,
            "processed_file_size": self.processed_file_size,
            "processing_time_minutes": self.processing_time_minutes,
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "total_text_matches": self.total_text_matches,
            "total_graphics_matches": self.total_graphics_matches,
            "total_image_matches": self.total_image_matches,
            "total_graphics_no_match": self.total_graphics_no_match,
            "total_image_no_match": self.total_image_no_match,
            "performance": self.performance.to_dict(),
            "match_details": [match.to_dict() for match in self.match_details]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class BatchReport:
    """
    Aggregated batch processing report that uses ProcessingResult instances.
    
    This model aggregates individual ProcessingResult instances to create
    comprehensive batch-level statistics and performance metrics.
    """
    timestamp: str = ""
    cli_parameters: CLIParameters = field(default_factory=CLIParameters)
    patterns: PatternInfo = field(default_factory=PatternInfo)
    file_reports: List[ProcessingResult] = field(default_factory=list)
    
    # Aggregated statistics
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_matches: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    
    # Aggregated performance metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    def aggregate(self) -> None:
        """Aggregate statistics from individual file reports."""
        if not self.file_reports:
            return
        
        self.total_documents = len(self.file_reports)
        self.successful_documents = sum(1 for r in self.file_reports if r.success)
        self.failed_documents = self.total_documents - self.successful_documents
        self.total_matches = sum(r.total_matches for r in self.file_reports)
        self.total_processing_time = sum(r.processing_time for r in self.file_reports)
        self.average_processing_time = self.total_processing_time / self.total_documents if self.total_documents > 0 else 0.0
        
        # Aggregate performance metrics
        self._aggregate_performance()
    
    def _aggregate_performance(self) -> None:
        """Aggregate performance metrics from individual results."""
        if not self.file_reports:
            return
        
        # Initialize aggregated metrics
        total_peak_memory = 0.0
        total_peak_cpu = 0.0
        total_file_size = 0
        total_bytes_read = 0
        total_bytes_written = 0
        max_processing_time = 0.0
        
        # Sum up metrics from all files
        for result in self.file_reports:
            perf = result.performance
            total_peak_memory = max(total_peak_memory, perf.peak_memory_mb)
            total_peak_cpu = max(total_peak_cpu, perf.peak_cpu_percent)
            total_file_size += result.doc_file_size + result.docx_file_size + result.processed_file_size
            total_bytes_read += perf.total_bytes_read
            total_bytes_written += perf.total_bytes_written
            max_processing_time = max(max_processing_time, result.processing_time)
        
        # Update aggregated performance metrics
        self.performance.total_files_processed = self.total_documents
        self.performance.successful_files = self.successful_documents
        self.performance.failed_files = self.failed_documents
        self.performance.total_matches_found = self.total_matches
        self.performance.processing_time_seconds = self.total_processing_time
        self.performance.average_processing_time_per_file = self.average_processing_time
        self.performance.max_processing_time_seconds = max_processing_time
        self.performance.total_file_size_bytes = total_file_size
        self.performance.average_file_size_mb = total_file_size / (1024 * 1024 * self.total_documents) if self.total_documents > 0 else 0.0
        self.performance.peak_memory_mb = total_peak_memory
        self.performance.peak_cpu_percent = total_peak_cpu
        self.performance.total_bytes_read = total_bytes_read
        self.performance.total_bytes_written = total_bytes_written
        
        # Calculate performance indicators
        if self.total_documents > 0:
            self.performance.success_rate_percent = (self.successful_documents / self.total_documents) * 100
            self.performance.processing_throughput_files_per_second = self.total_documents / self.total_processing_time if self.total_processing_time > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "cli_parameters": self.cli_parameters.to_dict(),
            "patterns": self.patterns.to_dict(),
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_matches": self.total_matches,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "performance": self.performance.to_dict(),
            "file_reports": [report.to_dict() for report in self.file_reports]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
