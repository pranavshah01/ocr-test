"""
Parallel processing manager for concurrent document processing.
Handles worker pool management, load balancing, and error isolation.
"""

import concurrent.futures
import multiprocessing
import threading
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
from queue import Queue

from .models import ProcessingResult, BatchProcessingResult, ProcessingStatistics
from .document_processor import DocumentProcessor
from ..utils.shared_constants import SharedUtilities

logger = logging.getLogger(__name__)

@dataclass
class WorkerStatus:
    """Status information for a worker."""
    worker_id: int
    is_busy: bool = False
    current_file: Optional[Path] = None
    start_time: Optional[float] = None
    processed_count: int = 0
    error_count: int = 0

class ProgressTracker:
    """Tracks processing progress across multiple workers."""
    
    def __init__(self, total_files: int):
        """
        Initialize progress tracker.
        
        Args:
            total_files: Total number of files to process
        """
        self.total_files = total_files
        self.completed_files = 0
        self.failed_files = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update_progress(self, success: bool):
        """
        Update progress counters.
        
        Args:
            success: Whether the file was processed successfully
        """
        with self._lock:
            if success:
                self.completed_files += 1
            else:
                self.failed_files += 1
    
    @property
    def processed_files(self) -> int:
        """Get total processed files."""
        return self.completed_files + self.failed_files
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage."""
        return (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining time in seconds."""
        if self.processed_files == 0:
            return 0.0
        
        avg_time_per_file = self.elapsed_time / self.processed_files
        remaining_files = self.total_files - self.processed_files
        return avg_time_per_file * remaining_files
    
    def get_status_message(self) -> str:
        """Get formatted status message."""
        return (f"Progress: {self.processed_files}/{self.total_files} "
                f"({self.progress_percentage:.1f}%) - "
                f"Success: {self.completed_files}, Failed: {self.failed_files} - "
                f"Elapsed: {self.elapsed_time:.1f}s, "
                f"ETA: {self.estimated_remaining_time:.1f}s")

class ParallelManager:
    """Manages parallel processing of multiple documents."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize parallel processing manager.
        
        Args:
            max_workers: Maximum number of workers (None for auto-detect)
            use_processes: Whether to use processes instead of threads
        """
        if max_workers is None:
            max_workers = max(1, min(multiprocessing.cpu_count() - 1, 8))
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.worker_statuses: Dict[int, WorkerStatus] = {}
        self.progress_tracker: Optional[ProgressTracker] = None
        
        logger.info(f"Parallel manager initialized with {max_workers} workers ({'processes' if use_processes else 'threads'})")
    
    def process_documents_parallel(self, documents: List[Path], processor_factory: Callable) -> BatchProcessingResult:
        """
        Process multiple documents in parallel.
        
        Args:
            documents: List of document paths to process
            processor_factory: Factory function to create DocumentProcessor instances
            
        Returns:
            BatchProcessingResult with processing details
        """
        start_time = time.time()
        results = []
        
        if not documents:
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                processing_time=0.0,
                results=[]
            )
        
        logger.info(f"Starting parallel processing of {len(documents)} documents with {self.max_workers} workers")
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(documents))
        
        # Initialize worker statuses
        for i in range(self.max_workers):
            self.worker_statuses[i] = WorkerStatus(worker_id=i)
        
        try:
            # Choose executor type
            executor_class = concurrent.futures.ProcessPoolExecutor if self.use_processes else concurrent.futures.ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_document = {}
                
                for doc_path in documents:
                    future = executor.submit(self._process_single_document, doc_path, processor_factory)
                    future_to_document[future] = doc_path
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_document):
                    doc_path = future_to_document[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        self.progress_tracker.update_progress(result.success)
                        
                        # Log progress periodically
                        if self.progress_tracker.processed_files % max(1, len(documents) // 10) == 0:
                            logger.info(self.progress_tracker.get_status_message())
                        
                    except Exception as e:
                        # Create error result
                        error_result = ProcessingResult(
                            success=False,
                            input_path=doc_path,
                            error_message=f"Worker exception: {e}"
                        )
                        results.append(error_result)
                        self.progress_tracker.update_progress(False)
                        logger.error(f"Worker failed for {doc_path}: {e}")
        
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            # Create error results for any unprocessed documents
            for doc_path in documents[len(results):]:
                error_result = ProcessingResult(
                    success=False,
                    input_path=doc_path,
                    error_message=f"Parallel processing error: {e}"
                )
                results.append(error_result)
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        successful_files = sum(1 for r in results if r.success)
        failed_files = len(results) - successful_files
        
        batch_result = BatchProcessingResult(
            total_files=len(documents),
            successful_files=successful_files,
            failed_files=failed_files,
            processing_time=processing_time,
            results=results
        )
        
        logger.info(f"Parallel processing completed: {successful_files}/{len(documents)} successful in {processing_time:.2f}s")
        
        return batch_result
    
    def _process_single_document(self, doc_path: Path, processor_factory: Callable) -> ProcessingResult:
        """
        Process a single document (worker function).
        
        Args:
            doc_path: Path to document
            processor_factory: Factory function to create DocumentProcessor
            
        Returns:
            ProcessingResult
        """
        worker_id = threading.get_ident() if not self.use_processes else multiprocessing.current_process().pid
        
        try:
            # Update worker status
            if worker_id in self.worker_statuses:
                status = self.worker_statuses[worker_id]
                status.is_busy = True
                status.current_file = doc_path
                status.start_time = time.time()
            
            # Create processor instance for this worker
            processor = processor_factory()
            
            # Process the document
            result = processor.process_document(doc_path)
            
            # Update worker statistics
            if worker_id in self.worker_statuses:
                status = self.worker_statuses[worker_id]
                status.processed_count += 1
                if not result.success:
                    status.error_count += 1
                status.is_busy = False
                status.current_file = None
                status.start_time = None
            
            return result
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed processing {doc_path}: {e}")
            
            # Update error count
            if worker_id in self.worker_statuses:
                status = self.worker_statuses[worker_id]
                status.error_count += 1
                status.is_busy = False
                status.current_file = None
                status.start_time = None
            
            return ProcessingResult(
                success=False,
                input_path=doc_path,
                error_message=str(e)
            )
    
    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get statistics about worker performance."""
        total_processed = sum(status.processed_count for status in self.worker_statuses.values())
        total_errors = sum(status.error_count for status in self.worker_statuses.values())
        busy_workers = sum(1 for status in self.worker_statuses.values() if status.is_busy)
        
        return {
            'total_workers': len(self.worker_statuses),
            'busy_workers': busy_workers,
            'idle_workers': len(self.worker_statuses) - busy_workers,
            'total_processed': total_processed,
            'total_errors': total_errors,
            'success_rate': ((total_processed - total_errors) / total_processed * 100) if total_processed > 0 else 0.0,
            'worker_details': {
                worker_id: {
                    'processed_count': status.processed_count,
                    'error_count': status.error_count,
                    'is_busy': status.is_busy,
                    'current_file': str(status.current_file) if status.current_file else None
                }
                for worker_id, status in self.worker_statuses.items()
            }
        }
    
    def get_progress_info(self) -> Optional[Dict[str, Any]]:
        """Get current progress information."""
        if not self.progress_tracker:
            return None
        
        return {
            'total_files': self.progress_tracker.total_files,
            'processed_files': self.progress_tracker.processed_files,
            'completed_files': self.progress_tracker.completed_files,
            'failed_files': self.progress_tracker.failed_files,
            'progress_percentage': self.progress_tracker.progress_percentage,
            'elapsed_time': self.progress_tracker.elapsed_time,
            'estimated_remaining_time': self.progress_tracker.estimated_remaining_time,
            'status_message': self.progress_tracker.get_status_message()
        }

class FileDiscovery:
    """Handles automatic file discovery and filtering."""
    
    @staticmethod
    def discover_documents(source_dir: Path, supported_formats: List[str] = None) -> List[Path]:
        """
        Discover document files in the source directory.
        
        Args:
            source_dir: Directory to search
            supported_formats: List of supported file extensions
            
        Returns:
            List of discovered document paths
        """
        if supported_formats is None:
            supported_formats = ['.doc', '.docx']
        
        documents = []
        
        try:
            if not source_dir.exists():
                logger.warning(f"Source directory does not exist: {source_dir}")
                return documents
            
            # Search for supported document files
            for format_ext in supported_formats:
                pattern = f"*{format_ext}"
                found_files = list(source_dir.glob(pattern))
                documents.extend(found_files)
            
            # Remove duplicates and sort
            documents = sorted(list(set(documents)))
            
            logger.info(f"Discovered {len(documents)} documents in {source_dir}")
            
        except Exception as e:
            logger.error(f"Error discovering documents in {source_dir}: {e}")
        
        return documents
    
    @staticmethod
    def filter_documents(documents: List[Path], max_size_mb: float = 100.0) -> List[Path]:
        """
        Filter documents by size and accessibility.
        
        Args:
            documents: List of document paths
            max_size_mb: Maximum file size in MB
            
        Returns:
            List of filtered document paths
        """
        filtered = []
        
        for doc_path in documents:
            try:
                # Check if file exists and is readable
                if not doc_path.exists():
                    logger.warning(f"File does not exist: {doc_path}")
                    continue
                
                # Check file size
                size_mb = doc_path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(f"File too large ({size_mb:.1f}MB): {doc_path}")
                    continue
                
                # Check if file is accessible
                if not os.access(doc_path, os.R_OK):
                    logger.warning(f"File not readable: {doc_path}")
                    continue
                
                filtered.append(doc_path)
                
            except Exception as e:
                logger.warning(f"Error checking file {doc_path}: {e}")
        
        logger.info(f"Filtered {len(filtered)}/{len(documents)} documents")
        return filtered

def create_parallel_manager(max_workers: Optional[int] = None, use_processes: bool = False) -> ParallelManager:
    """
    Factory function to create a ParallelManager instance.
    
    Args:
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        
    Returns:
        ParallelManager instance
    """
    return ParallelManager(max_workers, use_processes)