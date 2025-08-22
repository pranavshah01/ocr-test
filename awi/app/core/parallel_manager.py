"""
Parallel Manager for Document Processing Pipeline.

This module provides comprehensive parallel processing capabilities with proper
error handling, resource management, and performance optimization. It manages
concurrent document processing while ensuring system stability and efficiency.

Key Features:
- Configurable parallel processing with worker pools
- Comprehensive error handling and recovery
- Resource management and memory optimization
- Progress tracking and status reporting
- Graceful shutdown and cleanup
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import os

import queue
import signal
import sys

logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """Represents a processing task."""
    task_id: str
    file_path: Path
    operation: str
    priority: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    processing_time: float = 0.0
    memory_usage_mb: int = 0

class ParallelManager:
    """
    Manages parallel processing of documents with comprehensive error handling.
    
    This class provides a robust parallel processing framework that handles
    concurrent document processing while ensuring system stability, proper
    error handling, and resource management. It supports both thread-based
    and process-based parallelism with configurable worker pools.
    
    Attributes:
        max_workers: Maximum number of worker processes/threads
        executor_type: Type of executor ('thread' or 'process')
        executor: The actual executor instance
        task_queue: Queue for pending tasks
        results: Dictionary of completed results
        active_tasks: Set of currently active task IDs
        shutdown_event: Event to signal shutdown
        memory_manager: Reference to memory manager
        performance_monitor: Reference to performance monitor
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 executor_type: str = 'thread',
                 memory_manager=None,
                 performance_monitor=None,
                 timeout: int = 3600):  # 1 hour timeout
        """
        Initialize the parallel manager.
        
        Args:
            max_workers: Maximum number of worker processes/threads
            executor_type: Type of executor ('thread' or 'process')
            memory_manager: Optional memory manager for resource tracking
            performance_monitor: Optional performance monitor for metrics
            timeout: Timeout for individual tasks in seconds
        """
        self.max_workers = max_workers
        self.executor_type = executor_type
        self.timeout = timeout
        
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor
        
        self.task_queue = queue.Queue()
        self.results: Dict[str, ProcessingResult] = {}
        self.active_tasks = set()
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'start_time': time.time()
        }
        
        # Initialize executor
        self.executor = None
        self._initialize_executor()
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"Parallel Manager initialized with {max_workers} {executor_type} workers")
    
    def _initialize_executor(self):
        """Initialize the appropriate executor based on configuration."""
        try:
            if self.executor_type == 'process':
                self.executor = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    mp_context=None  # Use default multiprocessing context
                )
                logger.info("Process pool executor initialized")
            else:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="DocProcessor"
                )
                logger.info("Thread pool executor initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            # Fallback to thread executor
            self.executor_type = 'thread'
            self.executor = ThreadPoolExecutor(max_workers=1)
            logger.warning("Falling back to single-threaded execution")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            logger.warning(f"Could not set up signal handlers: {e}")
    
    def submit_task(self, task: ProcessingTask, processor_func: Callable) -> str:
        """
        Submit a processing task for execution.
        
        This method submits a task to the parallel processing queue and
        returns a future object that can be used to track the task's
        progress and retrieve results.
        
        Args:
            task: ProcessingTask object containing task information
            processor_func: Function to execute for the task
            
        Returns:
            Task ID for tracking the submitted task
        """
        if self.shutdown_event.is_set():
            raise RuntimeError("Parallel manager is shutting down")
        
        # Add task to active tasks
        self.active_tasks.add(task.task_id)
        self.stats['total_tasks'] += 1
        
        # Submit task to executor
        future = self.executor.submit(
            self._execute_task_with_monitoring,
            task,
            processor_func
        )
        
        logger.debug(f"Submitted task: {task.task_id} ({task.operation})")
        return task.task_id
    
    def _execute_task_with_monitoring(self, task: ProcessingTask, processor_func: Callable) -> ProcessingResult:
        """
        Execute a task with comprehensive monitoring and error handling.
        
        This method wraps the actual task execution with monitoring,
        error handling, and resource management. It ensures that all
        tasks are properly tracked and any errors are captured.
        
        Args:
            task: ProcessingTask object
            processor_func: Function to execute
            
        Returns:
            ProcessingResult object containing task results
        """
        start_time = time.time()
        operation_id = None
        
        try:
            # Start performance monitoring if available
            if self.performance_monitor:
                file_size_mb = task.file_path.stat().st_size / (1024 * 1024)
                operation_id = self.performance_monitor.start_operation(
                    task.operation, file_size_mb
                )
            
            # Execute the task
            logger.debug(f"Executing task: {task.task_id}")
            result = processor_func(task.file_path, **task.metadata)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get memory usage if available
            memory_usage = 0
            if self.memory_manager:
                memory_stats = self.memory_manager.get_memory_stats()
                memory_usage = memory_stats.process_memory
            
            # Create successful result
            processing_result = ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                memory_usage_mb=memory_usage
            )
            
            # End performance monitoring
            if self.performance_monitor and operation_id:
                self.performance_monitor.end_operation(operation_id, success=True)
            
            # Update statistics
            self.stats['completed_tasks'] += 1
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Task completed successfully: {task.task_id} ({processing_time:.3f}s)")
            return processing_result
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get memory usage if available
            memory_usage = 0
            if self.memory_manager:
                memory_stats = self.memory_manager.get_memory_stats()
                memory_usage = memory_stats.process_memory
            
            # Create error result
            processing_result = ProcessingResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message=str(e),
                processing_time=processing_time,
                memory_usage_mb=memory_usage
            )
            
            # End performance monitoring with error
            if self.performance_monitor and operation_id:
                self.performance_monitor.end_operation(
                    operation_id, 
                    success=False, 
                    error_message=str(e)
                )
            
            # Update statistics
            self.stats['failed_tasks'] += 1
            self.stats['total_processing_time'] += processing_time
            
            logger.error(f"Task failed: {task.task_id} - {e}")
            return processing_result
        
        finally:
            # Remove from active tasks
            self.active_tasks.discard(task.task_id)
    
    def process_batch(self, tasks: List[ProcessingTask], 
                     processor_func: Callable,
                     progress_callback: Optional[Callable] = None) -> Dict[str, ProcessingResult]:
        """
        Process a batch of tasks in parallel.
        
        This method processes multiple tasks concurrently and provides
        progress tracking and comprehensive error handling. It manages
        the entire batch processing lifecycle including submission,
        monitoring, and result collection.
        
        Args:
            tasks: List of ProcessingTask objects
            processor_func: Function to execute for each task
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping task IDs to ProcessingResult objects
        """
        if not tasks:
            logger.warning("No tasks provided for batch processing")
            return {}
        
        logger.info(f"Starting batch processing of {len(tasks)} tasks")
        
        # Submit all tasks
        futures = {}
        for task in tasks:
            future = self.executor.submit(
                self._execute_task_with_monitoring,
                task,
                processor_func
            )
            futures[future] = task.task_id
        
        # Collect results as they complete
        completed_count = 0
        total_count = len(tasks)
        
        for future in as_completed(futures, timeout=self.timeout):
            task_id = futures[future]
            
            try:
                result = future.result(timeout=60)  # 1 minute timeout for result retrieval
                self.results[task_id] = result
                completed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed_count, total_count, task_id, result)
                
                logger.debug(f"Batch progress: {completed_count}/{total_count} tasks completed")
                
            except Exception as e:
                logger.error(f"Error retrieving result for task {task_id}: {e}")
                # Create error result
                error_result = ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result=None,
                    error_message=f"Result retrieval failed: {e}",
                    processing_time=0.0
                )
                self.results[task_id] = error_result
                completed_count += 1
        
        logger.info(f"Batch processing completed: {completed_count}/{total_count} tasks")
        return self.results.copy()
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Task status ('pending', 'active', 'completed', 'failed', 'unknown')
        """
        if task_id in self.active_tasks:
            return 'active'
        elif task_id in self.results:
            result = self.results[task_id]
            return 'completed' if result.success else 'failed'
        else:
            return 'unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        total_time = time.time() - self.stats['start_time']
        
        return {
            'total_tasks': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'active_tasks': len(self.active_tasks),
            'success_rate': (self.stats['completed_tasks'] / max(1, self.stats['total_tasks']) * 100),
            'total_processing_time': self.stats['total_processing_time'],
            'average_processing_time': (self.stats['total_processing_time'] / 
                                      max(1, self.stats['completed_tasks'] + self.stats['failed_tasks'])),
            'total_time': total_time,
            'efficiency': (self.stats['total_processing_time'] / max(1, total_time) * 100),
            'executor_type': self.executor_type,
            'max_workers': self.max_workers
        }
    
    def shutdown(self, wait: bool = True, timeout: int = 30):
        """
        Shutdown the parallel manager gracefully.
        
        This method initiates a graceful shutdown of the parallel manager,
        ensuring that all active tasks are properly handled and resources
        are cleaned up.
        
        Args:
            wait: Whether to wait for active tasks to complete
            timeout: Maximum time to wait for shutdown in seconds
        """
        logger.info("Initiating parallel manager shutdown")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Cancel any pending tasks
        if hasattr(self.executor, '_threads'):
            for thread in self.executor._threads:
                if hasattr(thread, '_stop'):
                    thread._stop()
        
        # Shutdown executor
        if self.executor:
            if hasattr(self.executor, 'shutdown'):
                try:
                    self.executor.shutdown(wait=wait, timeout=timeout)
                except TypeError:
                    # Handle older versions that don't support timeout
                    self.executor.shutdown(wait=wait)
        
        # Wait for active tasks to complete if requested
        if wait:
            start_time = time.time()
            while self.active_tasks and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.active_tasks:
                logger.warning(f"Some tasks did not complete within timeout: {list(self.active_tasks)}")
        
        logger.info("Parallel manager shutdown completed")
    
    def cleanup(self):
        """Clean up parallel manager resources."""
        try:
            # Shutdown if not already done
            if not self.shutdown_event.is_set():
                self.shutdown(wait=False)
            
            # Clear results and statistics
            self.results.clear()
            self.active_tasks.clear()
            self.stats.clear()
            
            logger.info("Parallel manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during parallel manager cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


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
    def filter_documents(documents: List[Path], max_size_mb: float = 300.0) -> List[Path]:
        """
        Filter documents by size and accessibility.
        
        Args:
            documents: List of document paths
            max_size_mb: Maximum file size in MB (default: 300.0MB)
            
        Returns:
            List of filtered document paths
        """
        filtered = []
        
        for doc_path in documents:
            try:
                # Skip temporary Office files (start with ~$)
                if doc_path.name.startswith('~$'):
                    logger.debug(f"Skipping temporary Office file: {doc_path.name}")
                    continue
                
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
    if max_workers is None:
        # Auto-detect based on CPU count
        import psutil
        max_workers = max(1, min(psutil.cpu_count(logical=True) - 2, 8))
    
    executor_type = 'process' if use_processes else 'thread'
    return ParallelManager(max_workers=max_workers, executor_type=executor_type)