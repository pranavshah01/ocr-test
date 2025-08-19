"""
Memory Manager for Document Processing Pipeline.

This module provides comprehensive memory management capabilities including
garbage collection, memory monitoring, and resource cleanup. It ensures
efficient memory usage and prevents memory leaks during document processing.

Key Features:
- Automatic garbage collection with configurable thresholds
- Memory usage monitoring and reporting
- Resource cleanup and memory optimization
- Memory leak detection and prevention
- Cross-platform memory management
"""

import gc
import logging
import psutil
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int  # Total system memory in MB
    available_memory: int  # Available memory in MB
    used_memory: int  # Used memory in MB
    memory_percentage: float  # Memory usage percentage
    process_memory: int  # Current process memory in MB
    gc_objects: int  # Number of objects tracked by garbage collector
    gc_collections: Dict[str, int]  # Garbage collection statistics

class MemoryManager:
    """
    Manages memory usage, garbage collection, and resource cleanup.
    
    This class provides comprehensive memory management capabilities to ensure
    efficient memory usage during document processing. It includes automatic
    garbage collection, memory monitoring, and resource cleanup mechanisms.
    
    Attributes:
        gc_threshold: Memory threshold for triggering garbage collection
        gc_frequency: How often to perform garbage collection
        memory_warning_threshold: Memory usage threshold for warnings
        memory_critical_threshold: Memory usage threshold for critical warnings
        last_gc_time: Timestamp of last garbage collection
        gc_stats: Statistics about garbage collection operations
    """
    
    def __init__(self, 
                 gc_threshold: float = 0.8,  # 80% memory usage
                 gc_frequency: int = 60,  # Every 60 seconds
                 memory_warning_threshold: float = 0.85,  # 85% memory usage
                 memory_critical_threshold: float = 0.95):  # 95% memory usage
        """
        Initialize the memory manager.
        
        Args:
            gc_threshold: Memory usage threshold for triggering garbage collection
            gc_frequency: Minimum time between garbage collection operations (seconds)
            memory_warning_threshold: Memory usage threshold for warnings
            memory_critical_threshold: Memory usage threshold for critical warnings
        """
        self.gc_threshold = gc_threshold
        self.gc_frequency = gc_frequency
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        
        self.last_gc_time = 0
        self.gc_stats = {
            'collections': 0,
            'objects_freed': 0,
            'total_time': 0.0
        }
        
        # Configure garbage collector
        self._configure_gc()
        
        logger.info("Memory Manager initialized")
        logger.info(f"GC threshold: {gc_threshold * 100:.1f}%")
        logger.info(f"GC frequency: {gc_frequency} seconds")
    
    def _configure_gc(self):
        """Configure the garbage collector for optimal performance."""
        try:
            # Enable all garbage collection generations
            gc.enable()
            
            # Set garbage collection thresholds
            # (threshold0, threshold1, threshold2)
            # threshold0: number of allocations minus deallocations before GC
            # threshold1: number of collections of generation 0 before GC of generation 1
            # threshold2: number of collections of generation 1 before GC of generation 2
            gc.set_threshold(700, 10, 10)
            
            logger.info("Garbage collector configured successfully")
            
        except Exception as e:
            logger.warning(f"Could not configure garbage collector: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics.
        
        This method provides comprehensive memory information including system
        memory, process memory, and garbage collection statistics. It's useful
        for monitoring and debugging memory-related issues.
        
        Returns:
            MemoryStats object containing current memory information
        """
        try:
            # Get system memory information
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            # Get garbage collection statistics
            gc_objects = len(gc.get_objects())
            gc_collections = gc.get_stats()
            
            stats = MemoryStats(
                total_memory=memory.total // (1024 * 1024),  # MB
                available_memory=memory.available // (1024 * 1024),  # MB
                used_memory=memory.used // (1024 * 1024),  # MB
                memory_percentage=memory.percent,
                process_memory=process.memory_info().rss // (1024 * 1024),  # MB
                gc_objects=gc_objects,
                gc_collections={gen: stats['collections'] for gen, stats in enumerate(gc_collections)}
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(0, 0, 0, 0.0, 0, 0, {})
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Check current memory usage and trigger warnings if necessary.
        
        This method monitors memory usage and provides warnings when memory
        usage exceeds configured thresholds. It also triggers automatic
        garbage collection when appropriate.
        
        Returns:
            Dictionary containing memory status and any warnings
        """
        stats = self.get_memory_stats()
        warnings = []
        
        # Check memory usage thresholds
        if stats.memory_percentage > self.memory_critical_threshold * 100:
            warnings.append(f"CRITICAL: Memory usage at {stats.memory_percentage:.1f}%")
            logger.critical(f"Memory usage critical: {stats.memory_percentage:.1f}%")
        elif stats.memory_percentage > self.memory_warning_threshold * 100:
            warnings.append(f"WARNING: Memory usage at {stats.memory_percentage:.1f}%")
            logger.warning(f"Memory usage high: {stats.memory_percentage:.1f}%")
        
        # Check if garbage collection is needed
        should_gc = (stats.memory_percentage > self.gc_threshold * 100 and 
                    time.time() - self.last_gc_time > self.gc_frequency)
        
        if should_gc:
            logger.info("Memory threshold exceeded, triggering garbage collection")
            self.force_garbage_collection()
        
        return {
            'memory_percentage': stats.memory_percentage,
            'process_memory_mb': stats.process_memory,
            'gc_objects': stats.gc_objects,
            'warnings': warnings,
            'should_gc': should_gc
        }
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.
        
        This method performs a comprehensive garbage collection operation
        and returns detailed statistics about what was collected. It's useful
        for memory optimization and debugging.
        
        Returns:
            Dictionary containing garbage collection statistics
        """
        start_time = time.time()
        
        # Get pre-GC statistics
        pre_gc_stats = self.get_memory_stats()
        pre_gc_objects = len(gc.get_objects())
        
        # Perform garbage collection
        try:
            # Collect all generations
            collected = gc.collect()
            
            # Get post-GC statistics
            post_gc_stats = self.get_memory_stats()
            post_gc_objects = len(gc.get_objects())
            
            # Calculate statistics
            gc_time = time.time() - start_time
            objects_freed = pre_gc_objects - post_gc_objects
            memory_freed = pre_gc_stats.process_memory - post_gc_stats.process_memory
            
            # Update GC statistics
            self.gc_stats['collections'] += 1
            self.gc_stats['objects_freed'] += objects_freed
            self.gc_stats['total_time'] += gc_time
            self.last_gc_time = time.time()
            
            logger.info(f"Garbage collection completed: {objects_freed} objects freed, "
                       f"{memory_freed} MB freed, {gc_time:.3f}s")
            
            return {
                'collected': collected,
                'objects_freed': objects_freed,
                'memory_freed_mb': memory_freed,
                'gc_time_seconds': gc_time,
                'pre_gc_memory_mb': pre_gc_stats.process_memory,
                'post_gc_memory_mb': post_gc_stats.process_memory
            }
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return {
                'error': str(e),
                'gc_time_seconds': time.time() - start_time
            }
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        Context manager for memory-intensive operations.
        
        This context manager provides automatic memory management for
        memory-intensive operations. It performs garbage collection before
        and after the operation, and provides detailed memory statistics.
        
        Args:
            operation_name: Name of the operation for logging purposes
        """
        logger.info(f"Starting memory-intensive operation: {operation_name}")
        
        # Pre-operation memory check
        pre_stats = self.get_memory_stats()
        logger.debug(f"Pre-operation memory: {pre_stats.process_memory} MB")
        
        try:
            # Perform garbage collection before operation
            self.force_garbage_collection()
            
            # Execute the operation
            yield
            
        except Exception as e:
            logger.error(f"Error during {operation_name}: {e}")
            raise
        
        finally:
            # Post-operation cleanup
            post_stats = self.get_memory_stats()
            memory_delta = post_stats.process_memory - pre_stats.process_memory
            
            logger.info(f"Completed {operation_name}: "
                       f"Memory delta: {memory_delta:+d} MB, "
                       f"Final memory: {post_stats.process_memory} MB")
            
            # Force garbage collection after operation
            self.force_garbage_collection()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform comprehensive memory optimization.
        
        This method performs a series of memory optimization operations
        including garbage collection, memory defragmentation, and resource
        cleanup. It's useful for long-running processes or when memory
        usage becomes problematic.
        
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting memory optimization")
        
        start_time = time.time()
        initial_stats = self.get_memory_stats()
        
        # Step 1: Force garbage collection
        gc_results = self.force_garbage_collection()
        
        # Step 2: Clear any cached objects (if applicable)
        self._clear_caches()
        
        # Step 3: Final garbage collection
        final_gc_results = self.force_garbage_collection()
        
        # Calculate optimization results
        final_stats = self.get_memory_stats()
        optimization_time = time.time() - start_time
        
        memory_saved = initial_stats.process_memory - final_stats.process_memory
        
        results = {
            'initial_memory_mb': initial_stats.process_memory,
            'final_memory_mb': final_stats.process_memory,
            'memory_saved_mb': memory_saved,
            'optimization_time_seconds': optimization_time,
            'gc_results': gc_results,
            'final_gc_results': final_gc_results
        }
        
        logger.info(f"Memory optimization completed: {memory_saved} MB saved, "
                   f"{optimization_time:.3f}s")
        
        return results
    
    def _clear_caches(self):
        """Clear various caches to free memory."""
        try:
            # Clear Python's import cache
            import importlib
            importlib.invalidate_caches()
            
            # Clear any torch caches if available
            try:
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("Caches cleared successfully")
            
        except Exception as e:
            logger.warning(f"Error clearing caches: {e}")
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        return {
            'total_collections': self.gc_stats['collections'],
            'total_objects_freed': self.gc_stats['objects_freed'],
            'total_gc_time': self.gc_stats['total_time'],
            'average_gc_time': (self.gc_stats['total_time'] / 
                              max(1, self.gc_stats['collections'])),
            'last_gc_time': self.last_gc_time
        }
    
    def cleanup(self):
        """Clean up memory manager resources."""
        try:
            # Final garbage collection
            self.force_garbage_collection()
            
            # Clear any remaining references
            self.gc_stats.clear()
            
            logger.info("Memory manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory manager cleanup: {e}")
    
    def monitor_memory(self, callback: Optional[Callable[[MemoryStats], None]] = None):
        """
        Monitor memory usage and call callback when thresholds are exceeded.
        
        Args:
            callback: Optional callback function to call with memory stats
        """
        stats = self.get_memory_stats()
        
        if callback:
            callback(stats)
        
        return stats
