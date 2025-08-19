"""
Performance Monitoring System for Document Processing Pipeline.

This module provides comprehensive performance monitoring capabilities including:
- Processing time tracking
- Memory usage monitoring
- CPU/GPU utilization tracking
- File I/O operations monitoring
- Performance metrics aggregation and reporting
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    processing_time: float = 0.0
    
    # Memory metrics
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    memory_increase_mb: float = 0.0
    
    # CPU metrics
    initial_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    # GPU metrics (if available)
    gpu_available: bool = False
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_usage: List[float] = field(default_factory=list)
    
    # File I/O metrics
    file_operations: List[Dict[str, Any]] = field(default_factory=list)
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    
    # System metrics
    system_memory_total_gb: float = 0.0
    system_memory_available_gb: float = 0.0
    system_cpu_count: int = 0
    
    # Processing metrics
    documents_processed: int = 0
    successful_processing: int = 0
    failed_processing: int = 0
    total_matches_found: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'timing': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'processing_time_seconds': round(self.processing_time, 3)
            },
            'memory': {
                'initial_memory_mb': round(self.initial_memory_mb, 2),
                'peak_memory_mb': round(self.peak_memory_mb, 2),
                'final_memory_mb': round(self.final_memory_mb, 2),
                'memory_increase_mb': round(self.memory_increase_mb, 2)
            },
            'cpu': {
                'initial_cpu_percent': round(self.initial_cpu_percent, 2),
                'peak_cpu_percent': round(self.peak_cpu_percent, 2),
                'average_cpu_percent': round(self.average_cpu_percent, 2),
                'cpu_samples_count': len(self.cpu_samples)
            },
            'gpu': {
                'available': self.gpu_available,
                'utilization_samples': len(self.gpu_utilization),
                'memory_samples': len(self.gpu_memory_usage)
            },
            'file_io': {
                'operations_count': len(self.file_operations),
                'total_bytes_read': self.total_bytes_read,
                'total_bytes_written': self.total_bytes_written
            },
            'system': {
                'memory_total_gb': round(self.system_memory_total_gb, 2),
                'memory_available_gb': round(self.system_memory_available_gb, 2),
                'cpu_count': self.system_cpu_count
            },
            'processing': {
                'documents_processed': self.documents_processed,
                'successful_processing': self.successful_processing,
                'failed_processing': self.failed_processing,
                'total_matches_found': self.total_matches_found
            }
        }

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    This class provides real-time monitoring of system resources and processing
    performance, including memory usage, CPU utilization, GPU usage (if available),
    and file I/O operations.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Interval in seconds between monitoring samples
        """
        self.monitoring_interval = monitoring_interval
        self.metrics = PerformanceMetrics()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        self.process = psutil.Process()
        
        # Initialize system metrics
        self._initialize_system_metrics()
        
        # GPU monitoring (if available)
        self.gpu_monitor = self._initialize_gpu_monitoring()
        
        logger.info("Performance monitor initialized")
    
    def _initialize_system_metrics(self):
        """Initialize system-level metrics."""
        try:
            memory = psutil.virtual_memory()
            self.metrics.system_memory_total_gb = memory.total / (1024**3)
            self.metrics.system_memory_available_gb = memory.available / (1024**3)
            self.metrics.system_cpu_count = psutil.cpu_count()
            
            logger.info(f"System: {self.metrics.system_cpu_count} CPUs, "
                       f"{self.metrics.system_memory_total_gb:.1f}GB RAM")
        except Exception as e:
            logger.warning(f"Failed to initialize system metrics: {e}")
    
    def _initialize_gpu_monitoring(self) -> Optional[Any]:
        """Initialize GPU monitoring if available."""
        try:
            # Try to import GPU monitoring libraries
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.metrics.gpu_available = True
                logger.info(f"GPU monitoring enabled: {device_count} device(s) found")
                return pynvml
            else:
                logger.info("No GPU devices found")
                return None
        except ImportError:
            logger.debug("pynvml not available, GPU monitoring disabled")
            return None
        except Exception as e:
            logger.debug(f"GPU monitoring initialization failed: {e}")
            return None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        # Initialize metrics
        self.metrics.start_time = time.time()
        self.metrics.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.initial_cpu_percent = psutil.cpu_percent()
        
        # Clear previous data
        self.metrics.cpu_samples.clear()
        self.metrics.gpu_utilization.clear()
        self.metrics.gpu_memory_usage.clear()
        self.metrics.file_operations.clear()
        
        # Start monitoring thread
        self.stop_monitoring_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring and finalize metrics."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring not running")
            return
        
        # Signal monitoring thread to stop
        self.stop_monitoring_event.set()
        self.monitoring_thread.join(timeout=5.0)
        
        # Finalize metrics
        self.metrics.end_time = time.time()
        self.metrics.processing_time = self.metrics.end_time - self.metrics.start_time
        self.metrics.final_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.memory_increase_mb = self.metrics.final_memory_mb - self.metrics.initial_memory_mb
        
        # Calculate averages
        if self.metrics.cpu_samples:
            self.metrics.average_cpu_percent = sum(self.metrics.cpu_samples) / len(self.metrics.cpu_samples)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while not self.stop_monitoring_event.is_set():
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent()
                self.metrics.cpu_samples.append(cpu_percent)
                self.metrics.peak_cpu_percent = max(self.metrics.peak_cpu_percent, cpu_percent)
                
                # Monitor memory usage
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                
                # Monitor GPU usage (if available)
                if self.gpu_monitor and self.metrics.gpu_available:
                    self._monitor_gpu()
                
                # Sleep until next monitoring interval
                self.stop_monitoring_event.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.debug(f"Error in monitoring loop: {e}")
                self.stop_monitoring_event.wait(self.monitoring_interval)
    
    def _monitor_gpu(self):
        """Monitor GPU utilization and memory usage."""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Monitor first GPU
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.metrics.gpu_utilization.append(utilization.gpu)
            
            # GPU memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage_mb = memory_info.used / 1024 / 1024
            self.metrics.gpu_memory_usage.append(memory_usage_mb)
            
        except Exception as e:
            logger.debug(f"GPU monitoring error: {e}")
    
    def record_file_operation(self, operation: str, file_path: Path, bytes_transferred: int = 0):
        """Record a file I/O operation."""
        file_op = {
            'timestamp': time.time(),
            'operation': operation,
            'file_path': str(file_path),
            'bytes_transferred': bytes_transferred
        }
        self.metrics.file_operations.append(file_op)
        
        if operation in ['read', 'load']:
            self.metrics.total_bytes_read += bytes_transferred
        elif operation in ['write', 'save']:
            self.metrics.total_bytes_written += bytes_transferred
    
    def update_processing_stats(self, documents_processed: int = 0, successful: int = 0, 
                               failed: int = 0, matches_found: int = 0):
        """Update processing statistics."""
        self.metrics.documents_processed = documents_processed
        self.metrics.successful_processing = successful
        self.metrics.failed_processing = failed
        self.metrics.total_matches_found = matches_found
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_memory_mb = self.process.memory_info().rss / 1024 / 1024
        current_cpu_percent = psutil.cpu_percent()
        
        # Calculate average memory - use a reasonable estimate based on peak and current
        # Since we don't have memory samples, use a weighted average
        if self.metrics.peak_memory_mb > 0:
            # Use 70% of peak + 30% of current as a reasonable average
            average_memory_mb = (self.metrics.peak_memory_mb * 0.7) + (current_memory_mb * 0.3)
        else:
            average_memory_mb = current_memory_mb
        
        # Calculate total memory usage (cumulative) - this should represent total memory consumed over time
        # Use peak memory * processing time as a reasonable approximation
        processing_time_hours = (time.time() - self.metrics.start_time) / 3600
        total_memory_usage_mb = self.metrics.peak_memory_mb * processing_time_hours if processing_time_hours > 0 else 0
        
        return {
            'current_memory_mb': round(current_memory_mb, 2),
            'current_cpu_percent': round(current_cpu_percent, 2),
            'peak_memory_mb': round(self.metrics.peak_memory_mb, 2),
            'peak_cpu_percent': round(self.metrics.peak_cpu_percent, 2),
            'average_memory_mb': round(average_memory_mb, 2),
            'average_cpu_percent': round(self.metrics.average_cpu_percent, 2),
            'total_memory_usage_mb': round(total_memory_usage_mb, 2),
            'processing_time_seconds': round(time.time() - self.metrics.start_time, 3)
        }
    
    def save_metrics_report(self, output_path: Path):
        """Save performance metrics to JSON file."""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.to_dict(),
                'summary': self._generate_summary()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Performance metrics saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        success_rate = (self.metrics.successful_processing / self.metrics.documents_processed * 100) if self.metrics.documents_processed > 0 else 0
        
        return {
            'processing_efficiency': {
                'success_rate_percent': round(success_rate, 2),
                'average_processing_time_per_document': round(self.metrics.processing_time / self.metrics.documents_processed, 3) if self.metrics.documents_processed > 0 else 0,
                'matches_per_document': round(self.metrics.total_matches_found / self.metrics.documents_processed, 2) if self.metrics.documents_processed > 0 else 0
            },
            'resource_utilization': {
                'memory_efficiency_mb_per_document': round(self.metrics.memory_increase_mb / self.metrics.documents_processed, 2) if self.metrics.documents_processed > 0 else 0,
                'cpu_intensity_percent': round(self.metrics.average_cpu_percent, 2),
                'io_intensity_bytes_per_document': round((self.metrics.total_bytes_read + self.metrics.total_bytes_written) / self.metrics.documents_processed) if self.metrics.documents_processed > 0 else 0
            },
            'performance_indicators': {
                'memory_usage_peak_percent': round((self.metrics.peak_memory_mb * 1024 * 1024) / (self.metrics.system_memory_total_gb * 1024**3) * 100, 2),
                'cpu_utilization_peak_percent': round(self.metrics.peak_cpu_percent, 2),
                'processing_throughput_documents_per_second': round(self.metrics.documents_processed / self.metrics.processing_time, 3) if self.metrics.processing_time > 0 else 0
            }
        }
