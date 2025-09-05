
import gc
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import psutil

from .models import PerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceMonitor:

    def __init__(self, monitoring_interval: float = 1.0, max_samples: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.max_samples = max_samples
        self.metrics = PerformanceMetrics()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        self.process = psutil.Process()


        # FIXED: Thread-safe collections with bounded size to prevent memory leaks
        self._lock = threading.RLock()
        self._cpu_samples: List[float] = []
        self._gpu_utilization: List[float] = []
        self._gpu_memory_usage: List[float] = []
        self._file_operations: List[Dict[str, Any]] = []


        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 90.0
        self.disk_critical_threshold = 95.0


        self._initialize_system_metrics()


        self.gpu_monitor = self._initialize_gpu_monitoring()

        logger.info(f"Performance monitor initialized with max_samples={max_samples}")

    def _add_bounded_sample(self, collection: List, value: Any) -> None:
        """Thread-safe method to add sample with bounded size to prevent memory leaks."""
        with self._lock:
            collection.append(value)
            # Enforce max_samples limit to prevent unbounded growth
            if len(collection) > self.max_samples:
                # Remove oldest samples (FIFO)
                excess = len(collection) - self.max_samples
                del collection[:excess]

    def _get_bounded_samples(self, collection: List) -> List:
        """Thread-safe method to get samples."""
        with self._lock:
            return collection.copy()

    def _initialize_system_metrics(self):
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
        try:

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

    def check_system_resources(self) -> Dict[str, Any]:
        status = {
            'memory_ok': True,
            'disk_ok': True,
            'warnings': [],
            'critical': False
        }

        try:

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > self.memory_critical_threshold:
                status['memory_ok'] = False
                status['critical'] = True
                status['warnings'].append(f"CRITICAL: Memory usage {memory_percent:.1f}%")
            elif memory_percent > self.memory_warning_threshold:
                status['warnings'].append(f"WARNING: Memory usage {memory_percent:.1f}%")


            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            if disk_percent > self.disk_critical_threshold:
                status['disk_ok'] = False
                status['critical'] = True
                status['warnings'].append(f"CRITICAL: Disk usage {disk_percent:.1f}%")
            elif disk_percent > self.disk_warning_threshold:
                status['warnings'].append(f"WARNING: Disk usage {disk_percent:.1f}%")


            for warning in status['warnings']:
                logger.warning(warning)


            if memory_percent > self.memory_warning_threshold:
                logger.info("High memory usage detected, triggering garbage collection")
                gc.collect()

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            status['critical'] = True

        return status

    def start_monitoring(self):
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return


        self.metrics.start_time = time.time()
        self.metrics.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.initial_cpu_percent = psutil.cpu_percent()


        self._cpu_samples.clear()
        self._gpu_utilization.clear()
        self._gpu_memory_usage.clear()
        self._file_operations.clear()


        self.stop_monitoring_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring not running")
            return


        self.stop_monitoring_event.set()
        self.monitoring_thread.join(timeout=5.0)


        self.metrics.end_time = time.time()
        self.metrics.processing_time_seconds = self.metrics.end_time - self.metrics.start_time
        self.metrics.final_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.memory_increase_mb = self.metrics.final_memory_mb - self.metrics.initial_memory_mb


        if self._cpu_samples:
            self.metrics.average_cpu_percent = sum(self._cpu_samples) / len(self._cpu_samples)
            self.metrics.cpu_samples_count = len(self._cpu_samples)


        self.metrics.file_operations_count = len(self._file_operations)


        self.metrics.gpu_utilization_samples = len(self._gpu_utilization)
        self.metrics.gpu_memory_samples = len(self._gpu_memory_usage)


        self._calculate_performance_indicators()

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        while not self.stop_monitoring_event.is_set():
            try:

                cpu_percent = psutil.cpu_percent()
                self._add_bounded_sample(self._cpu_samples, cpu_percent)
                self.metrics.peak_cpu_percent = max(self.metrics.peak_cpu_percent, cpu_percent)
                self.metrics.current_cpu_percent = cpu_percent


                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                self.metrics.current_memory_mb = memory_mb


                if self.gpu_monitor and self.metrics.gpu_available:
                    self._monitor_gpu()


                self._apply_memory_limits()


                if len(self._cpu_samples) % 60 == 0:
                    self.check_system_resources()


                self.stop_monitoring_event.wait(self.monitoring_interval)

            except Exception as e:
                logger.debug(f"Error in monitoring loop: {e}")
                self.stop_monitoring_event.wait(self.monitoring_interval)

    def _apply_memory_limits(self):
        # FIXED: Bounds checking now handled by _add_bounded_sample method
        pass

    def _monitor_gpu(self):
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)


            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._add_bounded_sample(self._gpu_utilization, utilization.gpu)


            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage_mb = memory_info.used / 1024 / 1024
            self._add_bounded_sample(self._gpu_memory_usage, memory_usage_mb)

        except Exception as e:
            logger.debug(f"GPU monitoring error: {e}")

    def record_file_operation(self, operation: str, file_path: Path, bytes_transferred: int = 0):
        file_op = {
            'timestamp': time.time(),
            'operation': operation,
            'file_path': str(file_path),
            'bytes_transferred': bytes_transferred
        }
        self._add_bounded_sample(self._file_operations, file_op)

        if operation in ['read', 'load']:
            self.metrics.total_bytes_read += bytes_transferred
        elif operation in ['write', 'save']:
            self.metrics.total_bytes_written += bytes_transferred

    def update_processing_stats(self, documents_processed: int = 0, successful: int = 0,
                               failed: int = 0, matches_found: int = 0):
        self.metrics.total_files_processed = documents_processed
        self.metrics.successful_files = successful
        self.metrics.failed_files = failed
        self.metrics.total_matches_found = matches_found


        self._calculate_performance_indicators()

    def _calculate_performance_indicators(self):

        if self.metrics.total_files_processed > 0:
            self.metrics.success_rate_percent = (self.metrics.successful_files / self.metrics.total_files_processed) * 100


        if self.metrics.total_files_processed > 0:
            self.metrics.memory_efficiency_mb_per_file = self.metrics.memory_increase_mb / self.metrics.total_files_processed


        if self.metrics.total_files_processed > 0:
            self.metrics.io_intensity_bytes_per_file = (self.metrics.total_bytes_read + self.metrics.total_bytes_written) / self.metrics.total_files_processed


        if self.metrics.system_memory_total_gb > 0:
            self.metrics.memory_usage_peak_percent = (self.metrics.peak_memory_mb * 1024 * 1024) / (self.metrics.system_memory_total_gb * 1024**3) * 100


        self.metrics.cpu_utilization_peak_percent = self.metrics.peak_cpu_percent


        if self.metrics.processing_time_seconds > 0:
            self.metrics.processing_throughput_files_per_second = self.metrics.total_files_processed / self.metrics.processing_time_seconds


        if self.metrics.total_files_processed > 0:
            self.metrics.average_processing_time_per_file = self.metrics.processing_time_seconds / self.metrics.total_files_processed


        if self.metrics.total_files_processed > 0:
            self.metrics.average_file_size_mb = (self.metrics.total_bytes_read + self.metrics.total_bytes_written) / (1024 * 1024 * self.metrics.total_files_processed)


        if self.metrics.peak_memory_mb > 0:
            self.metrics.average_memory_mb = (self.metrics.peak_memory_mb * 0.7) + (self.metrics.current_memory_mb * 0.3)
        else:
            self.metrics.average_memory_mb = self.metrics.current_memory_mb


        processing_time_hours = self.metrics.processing_time_seconds / 3600
        self.metrics.total_memory_usage_mb = self.metrics.peak_memory_mb * processing_time_hours if processing_time_hours > 0 else 0

    def get_current_metrics(self) -> Dict[str, Any]:

        self.metrics.current_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.current_cpu_percent = psutil.cpu_percent()

        return {
            'current_memory_mb': round(self.metrics.current_memory_mb, 2),
            'current_cpu_percent': round(self.metrics.current_cpu_percent, 2),
            'peak_memory_mb': round(self.metrics.peak_memory_mb, 2),
            'peak_cpu_percent': round(self.metrics.peak_cpu_percent, 2),
            'average_memory_mb': round(self.metrics.average_memory_mb, 2),
            'average_cpu_percent': round(self.metrics.average_cpu_percent, 2),
            'total_memory_usage_mb': round(self.metrics.total_memory_usage_mb, 2),
            'processing_time_seconds': round(time.time() - self.metrics.start_time, 3)
        }

    def get_performance_metrics(self) -> PerformanceMetrics:
        return self.metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'processing_efficiency': {
                'success_rate_percent': round(self.metrics.success_rate_percent, 2),
                'average_processing_time_per_document': round(self.metrics.average_processing_time_per_file, 3),
                'matches_per_document': round(self.metrics.total_matches_found / self.metrics.total_files_processed, 2) if self.metrics.total_files_processed > 0 else 0
            },
            'resource_utilization': {
                'memory_efficiency_mb_per_document': round(self.metrics.memory_efficiency_mb_per_file, 2),
                'cpu_intensity_percent': round(self.metrics.average_cpu_percent, 2),
                'io_intensity_bytes_per_document': round(self.metrics.io_intensity_bytes_per_file, 2)
            },
            'performance_indicators': {
                'memory_usage_peak_percent': round(self.metrics.memory_usage_peak_percent, 2),
                'cpu_utilization_peak_percent': round(self.metrics.cpu_utilization_peak_percent, 2),
                'processing_throughput_documents_per_second': round(self.metrics.processing_throughput_files_per_second, 3)
            }
        }