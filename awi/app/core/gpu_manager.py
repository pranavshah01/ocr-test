"""
GPU Manager for Document Processing Pipeline.

This module handles GPU detection, initialization, and management across different
platforms and frameworks. It provides a unified interface for GPU operations
regardless of the underlying hardware and software stack.

Key Features:
- Automatic GPU detection (CUDA, MPS, CPU fallback)
- GPU memory management and monitoring
- Cross-platform compatibility (Windows, macOS, Linux)
- Framework-specific optimizations (PyTorch, TensorFlow)
- Graceful fallback to CPU when GPU is unavailable
"""

import logging
import platform
import psutil
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    name: str
    memory_total: Optional[int] = None  # Total memory in MB
    memory_available: Optional[int] = None  # Available memory in MB
    memory_used: Optional[int] = None  # Used memory in MB
    utilization: Optional[float] = None  # GPU utilization percentage
    temperature: Optional[float] = None  # GPU temperature in Celsius
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None
    is_primary: bool = False

@dataclass
class GPUConfig:
    """Configuration for GPU operations."""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_id: int = 0
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    allow_growth: bool = True  # Allow memory growth
    enable_mixed_precision: bool = True  # Enable mixed precision for speed
    max_memory_usage: Optional[int] = None  # Max memory usage in MB

class GPUManager:
    """
    Manages GPU detection, initialization, and operations.
    
    This class provides a unified interface for GPU operations across different
    platforms and frameworks. It handles automatic detection of available GPUs,
    memory management, and provides fallback mechanisms when GPU is unavailable.
    
    Attributes:
        available_gpus: Dictionary of detected GPUs
        primary_gpu: The primary GPU to use for processing
        config: GPU configuration settings
        is_initialized: Whether the GPU manager has been initialized
    """
    
    def __init__(self, config: GPUConfig):
        """
        Initialize the GPU manager.
        
        Args:
            config: GPU configuration settings
        """
        self.config = config
        self.available_gpus: Dict[str, GPUInfo] = {}
        self.primary_gpu: Optional[GPUInfo] = None
        self.is_initialized = False
        self._torch_available = False
        self._cuda_available = False
        self._mps_available = False
        
        logger.info(f"Initializing GPU Manager with device type: {config.device_type}")
    
    def detect_gpus(self) -> Dict[str, GPUInfo]:
        """
        Detect available GPUs on the system.
        
        This method scans for available GPUs across different platforms and
        frameworks. It supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU
        fallback. The detection is comprehensive and handles various edge cases.
        
        Returns:
            Dictionary of detected GPUs with their information
        """
        logger.info("Detecting available GPUs...")
        
        # Reset available GPUs
        self.available_gpus = {}
        
        # Detect PyTorch availability
        self._detect_pytorch()
        
        # Detect based on platform and available frameworks
        if platform.system() == "Darwin" and self._mps_available:
            self._detect_mps_gpus()
        elif self._cuda_available:
            self._detect_cuda_gpus()
        else:
            self._detect_cpu_fallback()
        
        # Set primary GPU
        if self.available_gpus:
            primary_key = list(self.available_gpus.keys())[0]
            self.primary_gpu = self.available_gpus[primary_key]
            self.primary_gpu.is_primary = True
            logger.info(f"Primary GPU set to: {self.primary_gpu.name}")
        
        logger.info(f"Detected {len(self.available_gpus)} GPU(s)")
        return self.available_gpus
    
    def _detect_pytorch(self):
        """Detect PyTorch availability and GPU support."""
        try:
            import torch
            self._torch_available = True
            
            # Check CUDA availability
            if torch.cuda.is_available():
                self._cuda_available = True
                logger.info("PyTorch CUDA support detected")
            
            # Check MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._mps_available = True
                logger.info("PyTorch MPS support detected")
                
        except ImportError:
            logger.warning("PyTorch not available - GPU acceleration disabled")
        except Exception as e:
            logger.warning(f"Error detecting PyTorch GPU support: {e}")
    
    def _detect_cuda_gpus(self):
        """Detect NVIDIA CUDA GPUs."""
        try:
            import torch
            
            gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {gpu_count} CUDA GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # MB
                
                gpu_info = GPUInfo(
                    name=gpu_name,
                    memory_total=memory_total,
                    driver_version=torch.version.cuda,
                    compute_capability=f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                )
                
                self.available_gpus[f"cuda:{i}"] = gpu_info
                logger.info(f"CUDA GPU {i}: {gpu_name} ({memory_total} MB)")
                
        except Exception as e:
            logger.error(f"Error detecting CUDA GPUs: {e}")
    
    def _detect_mps_gpus(self):
        """Detect Apple Silicon MPS GPU."""
        try:
            import torch
            
            # For Apple Silicon, we typically have one unified GPU
            gpu_info = GPUInfo(
                name="Apple Silicon GPU",
                memory_total=self._get_mps_memory(),
                driver_version="MPS"
            )
            
            self.available_gpus["mps:0"] = gpu_info
            logger.info(f"MPS GPU: Apple Silicon GPU ({gpu_info.memory_total} MB)")
            
        except Exception as e:
            logger.error(f"Error detecting MPS GPU: {e}")
    
    def _detect_cpu_fallback(self):
        """Set up CPU fallback when no GPU is available."""
        cpu_info = GPUInfo(
            name="CPU",
            memory_total=psutil.virtual_memory().total // (1024 * 1024),  # MB
            driver_version="CPU"
        )
        
        self.available_gpus["cpu:0"] = cpu_info
        logger.info(f"CPU Fallback: {cpu_info.name} ({cpu_info.memory_total} MB)")
    
    def _get_mps_memory(self) -> Optional[int]:
        """Get MPS GPU memory information."""
        try:
            # For Apple Silicon, we can estimate based on system memory
            total_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
            # Apple Silicon typically reserves about 30-50% of system memory for GPU
            estimated_gpu_memory = int(total_memory * 0.4)
            return estimated_gpu_memory
        except Exception:
            return None
    
    def initialize_gpu(self) -> bool:
        """
        Initialize the primary GPU for processing.
        
        This method sets up the GPU environment, allocates memory, and prepares
        the device for processing operations. It handles various initialization
        scenarios and provides detailed logging for debugging.
        
        Returns:
            True if GPU initialization was successful, False otherwise
        """
        if not self.available_gpus:
            logger.warning("No GPUs available for initialization")
            return False
        
        try:
            logger.info(f"Initializing GPU: {self.primary_gpu.name}")
            
            if self.config.device_type == "cuda" and self._cuda_available:
                success = self._initialize_cuda()
            elif self.config.device_type == "mps" and self._mps_available:
                success = self._initialize_mps()
            else:
                success = self._initialize_cpu()
            
            if success:
                self.is_initialized = True
                logger.info("GPU initialization completed successfully")
            else:
                logger.error("GPU initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during GPU initialization: {e}")
            return False
    
    def _initialize_cuda(self) -> bool:
        """Initialize CUDA GPU."""
        try:
            import torch
            
            # Set CUDA device
            torch.cuda.set_device(self.config.device_id)
            
            # Set memory fraction
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            
            # Enable mixed precision if requested
            if self.config.enable_mixed_precision:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Test GPU with a simple operation
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()
            
            logger.info("CUDA GPU initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            return False
    
    def _initialize_mps(self) -> bool:
        """Initialize MPS GPU."""
        try:
            import torch
            
            # Test MPS with a simple operation
            test_tensor = torch.randn(100, 100).to('mps')
            result = torch.mm(test_tensor, test_tensor)
            del test_tensor, result
            
            logger.info("MPS GPU initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MPS initialization failed: {e}")
            return False
    
    def _initialize_cpu(self) -> bool:
        """Initialize CPU processing."""
        try:
            # Set number of threads for CPU processing
            import torch
            torch.set_num_threads(psutil.cpu_count())
            
            logger.info("CPU processing initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CPU initialization failed: {e}")
            return False
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary containing memory usage information
        """
        if not self.primary_gpu:
            return {}
        
        try:
            if self.config.device_type == "cuda" and self._cuda_available:
                import torch
                memory_allocated = torch.cuda.memory_allocated() // (1024 * 1024)  # MB
                memory_reserved = torch.cuda.memory_reserved() // (1024 * 1024)  # MB
                
                return {
                    "device": self.primary_gpu.name,
                    "memory_allocated": memory_allocated,
                    "memory_reserved": memory_reserved,
                    "memory_total": self.primary_gpu.memory_total,
                    "memory_available": self.primary_gpu.memory_total - memory_reserved
                }
            else:
                return {
                    "device": self.primary_gpu.name,
                    "memory_total": self.primary_gpu.memory_total
                }
                
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {}
    
    def cleanup(self):
        """Clean up GPU resources."""
        try:
            if self.config.device_type == "cuda" and self._cuda_available:
                import torch
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            self.is_initialized = False
            logger.info("GPU cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
    
    def get_device_string(self) -> str:
        """Get the device string for PyTorch operations."""
        if self.config.device_type == "cuda":
            return f"cuda:{self.config.device_id}"
        elif self.config.device_type == "mps":
            return "mps"
        else:
            return "cpu"
