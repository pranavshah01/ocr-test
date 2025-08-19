"""
Processor Factory for Document Processing Pipeline.

This module provides a factory pattern for creating and managing different types
of document processors. It handles processor initialization, configuration,
and lifecycle management with proper error handling and resource management.

Key Features:
- Factory pattern for processor creation
- Processor initialization and configuration
- Resource management and cleanup
- Error handling and recovery
- Processor lifecycle management
"""

import logging
import time
from typing import Dict, Any, Optional, Type, Callable
from dataclasses import dataclass
from pathlib import Path

import importlib

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration for a processor."""
    processor_type: str
    enabled: bool = True
    priority: int = 0
    timeout: int = 300  # 5 minutes default
    max_retries: int = 3
    retry_delay: float = 1.0
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

@dataclass
class ProcessorInfo:
    """Information about a processor."""
    name: str
    processor_type: str
    version: str
    description: str
    supported_formats: list
    is_initialized: bool = False
    initialization_time: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0
    error_count: int = 0

class ProcessorFactory:
    """
    Factory for creating and managing document processors.
    
    This class provides a centralized factory for creating different types
    of document processors. It handles processor initialization, configuration,
    and lifecycle management with proper error handling and resource management.
    
    Attributes:
        processors: Dictionary of registered processors
        processor_configs: Configuration for each processor type
        processor_info: Information about each processor
        initialization_order: Order in which processors should be initialized
    """
    
    def __init__(self):
        """Initialize the processor factory."""
        self.processors: Dict[str, Any] = {}
        self.processor_configs: Dict[str, ProcessorConfig] = {}
        self.processor_info: Dict[str, ProcessorInfo] = {}
        self.initialization_order = []
        
        # Register default processor types
        self._register_default_processors()
        
        logger.info("Processor Factory initialized")
    
    def _register_default_processors(self):
        """Register default processor types."""
        default_processors = {
            'text': {
                'name': 'Text Processor',
                'type': 'text',
                'version': '1.0.0',
                'description': 'Processes DOCX documents with pattern matching and text replacement',
                'supported_formats': ['.docx'],
                'module_path': 'app.processors.text_processor',
                'class_name': 'TextProcessor'
            },
            'graphics': {
                'name': 'Graphics Processor',
                'type': 'graphics',
                'version': '1.0.0',
                'description': 'Processes graphics and diagrams (placeholder for future implementation)',
                'supported_formats': ['.svg', '.pdf'],
                'module_path': 'app.processors.graphics_processor',
                'class_name': 'GraphicsProcessor'
            },
            'image': {
                'name': 'Image Processor',
                'type': 'image',
                'version': '1.0.0',
                'description': 'Processes images with OCR capabilities (placeholder for future implementation)',
                'supported_formats': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
                'module_path': 'app.processors.image_processor',
                'class_name': 'ImageProcessor'
            }
        }
        
        for processor_type, info in default_processors.items():
            self.register_processor_type(processor_type, info)
        
        # Configure processors with default settings
        text_config = ProcessorConfig(
            processor_type="text",
            enabled=True,
            priority=1,
            timeout=300,
            max_retries=3,
            config={
                'mode': 'append',  # Default to append mode
                'patterns_file': 'patterns.json',
                'mappings_file': 'mapping.json'
            }
        )
        self.configure_processor("text", text_config)
        
        graphics_config = ProcessorConfig(
            processor_type="graphics",
            enabled=False,  # Disabled until implemented
            priority=2,
            timeout=300,
            max_retries=3,
            config={
                'mode': 'replace',  # Default to replace mode
                'patterns_file': 'patterns.json',
                'mappings_file': 'mapping.json'
            }
        )
        self.configure_processor("graphics", graphics_config)
        
        image_config = ProcessorConfig(
            processor_type="image",
            enabled=False,  # Disabled until implemented
            priority=3,
            timeout=300,
            max_retries=3,
            config={
                'ocr_mode': 'append',  # Default to append mode
                'ocr_engine': 'easyocr',  # Default OCR engine
                'use_gpu': True,  # Default to GPU acceleration
                'confidence_threshold': 0.4,  # Default confidence threshold
                'patterns_file': 'patterns.json',
                'mappings_file': 'mapping.json'
            }
        )
        self.configure_processor("image", image_config)
    
    def register_processor_type(self, processor_type: str, info: Dict[str, Any]):
        """
        Register a new processor type.
        
        Args:
            processor_type: Type identifier for the processor
            info: Dictionary containing processor information
        """
        processor_info = ProcessorInfo(
            name=info['name'],
            processor_type=info['type'],
            version=info['version'],
            description=info['description'],
            supported_formats=info['supported_formats']
        )
        
        self.processor_info[processor_type] = processor_info
        
        # Store module path and class name for dynamic loading
        if 'module_path' in info and 'class_name' in info:
            processor_info.module_path = info['module_path']
            processor_info.class_name = info['class_name']
        
        logger.info(f"Registered processor type: {processor_type} ({info['name']})")
    
    def configure_processor(self, processor_type: str, config: ProcessorConfig):
        """
        Configure a processor type.
        
        Args:
            processor_type: Type identifier for the processor
            config: Processor configuration
        """
        self.processor_configs[processor_type] = config
        logger.info(f"Configured processor: {processor_type}")
    
    def initialize_processor(self, processor_type: str, **kwargs) -> Optional[Any]:
        """
        Initialize a processor of the specified type.
        
        This method dynamically loads and initializes a processor based on
        its type. It handles module imports, class instantiation, and
        proper error handling during initialization.
        
        Args:
            processor_type: Type identifier for the processor
            **kwargs: Additional initialization parameters
            
        Returns:
            Initialized processor instance or None if initialization failed
        """
        if processor_type not in self.processor_info:
            logger.error(f"Unknown processor type: {processor_type}")
            return None
        
        processor_info = self.processor_info[processor_type]
        
        if processor_info.is_initialized:
            logger.debug(f"Processor {processor_type} already initialized")
            return self.processors.get(processor_type)
        
        start_time = time.time()
        
        try:
            logger.info(f"Initializing processor: {processor_type}")
            
            # Get processor configuration
            config = self.processor_configs.get(processor_type, ProcessorConfig(processor_type))
            
            # Check if processor is enabled
            if not config.enabled:
                logger.warning(f"Processor {processor_type} is disabled")
                return None
            
            # Dynamic import and instantiation
            processor = self._create_processor_instance(processor_type, processor_info, **kwargs)
            
            if processor:
                # Initialize the processor
                if hasattr(processor, 'initialize'):
                    processor.initialize(**config.config)
                
                # Store processor instance
                self.processors[processor_type] = processor
                processor_info.is_initialized = True
                processor_info.initialization_time = time.time() - start_time
                
                logger.info(f"Processor {processor_type} initialized successfully in {processor_info.initialization_time:.3f}s")
                return processor
            else:
                logger.error(f"Failed to create processor instance: {processor_type}")
                return None
                
        except Exception as e:
            processor_info.error_count += 1
            logger.error(f"Error initializing processor {processor_type}: {e}")
            return None
    
    def _create_processor_instance(self, processor_type: str, processor_info: ProcessorInfo, **kwargs) -> Optional[Any]:
        """
        Create a processor instance through dynamic loading.
        
        Args:
            processor_type: Type identifier for the processor
            processor_info: Processor information
            **kwargs: Additional initialization parameters
            
        Returns:
            Processor instance or None if creation failed
        """
        try:
            # Check if we have module path and class name
            if not hasattr(processor_info, 'module_path') or not hasattr(processor_info, 'class_name'):
                logger.warning(f"No module path or class name for processor: {processor_type}")
                return None
            
            # Import the module
            module = importlib.import_module(processor_info.module_path)
            
            # Get the class
            processor_class = getattr(module, processor_info.class_name)
            
            # Get processor configuration
            config = self.processor_configs.get(processor_type, ProcessorConfig(processor_type))
            
            # Prepare initialization parameters
            init_params = {}
            
            # Add configuration parameters if available
            if hasattr(config, 'config') and config.config:
                init_params.update(config.config)
            
            # Add additional kwargs (these will override config if there are conflicts)
            init_params.update(kwargs)
            
            # Create instance with combined parameters
            processor = processor_class(**init_params)
            
            return processor
            
        except ImportError as e:
            logger.error(f"Could not import module for processor {processor_type}: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Could not find class {processor_info.class_name} in module: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating processor instance {processor_type}: {e}")
            return None
    
    def get_processor(self, processor_type: str) -> Optional[Any]:
        """
        Get an initialized processor of the specified type.
        
        Args:
            processor_type: Type identifier for the processor
            
        Returns:
            Processor instance or None if not available
        """
        if processor_type in self.processors:
            processor_info = self.processor_info[processor_type]
            processor_info.last_used = time.time()
            processor_info.usage_count += 1
            return self.processors[processor_type]
        
        # Try to initialize if not already done
        return self.initialize_processor(processor_type)
    
    def get_processor_for_file(self, file_path: Path) -> Optional[Any]:
        """
        Get the appropriate processor for a given file.
        
        This method determines the best processor to use based on the file
        extension and processor capabilities. It automatically initializes
        the processor if needed.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Appropriate processor instance or None if no suitable processor found
        """
        file_extension = file_path.suffix.lower()
        
        # Find processor that supports this file type
        for processor_type, processor_info in self.processor_info.items():
            if file_extension in processor_info.supported_formats:
                logger.debug(f"Found processor {processor_type} for file type {file_extension}")
                return self.get_processor(processor_type)
        
        logger.warning(f"No processor found for file type: {file_extension}")
        return None
    
    def initialize_all_processors(self, **kwargs) -> Dict[str, bool]:
        """
        Initialize all registered processors.
        
        Args:
            **kwargs: Additional initialization parameters for all processors
            
        Returns:
            Dictionary mapping processor types to initialization success status
        """
        results = {}
        
        logger.info("Initializing all processors...")
        
        for processor_type in self.processor_info.keys():
            try:
                processor = self.initialize_processor(processor_type, **kwargs)
                results[processor_type] = processor is not None
            except Exception as e:
                logger.error(f"Error initializing processor {processor_type}: {e}")
                results[processor_type] = False
        
        # Log initialization results
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Processor initialization completed: {successful}/{total} successful")
        
        return results
    
    def get_processor_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all processors.
        
        Returns:
            Dictionary containing status information for each processor
        """
        status = {}
        
        for processor_type, processor_info in self.processor_info.items():
            config = self.processor_configs.get(processor_type)
            
            status[processor_type] = {
                'name': processor_info.name,
                'version': processor_info.version,
                'description': processor_info.description,
                'supported_formats': processor_info.supported_formats,
                'is_initialized': processor_info.is_initialized,
                'is_enabled': config.enabled if config else True,
                'initialization_time': processor_info.initialization_time,
                'last_used': processor_info.last_used,
                'usage_count': processor_info.usage_count,
                'error_count': processor_info.error_count,
                'priority': config.priority if config else 0
            }
        
        return status
    
    def cleanup_processor(self, processor_type: str):
        """
        Clean up a specific processor.
        
        Args:
            processor_type: Type identifier for the processor
        """
        if processor_type in self.processors:
            processor = self.processors[processor_type]
            
            try:
                # Call cleanup method if available
                if hasattr(processor, 'cleanup'):
                    processor.cleanup()
                
                # Remove from processors dictionary
                del self.processors[processor_type]
                
                # Update processor info
                processor_info = self.processor_info[processor_type]
                processor_info.is_initialized = False
                
                logger.info(f"Cleaned up processor: {processor_type}")
                
            except Exception as e:
                logger.error(f"Error cleaning up processor {processor_type}: {e}")
    
    def cleanup_all_processors(self):
        """Clean up all initialized processors."""
        logger.info("Cleaning up all processors...")
        
        for processor_type in list(self.processors.keys()):
            self.cleanup_processor(processor_type)
        
        logger.info("All processors cleaned up")
    
    def reset_processor(self, processor_type: str):
        """
        Reset a processor (cleanup and reinitialize).
        
        Args:
            processor_type: Type identifier for the processor
        """
        logger.info(f"Resetting processor: {processor_type}")
        
        # Clean up existing processor
        self.cleanup_processor(processor_type)
        
        # Reinitialize
        self.initialize_processor(processor_type)
    
    def get_supported_formats(self) -> Dict[str, list]:
        """
        Get all supported file formats by processor type.
        
        Returns:
            Dictionary mapping processor types to supported file formats
        """
        return {
            processor_type: processor_info.supported_formats
            for processor_type, processor_info in self.processor_info.items()
        }
    
    def is_format_supported(self, file_extension: str) -> bool:
        """
        Check if a file format is supported by any processor.
        
        Args:
            file_extension: File extension to check (e.g., '.pdf')
            
        Returns:
            True if the format is supported, False otherwise
        """
        file_extension = file_extension.lower()
        
        for processor_info in self.processor_info.values():
            if file_extension in processor_info.supported_formats:
                return True
        
        return False
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processor usage and performance.
        
        Returns:
            Dictionary containing processor statistics
        """
        total_processors = len(self.processor_info)
        initialized_processors = sum(1 for info in self.processor_info.values() if info.is_initialized)
        total_usage = sum(info.usage_count for info in self.processor_info.values())
        total_errors = sum(info.error_count for info in self.processor_info.values())
        
        return {
            'total_processors': total_processors,
            'initialized_processors': initialized_processors,
            'initialization_rate': (initialized_processors / total_processors * 100) if total_processors > 0 else 0,
            'total_usage_count': total_usage,
            'total_error_count': total_errors,
            'error_rate': (total_errors / max(1, total_usage) * 100),
            'average_initialization_time': (
                sum(info.initialization_time for info in self.processor_info.values()) / 
                max(1, initialized_processors)
            )
        }
