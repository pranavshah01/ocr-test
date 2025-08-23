"""
Comprehensive Error Handling System for Document Processing Pipeline.

This module provides robust error handling with multiple fallback mechanisms,
retry logic, and graceful degradation for all processing components.
"""

import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import functools

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    CONVERSION = "conversion"
    LOADING = "loading"
    PROCESSING = "processing"
    MEMORY = "memory"
    NETWORK = "network"
    PERMISSION = "permission"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

@dataclass
class ProcessingError:
    """Container for processing error information."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    fallback_used: bool = False
    fallback_method: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

class ErrorHandler:
    """
    Comprehensive error handling system with fallback mechanisms.
    
    This class provides robust error handling with:
    - Multiple fallback strategies
    - Retry logic with exponential backoff
    - Error classification and severity assessment
    - Graceful degradation
    - Comprehensive error logging and reporting
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will be exponential)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.errors: List[ProcessingError] = []
        self.fallback_strategies: Dict[str, List[Callable]] = {}
        self.retry_strategies: Dict[str, Callable] = {}
        
        # Initialize default fallback strategies
        self._initialize_default_fallbacks()
        
        logger.info("Error handler initialized")
    
    def _initialize_default_fallbacks(self):
        """Initialize default fallback strategies for common operations."""
        
        # Document conversion fallbacks
        self.fallback_strategies['conversion'] = [
            self._fallback_libreoffice_conversion,
            self._fallback_word_com_conversion,
            self._fallback_skip_conversion
        ]
        
        # Document loading fallbacks
        self.fallback_strategies['loading'] = [
            self._fallback_enhanced_parser,
            self._fallback_custom_parser,
            self._fallback_basic_parser
        ]
        
        # OCR fallbacks
        self.fallback_strategies['ocr'] = [
            self._fallback_easyocr,
            self._fallback_tesseract,
            self._fallback_hybrid_ocr
        ]
        
        # Processing fallbacks
        self.fallback_strategies['processing'] = [
            self._fallback_text_only_processing,
            self._fallback_basic_processing,
            self._fallback_skip_processing
        ]
    
    def handle_operation(self, operation_name: str, operation_func: Callable, 
                        *args, **kwargs) -> Any:
        """
        Execute operation with comprehensive error handling and fallbacks.
        
        Args:
            operation_name: Name of the operation for error tracking
            operation_func: Function to execute
            *args: Arguments for the operation function
            **kwargs: Keyword arguments for the operation function
            
        Returns:
            Result of the operation or fallback result
        """
        last_error = None
        
        # Try primary operation with retries
        for attempt in range(self.max_retries + 1):
            try:
                result = operation_func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_error = e
                error_info = self._analyze_error(e, operation_name, attempt)
                self.errors.append(error_info)
                
                logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    # Try retry strategy if available
                    if operation_name in self.retry_strategies:
                        try:
                            self.retry_strategies[operation_name](*args, **kwargs)
                            logger.info(f"Retry strategy executed for {operation_name}")
                        except Exception as retry_error:
                            logger.debug(f"Retry strategy failed: {retry_error}")
                    
                    # Wait before retry with exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying {operation_name} in {delay:.1f}s...")
                    time.sleep(delay)
        
        # All retries failed, try fallbacks
        return self._execute_fallbacks(operation_name, last_error, *args, **kwargs)
    
    def _analyze_error(self, error: Exception, operation_name: str, attempt: int) -> ProcessingError:
        """Analyze error and create ProcessingError object."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine severity based on error type and context
        severity = self._determine_severity(error, operation_name)
        
        # Determine category based on error type and message
        category = self._determine_category(error, operation_name)
        
        # Create context information
        context = {
            'operation': operation_name,
            'attempt': attempt,
            'error_type': error_type,
            'max_retries': self.max_retries
        }
        
        return ProcessingError(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            category=category,
            retry_count=attempt,
            max_retries=self.max_retries,
            context=context,
            stack_trace=traceback.format_exc()
        )
    
    def _determine_severity(self, error: Exception, operation_name: str) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['MemoryError', 'OSError', 'PermissionError']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['FileNotFoundError', 'ImportError', 'ConnectionError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (default)
        return ErrorSeverity.LOW
    
    def _determine_category(self, error: Exception, operation_name: str) -> ErrorCategory:
        """Determine error category based on error type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Conversion errors
        if 'conversion' in error_message or 'convert' in error_message:
            return ErrorCategory.CONVERSION
        
        # Loading errors
        if 'load' in error_message or 'parse' in error_message or 'open' in error_message:
            return ErrorCategory.LOADING
        
        # Memory errors
        if error_type == 'MemoryError' or 'memory' in error_message:
            return ErrorCategory.MEMORY
        
        # Permission errors
        if error_type == 'PermissionError' or 'permission' in error_message:
            return ErrorCategory.PERMISSION
        
        # Network errors
        if 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        
        # Validation errors
        if 'validation' in error_message or 'invalid' in error_message:
            return ErrorCategory.VALIDATION
        
        # Processing errors (default for most operations)
        return ErrorCategory.PROCESSING
    
    def _execute_fallbacks(self, operation_name: str, last_error: Exception, 
                          *args, **kwargs) -> Any:
        """Execute fallback strategies for failed operation."""
        logger.info(f"Executing fallbacks for {operation_name}")
        
        # Determine fallback category
        fallback_category = self._get_fallback_category(operation_name)
        
        if fallback_category not in self.fallback_strategies:
            logger.error(f"No fallback strategies available for {operation_name}")
            raise last_error
        
        # Try each fallback strategy
        for i, fallback_func in enumerate(self.fallback_strategies[fallback_category]):
            try:
                logger.info(f"Trying fallback {i + 1} for {operation_name}")
                result = fallback_func(*args, **kwargs)
                
                # Record successful fallback
                self._record_fallback_success(operation_name, fallback_func.__name__)
                
                logger.info(f"Fallback {i + 1} succeeded for {operation_name}")
                return result
                
            except Exception as fallback_error:
                logger.warning(f"Fallback {i + 1} failed for {operation_name}: {fallback_error}")
                continue
        
        # All fallbacks failed
        logger.error(f"All fallbacks failed for {operation_name}")
        raise last_error
    
    def _get_fallback_category(self, operation_name: str) -> str:
        """Determine fallback category based on operation name."""
        operation_lower = operation_name.lower()
        
        if 'convert' in operation_lower or 'doc' in operation_lower:
            return 'conversion'
        elif 'load' in operation_lower or 'parse' in operation_lower:
            return 'loading'
        elif 'ocr' in operation_lower or 'image' in operation_lower:
            return 'ocr'
        else:
            return 'processing'
    
    def _record_fallback_success(self, operation_name: str, fallback_method: str):
        """Record successful fallback usage."""
        # Update the last error to indicate fallback was used
        if self.errors:
            last_error = self.errors[-1]
            last_error.fallback_used = True
            last_error.fallback_method = fallback_method
    
    # Fallback strategy implementations
    
    def _fallback_libreoffice_conversion(self, *args, **kwargs):
        """Fallback: Try LibreOffice conversion."""
        logger.info("Trying LibreOffice conversion fallback")
        # Implementation would go here
        raise NotImplementedError("LibreOffice conversion not implemented")
    
    def _fallback_word_com_conversion(self, *args, **kwargs):
        """Fallback: Try Microsoft Word COM conversion."""
        logger.info("Trying Microsoft Word COM conversion fallback")
        # Implementation would go here
        raise NotImplementedError("Word COM conversion not implemented")
    
    def _fallback_skip_conversion(self, *args, **kwargs):
        """Fallback: Skip conversion and return original file."""
        logger.info("Skipping conversion fallback")
        # Return original file path
        return args[0] if args else None
    
    def _fallback_enhanced_parser(self, *args, **kwargs):
        """Fallback: Use enhanced parser for document loading."""
        logger.info("Trying enhanced parser fallback")
        # Implementation would go here
        raise NotImplementedError("Enhanced parser not implemented")
    
    def _fallback_custom_parser(self, *args, **kwargs):
        """Fallback: Use custom parser for document loading."""
        logger.info("Trying custom parser fallback")
        # Implementation would go here
        raise NotImplementedError("Custom parser not implemented")
    
    def _fallback_basic_parser(self, *args, **kwargs):
        """Fallback: Use basic parser for document loading."""
        logger.info("Trying basic parser fallback")
        # Implementation would go here
        raise NotImplementedError("Basic parser not implemented")
    
    def _fallback_easyocr(self, *args, **kwargs):
        """Fallback: Use EasyOCR for text extraction."""
        logger.info("Trying EasyOCR fallback")
        # Implementation would go here
        raise NotImplementedError("EasyOCR not implemented")
    
    def _fallback_tesseract(self, *args, **kwargs):
        """Fallback: Use Tesseract for text extraction."""
        logger.info("Trying Tesseract fallback")
        # Implementation would go here
        raise NotImplementedError("Tesseract not implemented")
    
    def _fallback_hybrid_ocr(self, *args, **kwargs):
        """Fallback: Use hybrid OCR approach."""
        logger.info("Trying hybrid OCR fallback")
        # Implementation would go here
        raise NotImplementedError("Hybrid OCR not implemented")
    
    def _fallback_text_only_processing(self, *args, **kwargs):
        """Fallback: Process only text content."""
        logger.info("Trying text-only processing fallback")
        # Implementation would go here
        raise NotImplementedError("Text-only processing not implemented")
    
    def _fallback_basic_processing(self, *args, **kwargs):
        """Fallback: Use basic processing approach."""
        logger.info("Trying basic processing fallback")
        # Implementation would go here
        raise NotImplementedError("Basic processing not implemented")
    
    def _fallback_skip_processing(self, *args, **kwargs):
        """Fallback: Skip processing and return empty result."""
        logger.info("Skipping processing fallback")
        # Return empty result
        return None
    
    def add_fallback_strategy(self, category: str, fallback_func: Callable):
        """Add a custom fallback strategy."""
        if category not in self.fallback_strategies:
            self.fallback_strategies[category] = []
        self.fallback_strategies[category].append(fallback_func)
        logger.info(f"Added fallback strategy for category: {category}")
    
    def add_retry_strategy(self, operation_name: str, retry_func: Callable):
        """Add a custom retry strategy."""
        self.retry_strategies[operation_name] = retry_func
        logger.info(f"Added retry strategy for operation: {operation_name}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.errors:
            return {
                'total_errors': 0,
                'category_distribution': {},
                'severity_distribution': {},
                'fallback_usage_count': 0,
                'fallback_usage_percent': 0.0,
                'most_common_category': None,
                'most_common_severity': None
            }
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        fallback_usage = 0
        
        for error in self.errors:
            # Category counts
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Severity counts
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Fallback usage
            if error.fallback_used:
                fallback_usage += 1
        
        return {
            'total_errors': len(self.errors),
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'fallback_usage_count': fallback_usage,
            'fallback_usage_percent': (fallback_usage / len(self.errors) * 100) if self.errors else 0,
            'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }
    
    def log_error_summary(self):
        """Log a summary of all errors encountered."""
        summary = self.get_error_summary()
        
        logger.info("=" * 60)
        logger.info("ERROR HANDLING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total errors: {summary['total_errors']}")
        logger.info(f"Fallback usage: {summary['fallback_usage_count']} ({summary['fallback_usage_percent']:.1f}%)")
        
        if summary['category_distribution']:
            logger.info("Error categories:")
            for category, count in summary['category_distribution'].items():
                logger.info(f"  {category}: {count}")
        
        if summary['severity_distribution']:
            logger.info("Error severities:")
            for severity, count in summary['severity_distribution'].items():
                logger.info(f"  {severity}: {count}")
        
        logger.info("=" * 60)
    
    def clear_errors(self):
        """Clear all error records."""
        self.errors.clear()
        logger.info("Error records cleared")
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ProcessingError]:
        """Get all errors of a specific category."""
        return [error for error in self.errors if error.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ProcessingError]:
        """Get all errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]
