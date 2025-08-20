"""
Main entry point for Document Processing Pipeline.

This module orchestrates the complete document processing workflow including:
- Configuration management
- Document discovery and filtering
- Parallel processing with enhanced error handling
- Performance monitoring
- Comprehensive reporting
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.document_processor import DocumentProcessor
from app.core.parallel_manager import ParallelManager
from app.core.performance_monitor import PerformanceMonitor
from app.core.error_handler import ErrorHandler
from app.processors.text_processor import create_text_processor
from app.processors.graphics_processor import create_graphics_processor
from app.processors.image_processor import create_image_processor

from app.utils.report_generator import ReportGenerator
from app.utils.platform_utils import validate_platform_support, get_system_info
from config import create_argument_parser, load_config_from_args

logger = logging.getLogger(__name__)

def setup_logging(config_obj) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        config_obj: Configuration object containing logging settings
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"processing_{timestamp}.log"
    
    # Determine log level based on verbose flag
    log_level = logging.DEBUG if config_obj.verbose else logging.INFO
    
    # Create custom formatter with colors
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colored output for different log levels."""
        
        # ANSI color codes
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        def format(self, record):
            # Add color to the levelname
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            
            # Format the message
            formatted = super().format(record)
            
            # Add color to the entire message for ERROR and WARNING
            if record.levelname == 'ERROR':
                formatted = f"{self.COLORS['ERROR']}{formatted}{self.COLORS['RESET']}"
            elif record.levelname == 'WARNING':
                formatted = f"{self.COLORS['WARNING']}{formatted}{self.COLORS['RESET']}"
            
            return formatted
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    logger.info("=" * 80)
    logger.info("DOCUMENT PROCESSING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Configuration: {config_obj}")

def discover_documents(config_obj) -> List[Path]:
    """
    Discover and filter documents for processing.
    
    Args:
        config_obj: Configuration object containing discovery settings
        
    Returns:
        List of document paths to process
    """
    logger.info("🔍 DISCOVERING DOCUMENTS")
    
    source_dir = Path(config_obj.source_dir)
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return []
    
    # Find all document files
    document_files = []
    supported_extensions = {'.docx', '.doc'}
    
    for file_path in source_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            # Check file size filter
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb >= config_obj.min_file_size and file_size_mb <= config_obj.max_file_size:
                document_files.append(file_path)
            else:
                logger.debug(f"Skipping {file_path.name} (size: {file_size_mb:.1f}MB)")
    
    logger.info(f"Found {len(document_files)} documents to process")
    
    # Log document details
    for doc_path in document_files:
        file_size_mb = doc_path.stat().st_size / (1024 * 1024)
        logger.info(f"  - {doc_path.name} ({file_size_mb:.1f}MB)")
    
    return document_files

def create_processor_factory(config_obj) -> Dict[str, Any]:
    """
    Create processor instances for different content types.
    
    Args:
        config_obj: Configuration object containing processor settings
        
    Returns:
        Dictionary containing processor instances
    """
    logger.info("🔄 CREATING PROCESSORS")
    
    processors = {}
    
    # Load patterns and mappings from files
    try:
        import json
        with open(config_obj.patterns_file, 'r') as f:
            patterns = json.load(f)
        with open(config_obj.mappings_file, 'r') as f:
            mappings = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load patterns/mappings: {e}")
        return processors
    
    # Create text processor
    logger.debug(f"TEXT PROCESSOR DEBUG: About to create text processor with patterns: {len(patterns)}, mappings: {len(mappings)}")
    try:
        text_processor = create_text_processor(
            patterns, 
            mappings, 
            config_obj.text_mode,
            config_obj.text_separator,
            config_obj.default_mapping
        )
        
        if text_processor:
            text_processor.initialize()
            processors['text'] = text_processor
            logger.info("✅ Text processor created and initialized")
        else:
            logger.warning("⚠️ Text processor creation failed")
    except Exception as e:
        logger.error(f"❌ Text processor creation failed: {e}")
    
    # Create graphics processor
    try:
        # Load patterns and mappings for graphics processor
        patterns = {}
        mappings = {}
        
        if config_obj.patterns_file and config_obj.patterns_file.exists():
            import json
            with open(config_obj.patterns_file, 'r') as f:
                patterns = json.load(f)
        
        if config_obj.mappings_file and config_obj.mappings_file.exists():
            import json
            with open(config_obj.mappings_file, 'r') as f:
                mappings = json.load(f)
        
        graphics_processor = create_graphics_processor(
            patterns=patterns,
            mappings=mappings,
            mode=config_obj.text_mode,
            separator=config_obj.text_separator,
            default_mapping=config_obj.default_mapping
        )
        
        if graphics_processor:
            graphics_processor.initialize()
            processors['graphics'] = graphics_processor
            logger.info("✅ Graphics processor created and initialized")
        else:
            logger.warning("⚠️ Graphics processor not available (placeholder)")
    except Exception as e:
        logger.debug(f"Graphics processor not available: {e}")
    
    # Create image processor
    try:
        image_processor = create_image_processor(
            patterns=patterns,
            mappings=mappings,
            mode=config_obj.ocr_mode,
            separator=config_obj.text_separator,
            default_mapping=config_obj.default_mapping,
            ocr_engine=config_obj.ocr_engine,
            use_gpu=config_obj.gpu_available,
            confidence_threshold=config_obj.confidence_min,
        )
        if image_processor:
            image_processor.initialize()
            processors['image'] = image_processor
            logger.info("✅ Image processor created and initialized")
        else:
            logger.warning("⚠️ Image processor not available (placeholder)")
    except Exception as e:
        logger.debug(f"Image processor not available: {e}")


    
    logger.info(f"Created {len(processors)} processor(s)")
    return processors

def main():
    """
    Main entry point for the document processing pipeline.
    
    This function orchestrates the complete workflow:
    1. Parse CLI arguments and load configuration
    2. Setup logging and validate platform support
    3. Discover documents for processing
    4. Create processor instances
    5. Initialize parallel processing manager
    6. Execute document processing with performance monitoring
    7. Generate comprehensive reports
    """
    try:
        # Parse CLI arguments and load configuration
        parser = create_argument_parser()
        args = parser.parse_args()
        config_obj = load_config_from_args(args)
        
        # Setup logging
        setup_logging(config_obj)
        
        # Validate platform support
        if not validate_platform_support():
            logger.error("Platform not supported")
            sys.exit(1)
        
        # Log system information
        system_info = get_system_info()
        logger.info(f"System: {system_info}")
        
        # Discover documents
        document_files = discover_documents(config_obj)
        if not document_files:
            logger.warning("No documents found for processing")
            return
        
        # Create processors
        processors = create_processor_factory(config_obj)
        if not processors:
            logger.error("No processors available")
            return
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor(monitoring_interval=1.0)
        
        # Initialize error handler
        error_handler = ErrorHandler(max_retries=3, retry_delay=1.0)
        
        # Create document processor
        document_processor = DocumentProcessor(config_obj)
        
        # Set processors in document processor
        if 'text' in processors:
            document_processor.text_processor = processors['text']
        if 'graphics' in processors:
            document_processor.graphics_processor = processors['graphics']
        if 'image' in processors:
            document_processor.image_processor = processors['image']
        
        # Initialize parallel manager
        parallel_manager = ParallelManager(
            max_workers=config_obj.max_workers,
            performance_monitor=performance_monitor
        )
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        
        logger.info("🚀 STARTING PROCESSING")
        
        # Track start time for batch processing
        start_time = time.time()
        
        # Process documents directly without complex parallel manager for now
        results = []
        for i, file_path in enumerate(document_files):
            try:
                logger.info(f"Processing document {i+1}/{len(document_files)}: {file_path.name}")
                
                # Record file operation in performance monitor
                try:
                    file_size = file_path.stat().st_size
                    performance_monitor.record_file_operation("read", file_path, file_size)
                except Exception as e:
                    logger.debug(f"Could not record file operation: {e}")
                
                # Process the document
                result = document_processor.process_document(file_path)
                results.append(result)
                
                if result.success:
                    logger.info(f"✅ Successfully processed: {file_path.name}")
                else:
                    logger.warning(f"⚠️ Failed to process: {file_path.name} - {result.error_message}")
                
            except Exception as e:
                logger.error(f"❌ Unexpected error processing {file_path.name}: {e}")
                
                # Create a failed result
                from app.core.processor_interface import ProcessingResult
                failed_result = ProcessingResult(
                    success=False,
                    processor_type="document_processor",
                    processing_time=0.0,
                    error_message=f"Unexpected error: {e}"
                )
                results.append(failed_result)
                
                # Continue with next file
                continue
        
        # Stop performance monitoring
        try:
            performance_monitor.stop_monitoring()
        except Exception as e:
            logger.warning(f"Error stopping performance monitoring: {e}")
        
        # Update performance metrics with processing results
        try:
            successful_count = sum(1 for result in results if result.success)
            failed_count = len(results) - successful_count
            total_matches = sum(result.matches_found for result in results if result.success)
            
            performance_monitor.update_processing_stats(
                documents_processed=len(results),
                successful=successful_count,
                failed=failed_count,
                matches_found=total_matches
            )
        except Exception as e:
            logger.warning(f"Error updating performance metrics: {e}")
        
        # Generate reports
        logger.info("📊 GENERATING REPORTS")
        report_generator = ReportGenerator(output_dir=config_obj.reports_dir, config=config_obj)
        
        try:
            # Generate individual document reports
            for result in results:
                try:
                    if result.success:
                        report_generator.generate_document_reports(result)
                except Exception as e:
                    logger.warning(f"Error generating document report: {e}")
            
            # Generate batch report
            try:
                # Create a batch result from individual results
                from app.core.models import BatchProcessingResult, ProcessingStatistics
                
                # Aggregate statistics
                stats = ProcessingStatistics()
                for result in results:
                    stats.add_result(result)
                
                # Create batch result with proper structure
                batch_result = BatchProcessingResult()
                batch_result.total_documents = len(results)  # Only current run
                batch_result.successful_documents = sum(1 for r in results if r.success)
                batch_result.failed_documents = sum(1 for r in results if not r.success)
                batch_result.total_processing_time = time.time() - start_time
                batch_result.total_matches_found = stats.total_matches
                
                # Add individual results as DocumentInfo objects
                for result in results:
                    from app.core.models import DocumentInfo
                    doc_info = DocumentInfo(
                        file_path=result.input_path,
                        file_size=result.input_path.stat().st_size if result.input_path.exists() else 0,
                        file_extension=result.input_path.suffix,
                        processing_status="success" if result.success else "error",
                        processing_time=result.processing_time,
                        matches_found=result.total_matches,
                        error_message=result.error_message
                    )
                    batch_result.add_document_result(doc_info)
                
                # Get performance metrics
                current_metrics = performance_monitor.get_current_metrics()
                
                # Add statistics and performance metrics to batch result
                batch_result.statistics = stats
                batch_result.individual_results = results
                batch_result.performance_metrics = current_metrics
                
                report_generator.generate_batch_reports(batch_result)
            except Exception as e:
                logger.warning(f"Error generating batch report: {e}")
                
        except Exception as e:
            logger.warning(f"Error in report generation: {e}")
        
        # Log error summary
        try:
            error_handler.log_error_summary()
        except Exception as e:
            logger.warning(f"Error logging error summary: {e}")
            # Try to get basic error info
            try:
                error_count = len(error_handler.errors) if hasattr(error_handler, 'errors') else 0
                logger.info(f"Total errors encountered: {error_count}")
            except Exception:
                logger.info("Error summary unavailable")
        
        # Log performance summary
        try:
            current_metrics = performance_monitor.get_current_metrics()
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Documents processed: {len(results)}")
            logger.info(f"Successful: {successful_count}")
            logger.info(f"Failed: {failed_count}")
            logger.info(f"Success rate: {(successful_count/len(results)*100):.1f}%" if results else "0%")
            logger.info(f"Total matches found: {total_matches}")
            logger.info(f"Total processing time: {current_metrics['processing_time_seconds']:.3f}s")
            logger.info(f"Peak memory usage: {current_metrics['peak_memory_mb']:.1f}MB")
            logger.info(f"Peak CPU usage: {current_metrics['peak_cpu_percent']:.1f}%")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Error logging performance summary: {e}")
        
        # Cleanup
        try:
            # Cleanup converted files
            if hasattr(document_processor, 'converter') and document_processor.converter:
                document_processor.converter.cleanup_converted_files()
            
            # Cleanup processors
            for processor in processors.values():
                try:
                    if hasattr(processor, 'cleanup'):
                        processor.cleanup()
                except Exception as e:
                    logger.debug(f"Error cleaning up processor: {e}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()