"""
Main entry point for the Document Processing Pipeline.
Provides CLI interface and coordinates the complete processing workflow.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add the parent directory to the path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import create_argument_parser, load_config_from_args, get_default_config
from app.core.document_processor import create_document_processor
from app.core.parallel_manager import create_parallel_manager, FileDiscovery
from app.processors.text_processor import create_text_processor
from app.processors.graphics_processor import create_graphics_processor
from app.processors.image_processor import create_image_processor
from app.utils.report_generator import create_report_generator
from app.utils.shared_constants import SharedUtilities, SUPPORTED_FORMATS
from app.utils.platform_utils import get_system_info, validate_platform_support

logger = logging.getLogger(__name__)

def setup_logging(config) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Processing configuration
        
    Returns:
        Configured logger
    """
    log_level = logging.DEBUG if config.verbose else logging.INFO
    
    # Create logs directory
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up detailed logger with file output
    log_file = config.logs_dir / "processing.log"
    main_logger = SharedUtilities.setup_detailed_logger(
        "document_processor", 
        str(log_file), 
        log_level
    )
    
    return main_logger

def display_system_info(config):
    """Display system information and configuration."""
    logger.info("=== Document Processing Pipeline ===")
    logger.info("System Information:")
    
    # Platform validation
    platform_supported = validate_platform_support()
    logger.info(f"Platform supported: {platform_supported}")
    
    # System info
    system_info = get_system_info()
    logger.info(f"Platform: {system_info['platform_info']['system']} {system_info['platform_info']['machine']}")
    logger.info(f"Available conversion tools: {system_info['available_tools']}")
    logger.info(f"Preferred conversion tool: {system_info['preferred_tool']}")
    
    # Configuration
    logger.info("Configuration:")
    logger.info(f"Text mode: {config.text_mode}")
    logger.info(f"OCR mode: {config.ocr_mode}")
    logger.info(f"GPU enabled: {config.use_gpu}")
    logger.info(f"GPU device: {getattr(config, 'gpu_device', 'unknown')}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(f"Confidence threshold: {config.confidence_min}")
    logger.info(f"Source directory: {config.source_dir}")
    logger.info(f"Output directory: {config.output_dir}")

def create_processor_factory(config):
    """
    Create a factory function for document processors.
    
    Args:
        config: Processing configuration
        
    Returns:
        Factory function that creates configured DocumentProcessor instances
    """
    def processor_factory():
        # Create main document processor
        processor = create_document_processor(config)
        
        # Create and set individual processors
        text_processor = create_text_processor(
            processor.patterns, 
            processor.mappings, 
            config.text_mode
        )
        
        graphics_processor = create_graphics_processor(
            processor.patterns, 
            processor.mappings, 
            config.text_mode  # Use same mode as text
        )
        
        image_processor = create_image_processor(
            processor.patterns, 
            processor.mappings, 
            config.ocr_mode,
            ocr_engine="easyocr",  # Default to EasyOCR
            use_gpu=config.use_gpu,
            confidence_threshold=config.confidence_min
        )
        
        # Set processors in main processor
        processor.set_processors(text_processor, graphics_processor, image_processor)
        
        return processor
    
    return processor_factory

def discover_and_filter_documents(config) -> list[Path]:
    """
    Discover and filter documents for processing.
    
    Args:
        config: Processing configuration
        
    Returns:
        List of document paths to process
    """
    logger.info(f"Discovering documents in: {config.source_dir}")
    
    # Discover documents
    documents = FileDiscovery.discover_documents(config.source_dir, SUPPORTED_FORMATS)
    
    if not documents:
        logger.warning("No documents found for processing")
        return []
    
    # Filter documents
    filtered_documents = FileDiscovery.filter_documents(documents, max_size_mb=100.0)
    
    logger.info(f"Found {len(filtered_documents)} documents to process")
    for doc in filtered_documents:
        logger.info(f"  - {doc.name}")
    
    return filtered_documents

def process_documents(config, documents: list[Path]):
    """
    Process documents using parallel processing.
    
    Args:
        config: Processing configuration
        documents: List of document paths to process
        
    Returns:
        Batch processing result
    """
    if not documents:
        logger.warning("No documents to process")
        return None
    
    logger.info(f"Starting processing of {len(documents)} documents with {config.max_workers} workers")
    
    # Create processor factory
    processor_factory = create_processor_factory(config)
    
    # Create parallel manager
    parallel_manager = create_parallel_manager(
        max_workers=config.max_workers,
        use_processes=False  # Use threads for better resource sharing
    )
    
    # Process documents
    batch_result = parallel_manager.process_documents_parallel(documents, processor_factory)
    
    logger.info(f"Processing completed: {batch_result.successful_files}/{batch_result.total_files} successful")
    
    return batch_result

def generate_reports(config, batch_result):
    """
    Generate processing reports.
    
    Args:
        config: Processing configuration
        batch_result: Batch processing result
    """
    if not batch_result:
        return
    
    logger.info("Generating reports...")
    
    # Create report generator
    report_generator = create_report_generator(config.reports_dir)
    
    # Generate batch summary reports
    batch_reports = report_generator.generate_batch_reports(batch_result)
    
    if batch_reports:
        logger.info("Batch reports generated:")
        for report_type, report_path in batch_reports.items():
            logger.info(f"  - {report_type.upper()}: {report_path}")
    
    # Generate individual document reports for successful processing
    successful_results = [r for r in batch_result.results if r.success]
    
    if successful_results:
        logger.info(f"Generating individual reports for {len(successful_results)} successful documents...")
        
        for result in successful_results:
            try:
                doc_reports = report_generator.generate_document_reports(result)
                if doc_reports:
                    logger.debug(f"Reports generated for {result.input_path.name}")
            except Exception as e:
                logger.error(f"Failed to generate reports for {result.input_path.name}: {e}")

def display_final_summary(batch_result):
    """
    Display final processing summary.
    
    Args:
        batch_result: Batch processing result
    """
    if not batch_result:
        return
    
    logger.info("=== Processing Summary ===")
    logger.info(f"Total files: {batch_result.total_files}")
    logger.info(f"Successful: {batch_result.successful_files}")
    logger.info(f"Failed: {batch_result.failed_files}")
    logger.info(f"Success rate: {batch_result.success_rate:.1f}%")
    logger.info(f"Total processing time: {batch_result.processing_time:.2f} seconds")
    logger.info(f"Average time per file: {batch_result.processing_time / batch_result.total_files:.2f} seconds")
    logger.info(f"Total matches found: {batch_result.total_matches}")
    
    # Show failed files if any
    failed_results = [r for r in batch_result.results if not r.success]
    if failed_results:
        logger.warning("Failed files:")
        for result in failed_results:
            logger.warning(f"  - {result.input_path.name}: {result.error_message}")

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Load configuration
        config = load_config_from_args(args)
        
        # Set up logging
        setup_logging(config)
        
        # Display system information
        display_system_info(config)
        
        # Discover documents
        documents = discover_and_filter_documents(config)
        
        if not documents:
            logger.error("No documents found to process. Exiting.")
            sys.exit(1)
        
        # Process documents
        batch_result = process_documents(config, documents)
        
        # Generate reports
        generate_reports(config, batch_result)
        
        # Display final summary
        display_final_summary(batch_result)
        
        # Exit with appropriate code
        if batch_result and batch_result.successful_files > 0:
            logger.info("Processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("Processing failed for all documents!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()