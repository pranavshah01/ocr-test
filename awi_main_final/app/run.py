"""
Main entry point for Document Processing Pipeline.

This module orchestrates the complete document processing workflow including:
- Configuration management
- Document discovery and filtering
- Parallel processing with enhanced error handling
- Performance monitoring
- Comprehensive reporting
- Simple file management (move processed files to processed folder)
"""

import sys
import time
import logging
import json
import gc
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import performance monitor
from core.performance_monitor import PerformanceMonitor

# Import the new reporting system
from utils.report_generator import ReportGenerator
from core.models import (
    ProcessingResult, BatchReport, CLIParameters, 
    PerformanceMetrics, PatternInfo, ProcessingStatus
)

# Add the parent directory to the Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import create_argument_parser, load_config_from_args

logger = logging.getLogger(__name__)

# Simple processing log
PROCESSING_LOG_FILE = "processing_log.json"


def save_processing_log(results: List[ProcessingResult], config_obj) -> bool:
    """
    Save simple processing log with results.
    
    Args:
        results: List of processing results
        config_obj: Configuration object
        
    Returns:
        True if log saved successfully, False otherwise
    """
    try:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(results),
            'successful': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success),
            'total_matches': sum(r.matches_found for r in results),
            'results': [r.to_dict() for r in results]
        }
        
        log_path = Path(config_obj.reports_dir) / PROCESSING_LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Processing log saved: {len(results)} documents processed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save processing log: {e}")
        return False


def move_source_file_to_complete(file_path: Path, config_obj) -> bool:
    """
    Move source file to complete folder after processing.
    
    Args:
        file_path: Path to the source file
        config_obj: Configuration object
        
    Returns:
        True if moved successfully, False otherwise
    """
    try:
        # Create complete folder
        complete_dir = Path(config_obj.complete_dir)
        complete_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the source file to complete directory
        dest_path = complete_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        
        logger.info(f"Moved source file {file_path.name} to complete folder")
        return True
        
    except Exception as e:
        logger.error(f"Failed to move source file {file_path.name}: {e}")
        return False


def save_processed_file_with_suffix(document, file_path: Path, config_obj) -> bool:
    """
    Save processed document with _12NC suffix to processed folder.
    
    Args:
        document: The processed document to save
        file_path: Path to the original source file
        config_obj: Configuration object
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create processed folder
        processed_dir = Path(config_obj.output_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with _12NC suffix
        processed_filename = f"{file_path.stem}{config_obj.suffix}{file_path.suffix}"
        processed_path = processed_dir / processed_filename
        
        # Save the processed document
        document.save(str(processed_path))
        
        logger.info(f"Saved processed file {processed_filename} to processed folder")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save processed file {file_path.name}: {e}")
        return False


def process_with_retry(doc_path: Path, doc_processor, max_retries: int = 3, 
                      base_delay: float = 2.0) -> Optional[ProcessingResult]:
    """
    Process document with retry logic for robustness.
    
    Args:
        doc_path: Path to document to process
        doc_processor: Document processor instance
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        
    Returns:
        ProcessingResult if successful, None if failed after all retries
    """
    for attempt in range(max_retries):
        try:
            # Check system resources before processing
            if hasattr(doc_processor, 'performance_monitor') and doc_processor.performance_monitor:
                resource_status = doc_processor.performance_monitor.check_system_resources()
                if resource_status['critical']:
                    logger.error("System resources critical, stopping processing")
                    return None
            
            # Process the document
            if doc_path.suffix.lower() == '.doc':
                # Convert .doc to .docx first
                success, out_path, msg = doc_processor.convert_if_needed(doc_path)
                if not success:
                    raise Exception(f"Conversion failed: {msg}")
                doc_path = out_path
            
            # Load and process document
            from docx import Document
            document = Document(doc_path)
            
            # Create processing result
            result = create_pending_result(doc_path, doc_processor.config, doc_processor)
            
            # Process with text detection
            success = doc_processor.process_document_text(document, result)
            
            if success:
                result.status = ProcessingStatus.PROCESSED
                result.success = True
                logger.info(f"✓ Processed {doc_path.name}: {result.matches_found} matches found")
                # Return both the result and the modified document
                return result, document
            else:
                result.status = ProcessingStatus.ERROR
                result.success = False
                raise Exception("Text processing failed")
                
        except (OSError, IOError) as e:
            # File system errors - retry with exponential backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"File system error on attempt {attempt + 1}, retrying in {delay}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts due to file system error: {e}")
                return None
                
        except MemoryError as e:
            # Memory issues - trigger cleanup and retry once
            logger.warning(f"Memory error on attempt {attempt + 1}: {e}")
            gc.collect()
            if attempt < max_retries - 1:
                logger.info("Triggered garbage collection, retrying...")
                time.sleep(5)
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts due to memory error: {e}")
                return None
                
        except Exception as e:
            # Other errors - log and continue
            logger.error(f"Unexpected error processing {doc_path.name}: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                return None
    
    return None


def log_progress(current: int, total: int, start_time: float, 
                successful: int, failed: int, performance_monitor: Optional[PerformanceMonitor] = None):
    """
    Log detailed progress information for long-running operations.
    
    Args:
        current: Current document number
        total: Total documents to process
        start_time: Start time timestamp
        successful: Number of successful documents
        failed: Number of failed documents
        performance_monitor: Performance monitor instance
    """
    elapsed = time.time() - start_time
    elapsed_hours = elapsed / 3600
    
    # Calculate progress percentage
    progress_pct = (current / total * 100) if total > 0 else 0
    
    # Calculate estimated time remaining
    if current > 0:
        avg_time_per_doc = elapsed / current
        remaining_docs = total - current
        eta_seconds = remaining_docs * avg_time_per_doc
        eta_hours = eta_seconds / 3600
        eta_str = f"{eta_hours:.1f}h remaining"
    else:
        eta_str = "calculating..."
    
    # Get performance metrics if available
    perf_info = ""
    if performance_monitor:
        metrics = performance_monitor.get_current_metrics()
        perf_info = f" | Memory: {metrics['current_memory_mb']:.1f}MB | CPU: {metrics['current_cpu_percent']:.1f}%"
    
    # Log progress
    logger.info(f"Progress: {current}/{total} ({progress_pct:.1f}%) | "
               f"Success: {successful} | Failed: {failed} | "
               f"Elapsed: {elapsed_hours:.1f}h | {eta_str}{perf_info}")
    
    # Log every 10% progress
    if current > 0 and current % max(1, total // 10) == 0:
        logger.info(f"=== {progress_pct:.0f}% COMPLETE ===")


def main():
    """
    Main entry point for the document processing pipeline.
    
    This function parses CLI arguments, discovers files, processes them,
    moves processed files to processed folder, and generates reports.
    """
    # Initialize performance monitor
    performance_monitor = None
    start_time = time.time()
    
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        config_obj = load_config_from_args(args)
        
        # Configure logging to show INFO level messages
        log_file = Path(config_obj.logs_dir) / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("=" * 80)
        logger.info("DOCUMENT PROCESSING PIPELINE STARTED")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Configuration: {config_obj}")
        
        # Initialize performance monitoring
        print("\nINITIALIZING PERFORMANCE MONITOR...")
        performance_monitor = PerformanceMonitor(monitoring_interval=1.0, max_samples=1000)
        performance_monitor.start_monitoring()
        
        print("\nDISCOVERING FILES...")
        discovered = discover_input_files(config_obj)
        logger.info(f"Found {len(discovered)} documents to process")
        
        # Record file discovery operation
        if performance_monitor:
            total_size = sum(p.stat().st_size for p in discovered)
            performance_monitor.record_file_operation("discover", Path("source"), total_size)
        
        print("\nINITIALIZING DOCUMENT PROCESSOR...")
        from app.core.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor(config_obj)
        doc_processor.initialize()
        
        # Add performance monitor to doc processor for resource checking
        doc_processor.performance_monitor = performance_monitor
        
        print("\nPROCESSING DOCUMENTS...")
        successful_processing = 0
        failed_processing = 0
        total_matches = 0
        results = []
        
        # Process documents with robust error handling
        for i, doc_path in enumerate(discovered):
            try:
                # Log progress every 10 documents
                if i % 10 == 0:
                    log_progress(i, len(discovered), start_time, successful_processing, 
                               failed_processing, performance_monitor)
                
                # Process document with retry logic
                process_result = process_with_retry(doc_path, doc_processor)
                
                if process_result:
                    # Unpack the result and modified document
                    if isinstance(process_result, tuple):
                        result, modified_document = process_result
                    else:
                        result = process_result
                        modified_document = None
                    
                    successful_processing += 1
                    total_matches += result.matches_found
                    results.append(result)
                    
                    # Save processed document with _12NC suffix to processed folder
                    if result.processed_file_name:
                        # Create the processed document path
                        processed_path = Path(config_obj.output_dir) / result.processed_file_name
                        processed_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save the modified document (with reconstruction applied)
                        if modified_document:
                            modified_document.save(str(processed_path))
                            logger.info(f"Saved processed file with reconstruction: {result.processed_file_name}")
                        else:
                            # Fallback: load the original document (this shouldn't happen if reconstruction worked)
                            from docx import Document
                            document = Document(doc_path)
                            document.save(str(processed_path))
                            logger.warning(f"Saved original document (no modified document found): {result.processed_file_name}")
                    
                    # Move source file to complete folder
                    move_source_file_to_complete(doc_path, config_obj)
                    
                    # Generate individual report
                    report_generator = ReportGenerator(output_dir=config_obj.reports_dir, config=config_obj)
                    report_generator.generate_document_reports(result)
                    
                else:
                    failed_processing += 1
                    # Create failed result
                    failed_result = create_pending_result(doc_path, config_obj, doc_processor)
                    failed_result.status = ProcessingStatus.ERROR
                    failed_result.success = False
                    failed_result.error_message = "Processing failed after all retries"
                    results.append(failed_result)
                
                # Check system resources
                if performance_monitor:
                    resource_status = performance_monitor.check_system_resources()
                    if resource_status['critical']:
                        logger.error("System resources critical, stopping processing")
                        break
                
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                failed_processing += 1
                logger.error(f"Unexpected error processing {doc_path.name}: {e}")
                # Create failed result
                failed_result = create_pending_result(doc_path, config_obj, doc_processor)
                failed_result.status = ProcessingStatus.ERROR
                failed_result.success = False
                failed_result.error_message = str(e)
                results.append(failed_result)
        
        # Final progress log
        log_progress(len(discovered), len(discovered), start_time, successful_processing, 
                   failed_processing, performance_monitor)
        
        # Update performance monitor with final processing stats
        if performance_monitor:
            performance_monitor.update_processing_stats(
                documents_processed=len(results),
                successful=successful_processing,
                failed=failed_processing,
                matches_found=total_matches
            )
        
        print(f"\nPROCESSING COMPLETED: {successful_processing} successful, {failed_processing} failed, {total_matches} total matches")
        
        print("\nGENERATING REPORTS...")
        generate_reports_from_results(config_obj, results, performance_monitor, args)
        
        # Save simple processing log
        save_processing_log(results, config_obj)
        
        print("Configuration loaded successfully!")
        print("Reports generated successfully!")
        print(f"Source files moved to: {config_obj.complete_dir}/")
        print(f"Processed files saved to: {config_obj.output_dir}/")
        
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        if performance_monitor:
            performance_monitor.stop_monitoring()
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        if performance_monitor:
            performance_monitor.stop_monitoring()
        sys.exit(1)
    finally:
        # Clean up processors
        if 'doc_processor' in locals():
            doc_processor.cleanup()
        
        # Stop performance monitoring
        if performance_monitor:
            print("\nSTOPPING PERFORMANCE MONITOR...")
            performance_monitor.stop_monitoring()


def discover_input_files(config_obj):
    """Find .doc and .docx files in the source directory within size limits.

    Rules:
    - Exclude any files under folders named 'orig_doc_files'
    - Exclude files that already look processed (match processed suffix)
    - Exclude files already in processed folder
    - Dedupe by stem: if both stem.doc and stem.docx exist, prefer .doc
    """
    source_dir = Path(config_obj.source_dir)
    if not source_dir.exists():
        return []
    supported = {".docx", ".doc"}
    candidates = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        # Skip anything inside orig_doc_files
        if "orig_doc_files" in {p.name for p in path.parents}:
            continue
        # Skip already processed outputs (processed suffix)
        if path.suffix.lower() == ".docx" and path.stem.endswith(config_obj.suffix):
            continue
        # Skip files already in processed folder
        if "processed" in {p.name for p in path.parents}:
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= config_obj.min_file_size and size_mb <= config_obj.max_file_size:
            candidates.append(path)
    # Dedupe by stem: prefer .doc if both exist
    by_stem = {}
    for p in candidates:
        stem = p.stem
        if p.suffix.lower() == ".docx" and stem.endswith(config_obj.suffix):
            # treat processed suffix as separate stem (already excluded above, but guard)
            continue
        if stem not in by_stem:
            by_stem[stem] = p
        else:
            existing = by_stem[stem]
            if existing.suffix.lower() == ".docx" and p.suffix.lower() == ".doc":
                by_stem[stem] = p
    return list(by_stem.values())


def create_pending_result(file_path: Path, config_obj, doc_processor=None):
    """Create a ProcessingResult marked as PENDING with basic file info."""
    from core.models import ProcessingResult, ProcessingStatus
    # While pending, we don't display a processed file name yet
    intended_processed_name = f"{file_path.stem}{config_obj.suffix}.docx"
    # Parity fields
    src_name = file_path.name
    src_size = file_path.stat().st_size
    is_doc = file_path.suffix.lower() == ".doc"
    is_docx = file_path.suffix.lower() == ".docx"
    
    # Detect parser if document processor is available
    parser_type = "pending"
    if doc_processor:
        try:
            # Use fallback logic to handle "AttValue too large" errors
            parser_type, error_msg = doc_processor.try_parser_with_fallback(file_path)
            if error_msg:
                print(f"Warning: Parser fallback completed with error for {file_path}: {error_msg}")
        except Exception as e:
            print(f"Warning: Failed to detect parser for {file_path}: {e}")
            parser_type = "error"
    
    result = ProcessingResult(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        success=False,
        processor_type="document_discovery",
        parser=parser_type,  # Now set based on file size detection
        processing_time=0.0,
        error_message="",
        input_path=file_path,
        output_path=Path(config_obj.output_dir) / intended_processed_name,
        matches_found=0,
        total_matches=0,
        file_name=src_name,
        doc_file_name=src_name if is_doc else "",
        doc_file_size=src_size if is_doc else 0,
        docx_file_name=src_name if is_docx else "",
        docx_file_size=src_size if is_docx else 0,
        processed_file_name=intended_processed_name,
        processed_file_size=0,
        processing_time_minutes=0.0,
        status=ProcessingStatus.PENDING,
        failure_reason="",
        total_text_matches=0,
        total_graphics_matches=0,
        total_image_matches=0,
        total_graphics_no_match=0,
        total_image_no_match=0,
        match_details=[]
    )
    return result


def generate_reports_from_results(config_obj, results, performance_monitor=None, original_args=None):
    """Aggregate results and generate batch report."""
    report_generator = ReportGenerator(output_dir=config_obj.reports_dir, config=config_obj)
    
    # Create CLI parameters and pattern info
    cli_parameters = CLIParameters(
        text_mode=config_obj.text_mode,
        text_separator=config_obj.text_separator,
        default_mapping=config_obj.default_mapping,
        ocr_mode=config_obj.ocr_mode,
        ocr_engine=config_obj.ocr_engine,
        use_gpu=config_obj.use_gpu,
        gpu_device=config_obj.gpu_device,
        gpu_available=config_obj.gpu_available,
        max_workers=config_obj.max_workers,
        confidence_min=config_obj.confidence_min,
        verbose=config_obj.verbose,
        patterns_file=str(config_obj.patterns_file),
        mappings_file=str(config_obj.mappings_file),
        source_dir=str(config_obj.source_dir),
        output_dir=str(config_obj.output_dir),
        reports_dir=str(config_obj.reports_dir),
        logs_dir=str(config_obj.logs_dir),
        suffix=config_obj.suffix,
        processing_timeout=config_obj.processing_timeout,
        max_retries=config_obj.max_retries,
        min_file_size=config_obj.min_file_size,
        max_file_size=config_obj.max_file_size
    )
    
    # For now, we'll use the original approach of comparing values
    # This will be improved in a future version with better argument tracking
    cli_parameters.user_provided = set()
    
    # Create pattern info
    patterns = {}
    total_patterns = 0
    try:
        if config_obj.patterns_file.exists():
            import json
            with open(config_obj.patterns_file, 'r') as f:
                raw = json.load(f)
            patterns = {k: v for k, v in raw.items() if not str(k).startswith("_")}
            total_patterns = len(patterns)
    except Exception:
        pass
    
    pattern_info = PatternInfo(
        patterns_file=str(config_obj.patterns_file),
        total_patterns=total_patterns,
        patterns=patterns
    )
    
    # Create batch report with unified ProcessingResult instances
    batch_report = BatchReport(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        cli_parameters=cli_parameters,
        patterns=pattern_info,
        file_reports=results
    )
    
    # Aggregate the batch report
    batch_report.aggregate()
    
    print("Generating batch reports...")
    report_generator.generate_batch_reports(batch_report)
    
    # Skip file-level while pending
    pending_count = sum(1 for r in results if r.status == ProcessingStatus.PENDING)
    if pending_count:
        print(f"{pending_count} file(s) pending - skipping file-level reports.")
    print("All reports generated successfully!")


if __name__ == "__main__":
    main()