
import sys
import time
import logging
import json
import gc
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


sys.path.insert(0, str(Path(__file__).parent))


from core.performance_monitor import PerformanceMonitor


from utils.report_generator import ReportGenerator
from core.models import (
    ProcessingResult, BatchReport, CLIParameters,
    PerformanceMetrics, PatternInfo, ProcessingStatus
)


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import create_argument_parser, load_config_from_args

logger = logging.getLogger(__name__)


PROCESSING_LOG_FILE = "processing_log.json"


def save_processing_log(results: List[ProcessingResult], config_obj) -> bool:
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

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Processing log saved: {len(results)} documents processed")
        return True

    except Exception as e:
        logger.error(f"Failed to save processing log: {e}")
        return False


def move_source_file_to_complete(file_path: Path, config_obj) -> bool:
    try:

        complete_dir = Path(config_obj.complete_dir)
        complete_dir.mkdir(parents=True, exist_ok=True)


        dest_path = complete_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))

        logger.info(f"Moved source file {file_path.name} to complete folder")
        return True

    except Exception as e:
        logger.error(f"Failed to move source file {file_path.name}: {e}")
        return False


def save_processed_file_with_suffix(document, file_path: Path, config_obj) -> bool:
    try:

        processed_dir = Path(config_obj.output_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)


        processed_filename = f"{file_path.stem}{config_obj.suffix}{file_path.suffix}"
        processed_path = processed_dir / processed_filename


        document.save(str(processed_path))

        logger.info(f"Saved processed file {processed_filename} to processed folder")
        return True

    except Exception as e:
        logger.error(f"Failed to save processed file {file_path.name}: {e}")
        return False


def process_with_retry(doc_path: Path, doc_processor, max_retries: int = 3,
                      base_delay: float = 2.0) -> Optional[ProcessingResult]:
    for attempt in range(max_retries):
        try:

            # Enhanced circuit breaker protection
            if hasattr(doc_processor, 'performance_monitor') and doc_processor.performance_monitor:
                resource_status = doc_processor.performance_monitor.check_system_resources()
                if resource_status['critical']:
                    logger.error("System resources critical, stopping processing")
                    return None
            elif hasattr(doc_processor, '_check_system_resources'):
                # Use built-in circuit breaker if performance monitor not available
                if not doc_processor._check_system_resources():
                    logger.error("System resources critical (circuit breaker), stopping processing")
                    return None


            if doc_path.suffix.lower() == '.doc':

                success, out_path, msg = doc_processor.convert_if_needed(doc_path)
                if not success:
                    raise Exception(f"Conversion failed: {msg}")
                doc_path = out_path


            from docx import Document
            
            # Check if this is a large file and prepare accordingly
            file_size_mb = doc_path.stat().st_size / (1024 * 1024)
            if hasattr(doc_processor, 'prepare_for_large_file'):
                doc_processor.prepare_for_large_file(file_size_mb)
            
            # Use the existing parser fallback mechanism from document_processor
            parser_type, error_msg = doc_processor.try_parser_with_fallback(doc_path)
            
            # Check if parser fallback failed
            if parser_type == "failed":
                raise Exception(f"All parsers failed for {doc_path.name}: {error_msg}")
            
            # Load document with the determined parser
            document = Document(doc_path)


            result = create_pending_result(doc_path, doc_processor.config, doc_processor)
            
            # Update the parser field in the result
            result.parser = parser_type

            # Process document with centralized AttValue error handling
            try:
                success = doc_processor.process_document_text(document, result)
            except Exception as text_error:
                if doc_processor.is_attvalue_error(str(text_error)):
                    logger.warning(f"AttValue error in text processing, skipping problematic attributes: {text_error}")
                    # Continue with graphics processing even if text processing failed due to AttValue
                    success = False
                    result.error_message = f"Text processing skipped due to AttValue error: {text_error}"
                else:
                    raise text_error

            try:
                graphics_success = doc_processor.process_document_graphics(document, result)
            except Exception as graphics_error:
                if doc_processor.is_attvalue_error(str(graphics_error)):
                    logger.warning(f"AttValue error in graphics processing, skipping problematic attributes: {graphics_error}")
                    # Continue even if graphics processing failed due to AttValue
                    graphics_success = False
                    if result.error_message:
                        result.error_message += f"; Graphics processing skipped due to AttValue error: {graphics_error}"
                    else:
                        result.error_message = f"Graphics processing skipped due to AttValue error: {graphics_error}"
                else:
                    raise graphics_error


            if success or graphics_success:
                result.status = ProcessingStatus.PROCESSED
                result.success = True
                total_matches = result.matches_found + result.total_graphics_matches
                logger.info(f"Processed {doc_path.name}: {result.matches_found} text matches, {result.total_graphics_matches} graphics matches found")

                return result, document
            else:
                result.status = ProcessingStatus.ERROR
                result.success = False
                raise Exception("Text and graphics processing failed")

        except (OSError, IOError) as e:

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"File system error on attempt {attempt + 1}, retrying in {delay}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts due to file system error: {e}")
                return None

        except MemoryError as e:
            # Force garbage collection on memory errors
            logger.warning(f"Memory error on attempt {attempt + 1}: {e}")
            if hasattr(doc_processor, 'clear_memory'):
                doc_processor.clear_memory(force_gc=True)
            else:
                gc.collect()
            if attempt < max_retries - 1:
                logger.info("Triggered garbage collection, retrying...")
                time.sleep(5)
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts due to memory error: {e}")
                return None

        except Exception as e:

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
    elapsed = time.time() - start_time
    elapsed_hours = elapsed / 3600


    progress_pct = (current / total * 100) if total > 0 else 0


    if current > 0:
        avg_time_per_doc = elapsed / current
        remaining_docs = total - current
        eta_seconds = remaining_docs * avg_time_per_doc
        eta_hours = eta_seconds / 3600
        eta_str = f"{eta_hours:.1f}h remaining"
    else:
        eta_str = "calculating..."


    perf_info = ""
    if performance_monitor:
        metrics = performance_monitor.get_current_metrics()
        perf_info = f" | Memory: {metrics['current_memory_mb']:.1f}MB | CPU: {metrics['current_cpu_percent']:.1f}%"


    logger.info(f"Progress: {current}/{total} ({progress_pct:.1f}%) | "
               f"Success: {successful} | Failed: {failed} | "
               f"Elapsed: {elapsed_hours:.1f}h | {eta_str}{perf_info}")


    if current > 0 and current % max(1, total // 10) == 0:
        logger.info(f"=== {progress_pct:.0f}% COMPLETE ===")


def main():

    performance_monitor = None
    start_time = time.time()

    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        config_obj = load_config_from_args(args)


        log_file = Path(config_obj.logs_dir) / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)


        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),  # Fix for Windows charset issues
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )

        logger.info("=" * 80)
        logger.info("DOCUMENT PROCESSING PIPELINE STARTED")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Configuration: {config_obj}")





        print("\nINITIALIZING PERFORMANCE MONITOR...")
        performance_monitor = PerformanceMonitor(monitoring_interval=1.0, max_samples=1000)
        performance_monitor.start_monitoring()

        print("\nDISCOVERING FILES...")
        discovered = discover_input_files(config_obj)
        logger.info(f"Found {len(discovered)} documents to process")


        if performance_monitor:
            total_size = sum(p.stat().st_size for p in discovered)
            performance_monitor.record_file_operation("discover", Path("source"), total_size)

        print("\nINITIALIZING DOCUMENT PROCESSOR...")
        from app.core.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor(config_obj)
        doc_processor.initialize()


        doc_processor.performance_monitor = performance_monitor

        print("\nPROCESSING DOCUMENTS...")
        successful_processing = 0
        failed_processing = 0
        total_matches = 0
        results = []


        for i, doc_path in enumerate(discovered):
            try:

                if i % 10 == 0:
                    log_progress(i, len(discovered), start_time, successful_processing,
                               failed_processing, performance_monitor)


                process_result = process_with_retry(doc_path, doc_processor)

                if process_result:

                    if isinstance(process_result, tuple):
                        result, modified_document = process_result
                    else:
                        result = process_result
                        modified_document = None

                    successful_processing += 1
                    total_matches += result.matches_found
                    results.append(result)


                    if result.processed_file_name:

                        processed_path = Path(config_obj.output_dir) / result.processed_file_name
                        processed_path.parent.mkdir(parents=True, exist_ok=True)


                        if modified_document:
                            modified_document.save(str(processed_path))
                            logger.info(f"Saved processed file with reconstruction: {result.processed_file_name}")
                        else:

                            from docx import Document
                            document = Document(doc_path)
                            document.save(str(processed_path))
                            logger.warning(f"Saved original document (no modified document found): {result.processed_file_name}")


                    move_source_file_to_complete(doc_path, config_obj)


                    report_generator = ReportGenerator(output_dir=config_obj.reports_dir, config=config_obj)
                    report_generator.generate_document_reports(result)

                else:
                    failed_processing += 1

                    failed_result = create_pending_result(doc_path, config_obj, doc_processor)
                    failed_result.status = ProcessingStatus.ERROR
                    failed_result.success = False
                    failed_result.error_message = "Processing failed after all retries"
                    results.append(failed_result)


                # Memory cleanup after each document to prevent accumulation
                if hasattr(doc_processor, 'clear_memory'):
                    # More aggressive cleanup every 10 documents
                    if (i + 1) % 10 == 0:
                        doc_processor.clear_memory(force_gc=True)  # Force GC every 10 documents
                        logger.info(f"Memory cleanup with garbage collection performed after processing document {i+1}/{len(discovered)}")
                    else:
                        doc_processor.clear_memory(force_gc=False)  # Light cleanup between documents
                        logger.debug(f"Memory cleanup performed after processing document {i+1}/{len(discovered)}")

                # Circuit breaker: Check system resources after each document
                if performance_monitor:
                    resource_status = performance_monitor.check_system_resources()
                    if resource_status['critical']:
                        logger.error("System resources critical, stopping processing")
                        break
                elif hasattr(doc_processor, '_check_system_resources'):
                    # Use built-in circuit breaker if performance monitor not available
                    if not doc_processor._check_system_resources():
                        logger.error("System resources critical (circuit breaker), stopping processing")
                        break

            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                failed_processing += 1
                logger.error(f"Unexpected error processing {doc_path.name}: {e}")

                failed_result = create_pending_result(doc_path, config_obj, doc_processor)
                failed_result.status = ProcessingStatus.ERROR
                failed_result.success = False
                failed_result.error_message = str(e)
                results.append(failed_result)


        log_progress(len(discovered), len(discovered), start_time, successful_processing,
                   failed_processing, performance_monitor)


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

        if 'doc_processor' in locals():
            doc_processor.cleanup()


        if performance_monitor:
            print("\nSTOPPING PERFORMANCE MONITOR...")
            performance_monitor.stop_monitoring()


# Cache for file discovery to avoid repeated expensive recursive scans
_FILE_DISCOVERY_CACHE = {}
_CACHE_REFRESH_INTERVAL = 60  # seconds

def discover_input_files(config_obj, force_refresh: bool = False):
    """Optimized file discovery with caching to avoid repeated recursive scans."""
    import time
    
    source_dir = Path(config_obj.source_dir)
    if not source_dir.exists():
        return []
        
    cache_key = str(source_dir)
    current_time = time.time()
    
    # Check if we have valid cached results
    if (not force_refresh and 
        cache_key in _FILE_DISCOVERY_CACHE and 
        current_time - _FILE_DISCOVERY_CACHE[cache_key]['timestamp'] < _CACHE_REFRESH_INTERVAL):
        logger.debug(f"Using cached file discovery for {source_dir}")
        return _FILE_DISCOVERY_CACHE[cache_key]['files']
    
    # Perform fresh discovery
    logger.debug(f"Performing fresh file discovery for {source_dir}")
    supported = {".docx", ".doc"}
    candidates = []
    
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue

        # Skip excluded directories
        parent_names = {p.name for p in path.parents}
        if "orig_doc_files" in parent_names or "processed" in parent_names:
            continue

        # Skip already processed files
        if path.suffix.lower() == ".docx" and path.stem.endswith(config_obj.suffix):
            continue
        
        # Check file size constraints
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= config_obj.min_file_size and size_mb <= config_obj.max_file_size:
            candidates.append(path)

    # Deduplicate by stem, preferring .doc over .docx for conversion
    by_stem = {}
    for p in candidates:
        stem = p.stem
        if p.suffix.lower() == ".docx" and stem.endswith(config_obj.suffix):
            continue
        if stem not in by_stem:
            by_stem[stem] = p
        else:
            existing = by_stem[stem]
            if existing.suffix.lower() == ".docx" and p.suffix.lower() == ".doc":
                by_stem[stem] = p
    
    final_files = list(by_stem.values())
    
    # Cache the results
    _FILE_DISCOVERY_CACHE[cache_key] = {
        'files': final_files,
        'timestamp': current_time
    }
    
    logger.info(f"Discovered {len(final_files)} files in {source_dir}")
    return final_files


def create_pending_result(file_path: Path, config_obj, doc_processor=None):
    from core.models import ProcessingResult, ProcessingStatus

    intended_processed_name = f"{file_path.stem}{config_obj.suffix}.docx"

    src_name = file_path.name
    src_size = file_path.stat().st_size
    is_doc = file_path.suffix.lower() == ".doc"
    is_docx = file_path.suffix.lower() == ".docx"


    parser_type = "pending"
    if doc_processor:
        try:

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
        parser=parser_type,
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
    report_generator = ReportGenerator(output_dir=config_obj.reports_dir, config=config_obj)


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


    cli_parameters.user_provided = set()


    patterns = {}
    total_patterns = 0
            # Pattern loading moved to cached utilities - no longer needed here

    pattern_info = PatternInfo(
        patterns_file=str(config_obj.patterns_file),
        total_patterns=total_patterns,
        patterns=patterns
    )


    batch_report = BatchReport(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        cli_parameters=cli_parameters,
        patterns=pattern_info,
        file_reports=results
    )


    batch_report.aggregate()

    print("Generating batch reports...")
    report_generator.generate_batch_reports(batch_report)


    pending_count = sum(1 for r in results if r.status == ProcessingStatus.PENDING)
    if pending_count:
        print(f"{pending_count} file(s) pending - skipping file-level reports.")
    print("All reports generated successfully!")


if __name__ == "__main__":
    main()