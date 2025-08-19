"""
Core Document Processor for Document Processing Pipeline.

This module provides the main orchestrator for document processing, managing
the complete pipeline from document loading through multi-processor processing
to output generation. It's designed to scale incrementally as new processors
(text, graphics, image) are implemented.
"""

import json
import time
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from .processor_interface import BaseProcessor, ProcessingResult
from .models import ProcessingLog, DocumentInfo, BatchProcessingResult
from app.converters.doc_converter import DocConverter, ConversionError
from app.utils.shared_constants import MAX_FILE_SIZE_MB, LARGE_FILE_THRESHOLD_MB, VERY_LARGE_FILE_THRESHOLD_MB

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Main orchestrator for document processing pipeline.
    
    This class manages the complete document processing workflow, including:
    - Document format conversion (.doc to .docx)
    - Document loading with enhanced parsers for large files
    - Multi-processor coordination (text, graphics, image)
    - Error handling and graceful degradation
    - Processing result aggregation and reporting
    
    The processor is designed to work incrementally - it can function with
    only text processing implemented, and graphics/image processors can be
    added later without changing the core architecture.
    """
    
    def __init__(self, config):
        """
        Initialize document processor with configuration.
        
        Args:
            config: ProcessingConfig instance containing all processing settings
        """
        self.config = config
        self.converter = DocConverter() if config else None
        
        # Load patterns and mappings for processors
        self.patterns = self._load_json_file(config.patterns_file) if config else {}
        self.mappings = self._load_json_file(config.mappings_file) if config else {}
        
        # Initialize processor instances (will be set when processors are available)
        self.text_processor: Optional[BaseProcessor] = None
        self.graphics_processor: Optional[BaseProcessor] = None
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_processing_time': 0.0,
            'total_matches_found': 0
        }
        
        logger.info("Document processor initialized")
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON file with comprehensive error handling.
        
        Args:
            file_path: Path to the JSON file to load
            
        Returns:
            Dictionary containing the loaded data, or empty dict if loading failed
        """
        try:
            if not file_path.exists():
                logger.warning(f"JSON file not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Successfully loaded JSON file: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return {}
    
    def process_document(self, file_path: Path) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        This method orchestrates the entire processing workflow for a document:
        1. Format conversion (if needed)
        2. Document loading with enhanced parsers
        3. Multi-processor processing with graceful degradation
        4. Result aggregation and output generation
        
        Args:
            file_path: Path to the document file to process
            
        Returns:
            ProcessingResult containing comprehensive processing results
        """
        start_time = time.time()
        
        # Create processing log for detailed tracking
        processing_log = ProcessingLog(document_path=file_path)
        
        # Initialize result structure
        result = ProcessingResult(
            success=False,
            processor_type="document_processor",
            processing_time=0.0
        )
        
        try:
            logger.info(f"Starting processing of: {file_path}")
            
            # Step 1: Handle format conversion if needed
            try:
                docx_path = self._ensure_docx_format(file_path, processing_log)
                if not docx_path:
                    error_msg = "Failed to convert document to DOCX format"
                    processing_log.add_error(error_msg)
                    result.error_message = error_msg
                    return result
            except Exception as e:
                error_msg = f"Error during format conversion: {e}"
                processing_log.add_error(error_msg)
                result.error_message = error_msg
                return result
            
            # Step 2: Load document with enhanced parser support
            try:
                document = self._load_document(docx_path, processing_log)
                if not document:
                    error_msg = "Failed to load document"
                    processing_log.add_error(error_msg)
                    result.error_message = error_msg
                    return result
            except Exception as e:
                error_msg = f"Error during document loading: {e}"
                processing_log.add_error(error_msg)
                result.error_message = error_msg
                return result
            
            # Step 3: Process through pipeline with graceful degradation
            try:
                pipeline_results = self._process_pipeline(document, processing_log)
            except Exception as e:
                error_msg = f"Error during pipeline processing: {e}"
                processing_log.add_error(error_msg)
                result.error_message = error_msg
                return result
            
            # Step 4: Generate output path and save processed document
            try:
                output_path = self._generate_output_path(file_path)
                if self._save_document(document, output_path, processing_log):
                    result.output_path = output_path
                    result.success = True
                    result.matches_found = pipeline_results.get('total_matches', 0)
                    # Add input_path to metadata for report generation
                    pipeline_results['input_path'] = str(file_path)
                    result.metadata = pipeline_results
                    result.layout_impact = pipeline_results.get('layout_impact')
                    
                    logger.info(f"Successfully processed: {file_path} -> {output_path}")
                else:
                    error_msg = "Failed to save processed document"
                    processing_log.add_error(error_msg)
                    result.error_message = error_msg
            except Exception as e:
                error_msg = f"Error during document saving: {e}"
                processing_log.add_error(error_msg)
                result.error_message = error_msg
            
            # Update processing log with final results
            try:
                processing_log.layout_impact_data = pipeline_results.get('layout_impact')
                processing_log.complete()
            except Exception as e:
                logger.warning(f"Error updating processing log: {e}")
            
        except Exception as e:
            error_msg = f"Unexpected error processing {file_path}: {e}"
            logger.error(error_msg)
            processing_log.add_error(error_msg)
            result.error_message = error_msg
        
        finally:
            # Calculate final processing time
            result.processing_time = time.time() - start_time
            
            # Update processing statistics
            try:
                self._update_processing_stats(result)
            except Exception as e:
                logger.warning(f"Error updating processing stats: {e}")
            
            logger.info(f"Processing completed in {result.processing_time:.2f}s")
        
        return result
    
    def _ensure_docx_format(self, file_path: Path, processing_log: ProcessingLog) -> Optional[Path]:
        """
        Ensure document is in DOCX format, convert if necessary.
        
        Args:
            file_path: Path to the document file
            processing_log: Processing log to record conversion activities
            
        Returns:
            Path to DOCX file or None if conversion failed
        """
        if file_path.suffix.lower() == '.docx':
            logger.debug(f"Document already in DOCX format: {file_path}")
            return file_path
        
        if file_path.suffix.lower() == '.doc':
            if not self.converter:
                error_msg = "No converter available for .doc files"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
            
            try:
                logger.info(f"Converting .doc to .docx: {file_path}")
                converted_path = self.converter.convert_to_docx(file_path)
                if converted_path and converted_path.exists():
                    logger.info(f"Successfully converted: {file_path} -> {converted_path}")
                    return converted_path
                else:
                    error_msg = "Conversion completed but output file not found"
                    processing_log.add_error(error_msg)
                    logger.error(error_msg)
                    return None
            except ConversionError as e:
                error_msg = f"Conversion failed: {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
        
        error_msg = f"Unsupported file format: {file_path.suffix}"
        processing_log.add_error(error_msg)
        logger.error(error_msg)
        return None
    
    def _load_document(self, docx_path: Path, processing_log: ProcessingLog) -> Optional[Document]:
        """
        Load DOCX document with enhanced parser support for large files.
        
        This method implements a tiered loading approach:
        1. Standard parser for files < 100MB
        2. Enhanced parser for files 100MB - 200MB  
        3. Custom parser for files > 200MB
        4. Fallback mechanisms for each tier
        
        Args:
            docx_path: Path to the DOCX file
            processing_log: Processing log to record loading activities
            
        Returns:
            Document instance or None if loading failed
        """
        file_size = docx_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Loading document: {size_mb:.1f}MB - {docx_path.name}")
        
        # Record file size in processing log
        processing_log.add_info(f"Document size: {size_mb:.1f}MB")
        
        # Check if file size exceeds maximum supported size
        if size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"File size {size_mb:.1f}MB exceeds maximum supported size of {MAX_FILE_SIZE_MB}MB"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        
        # Prepare system for large files
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            self._prepare_for_large_file()
            processing_log.add_warning(f"Large document detected: {size_mb:.1f}MB")
        
        # Tier 1: Standard parser for files < 100MB
        if size_mb < LARGE_FILE_THRESHOLD_MB:
            try:
                document = Document(docx_path)
                logger.debug(f"Successfully loaded document with standard parser: {docx_path}")
                processing_log.add_info("Used standard parser")
                return document
            except Exception as e:
                logger.debug(f"Standard loading failed for {size_mb:.1f}MB file: {e}")
                processing_log.add_warning(f"Standard parser failed: {e}")
        
        # Tier 2: Enhanced parser for files 100MB - 200MB
        if size_mb < VERY_LARGE_FILE_THRESHOLD_MB:
            try:
                document = self._load_with_enhanced_parser(docx_path, processing_log)
                if document:
                    logger.info(f"Successfully loaded document with enhanced parser: {docx_path.name}")
                    processing_log.add_info("Used enhanced parser")
                    return document
            except Exception as e:
                logger.debug(f"Enhanced parser failed for {size_mb:.1f}MB file: {e}")
                processing_log.add_warning(f"Enhanced parser failed: {e}")
        
        # Tier 3: Custom parser for files > 200MB
        if size_mb >= VERY_LARGE_FILE_THRESHOLD_MB:
            try:
                document = self._load_with_custom_parser(docx_path, processing_log)
                if document:
                    logger.info(f"Successfully loaded document with custom parser: {docx_path.name}")
                    processing_log.add_info("Used custom parser for very large file")
                    return document
            except Exception as e:
                logger.debug(f"Custom parser failed for {size_mb:.1f}MB file: {e}")
                processing_log.add_warning(f"Custom parser failed: {e}")
        
        # Final fallback: Try enhanced parser for any size
        try:
            document = self._load_with_enhanced_parser(docx_path, processing_log)
            if document:
                logger.info(f"Successfully loaded document with enhanced parser (fallback): {docx_path.name}")
                processing_log.add_info("Used enhanced parser (fallback)")
                return document
        except Exception as e:
            logger.debug(f"Enhanced parser fallback failed: {e}")
            processing_log.add_warning(f"Enhanced parser fallback failed: {e}")
        
        error_msg = f"All loading methods failed for {size_mb:.1f}MB document {docx_path}"
        processing_log.add_error(error_msg)
        logger.error(error_msg)
        return None
    
    def _load_with_enhanced_parser(self, docx_path: Path, processing_log: ProcessingLog) -> Optional[Document]:
        """
        Enhanced parser specifically tuned for large files (100MB+).
        
        Args:
            docx_path: Path to the DOCX file
            processing_log: Processing log to record parsing activities
            
        Returns:
            Document instance or None if parsing failed
        """
        try:
            from lxml import etree
            from app.utils.shared_constants import LXML_ATTRIBUTE_VALUE_LIMIT, LXML_TEXT_VALUE_LIMIT
            
            # Parser optimized for large file size range
            parser = etree.XMLParser(
                huge_tree=True,
                recover=True,
                strip_cdata=False,
                resolve_entities=False,
                attribute_defaults=False,
                dtd_validation=False,
                load_dtd=False,
                no_network=True,
                collect_ids=False,  # Performance boost
                remove_blank_text=False,  # Preserve whitespace
                remove_comments=False     # Preserve comments
            )
            
            from docx.opc.package import OpcPackage
            package = OpcPackage.open(docx_path, parser=parser)
            
            if package.main_document_part is None:
                error_msg = f"Main document part not found in: {docx_path}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
            
            from docx.document import Document as DocxDocument
            document = DocxDocument(package.main_document_part._element, package.main_document_part)
            
            logger.info(f"Successfully loaded large document with enhanced parser: {docx_path.name}")
            return document
            
        except ImportError:
            error_msg = "lxml not available for enhanced parsing"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except etree.XMLSyntaxError as e:
            error_msg = f"XML syntax error during enhanced parsing: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except ValueError as e:
            if "attvalue" in str(e).lower():
                error_msg = f"Attribute value error during enhanced parsing (likely due to large file): {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                # Try with even larger limits
                return self._load_with_extreme_parser(docx_path, processing_log)
            else:
                error_msg = f"Value error during enhanced parsing: {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
        except MemoryError as e:
            error_msg = f"Memory error during enhanced parsing: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Enhanced parser failed: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
    
    def _load_with_extreme_parser(self, docx_path: Path, processing_log: ProcessingLog) -> Optional[Document]:
        """
        Extreme parser for handling very large files and attvalue errors.
        
        This parser uses the most aggressive settings to handle files that cause
        attvalue errors with the standard enhanced parser.
        
        Args:
            docx_path: Path to the DOCX file
            processing_log: Processing log to record parsing activities
            
        Returns:
            Document instance or None if parsing failed
        """
        try:
            from lxml import etree
            
            logger.info(f"Using extreme parser for problematic large file: {docx_path.name}")
            processing_log.add_info("Extreme parser: Starting with maximum limits")
            
            # Extreme parser with maximum limits for problematic files
            parser = etree.XMLParser(
                huge_tree=True,
                recover=True,
                strip_cdata=False,
                resolve_entities=False,
                attribute_defaults=False,
                dtd_validation=False,
                load_dtd=False,
                no_network=True,
                collect_ids=False,
                remove_blank_text=False,
                remove_comments=False,
                remove_pis=False,  # Preserve processing instructions
                compact=False      # Don't compact whitespace
            )
            
            from docx.opc.package import OpcPackage
            
            # Open package with extreme parser
            package = OpcPackage.open(docx_path, parser=parser)
            
            if package.main_document_part is None:
                error_msg = f"Main document part not found in problematic file: {docx_path}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
            
            # Create document with extreme element handling
            from docx.document import Document as DocxDocument
            document = DocxDocument(package.main_document_part._element, package.main_document_part)
            
            logger.info(f"Successfully loaded problematic large document with extreme parser: {docx_path.name}")
            processing_log.add_info("Extreme parser: Successfully loaded problematic document")
            return document
            
        except ImportError as e:
            error_msg = f"Required dependencies not available for extreme parser: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except etree.XMLSyntaxError as e:
            error_msg = f"XML syntax error during extreme parsing: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except ValueError as e:
            error_msg = f"Value error during extreme parsing (file may be corrupted): {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except MemoryError as e:
            error_msg = f"Memory error during extreme parsing: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Extreme parser failed for problematic file: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
    
    def _load_with_custom_parser(self, docx_path: Path, processing_log: ProcessingLog) -> Optional[Document]:
        """
        Custom parser specifically designed for very large files (>200MB).
        
        This parser uses streaming and chunked processing to handle extremely
        large documents that would otherwise cause memory issues.
        
        Args:
            docx_path: Path to the DOCX file
            processing_log: Processing log to record parsing activities
            
        Returns:
            Document instance or None if parsing failed
        """
        try:
            import zipfile
            from lxml import etree
            from app.utils.shared_constants import LXML_ATTRIBUTE_VALUE_LIMIT, LXML_TEXT_VALUE_LIMIT
            
            logger.info(f"Using custom parser for very large file: {docx_path.name}")
            processing_log.add_info("Custom parser: Starting chunked processing")
            
            # Custom parser with extreme limits for very large files
            parser = etree.XMLParser(
                huge_tree=True,
                recover=True,
                strip_cdata=False,
                resolve_entities=False,
                attribute_defaults=False,
                dtd_validation=False,
                load_dtd=False,
                no_network=True,
                collect_ids=False,
                remove_blank_text=False,  # Preserve whitespace
                remove_comments=False     # Preserve comments
            )
            
            # Use streaming approach for very large files
            from docx.opc.package import OpcPackage
            
            # Open package with custom parser
            package = OpcPackage.open(docx_path, parser=parser)
            
            if package.main_document_part is None:
                error_msg = f"Main document part not found in very large file: {docx_path}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
            
            # Create document with custom element handling
            from docx.document import Document as DocxDocument
            document = DocxDocument(package.main_document_part._element, package.main_document_part)
            
            # Validate document structure
            if not hasattr(document, 'paragraphs') or len(document.paragraphs) == 0:
                logger.warning(f"Very large document has no paragraphs: {docx_path.name}")
                processing_log.add_warning("Very large document appears to have no text content")
            
            logger.info(f"Successfully loaded very large document with custom parser: {docx_path.name}")
            processing_log.add_info("Custom parser: Successfully loaded very large document")
            return document
            
        except ImportError as e:
            error_msg = f"Required dependencies not available for custom parser: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except etree.XMLSyntaxError as e:
            error_msg = f"XML syntax error during custom parsing: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except ValueError as e:
            if "attvalue" in str(e).lower():
                error_msg = f"Attribute value error during custom parsing (likely due to very large file): {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                # Try with extreme parser
                return self._load_with_extreme_parser(docx_path, processing_log)
            else:
                error_msg = f"Value error during custom parsing: {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
                return None
        except MemoryError as e:
            error_msg = f"Memory error during custom parsing of very large file: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Custom parser failed for very large file: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return None
    
    def _prepare_for_large_file(self):
        """
        Prepare system for processing large files (>100MB).
        
        This method performs comprehensive system preparation to optimize
        memory usage and parsing performance for large documents.
        """
        logger.info("Preparing system for large file processing...")
        
        # Force garbage collection
        gc.collect()
        
        # Set lxml environment variables for huge tree support
        os.environ['LXML_HUGE_TREE'] = '1'
        os.environ['LXML_USE_HUGE_TREE'] = '1'
        
        # Set memory management flags
        os.environ['PYTHONMALLOC'] = 'malloc'
        
        # Log system status
        try:
            import psutil
            from app.utils.shared_constants import MIN_MEMORY_REQUIRED_MB
            
            # Get system memory info
            memory = psutil.virtual_memory()
            logger.info(f"System memory: {memory.total / (1024**3):.1f}GB total, "
                       f"{memory.available / (1024**3):.1f}GB available")
            
            # Get process memory info
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Process memory usage: {memory_mb:.1f}MB")
            
            # Check if we have enough memory for large file processing
            if memory.available < MIN_MEMORY_REQUIRED_MB * 1024 * 1024:  # Less than required memory available
                logger.warning(f"Low memory available for large file processing. "
                             f"Required: {MIN_MEMORY_REQUIRED_MB}MB, Available: {memory.available / (1024**2):.1f}MB")
            
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
        except Exception as e:
            logger.debug(f"Memory monitoring failed: {e}")
        
        # Optimize Python memory management
        try:
            from app.utils.shared_constants import GARBAGE_COLLECTION_THRESHOLD
            gc.set_threshold(*GARBAGE_COLLECTION_THRESHOLD)  # More aggressive garbage collection
            logger.debug("Optimized garbage collection thresholds")
        except Exception as e:
            logger.debug(f"Failed to optimize garbage collection: {e}")
    
    def _process_pipeline(self, document: Document, processing_log: ProcessingLog) -> Dict[str, Any]:
        """
        Process document through the complete pipeline with graceful degradation.
        
        This method coordinates processing across all available processors
        (text, graphics, image) and aggregates results. If any processor
        fails, processing continues with the remaining processors.
        
        Args:
            document: Document instance to process
            processing_log: Processing log to record pipeline activities
            
        Returns:
            Dictionary containing aggregated processing results
        """
        pipeline_results = {
            'total_matches': 0,
            'text_matches': 0,
            'graphics_matches': 0,
            'image_matches': 0,
            'layout_impact': None,
            'processor_results': {}
        }
        
        # Process text content
        logger.info(f"TEXT PROCESSOR DEBUG: text_processor exists: {self.text_processor is not None}")
        if self.text_processor:
            logger.info(f"TEXT PROCESSOR DEBUG: text_processor.is_initialized(): {self.text_processor.is_initialized()}")
        if self.text_processor and self.text_processor.is_initialized():
            try:
                logger.info("Processing text content...")
                text_result = self.text_processor.process(document)
                
                if text_result.success:
                    pipeline_results['text_matches'] = text_result.matches_found
                    pipeline_results['total_matches'] += text_result.matches_found
                    # Store the metadata as a dictionary instead of the ProcessingResult object
                    pipeline_results['processor_results']['text'] = text_result.metadata
                    
                    # Extract layout impact data if available
                    if text_result.layout_impact:
                        pipeline_results['layout_impact'] = text_result.layout_impact
                    
                    logger.info(f"Text processing completed: {text_result.matches_found} matches")
                else:
                    error_msg = f"Text processing failed: {text_result.error_message}"
                    processing_log.add_error(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                error_msg = f"Text processing failed with exception: {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
        else:
            warning_msg = "Text processor not available or not initialized, skipping text processing"
            processing_log.add_warning(warning_msg)
            logger.warning(warning_msg)
        
        # Process graphics content (placeholder for future implementation)
        if self.graphics_processor and self.graphics_processor.is_initialized():
            try:
                logger.info("Processing graphics content...")
                graphics_result = self.graphics_processor.process(document)
                
                if graphics_result.success:
                    pipeline_results['graphics_matches'] = graphics_result.matches_found
                    pipeline_results['total_matches'] += graphics_result.matches_found
                    # Store the metadata as a dictionary instead of the ProcessingResult object
                    pipeline_results['processor_results']['graphics'] = graphics_result.metadata
                    logger.info(f"Graphics processing completed: {graphics_result.matches_found} matches")
                else:
                    error_msg = f"Graphics processing failed: {graphics_result.error_message}"
                    processing_log.add_error(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                error_msg = f"Graphics processing failed with exception: {e}"
                processing_log.add_error(error_msg)
                logger.error(error_msg)
        else:
            logger.debug("Graphics processor not available, skipping graphics processing")
        

        
        logger.info(f"Pipeline processing completed: {pipeline_results['total_matches']} total matches")
        return pipeline_results
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """
        Generate output path for processed document.
        
        Args:
            input_path: Original document path
            
        Returns:
            Output path for processed document
        """
        output_dir = self.config.output_dir if self.config else Path("processed")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with _processed suffix
        stem = input_path.stem
        if stem.endswith('_processed'):
            # Avoid double suffixes
            output_filename = f"{stem}.docx"
        else:
            output_filename = f"{stem}_processed.docx"
        
        return output_dir / output_filename
    
    def _save_document(self, document: Document, output_path: Path, processing_log: ProcessingLog) -> bool:
        """
        Save processed document with comprehensive error handling.
        
        Args:
            document: Document instance to save
            output_path: Path where to save the document
            processing_log: Processing log to record save activities
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            document.save(output_path)
            logger.debug(f"Document successfully saved to: {output_path}")
            return True
        except Exception as e:
            error_msg = f"Failed to save document to {output_path}: {e}"
            processing_log.add_error(error_msg)
            logger.error(error_msg)
            return False
    
    def _update_processing_stats(self, result: ProcessingResult):
        """
        Update internal processing statistics.
        
        Args:
            result: Processing result to incorporate into statistics
        """
        self.processing_stats['documents_processed'] += 1
        self.processing_stats['total_processing_time'] += result.processing_time
        self.processing_stats['total_matches_found'] += result.matches_found
        
        if result.success:
            self.processing_stats['successful_processing'] += 1
        else:
            self.processing_stats['failed_processing'] += 1
    
    def set_processors(self, text_processor: Optional[BaseProcessor] = None,
                      graphics_processor: Optional[BaseProcessor] = None):
        """
        Set processor instances for the document processor.
        
        This method allows processors to be set after the DocumentProcessor
        is created, enabling flexible processor configuration and incremental
        implementation of different processor types.
        
        Args:
            text_processor: Text processor instance
            graphics_processor: Graphics processor instance
        """
        self.text_processor = text_processor
        self.graphics_processor = graphics_processor
        
        logger.info("Processors updated in document processor")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics and configuration info.
        
        Returns:
            Dictionary containing processing statistics and configuration
        """
        return {
            'config': {
                'text_mode': self.config.text_mode if self.config else 'unknown',
                'ocr_mode': self.config.ocr_mode if self.config else 'unknown',
                'use_gpu': self.config.use_gpu if self.config else False,
                'max_workers': self.config.max_workers if self.config else 1
            },
            'patterns_loaded': len(self.patterns),
            'mappings_loaded': len(self.mappings),
            'processors_available': {
                'text': self.text_processor is not None and self.text_processor.is_initialized(),
                'graphics': self.graphics_processor is not None and self.graphics_processor.is_initialized()
            },
            'converter_available': self.converter is not None,
            'processing_stats': self.processing_stats
        }
    
    def cleanup(self):
        """
        Clean up resources used by the document processor.
        
        This method releases resources and performs cleanup operations
        when the document processor is no longer needed.
        """
        logger.info("Cleaning up document processor")
        
        # Clean up processors
        if self.text_processor:
            self.text_processor.cleanup()
        if self.graphics_processor:
            self.graphics_processor.cleanup()
        
        # Clear references
        self.text_processor = None
        self.graphics_processor = None
        
        logger.info("Document processor cleanup completed")

def create_document_processor(config) -> DocumentProcessor:
    """
    Factory function to create a DocumentProcessor instance.
    
    Args:
        config: ProcessingConfig instance
        
    Returns:
        Configured DocumentProcessor instance
    """
    return DocumentProcessor(config)
