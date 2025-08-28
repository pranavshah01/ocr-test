"""
Text Processor for DOCX document processing.

This module provides text processing capabilities for DOCX documents including:
- Pattern matching using regex patterns
- Text replacement with formatting preservation
- Comprehensive detection reporting
- Integration with ProcessingResult for unified reporting
"""

import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from ..core.models import ProcessingResult, MatchDetail, ProcessorType, MatchFlag, FallbackFlag
from ..utils.text_utils.text_docx_utils import TextReconstructor, FontManager, PatternMatcher, create_pattern_matcher, load_patterns_and_mappings
from config import DEFAULT_MAPPING

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processor for DOCX document processing with comprehensive detection."""
    
    def __init__(self, config):
        """
        Initialize text processor.
        
        Args:
            config: Configuration object containing patterns and mappings
        """
        self.config = config
        self.initialized = False
        self.patterns = {}
        self.mappings = {}
        self.pattern_matcher = None
        
        # Load patterns and mappings
        self._load_patterns_and_mappings()
    
    def _load_patterns_and_mappings(self):
        """Load patterns and mappings from configuration files."""
        self.patterns, self.mappings = load_patterns_and_mappings(self.config)
    
    def initialize(self) -> bool:
        """Initialize the text processor."""
        try:
            # Create pattern matcher once during initialization
            self.pattern_matcher = create_pattern_matcher(self.patterns, self.mappings)
            logger.info(f"Pattern matcher created with {len(self.patterns)} patterns, {len(self.mappings)} mappings")
            
            self.initialized = True
            logger.info(f"Text processor initialized with {len(self.patterns)} patterns, {len(self.mappings)} mappings")
            return True
        except Exception as e:
            logger.error(f"Error initializing Text Processor: {e}")
            self.initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if the processor is initialized."""
        return self.initialized
    
    def process_document(self, document: Document, processing_result: ProcessingResult) -> ProcessingResult:
        """
        Process document text and update ProcessingResult with detection details.
        
        Args:
            document: The DOCX document to process
            processing_result: ProcessingResult object to update with detection details
            
        Returns:
            Updated ProcessingResult object
        """
        if not self.initialized:
            logger.error("Text processor not initialized")
            processing_result.error_message = "Text processor not initialized"
            return processing_result
        
        start_time = time.time()
        logger.info("Starting text processing...")
        
        try:
            # Process all text content
            all_detections = []
            
            # Process main document body paragraphs
            body_detections = self._process_paragraphs(document.paragraphs, "body")
            all_detections.extend(body_detections)
            
            # Process tables
            table_detections = self._process_tables(document.tables)
            all_detections.extend(table_detections)
            
            # Process headers and footers
            header_footer_detections = self._process_headers_footers(document)
            all_detections.extend(header_footer_detections)
            
            # Log detailed detection information
            logger.info(f"Found {len(all_detections)} pattern matches:")
            for i, detection in enumerate(all_detections, 1):
                pattern_name = detection.get('pattern_name', 'Unknown')
                matched_text = detection.get('matched_text', 'Unknown')
                replacement_text = detection.get('replacement_text', 'Unknown')
                location = detection.get('location', 'Unknown')
                logger.info(f"  {i}. Pattern '{pattern_name}' matched '{matched_text}' -> '{replacement_text}' at {location}")
            
            # Update ProcessingResult with detection details
            self._update_processing_result(processing_result, all_detections)
            
            processing_time = time.time() - start_time
            logger.info(f"Text processing completed in {processing_time:.2f}s: {len(all_detections)} detections")
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Error during text processing: {e}"
            logger.error(error_msg)
            processing_result.error_message = error_msg
            return processing_result
    
    def _process_paragraphs(self, paragraphs: List[Paragraph], location: str) -> List[Dict[str, Any]]:
        """
        Process paragraphs for pattern detection.
        
        Args:
            paragraphs: List of paragraphs to process
            location: Location description
            
        Returns:
            List of detection dictionaries
        """
        all_detections = []
        
        for i, paragraph in enumerate(paragraphs):
            try:
                paragraph_detections = self._process_paragraph(
                    paragraph, f"{location}_paragraph_{i}"
                )
                all_detections.extend(paragraph_detections)
            except Exception as e:
                logger.error(f"Error processing paragraph {i} in {location}: {e}")
        
        return all_detections
    
    def _process_paragraph(self, paragraph: Paragraph, location: str) -> List[Dict[str, Any]]:
        """
        Process individual paragraph for pattern detection (same logic as old AWI).
        
        Args:
            paragraph: Paragraph to process
            location: Location description
            
        Returns:
            List of detection dictionaries
        """
        if not paragraph.runs:
            return []
        
        # Reconstruct full text from runs (same as old AWI)
        full_text, runs = TextReconstructor.reconstruct_paragraph_text(paragraph)
        
        if not full_text.strip():
            return []
        
        # Find ALL pattern matches (including those without mappings for comprehensive reporting)
        all_pattern_matches = self.pattern_matcher.find_all_pattern_matches(full_text)
        
        detection_results = []
        
        for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:
            # Get replacement text
            replacement_text = self.pattern_matcher.get_replacement(matched_text)
            
            # If no replacement found, use default mapping for append mode
            if not replacement_text:
                replacement_text = DEFAULT_MAPPING
            
            # Determine if this pattern was successfully matched
            is_matched = replacement_text is not None
            
            # Get the actual pattern from patterns
            actual_pattern = self.patterns.get(pattern_name, pattern_name)
            
            # Get font information for this detection using consolidated method
            font_info = FontManager.extract_font_info_for_detection(
                runs, matched_text, start_pos, getattr(self, 'document', None)
            )
            
            # Determine content type and dimensions (same as old AWI)
            content_type = "Paragraph"
            dimension = ""
            
            if "table" in location.lower():
                content_type = "Table"
                # For tables, we'll need to get cell dimensions
                dimension = "Cell size"  # Placeholder - would need table cell info
            elif "header" in location.lower():
                content_type = "Header"
            elif "footer" in location.lower():
                content_type = "Footer"
            
            # Create detection result in ProcessingResult format (same as old AWI)
            detection_result = {
                'pattern_name': pattern_name,
                'actual_pattern': actual_pattern,
                'matched_text': matched_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'replacement_text': replacement_text,
                'location': location,
                'content_type': content_type,
                'dimension': dimension,
                'processor': 'Text',
                'font_info': font_info,
                'is_matched': is_matched,
                'confidence': 1.0
            }
            
            detection_results.append(detection_result)
        
        return detection_results
    
    def _process_tables(self, tables: List[Table], location_prefix: str = "") -> List[Dict[str, Any]]:
        """
        Process tables for pattern detection, handling vMerge cells properly.
        
        Args:
            tables: List of tables to process
            location_prefix: Prefix for location description
            
        Returns:
            List of detection dictionaries
        """
        all_detections = []
        
        # Import the table utilities
        from ..utils.table_utils import get_table_cells_to_process
        
        for table_idx, table in enumerate(tables):
            try:
                # Get cells that should be processed (skipping vMerge continue cells)
                cells_to_process = get_table_cells_to_process(table)
                
                for row_idx, cell_idx, cell in cells_to_process:
                    if location_prefix:
                        location = f"{location_prefix}_table_{table_idx}_row_{row_idx}_cell_{cell_idx}"
                    else:
                        location = f"table_{table_idx}_row_{row_idx}_cell_{cell_idx}"
                    
                    cell_detections = self._process_paragraphs(cell.paragraphs, location)
                    all_detections.extend(cell_detections)
                    
                    logger.debug(f"Processed cell at {location} with {len(cell_detections)} detections")
                    
            except Exception as e:
                logger.error(f"Error processing table {table_idx}: {e}")
        
        return all_detections
    
    def _process_headers_footers(self, document: Document) -> List[Dict[str, Any]]:
        """
        Process headers and footers for pattern detection (including tables).
        
        Args:
            document: Document to process
            
        Returns:
            List of detection dictionaries
        """
        all_detections = []
        
        try:
            # Process headers and footers (including tables)
            for section_idx, section in enumerate(document.sections):
                # Process header paragraphs and tables
                if section.header:
                    header_detections = self._process_paragraphs(section.header.paragraphs, f"header_section_{section_idx}")
                    all_detections.extend(header_detections)
                    
                    # Process tables in header
                    header_table_detections = self._process_tables(section.header.tables, f"header_section_{section_idx}")
                    all_detections.extend(header_table_detections)
                
                # Process footer paragraphs and tables
                if section.footer:
                    footer_detections = self._process_paragraphs(section.footer.paragraphs, f"footer_section_{section_idx}")
                    all_detections.extend(footer_detections)
                    
                    # Process tables in footer
                    footer_table_detections = self._process_tables(section.footer.tables, f"footer_section_{section_idx}")
                    all_detections.extend(footer_table_detections)
        except Exception as e:
            logger.error(f"Error processing headers/footers: {e}")
        
        return all_detections
    
    def _update_processing_result(self, processing_result: ProcessingResult, detections: List[Dict[str, Any]]):
        """
        Update ProcessingResult with detection details.
        
        Args:
            processing_result: ProcessingResult to update
            detections: List of detection dictionaries
        """
        # Count matches by type
        text_matches = 0
        graphics_matches = 0
        image_matches = 0
        graphics_no_match = 0
        image_no_match = 0
        
        # Convert detections to MatchDetail objects
        match_details = []
        sr_no = 1
        
        for detection in detections:
            # Count matches
            if detection.get('processor') == 'Text':
                if detection.get('is_matched', False):
                    text_matches += 1
            elif detection.get('processor') == 'Graphics':
                if detection.get('is_matched', False):
                    graphics_matches += 1
                else:
                    graphics_no_match += 1
            elif detection.get('processor') == 'Image':
                if detection.get('is_matched', False):
                    image_matches += 1
                else:
                    image_no_match += 1
            
            # Convert to MatchDetail
            match_detail_dict = self._convert_detection_to_match_detail(
                detection, sr_no, ProcessorType.TEXT.value
            )
            
            # Create MatchDetail object
            match_detail = MatchDetail(
                sr_no=match_detail_dict['sr_no'],
                type=ProcessorType.TEXT,
                orig_id_name=match_detail_dict['orig_id_name'],
                src_text=match_detail_dict['src_text'],
                src_text_font=match_detail_dict['src_text_font'],
                src_text_color=match_detail_dict['src_text_color'],
                src_text_size=match_detail_dict['src_text_size'],
                src_dimension=match_detail_dict['src_dimension'],
                mapped_text=match_detail_dict['mapped_text'],
                mapped_text_font=match_detail_dict['mapped_text_font'],
                mapped_text_color=match_detail_dict['mapped_text_color'],
                mapped_text_size=match_detail_dict['mapped_text_size'],
                match_flag=MatchFlag.YES if match_detail_dict['match_flag'] == 'Y' else MatchFlag.NO,
                is_fallback=FallbackFlag.NO,
                reasoning=None
            )
            
            match_details.append(match_detail)
            sr_no += 1
        
        # Update ProcessingResult
        processing_result.total_text_matches = text_matches
        processing_result.total_graphics_matches = graphics_matches
        processing_result.total_image_matches = image_matches
        processing_result.total_graphics_no_match = graphics_no_match
        processing_result.total_image_no_match = image_no_match
        processing_result.matches_found = text_matches + graphics_matches + image_matches
        processing_result.total_matches = len(detections)
        processing_result.match_details = match_details
        
        # Update processor type
        processing_result.processor_type = "text_processor"
    
    def _convert_detection_to_match_detail(
        self,
        detection: Dict[str, Any],
        sr_no: int,
        processor_type: str
    ) -> Dict[str, Any]:
        """
        Convert detection dictionary to MatchDetail format for ProcessingResult.
        
        Args:
            detection: Detection dictionary
            sr_no: Serial number for the match
            processor_type: Type of processor (TEXT, GRAPHICS, IMAGE)
            
        Returns:
            Dictionary in MatchDetail format
        """
        # Extract font information from the detection
        font_info = detection.get('font_info', {})
        font_family = font_info.get('font_family', 'Arial')
        font_size = font_info.get('font_size', '12.0')
        font_color = font_info.get('color', '000000')
        
        return {
            'sr_no': sr_no,
            'type': processor_type,
            'orig_id_name': detection.get('location', ''),
            'src_text': detection.get('matched_text', ''),
            'src_text_font': font_family,
            'src_text_color': font_color,
            'src_text_size': font_size,
            'src_dimension': detection.get('dimension', ''),
            'mapped_text': detection.get('replacement_text', ''),
            'mapped_text_font': font_family,  # Use same font for mapped text
            'mapped_text_color': font_color,  # Use same color for mapped text
            'mapped_text_size': font_size,    # Use same size for mapped text
            'match_flag': 'N' if detection.get('replacement_text', '') == DEFAULT_MAPPING else 'Y',
            'is_fallback': 'N',  # Will be updated during processing
            'reasoning': None  # Will be populated by specific processors
        }
    
    def cleanup(self):
        """Clean up resources used by the text processor."""
        logger.info("Cleaning up text processor")
        self.initialized = False


def create_text_processor(config) -> TextProcessor:
    """Factory function to create a TextProcessor instance."""
    return TextProcessor(config)
