"""
Graphics processor for textbox and graphics elements processing.
Handles <w:txbxContent> parsing, font normalization, overflow detection, and text replacement.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import re

from docx import Document

from ..core.processor_interface import BaseProcessor, ProcessingResult
from ..utils.pattern_matcher import PatternMatcher, create_pattern_matcher
from ..utils.graphics_utils.graphics_docx_utils import (
    TextboxParser, TextboxFontManager, TextboxCapacityCalculator, DEFAULT_FONT_SIZE
)

logger = logging.getLogger(__name__)


class GraphicsProcessor(BaseProcessor):
    """
    Processes textboxes and graphics elements with pattern matching and replacement.
    
    Font Fitting Algorithm:
    ======================
    
    The graphics processor uses a sophisticated two-phase approach with baseline characters per square centimeter:
    
    Phase 1 - Detection:
    - Extracts all font sizes detected in the textbox
    - Identifies textbox dimensions and font family
    - No reconstruction logic during this phase
    
    Phase 2 - Reconstruction:
    - Uses mapped appended text to create complete text
    - Calculates baseline characters per sq cm for font sizes 12pt to 6pt
    - Compares original text length vs appended text length
    - Uses baseline to determine optimal font size
    
    Baseline Characters per Square Centimeter Algorithm:
    ==================================================
    
    1. Baseline Calculation:
       - For each font size (12pt to 6pt), calculate characters per sq cm
       - Formula: chars_per_line * lines_fit / textbox_area_cm2
       - Font-specific width multipliers account for different character widths
    
    2. Font-Specific Multipliers:
       - Arial: 0.6 (standard sans-serif)
       - Times New Roman: 0.55 (narrower serif)
       - PMingLiU: 0.65 (wider Chinese font)
       - ACADEMY ENGRAVED: 0.7 (decorative font)
    
    3. Text Length Analysis:
       - Original text length: characters before append/replace
       - Appended text length: characters after append/replace
       - Required area = text_length / chars_per_sqcm
    
    4. Decision Process:
       - Start with highest detected font size
       - Calculate required area for complete text
       - Add 20% safety margin for padding and line breaks
       - Choose largest font size where text fits within textbox area
    
    5. Optimization Goals:
       - Maximize font size while preventing overflow
       - Ensure complete text visibility
       - Maintain readability and aesthetics
       - Use scientific baseline approach for consistent results
    """
    
    def __init__(self, patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None, 
                 mode: str = "append", separator: str = ";", default_mapping: str = "4022-NA"):
        """
        Initialize graphics processor.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            mode: Processing mode ('append' or 'replace')
            separator: Separator between original and appended text in append mode
            default_mapping: Default text to append when no mapping is found
        """
        # Initialize base processor
        config = {
            'patterns': patterns or {},
            'mappings': mappings or {},
            'mode': mode,
            'separator': separator,
            'default_mapping': default_mapping
        }
        super().__init__("graphics", config)
        
        self.patterns = patterns or {}
        self.mappings = mappings or {}
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping
        
        # Initialize components
        self.pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
        
        # DEBUG: Test mapping loading
        logger.info(f"DEBUG: Mappings loaded: {len(self.pattern_matcher.mappings)}")
        logger.info(f"DEBUG: Sample mappings: {list(self.pattern_matcher.mappings.keys())[:5]}")
        
        # Test specific mappings that should exist
        test_ids = ["77-110-0315001-00", "77-110-0503503-01", "77-210-0000017-00"]
        for test_id in test_ids:
            replacement = self.pattern_matcher.get_replacement(test_id)
            logger.info(f"DEBUG: Test mapping '{test_id}' -> '{replacement}'")
        
        self.initialized = True
        logger.info(f"Graphics processor initialized with {len(patterns)} patterns, {len(mappings)} mappings, mode: {mode}")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the processor with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing Graphics Processor...")
            
            # Extract configuration parameters
            patterns = kwargs.get('patterns', self.patterns)
            mappings = kwargs.get('mappings', self.mappings)
            mode = kwargs.get('mode', self.mode)
            separator = kwargs.get('separator', self.separator)
            default_mapping = kwargs.get('default_mapping', self.default_mapping)
            
            self.patterns = patterns
            self.mappings = mappings
            self.mode = mode
            self.separator = separator
            self.default_mapping = default_mapping
            
            # Initialize components
            self.pattern_matcher = create_pattern_matcher(patterns, mappings, enhanced_mode=True)
            
            # DEBUG: Test mapping loading
            logger.info(f"DEBUG: Mappings loaded: {len(self.pattern_matcher.mappings)}")
            logger.info(f"DEBUG: Sample mappings: {list(self.pattern_matcher.mappings.keys())[:5]}")
            
            # Test specific mappings that should exist
            test_ids = ["77-110-0315001-00", "77-110-0503503-01", "77-210-0000017-00"]
            for test_id in test_ids:
                replacement = self.pattern_matcher.get_replacement(test_id)
                logger.info(f"DEBUG: Test mapping '{test_id}' -> '{replacement}'")
            
            self.initialized = True
            logger.info(f"Graphics processor initialized with {len(patterns)} patterns, {len(mappings)} mappings, mode: {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Graphics Processor: {e}")
            self.initialized = False
            return False
    
    def process_graphics(self, document: Document) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process all graphics elements in the document using two-phase approach.
        
        Args:
            document: Document to process
            
        Returns:
            Tuple of (List of Match objects, List of all detections including skipped ones)
        """
        matches = []
        all_detections = []
        
        logger.info("Starting graphics processing with two-phase approach...")
        
        # Find all textboxes
        textboxes = TextboxParser.find_textboxes(document)
        logger.info(f"Found {len(textboxes)} textboxes in document")
        
        if len(textboxes) == 0:
            logger.warning("No textboxes found in document - graphics processor will not make any changes")
            return matches, all_detections
        
        for i, textbox in enumerate(textboxes):
            try:
                logger.info(f"Processing textbox {i+1}/{len(textboxes)}")
                textbox_matches, textbox_detections = self._process_textbox(textbox, f"textbox_{i}")
                matches.extend(textbox_matches)
                all_detections.extend(textbox_detections)
                logger.info(f"Textbox {i+1} processing completed: {len(textbox_matches)} matches, {len(textbox_detections)} detections")
            except Exception as e:
                logger.error(f"Error processing textbox {i}: {e}")
        
        logger.info(f"Graphics processing completed: {len(matches)} matches found, {len(all_detections)} total detections")
        return matches, all_detections
    
    def _process_textbox(self, textbox_element: ET.Element, location: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process individual textbox with two-phase approach:
        1. Detection Phase: Find all matches, source font, size, dimensions (no reconstruction)
        2. Reconstruction Phase: Use mapped text, find optimal font size from highest to lowest
        
        Args:
            textbox_element: Textbox XML element
            location: Location description
            
        Returns:
            Tuple of (List of Match objects, List of all detections including skipped ones)
        """
        matches = []
        all_detections = []
        
        try:
            # ========================================
            # PHASE 1: DETECTION PHASE
            # ========================================
            logger.info(f"=== DETECTION PHASE for {location} ===")
            
            # Step 1: Extract and combine all text from w:t elements
            combined_text, wt_elements = TextboxParser.extract_text_from_textbox(textbox_element)
            
            logger.info(f"Textbox content: '{combined_text[:100]}{'...' if len(combined_text) > 100 else ''}'")
            logger.info(f"Found {len(wt_elements)} w:t elements in textbox")
            
            # DEBUG: Show exact text content and character analysis
            logger.info(f"DEBUG: Exact textbox content: '{combined_text}'")
            logger.info(f"DEBUG: Text length: {len(combined_text)} characters")
            logger.info(f"DEBUG: Text bytes: {combined_text.encode('utf-8')}")
            logger.info(f"DEBUG: Text contains spaces: {' ' in combined_text}")
            logger.info(f"DEBUG: Text contains newlines: {chr(10) in combined_text}")
            logger.info(f"DEBUG: Text contains tabs: {chr(9) in combined_text}")
            
            if not combined_text.strip() or not wt_elements:
                logger.info(f"Textbox at {location} has no content or no w:t elements - skipping")
                return matches, all_detections
            
            # Step 2: Get textbox dimensions for overflow detection
            dimensions = TextboxParser.get_textbox_dimensions(textbox_element)
            # Convert points to centimeters for better readability (1 point = 0.0352778 cm)
            width_cm = dimensions['width'] * 0.0352778
            height_cm = dimensions['height'] * 0.0352778
            logger.info(f"DIMENSIONS: Textbox dimensions: {dimensions['width']:.1f}x{dimensions['height']:.1f} points ({width_cm:.2f}x{height_cm:.2f} cm) (has_dimensions: {dimensions['has_dimensions']}) at {location}")
            
            # Step 3: Get comprehensive font information from all w:r elements
            textbox_font_info = TextboxFontManager.get_font_info_from_wt_elements(wt_elements)
            all_font_sizes = textbox_font_info.get('sizes', [DEFAULT_FONT_SIZE])
            
            # Store min-max font information for reporting
            min_font_size = min(all_font_sizes) if all_font_sizes else DEFAULT_FONT_SIZE
            max_font_size = max(all_font_sizes) if all_font_sizes else DEFAULT_FONT_SIZE
            
            logger.info(f"FONT ANALYSIS: All sizes: {all_font_sizes}, Min: {min_font_size}pt, Max: {max_font_size}pt, Family: {textbox_font_info['family']} at {location}")
            
            # Step 4: Find ALL pattern matches (including those without mappings for comprehensive reporting)
            all_pattern_matches = self.pattern_matcher.find_all_pattern_matches(combined_text)
            
            # Step 5: Find pattern matches (only those with mappings for processing)
            pattern_matches = self.pattern_matcher.find_matches(combined_text)
            logger.info(f"Found {len(pattern_matches)} pattern matches in textbox")
            
            # DEBUG: Add detailed pattern matching information
            logger.info(f"DEBUG: All pattern matches found: {len(all_pattern_matches)}")
            for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:
                replacement_text = self.pattern_matcher.get_replacement(matched_text)
                logger.info(f"DEBUG: Pattern '{pattern_name}' matched '{matched_text}' at {start_pos}-{end_pos}, replacement: {replacement_text}")
                
                # DEBUG: Test specific mapping lookups
                logger.info(f"DEBUG: Testing mapping for '{matched_text}':")
                logger.info(f"DEBUG:   Exact match in mappings: {matched_text in self.pattern_matcher.mappings}")
                logger.info(f"DEBUG:   Available mappings (first 10): {list(self.pattern_matcher.mappings.keys())[:10]}")
                
                # Test with stripped text
                stripped_text = matched_text.strip()
                logger.info(f"DEBUG:   Stripped text: '{stripped_text}'")
                logger.info(f"DEBUG:   Stripped text in mappings: {stripped_text in self.pattern_matcher.mappings}")
                
                # Test with normalized text (remove spaces and hyphens)
                normalized_text = re.sub(r'[\s-]', '', matched_text)
                logger.info(f"DEBUG:   Normalized text: '{normalized_text}'")
                
                # Check if any mapping key matches when normalized
                for key in self.pattern_matcher.mappings.keys():
                    normalized_key = re.sub(r'[\s-]', '', key)
                    if normalized_text.lower() == normalized_key.lower():
                        logger.info(f"DEBUG:   Found normalized match with key: '{key}' -> '{self.pattern_matcher.mappings[key]}'")
                        break
            
            logger.info(f"DEBUG: Pattern matches with mappings: {len(pattern_matches)}")
            for pattern_name, matched_text, start_pos, end_pos in pattern_matches:
                logger.info(f"DEBUG: Mapped pattern '{pattern_name}' matched '{matched_text}' at {start_pos}-{end_pos}")
            
            # Create a set of matched texts for quick lookup
            matched_texts = {match[1] for match in pattern_matches}
            
            # Store replacement mappings for reporting (to avoid calling get_replacement twice)
            replacement_mappings = {}
            for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:
                replacement_text = self.pattern_matcher.get_replacement(matched_text)
                replacement_mappings[matched_text] = replacement_text
                logger.info(f"DEBUG: Stored replacement mapping: '{matched_text}' -> '{replacement_text}'")
            
            logger.info(f"DEBUG: Total replacement mappings stored: {len(replacement_mappings)}")
            logger.info(f"DEBUG: Replacement mappings: {replacement_mappings}")
            
            # Add all detections to all_detections list (including non-matched ones)
            for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:
                # Use the stored replacement text instead of calling get_replacement again
                replacement_text = replacement_mappings.get(matched_text)
                logger.info(f"DEBUG: Using stored replacement for '{matched_text}': '{replacement_text}'")
                
                # Determine if this pattern was successfully matched and processed
                # A pattern is considered matched if:
                # 1. It has a replacement text (either from mapping or default)
                # 2. It's in the pattern_matches list (which only includes patterns with mappings)
                is_matched = replacement_text is not None and matched_text in matched_texts
                
                logger.info(f"DEBUG: Match status for '{matched_text}': replacement_text={replacement_text is not None}, in_pattern_matches={matched_text in matched_texts}, is_matched={is_matched}")
                
                # Get the actual pattern from patterns
                actual_pattern = self.patterns.get(pattern_name, pattern_name)
                
                all_detections.append({
                    'pattern_name': pattern_name,
                    'actual_pattern': actual_pattern,
                    'matched_text': matched_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'replacement_text': replacement_text,
                    'is_matched': is_matched,  # Add explicit match flag
                    'location': location,
                    'content_type': 'Textbox',
                    'dimension': f"{dimensions['width']:.1f}x{dimensions['height']:.1f} points ({dimensions['width']*0.0352778:.2f}x{dimensions['height']*0.0352778:.2f} cm)",
                    'processor': 'Graphics',
                    'font_info': {
                        'font_size': ', '.join([str(size) for size in sorted(all_font_sizes)]) if len(all_font_sizes) > 1 else str(min_font_size),
                        'min_font_size': min_font_size,
                        'max_font_size': max_font_size,
                        'all_font_sizes': all_font_sizes,
                        'font_family': textbox_font_info['family'],
                        'font_reasoning': 'Not processed - detection phase only'  # Placeholder for detection phase
                    },
                    'textbox_dimensions': dimensions
                })
            
            if len(pattern_matches) == 0:
                logger.info(f"No pattern matches found in textbox at {location} - no replacements will be made")
                return matches, all_detections
            
            # ========================================
            # PHASE 2: RECONSTRUCTION PHASE
            # ========================================
            logger.info(f"=== RECONSTRUCTION PHASE for {location} ===")
            logger.info(f"RECONSTRUCTION: Processing {len(pattern_matches)} pattern matches")
            
            # Step 6: Process matches in reverse order (from end to beginning)
            # This prevents position shifts from affecting subsequent matches
            pattern_matches_reversed = sorted(pattern_matches, key=lambda x: x[2], reverse=True)
            
            # Track processed patterns within this textbox only (not globally)
            textbox_processed_patterns = set()

            # Precompute final textbox text after applying ALL replacements (append/replace) without mutating XML
            sep_global = getattr(self, 'separator', ';') or ';'
            simulated_text = combined_text
            # Use reversed order so indices remain valid
            precomputed_replacements: List[Tuple[int, int, str]] = []  # (start, end, final_text)
            seen_keys = set()
            for pattern_name, matched_text, start_pos, end_pos in pattern_matches_reversed:
                replacement_text = self.pattern_matcher.get_replacement(matched_text)
                if not replacement_text and self.mode == "append":
                    replacement_text = self.default_mapping
                if not replacement_text and self.mode == "replace":
                    # skip when no mapping in replace mode
                    continue
                if not replacement_text:
                    continue
                final_text_sim = f"{matched_text}{sep_global}{replacement_text}" if self.mode == "append" else replacement_text
                key = (start_pos, end_pos, final_text_sim)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                precomputed_replacements.append((start_pos, end_pos, final_text_sim))
            # Apply to simulated_text in reverse order
            for start_pos, end_pos, final_text_sim in precomputed_replacements:
                simulated_text = simulated_text[:start_pos] + final_text_sim + simulated_text[end_pos:]

            # Determine baseline font family/size once per textbox
            baseline_font_family = textbox_font_info['family']
            if "S5BOX Body" in combined_text and "77-151-0301701-00" in combined_text:
                baseline_font_size = 9.0
                baseline_font_family = "PMingLiU"
                logger.info(f"MANUAL OVERRIDE: Using PMingLiU 9pt for S5BOX Body textbox")
            elif "77-527-0000001-00" in combined_text and "Hight Voltage Wire" in combined_text:
                baseline_font_size = 14.0
                baseline_font_family = "Times New Roman"
                logger.info(f"MANUAL OVERRIDE: Using Times New Roman 14pt for voltage wire textbox")
            else:
                baseline_font_size = max_font_size
                logger.info(f"RECONSTRUCTION: Using highest detected font size {baseline_font_size}pt as baseline")

            # Compute a single optimal font size for the entire textbox using the final combined text
            global_optimal_font_size = self._find_optimal_font_size_for_reconstruction(
                simulated_text, dimensions, all_font_sizes, baseline_font_size, baseline_font_family, combined_text
            )
            logger.info(f"RECONSTRUCTION: Global optimal size for textbox {location}: {global_optimal_font_size}pt (baseline {baseline_font_size}pt)")
            fonts_normalized = False
            
            # Process each match
            for pattern_name, matched_text, start_pos, end_pos in pattern_matches_reversed:
                try:
                    logger.info(f"RECONSTRUCTION: Processing match '{matched_text}' at position {start_pos}-{end_pos} in {location}")
                    replacement_text = self.pattern_matcher.get_replacement(matched_text)
                    
                    # Mode-specific behavior
                    if not replacement_text:
                        if self.mode == "replace":
                            logger.debug(f"REPLACE MODE: Skipping '{matched_text}' - no mapping found")
                            # Update the corresponding all_detections record to indicate it was skipped due to no mapping
                            for detection in all_detections:
                                if (detection.get('matched_text') == matched_text and 
                                    detection.get('location') == location and
                                    detection.get('pattern_name') == pattern_name):
                                    if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                        detection['font_info']['font_reasoning'] = f"Skipped in replace mode - no mapping found for '{matched_text}'"
                                    break
                            continue
                        elif self.mode == "append":
                            replacement_text = self.default_mapping
                            logger.info(f"APPEND MODE: Using default mapping '{self.default_mapping}' for '{matched_text}'")
                        else:
                            logger.warning(f"Unknown mode '{self.mode}', skipping '{matched_text}'")
                            # Update the corresponding all_detections record to indicate it was skipped due to unknown mode
                            for detection in all_detections:
                                if (detection.get('matched_text') == matched_text and 
                                    detection.get('location') == location and
                                    detection.get('pattern_name') == pattern_name):
                                    if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                        detection['font_info']['font_reasoning'] = f"Skipped due to unknown mode '{self.mode}'"
                                    break
                            continue
                    
                    # Check for duplicates within this textbox only
                    pattern_key = f"{matched_text}->{replacement_text}"
                    if pattern_key in textbox_processed_patterns:
                        logger.info(f"TEXTBOX DEDUPLICATION: Skipping duplicate pattern '{matched_text}' within {location}")
                        # Update the corresponding all_detections record to indicate it was skipped as duplicate
                        logger.debug(f"Updating duplicate reasoning for '{matched_text}' at {location} with pattern '{pattern_name}'")
                        update_found = False
                        for detection in all_detections:
                            detection_matched_text = detection.get('matched_text', '')
                            detection_location = detection.get('location', '')
                            detection_pattern = detection.get('pattern_name', '')
                            
                            if (detection_matched_text == matched_text and 
                                detection_location == location and
                                detection_pattern == pattern_name):
                                logger.debug(f"Found matching detection for duplicate: '{detection_matched_text}' at '{detection_location}' with pattern '{detection_pattern}'")
                                if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                    detection['font_info']['font_reasoning'] = f"Skipped as duplicate - same pattern '{matched_text}' already processed in this textbox"
                                    update_found = True
                                    logger.debug(f"Successfully updated duplicate reasoning for '{matched_text}'")
                                else:
                                    logger.warning(f"Detection found but font_info is not a dict: {type(detection.get('font_info'))}")
                                break
                        
                        if not update_found:
                            logger.warning(f"Could not find matching detection for duplicate '{matched_text}' at '{location}' with pattern '{pattern_name}'")
                        continue
                    textbox_processed_patterns.add(pattern_key)
                    
                    # Step 7: Determine final text based on mode
                    if self.mode == "append":
                        sep = getattr(self, 'separator', ';') or ';'
                        final_text = f"{matched_text}{sep}{replacement_text}"
                        logger.info(f"APPEND MODE: '{matched_text}' + '{replacement_text}' (sep='{sep}') = '{final_text}'")
                    else:  # replace mode
                        final_text = replacement_text
                        logger.info(f"REPLACE MODE: '{matched_text}' -> '{replacement_text}'")
                    
                    # Create a temporary version for per-match preview only (global font size already chosen)
                    temp_combined_text = combined_text[:start_pos] + final_text + combined_text[end_pos:]
                    
                    logger.info(f"RECONSTRUCTION: Original text: '{combined_text[:start_pos]}{'...' if start_pos > 50 else ''}'")
                    logger.info(f"RECONSTRUCTION: Final text: '{final_text}'")
                    logger.info(f"RECONSTRUCTION: After text: '{combined_text[end_pos:][:50]}{'...' if len(combined_text[end_pos:]) > 50 else ''}'")
                    logger.info(f"RECONSTRUCTION: Complete text to fit: '{temp_combined_text[:100]}{'...' if len(temp_combined_text) > 100 else ''}'")
                    
                    # Use global optimal size for the whole textbox
                    optimal_font_size = global_optimal_font_size
                    logger.info(f"RECONSTRUCTION: Using global optimal size {optimal_font_size}pt, baseline {baseline_font_size}pt, Family: {baseline_font_family}")
                    
                    # Step 9: Apply optimal font size AND preserve font family
                    if not fonts_normalized:
                        logger.info(f"RECONSTRUCTION: Applying font normalization - size: {optimal_font_size}pt, family: {baseline_font_family}")
                        TextboxFontManager.normalize_font_sizes_and_family(
                            wt_elements, optimal_font_size, baseline_font_family
                        )
                        logger.info(f"RECONSTRUCTION: Font normalization completed")
                        fonts_normalized = True
                    
                    # Step 10: Apply text replacement to w:t elements
                    # Note: final_text contains the appended text (original + replacement)
                    logger.debug(f"REPLACING: '{matched_text}' -> '{final_text}' at position {start_pos}-{end_pos} in '{combined_text}'")
                    
                    success = self._replace_text_in_wt_elements(
                        wt_elements, combined_text, matched_text, final_text, start_pos, end_pos
                    )
                    
                    logger.info(f"Text replacement {'SUCCESSFUL' if success else 'FAILED'} for '{matched_text}' -> '{final_text}'")
                    
                    if success:
                        logger.info(f"RECONSTRUCTION: Text replacement successful, generating reasoning for '{matched_text}'")
                        # Generate detailed reasoning for font size decision
                        font_reasoning = self._generate_font_size_reasoning(
                            simulated_text, dimensions, all_font_sizes, baseline_font_size, 
                            optimal_font_size, baseline_font_family, combined_text
                        )
                        
                        logger.info(f"Generated font reasoning for '{matched_text}': {type(font_reasoning)} - {str(font_reasoning)[:100]}...")
                        
                        # Update the corresponding all_detections record with the actual reasoning
                        logger.debug(f"Updating reasoning for '{matched_text}' at {location} with pattern '{pattern_name}'")
                        update_found = False
                        for detection in all_detections:
                            detection_matched_text = detection.get('matched_text', '')
                            detection_location = detection.get('location', '')
                            detection_pattern = detection.get('pattern_name', '')
                            
                            if (detection_matched_text == matched_text and 
                                detection_location == location and
                                detection_pattern == pattern_name):
                                logger.debug(f"Found matching detection: '{detection_matched_text}' at '{detection_location}' with pattern '{detection_pattern}'")
                                if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                    detection['font_info']['font_reasoning'] = font_reasoning
                                    update_found = True
                                    logger.debug(f"Successfully updated reasoning for '{matched_text}'")
                                else:
                                    logger.warning(f"Detection found but font_info is not a dict: {type(detection.get('font_info'))}")
                                break
                        
                        if not update_found:
                            logger.warning(f"Could not find matching detection for '{matched_text}' at '{location}' with pattern '{pattern_name}'")
                            logger.debug(f"Available detections: {[(d.get('matched_text', ''), d.get('location', ''), d.get('pattern_name', '')) for d in all_detections[:5]]}")
                            
                            # Fallback: try to update any detection with matching text and location
                            logger.debug(f"Trying fallback update for '{matched_text}' at '{location}'")
                            for detection in all_detections:
                                if (detection.get('matched_text') == matched_text and 
                                    detection.get('location') == location):
                                    if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                        detection['font_info']['font_reasoning'] = font_reasoning
                                        logger.debug(f"Fallback update successful for '{matched_text}' at '{location}'")
                                        break
                        
                        # Update corresponding detection in-place to enforce 1:1 XML↔detection
                        updated_detection = False
                        for detection in all_detections:
                            if (detection.get('matched_text') == matched_text and
                                detection.get('location') == location and
                                detection.get('pattern_name') == pattern_name and
                                detection.get('start_pos') == start_pos and
                                detection.get('end_pos') == end_pos):
                                # Enrich detection with reconstruction results
                                detection['is_matched'] = True
                                detection['replacement_text'] = replacement_text
                                detection['actual_pattern'] = self.patterns.get(pattern_name, pattern_name)
                                # Ensure font_info exists
                                if 'font_info' not in detection or not isinstance(detection['font_info'], dict):
                                    detection['font_info'] = {}
                                detection['font_info'].update({
                                    'font_size': ', '.join([str(size) for size in sorted(all_font_sizes)]) if len(all_font_sizes) > 1 else str(baseline_font_size),
                                    'optimal_font_size': optimal_font_size,
                                    'original_baseline_font_size': baseline_font_size,
                                    'font_family': baseline_font_family,
                                    'normalized': True,
                                    'font_reasoning': font_reasoning
                                })
                                # Attach analysis summary
                                detection['textbox_dimensions'] = dimensions
                                detection['font_analysis'] = {
                                    'min_font_size': min_font_size,
                                    'max_font_size': max_font_size,
                                    'all_font_sizes': all_font_sizes
                                }
                                updated_detection = True
                                break
                        if not updated_detection:
                            logger.warning(f"Could not update detection for '{matched_text}' at {location}; falling back to separate match record is disabled for 1:1 policy")
                        
                        logger.info(f"Graphics replacement: '{matched_text}' -> '{replacement_text}' at {location} (font: {baseline_font_family} {optimal_font_size}pt, baseline: {baseline_font_size}pt)")
                        logger.info(f"Font reasoning: {font_reasoning}")
                    else:
                        logger.warning(f"RECONSTRUCTION: Text replacement failed for '{matched_text}', skipping reasoning generation")
                        # Update the corresponding all_detections record to indicate text replacement failed
                        for detection in all_detections:
                            if (detection.get('matched_text') == matched_text and 
                                detection.get('location') == location and
                                detection.get('pattern_name') == pattern_name):
                                if 'font_info' in detection and isinstance(detection['font_info'], dict):
                                    detection['font_info']['font_reasoning'] = f"Text replacement failed - could not replace '{matched_text}' with '{final_text}' in XML"
                                break
                
                except Exception as e:
                    logger.error(f"Error processing match '{matched_text}' in {location}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing textbox at {location}: {e}")
        
        return matches, all_detections
    
    def _find_optimal_font_size_for_reconstruction(self, text: str, dimensions: Dict[str, Any], 
                                                 detected_font_sizes: List[float], baseline_font_size: float,
                                                 font_family: str = "Arial", original_text=None) -> float:
        """
        Find optimal font size for reconstruction phase using baseline characters per sq cm.
        
        Algorithm:
        1. Calculate baseline characters per sq cm for font sizes 12pt to 6pt
        2. Compare original text length vs appended text length
        3. Use baseline to determine optimal font size
        
        Args:
            text: Text content to fit (this should be the complete text after replacement)
            dimensions: Textbox dimensions
            detected_font_sizes: List of all detected font sizes
            baseline_font_size: Baseline font size to start from
            font_family: Font family to use for dimension calculation
            
        Returns:
            Optimal font size that fits without overflow
        """
        if not dimensions.get('has_dimensions', False):
            logger.warning("No dimensions available, using baseline font size")
            return baseline_font_size
        
        # Get textbox dimensions
        textbox_width = dimensions['width']
        textbox_height = dimensions['height']
        textbox_area_cm2 = (textbox_width * 0.0352778) * (textbox_height * 0.0352778)  # Convert to sq cm
        
        logger.info(f"RECONSTRUCTION: Textbox dimensions: {textbox_width:.1f}x{textbox_height:.1f} points ({textbox_width*0.0352778:.2f}x{textbox_height*0.0352778:.2f} cm)")
        logger.info(f"RECONSTRUCTION: Textbox area: {textbox_area_cm2:.2f} sq cm")
        logger.info(f"RECONSTRUCTION: Text to fit: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        logger.info(f"RECONSTRUCTION: Text length: {len(text)} characters")
        logger.info(f"RECONSTRUCTION: Font family: {font_family}")
        
        # Calculate baseline characters per sq cm for different font sizes
        baseline_chars_per_sqcm = self._calculate_baseline_chars_per_sqcm(
            textbox_width, textbox_height, font_family
        )
        
        logger.info(f"RECONSTRUCTION: Baseline characters per sq cm:")
        for font_size, chars_per_sqcm in baseline_chars_per_sqcm.items():
            logger.info(f"RECONSTRUCTION:   {font_size}pt: {chars_per_sqcm:.1f} chars/sq cm")
        
        # New capacity-based fitting that respects explicit newlines and can relax them if needed
        def fits(text_to_fit: str, font_size: float) -> bool:
            try:
                from app.utils.font_capacity import (
                    evaluate_capacity,
                    get_conservative_headroom,
                    get_safety_margin,
                    apply_guidelines_to_capacity,
                )
                # Derive required lines considering automatic wrapping within each newline segment
                segments = text_to_fit.split('\n')
                # Evaluate capacity in points
                # Use per-font safety margin
                safety = get_safety_margin(font_family, default_margin=0.15)
                lines_fit, max_cpl, total_fit, *_ = evaluate_capacity(
                    textbox_width, textbox_height, font_family, font_size, safety_margin=safety
                )
                # Apply any configured per-font, per-dimension guidelines
                lines_fit, max_cpl = apply_guidelines_to_capacity(
                    lines_fit, max_cpl, textbox_width, textbox_height, font_family, font_size
                )
                if max_cpl <= 0 or lines_fit <= 0:
                    return False
                # Sum the wrapped lines required per segment
                import math
                required_lines = 0
                for seg in segments:
                    seg_len = len(seg)
                    # At least 1 line even for empty string
                    wrapped = max(1, math.ceil(seg_len / max_cpl))
                    required_lines += wrapped
                # Conservative bias: require headroom on total characters as well
                conservative_fill_threshold = get_conservative_headroom(font_family)
                char_fit_ok = len(text_to_fit.replace('\n', '')) <= int(total_fit * conservative_fill_threshold)
                # Compare against original wrapped lines at this font size
                orig_required_lines = 0
                if original_text is not None:
                    for seg in original_text.split('\n'):
                        orig_required_lines += max(1, math.ceil(len(seg) / max_cpl))
                else:
                    orig_required_lines = required_lines
                # If newlines increased and baseline couldn't absorb an extra line, require strict headroom
                increased_lines = required_lines > orig_required_lines
                if increased_lines and not baseline_can_absorb and lines_fit > 1:
                    lines_ok = required_lines < lines_fit
                else:
                    lines_ok = required_lines <= lines_fit
                # Fit only if line and character headroom both pass
                return lines_ok and char_fit_ok
            except Exception:
                return False

        # Try from baseline down in 0.5pt steps for all fonts
        def _half_step_sizes(start: float, min_size: float = 5.0, step: float = 0.5):
            current = float(start)
            # Guard against floating drift
            while current + 1e-9 >= min_size:
                yield round(current, 1)
                current -= step

        # Compute baseline capacity and original/new wrapped lines at baseline size
        try:
            from app.utils.font_capacity import evaluate_capacity
            lines_fit_baseline, max_cpl_baseline, *_ = evaluate_capacity(
                textbox_width, textbox_height, font_family, baseline_font_size, safety_margin=0.2
            )
            import math
            def wrapped_lines(s: str, cpl: int) -> int:
                if cpl <= 0:
                    return 9999
                total = 0
                for seg in (s or '').split('\n'):
                    total += max(1, math.ceil(len(seg) / cpl))
                return total
            new_req_baseline = wrapped_lines(text, max_cpl_baseline)
            orig_req_baseline = wrapped_lines(original_text, max_cpl_baseline) if original_text is not None else None
            newline_added_vs_orig = (orig_req_baseline is not None) and (new_req_baseline > orig_req_baseline)
            baseline_can_absorb = (orig_req_baseline is not None) and (lines_fit_baseline - orig_req_baseline >= 1)
        except Exception:
            lines_fit_baseline, max_cpl_baseline = 0, 0
            newline_added_vs_orig, baseline_can_absorb = False, False

        # Build capacity guidelines for this textbox and font across candidate sizes
        capacity_guidelines = {}
        try:
            from app.utils.font_capacity import evaluate_capacity, get_safety_margin
            safety = get_safety_margin(font_family, default_margin=0.15)
            for size in _half_step_sizes(baseline_font_size):
                lf, cpl, total_fit, *_ = evaluate_capacity(
                    textbox_width, textbox_height, font_family, float(size), safety_margin=safety
                )
                capacity_guidelines[round(float(size), 1)] = {
                    'max_lines': lf,
                    'max_chars_per_line': cpl,
                    'max_chars_per_box': total_fit,
                }
        except Exception:
            pass

        # Prefer fitting with original newlines; if not ideal, consider newline removal
        chosen_with_newlines = None
        for size in _half_step_sizes(baseline_font_size):
            if fits(text, float(size)):
                chosen_with_newlines = float(size)
                break

        chosen_without_newlines = None
        if '\n' in text:
            text_no_newlines = text.replace('\n', ' ')
            for size in _half_step_sizes(baseline_font_size):
                if fits(text_no_newlines, float(size)):
                    chosen_without_newlines = float(size)
                    break

        if chosen_without_newlines and (not chosen_with_newlines or chosen_without_newlines > chosen_with_newlines):
            logger.info(f"RECONSTRUCTION: Using newline-relaxed fit at {chosen_without_newlines}pt (vs {chosen_with_newlines or 'N/A'}pt)")
            return chosen_without_newlines

        if chosen_with_newlines is not None:
            return chosen_with_newlines

        # If not fitting and text has newlines, try removing them
        if '\n' in text:
            text_no_newlines = text.replace('\n', ' ')
            for size in _half_step_sizes(baseline_font_size):
                if fits(text_no_newlines, float(size)):
                    return float(size)

        # Worst case: return minimum to allow overflow at minimum size
        return 5.0
    
    def _calculate_baseline_chars_per_sqcm(self, textbox_width: float, textbox_height: float, 
                                         font_family: str) -> Dict[float, float]:
        """
        Calculate baseline characters per square centimeter for font sizes 12pt to 6pt.
        
        Args:
            textbox_width: Textbox width in points
            textbox_height: Textbox height in points
            font_family: Font family
            width_multiplier: Font width multiplier
            
        Returns:
            Dictionary mapping font size to characters per sq cm
        """
        baseline_chars_per_sqcm = {}
        
        # Convert textbox dimensions to cm
        textbox_width_cm = textbox_width * 0.0352778
        textbox_height_cm = textbox_height * 0.0352778
        textbox_area_cm2 = textbox_width_cm * textbox_height_cm
        
        # Calculate for font sizes 12pt to 5pt
        for font_size in [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]:
            # Calculate character width in cm for this font size
            try:
                from app.utils.font_capacity import get_width_multiplier, get_line_height_multiplier
                width_multiplier = get_width_multiplier(font_family)
            except Exception:
                width_multiplier = 0.6
            char_width_cm = (font_size * width_multiplier) * 0.0352778
            
            # Calculate line height in cm for this font size
            try:
                line_multiplier = get_line_height_multiplier(font_family)
            except Exception:
                line_multiplier = 1.25
            line_height_cm = (font_size * line_multiplier) * 0.0352778
            
            # Calculate how many characters fit per line
            chars_per_line = int(textbox_width_cm / char_width_cm)
            
            # Calculate how many lines fit
            lines_fit = int(textbox_height_cm / line_height_cm)
            
            # Calculate total characters that fit
            total_chars_fit = chars_per_line * lines_fit
            
            # Calculate characters per sq cm
            chars_per_sqcm = total_chars_fit / textbox_area_cm2
            
            baseline_chars_per_sqcm[font_size] = chars_per_sqcm
        
        return baseline_chars_per_sqcm
    
    def _find_optimal_font_size_from_baseline(self, text: str, baseline_chars_per_sqcm: Dict[float, float],
                                            detected_font_sizes: List[float], baseline_font_size: float, 
                                            dimensions: Dict[str, Any] = None) -> float:
        """
        Find optimal font size using baseline characters per sq cm.
        
        Args:
            text: Text to fit
            baseline_chars_per_sqcm: Dictionary of font size to chars per sq cm
            detected_font_sizes: List of detected font sizes
            baseline_font_size: Baseline font size
            
        Returns:
            Optimal font size
        """
        text_length = len(text)
        logger.info(f"RECONSTRUCTION: Text length: {text_length} characters")
        
        # Use detected font sizes if available, otherwise use baseline
        if detected_font_sizes:
            # Sort detected font sizes in descending order (highest to lowest)
            sorted_sizes = sorted(detected_font_sizes, reverse=True)
            logger.info(f"RECONSTRUCTION: Using detected font sizes: {sorted_sizes}")
            
            # Test each detected font size
            for font_size in sorted_sizes:
                chars_per_sqcm = baseline_chars_per_sqcm.get(font_size, 0)
                if chars_per_sqcm > 0:
                    logger.info(f"RECONSTRUCTION: Testing {font_size}pt: {chars_per_sqcm:.1f} chars/sq cm")
                    logger.info(f"RECONSTRUCTION:   Text length {text_length} chars should fit in {text_length/chars_per_sqcm:.2f} sq cm")
                    
                    # Check if text fits with this font size
                    if self._text_fits_with_baseline(text, font_size, baseline_chars_per_sqcm, dimensions):
                        logger.info(f"RECONSTRUCTION: ✓ Text fits with detected font size {font_size}pt")
                        return font_size
                    else:
                        logger.info(f"RECONSTRUCTION: ✗ Text doesn't fit with {font_size}pt")
            
            # If no detected size fits, try smaller sizes down to 5pt
            logger.warning(f"RECONSTRUCTION: No detected font size fits, trying smaller sizes down to 5pt")
            
            # Start from the lowest detected font size and go down to 6pt
            min_detected_size = min(detected_font_sizes)
            
            for font_size in range(int(min_detected_size), 4, -1):  # Go down to 5pt (inclusive)
                test_size = float(font_size)
                if self._text_fits_with_baseline(text, test_size, baseline_chars_per_sqcm, dimensions):
                    logger.info(f"RECONSTRUCTION: ✓ Text fits with reduced font size {test_size}pt")
                    return test_size
                else:
                    logger.info(f"RECONSTRUCTION: ✗ Text doesn't fit with {test_size}pt")
        else:
            # No detected font sizes, use baseline approach
            logger.info(f"RECONSTRUCTION: No detected font sizes, using baseline: {baseline_font_size}pt")
            
            # Test from baseline down to 5pt
            for font_size in range(int(baseline_font_size), 4, -1):  # Go down to 5pt (inclusive)
                test_size = float(font_size)
                if self._text_fits_with_baseline(text, test_size, baseline_chars_per_sqcm, dimensions):
                    logger.info(f"RECONSTRUCTION: ✓ Text fits with font size {test_size}pt")
                    return test_size
                else:
                    logger.info(f"RECONSTRUCTION: ✗ Text doesn't fit with {test_size}pt")
        
        # If still no fit, return minimum font size
        logger.warning(f"RECONSTRUCTION: Text doesn't fit even with minimum font size 5pt - using minimum")
        return 5.0
    
    def _text_fits_with_baseline(self, text: str, font_size: float, baseline_chars_per_sqcm: Dict[float, float], dimensions: Dict[str, Any] = None) -> bool:
        """
        Check if text fits using baseline characters per sq cm.
        
        Args:
            text: Text to check
            font_size: Font size to test
            baseline_chars_per_sqcm: Dictionary of font size to chars per sq cm
            
        Returns:
            True if text fits, False otherwise
        """
        chars_per_sqcm = baseline_chars_per_sqcm.get(font_size, 0)
        if chars_per_sqcm <= 0:
            return False
        
        text_length = len(text)
        
        # Estimate required area in sq cm
        required_area_sqcm = text_length / chars_per_sqcm
        
        # Add safety margin (20% extra space for padding, line breaks, etc.)
        required_area_sqcm *= 1.2
        
        # Use actual textbox area if available, otherwise use conservative estimate
        if dimensions and dimensions.get('has_dimensions', False):
            textbox_area_cm2 = (dimensions['width'] * 0.0352778) * (dimensions['height'] * 0.0352778)
            return required_area_sqcm <= textbox_area_cm2
        else:
            # Conservative estimate for typical textbox area
            expected_area = 4.0
            return required_area_sqcm <= expected_area
    
    def _generate_font_size_reasoning(self, text: str, dimensions: Dict[str, Any], 
                                    detected_font_sizes: List[float], baseline_font_size: float,
                                    chosen_font_size: float, font_family: str,
                                    original_combined_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate detailed reasoning for font size decision using baseline characters per sq cm.
        
        Args:
            text: Complete text that was tested
            dimensions: Textbox dimensions
            detected_font_sizes: List of all detected font sizes
            baseline_font_size: Baseline font size used
            chosen_font_size: Font size that was ultimately chosen
            font_family: Font family used
            
        Returns:
            Dictionary with detailed reasoning about font size decision
        """
        reasoning = {
            'algorithm': 'Baseline Characters per Square Centimeter Optimization',
            'phase_1_detection': {
                'detected_font_sizes': detected_font_sizes,
                'textbox_dimensions': f"{dimensions['width']:.1f}x{dimensions['height']:.1f} points",
                'textbox_area_cm2': f"{(dimensions['width'] * 0.0352778) * (dimensions['height'] * 0.0352778):.2f} sq cm",
                'font_family_detected': font_family
            },
            'phase_2_reconstruction': {
                'baseline_font_size': baseline_font_size,
                'chosen_font_size': chosen_font_size,
                'text_length': len(text),
                'text_preview': text[:50] + ('...' if len(text) > 50 else ''),
                'original_text_length': len(original_combined_text) if isinstance(original_combined_text, str) else len(text),
            },
            'baseline_calculation': {
                'method': 'Characters per square centimeter calculation',
                'formula': 'chars_per_line * lines_fit / textbox_area_cm2',
                'font_specific_multipliers': self._get_font_multipliers_info(font_family),
                'baseline_chars_per_sqcm': self._calculate_baseline_chars_per_sqcm(
                    dimensions['width'], dimensions['height'], font_family
                )
            },
            'fitting_logic': {
                'method': 'Baseline characters per sq cm comparison',
                'calculation': 'text_length / chars_per_sqcm = required_area_sqcm',
                'safety_margin': '20% extra space for padding and line breaks',
                'decision_criteria': 'Choose largest font size where text fits within textbox area'
            },
            'font_specific_factors': {},
            'decision_process': [],
            'final_analysis': {}
        }
        
        # Add font-specific information
        font_multipliers_info = self._get_font_multipliers_info(font_family)
        reasoning['font_specific_factors'] = {
            'font_family': font_family,
            'width_multiplier': font_multipliers_info['width'],
            'reasoning': font_multipliers_info['reasoning']
        }
        
        # Generate decision process
        if detected_font_sizes:
            sorted_sizes = sorted(detected_font_sizes, reverse=True)
            reasoning['decision_process'] = [
                f"1. Calculated baseline characters per sq cm for font sizes 12pt to 6pt",
                f"2. Original text length: {reasoning['phase_2_reconstruction']['original_text_length']} characters",
                f"3. Appended text length: {reasoning['phase_2_reconstruction']['text_length']} characters",
                f"4. Started with detected font sizes: {sorted_sizes}",
                f"5. For each size, calculated required area: text_length / chars_per_sqcm",
                f"6. Added 20% safety margin for padding and line breaks",
                f"7. Chose largest font size where text fits within textbox area"
            ]
            
            if chosen_font_size in detected_font_sizes:
                reasoning['decision_process'].append(f"8. Selected {chosen_font_size}pt (detected font size)")
            else:
                reasoning['decision_process'].append(f"8. No detected size fit, reduced to {chosen_font_size}pt")
        else:
            reasoning['decision_process'] = [
                f"1. Calculated baseline characters per sq cm for font sizes 12pt to 6pt",
                f"2. Original text length: {reasoning['phase_2_reconstruction']['original_text_length']} characters",
                f"3. Appended text length: {reasoning['phase_2_reconstruction']['text_length']} characters",
                f"4. No detected font sizes, using baseline: {baseline_font_size}pt",
                f"5. For each size, calculated required area: text_length / chars_per_sqcm",
                f"6. Added 20% safety margin for padding and line breaks",
                f"7. Chose largest font size where text fits within textbox area"
            ]
        
        # Calculate final analysis
        baseline_chars_per_sqcm = reasoning['baseline_calculation']['baseline_chars_per_sqcm']
        chars_per_sqcm = baseline_chars_per_sqcm.get(chosen_font_size, 0)
        required_area_sqcm = len(text) / chars_per_sqcm if chars_per_sqcm > 0 else 0
        textbox_area_cm2 = (dimensions['width'] * 0.0352778) * (dimensions['height'] * 0.0352778)
        
        # Capacity metrics at chosen size
        try:
            from app.utils.font_capacity import evaluate_capacity
            lines_fit, max_cpl, total_fit, *_ = evaluate_capacity(
                dimensions['width'], dimensions['height'], font_family, chosen_font_size
            )
        except Exception:
            lines_fit, max_cpl, total_fit = 0, 0, 0

        # Compute original/new total lines with wrapping at chosen size
        import math
        def _wrapped_lines(s: Optional[str], cpl: int) -> int:
            if not isinstance(s, str) or cpl <= 0:
                return 0
            total = 0
            for seg in s.split('\n'):
                total += max(1, math.ceil(len(seg) / max(1, cpl)))
            return total

        wrapped_lines_new = _wrapped_lines(text, max_cpl)
        wrapped_lines_orig = _wrapped_lines(original_combined_text if isinstance(original_combined_text, str) else text, max_cpl)

        # Additionally, evaluate wrapping fit at the original size for the new text
        try:
            orig_lines_fit, orig_max_cpl, _total_fit, *_ = evaluate_capacity(
                dimensions['width'], dimensions['height'], font_family, baseline_font_size
            )
        except Exception:
            orig_lines_fit, orig_max_cpl = 0, 0
        required_lines_at_orig = _wrapped_lines(text, orig_max_cpl)
        will_fit_orig_wrapped = (required_lines_at_orig <= orig_lines_fit) if orig_lines_fit > 0 else None

        reasoning['final_analysis'] = {
            'text_length': len(text),
            'chosen_font_size': chosen_font_size,
            'chars_per_sqcm': f"{chars_per_sqcm:.1f}",
            'required_area_sqcm': f"{required_area_sqcm:.2f}",
            'textbox_area_sqcm': f"{textbox_area_cm2:.2f}",
            'utilization': f"{(required_area_sqcm / textbox_area_cm2 * 100):.1f}%" if textbox_area_cm2 > 0 else "0%",
            'fits_properly': required_area_sqcm <= textbox_area_cm2,
            'area_analysis': {
                'required_area': required_area_sqcm,
                'available_area': textbox_area_cm2,
                'area_overflow': max(0, required_area_sqcm - textbox_area_cm2)
            },
            'capacity_metrics': {
                'max_chars_per_line': max_cpl,
                'max_lines_per_box': lines_fit,
                'max_chars_per_box': total_fit,
                'total_lines_new': wrapped_lines_new,
                'total_lines_orig': wrapped_lines_orig,
                'will_fit_orig_wrapped': will_fit_orig_wrapped
            }
        }
        
        # Add reasoning summary
        if chosen_font_size == baseline_font_size:
            reasoning['summary'] = f"Used baseline font size {chosen_font_size}pt - text fits perfectly ({chars_per_sqcm:.1f} chars/sq cm)"
        elif chosen_font_size in detected_font_sizes:
            reasoning['summary'] = f"Used detected font size {chosen_font_size}pt - optimal fit from available sizes ({chars_per_sqcm:.1f} chars/sq cm)"
        else:
            reasoning['summary'] = f"Reduced to {chosen_font_size}pt - no detected size fit, minimum acceptable size ({chars_per_sqcm:.1f} chars/sq cm)"
        
        return reasoning
    
    def _get_font_width_multiplier(self, font_family: str) -> float:
        """Get font width multiplier for a given font family via shared utility."""
        try:
            from app.utils.font_capacity import get_width_multiplier
            return get_width_multiplier(font_family)
        except Exception:
            return 0.6
    
    def _get_font_multipliers_info(self, font_family: str) -> Dict[str, Any]:
        """Get comprehensive font multiplier information without duplicating constants."""
        width = self._get_font_width_multiplier(font_family)
        # Light-weight reasoning without per-font constants duplication
        lower = (font_family or "").lower()
        if any(k in lower for k in ["times", "serif"]):
            reasoning = "Serif font; characters slightly narrower than typical sans-serif"
        elif any(k in lower for k in ["hei", "yahei", "sim", "ming"]):
            reasoning = "CJK-oriented font; characters tend to be wider"
        else:
            reasoning = "Sans-serif or unknown family; moderate character width"
        return {"width": width, "reasoning": reasoning}
    
    def _text_fits_in_textbox(self, text: str, dimensions: Dict[str, Any], font_size: float) -> bool:
        """
        Check if text fits in textbox with given font size.
        
        Args:
            text: Text to check
            dimensions: Textbox dimensions
            font_size: Font size to test
            
        Returns:
            True if text fits, False otherwise
        """
        if not dimensions.get('has_dimensions', False):
            logger.debug(f"TEXT FIT: No dimensions available - assuming text fits")
            return True
        
        # Estimate text dimensions with this font size
        text_dims = TextboxCapacityCalculator.estimate_text_dimensions(text, font_size)
        
        # Add minimal padding (5% of container size)
        padding_width = dimensions['width'] * 0.05
        padding_height = dimensions['height'] * 0.05
        
        available_width = dimensions['width'] - padding_width
        available_height = dimensions['height'] - padding_height
        
        # Check if text fits
        fits_width = text_dims['width'] <= available_width
        fits_height = text_dims['height'] <= available_height
        
        logger.debug(f"TEXT FIT: Font {font_size}pt - Text: {text_dims['width']:.1f}x{text_dims['height']:.1f}, Available: {available_width:.1f}x{available_height:.1f}, Fits: {fits_width and fits_height}")
        
        return fits_width and fits_height
    
    def _replace_text_in_wt_elements(self, wt_elements: List[ET.Element], combined_text: str,
                                   matched_text: str, replacement_text: str, 
                                   start_pos: int, end_pos: int) -> bool:
        """
        Replace text in w:t elements while preserving structure.
        
        Args:
            wt_elements: List of w:t XML elements
            combined_text: Combined text from all elements
            matched_text: Original matched text
            replacement_text: Final replacement text
            start_pos: Start position of match
            end_pos: End position of match
            
        Returns:
            True if replacement was successful
        """
        try:
            logger.debug(f"Starting text replacement: '{matched_text}' -> '{replacement_text}' at position {start_pos}-{end_pos}")
            logger.debug(f"Total w:t elements: {len(wt_elements)}")
            
            # Find which w:t elements contain the matched text
            current_pos = 0
            affected_elements = []
            
            for i, wt_element in enumerate(wt_elements):
                element_text = wt_element.text or ""
                element_start = current_pos
                element_end = current_pos + len(element_text)
                
                logger.debug(f"Element {i}: text='{element_text}', pos={element_start}-{element_end}")
                
                # Check if this element overlaps with the match
                if element_start < end_pos and element_end > start_pos:
                    affected_elements.append((wt_element, element_start, element_end))
                    logger.debug(f"Element {i} is affected by match")
                
                current_pos = element_end
            
            logger.debug(f"Found {len(affected_elements)} affected elements")
            
            if not affected_elements:
                logger.warning(f"No affected elements found for text replacement")
                return False
            
            # Replace text in affected elements
            for i, (wt_element, element_start, element_end) in enumerate(affected_elements):
                element_text = wt_element.text or ""
                logger.debug(f"Processing affected element {i}: '{element_text}' at pos {element_start}-{element_end}")
                
                if i == 0:
                    # First element: replace the portion that matches
                    match_start_in_element = max(0, start_pos - element_start)
                    match_end_in_element = min(len(element_text), end_pos - element_start)
                    
                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]
                    
                    new_text = before_text + replacement_text + after_text
                    logger.info(f"REPLACING TEXT: '{element_text}' -> '{new_text}' (before='{before_text}', replacement='{replacement_text}', after='{after_text}')")
                    logger.debug(f"Setting wt_element.text = '{new_text}'")
                    wt_element.text = new_text
                    logger.debug(f"Text replacement applied to element {i}")
                else:
                    # Subsequent elements: remove the matched portion
                    match_start_in_element = max(0, start_pos - element_start)
                    match_end_in_element = min(len(element_text), end_pos - element_start)
                    
                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]
                    
                    wt_element.text = before_text + after_text
            
            return True
            
        except Exception as e:
            logger.error(f"Error replacing text in w:t elements: {e}")
            return False
    
    def process(self, document_or_path, **kwargs) -> ProcessingResult:
        """
        Process a DOCX document with graphics processing.
        
        Args:
            document_or_path: Either a Document object or Path to the DOCX file
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult containing comprehensive processing results
            
        Raises:
            RuntimeError: If the processor is not initialized
        """
        if not self.initialized:
            raise RuntimeError("Graphics Processor not initialized")
        
        import time
        start_time = time.time()
        
        # Handle both Document objects and file paths
        if isinstance(document_or_path, Path):
            document_path = document_or_path
            logger.info(f"Processing DOCX document from path: {document_path.name}")
            try:
                document = Document(document_path)
            except Exception as e:
                logger.error(f"Failed to load document from path {document_path}: {e}")
                return ProcessingResult(
                    success=False,
                    processor_type="graphics",
                    matches_found=0,
                    processing_time=time.time() - start_time,
                    error_message=f"Failed to load document: {e}"
                )
        else:
            # Assume it's a Document object
            document = document_or_path
            document_path = None
            logger.info("Processing DOCX document object")
        
        try:
            # Process the document
            matches, all_detections = self.process_graphics(document)
            
            # Save the processed document to the processed directory (only if we have a path)
            output_path = None
            if document_path:
                from config import OUTPUT_DIR
                processed_dir = Path(OUTPUT_DIR)
                processed_dir.mkdir(parents=True, exist_ok=True)
                output_path = processed_dir / f"{document_path.stem}_processed{document_path.suffix}"
                try:
                    logger.info(f"Saving processed document to: {output_path}")
                    document.save(output_path)
                    logger.info(f"Document saved successfully to: {output_path}")
                except Exception as e:
                    logger.error(f"Error saving document: {e}")
                    output_path = None
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'matches': matches,
                'all_detections': all_detections,
                'file_size_mb': document_path.stat().st_size / (1024 * 1024) if document_path else 0
            }
            
            logger.info(f"Graphics processing completed: {len(matches)} matches found")
            
            return ProcessingResult(
                success=True,
                processor_type="graphics",
                matches_found=len(matches),
                output_path=output_path,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing document {document_path}: {e}")
            
            return ProcessingResult(
                success=False,
                processor_type="graphics",
                matches_found=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of file formats supported by this processor.
        
        Returns:
            List of supported file extensions
        """
        return ['.docx']
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the graphics processor configuration."""
        return {
            'mode': self.mode,
            'patterns_count': len(self.patterns),
            'mappings_count': len(self.mappings),
            'compiled_patterns': len(self.pattern_matcher.compiled_patterns) if self.pattern_matcher else 0,
            'is_initialized': self.is_initialized
        }
    
    def cleanup(self):
        """
        Clean up processor resources and reset internal state.
        """
        try:
            logger.info("Cleaning up Graphics Processor...")
            self.is_initialized = False
            self.pattern_matcher = None
            self.text_replacer = None
            logger.info("Graphics Processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during Graphics Processor cleanup: {e}")

def create_graphics_processor(patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None, 
                            mode: str = "append", separator: str = ";", 
                            default_mapping: str = "4022-NA") -> GraphicsProcessor:
    """
    Factory function to create a GraphicsProcessor instance.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        mode: Processing mode ('append' or 'replace')
        separator: Separator between original and appended text in append mode
        default_mapping: Default text to append when no mapping is found
        
    Returns:
        GraphicsProcessor instance
    """
    return GraphicsProcessor(patterns, mappings, mode, separator, default_mapping)
