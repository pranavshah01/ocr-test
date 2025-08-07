from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import re
from .text_extractor import EnhancedPatternMatcher, find_patterns_across_split_tags
from .ocr_engine import EnhancedOCREngine
from .comprehensive_image_detector import ComprehensiveImageDetector
from .ocr_text_replacement import OCRTextReplacementProcessor
from .enhanced_textbox_ocr_processor import EnhancedTextboxOCRProcessor
from .formatting_preservation import FormattingPreserver
from .document_sections_processor import DocumentSectionsProcessor
from .shared_constants import SharedUtilities
from .report import EnhancedReportGenerator
import shutil
from datetime import datetime
import zipfile
from lxml import etree
import time


class EnhancedMappingProcessor:
    """
    Enhanced mapping and replacement processor that integrates with the pattern matching from Task 2.1.
    Handles exact pattern matching, mapping lookup, and text replacement with formatting preservation.
    """
    
    def __init__(self, text_mode: str = "replace"):
        self.text_mode = text_mode
        self.pattern_matcher = EnhancedPatternMatcher()
        self.formatting_preserver = FormattingPreserver()
        self.replacements_made = 0
        self.mapping_hits = 0
        self.mapping_misses = 0
        self.formatting_preserved = 0
        self.processing_log = []
    
    def load_patterns_from_json(self, patterns_path: Path) -> List[str]:
        """
        Load patterns from patterns.json file.
        
        Args:
            patterns_path: Path to patterns.json file
            
        Returns:
            List of regex pattern strings
        """
        try:
            with open(patterns_path, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            # Handle both old array format and new object format
            if isinstance(patterns_data, list):
                return patterns_data
            elif isinstance(patterns_data, dict):
                # Extract pattern values, excluding metadata
                patterns = []
                for key, value in patterns_data.items():
                    if not key.startswith('_') and isinstance(value, str):
                        patterns.append(value)
                return patterns
            else:
                raise ValueError("Invalid patterns.json format")
                
        except Exception as e:
            self.processing_log.append(f"Error loading patterns: {e}")
            return []
    
    def find_mapping_match(self, matched_text: str, mapping_dict: Dict[str, str]) -> Optional[str]:
        """
        Enhanced mapping lookup with multiple normalization strategies.
        
        Args:
            matched_text: Text that was matched by pattern
            mapping_dict: Dictionary of old -> new text mappings
            
        Returns:
            Replacement text if found, None otherwise
        """
        # Clean the matched text first
        cleaned_text = matched_text.strip()
        
        # Strategy 1: Direct exact match
        if cleaned_text in mapping_dict:
            self.mapping_hits += 1
            self.processing_log.append(f"Found direct mapping: '{cleaned_text}' -> '{mapping_dict[cleaned_text]}'")
            return mapping_dict[cleaned_text]
        
        # Strategy 2: Normalized whitespace and separators
        normalized_variants = self._normalize_text_for_mapping(cleaned_text)
        for variant in normalized_variants:
            if variant in mapping_dict:
                self.mapping_hits += 1
                self.processing_log.append(f"Found mapping via normalization: '{cleaned_text}' -> '{variant}'")
                return mapping_dict[variant]
        
        # Strategy 3: Case-insensitive matching
        for key, value in mapping_dict.items():
            if key.lower() == cleaned_text.lower():
                self.mapping_hits += 1
                self.processing_log.append(f"Found mapping via case-insensitive: '{cleaned_text}' -> '{key}'")
                return value
        
        # Strategy 4: Alphanumeric-only matching (for OCR variations)
        matched_clean = re.sub(r'[^A-Z0-9]', '', cleaned_text.upper())
        for key, value in mapping_dict.items():
            key_clean = re.sub(r'[^A-Z0-9]', '', key.upper())
            if key_clean == matched_clean and len(key_clean) > 0:
                self.mapping_hits += 1
                self.processing_log.append(f"Found mapping via alphanumeric: '{cleaned_text}' -> '{key}'")
                return value
        
        # Strategy 5: Fuzzy matching for close matches
        for key, value in mapping_dict.items():
            if abs(len(key) - len(cleaned_text)) <= 2:  # Similar length
                # Check character similarity
                common_chars = sum(1 for a, b in zip(key.upper(), cleaned_text.upper()) if a == b)
                similarity = common_chars / max(len(key), len(cleaned_text))
                if similarity >= 0.9:  # 90% similarity
                    self.mapping_hits += 1
                    self.processing_log.append(f"Found mapping via fuzzy match: '{cleaned_text}' -> '{key}' (similarity: {similarity:.2f})")
                    return value
        
        # No mapping found
        self.mapping_misses += 1
        self.processing_log.append(f"No mapping found for: '{cleaned_text}' (length: {len(cleaned_text)})")
        return None
    
    def _normalize_text_for_mapping(self, text: str) -> List[str]:
        """
        Generate normalized variants of text for mapping lookup.
        
        Args:
            text: Input text to normalize
            
        Returns:
            List of normalized text variants
        """
        variants = []
        
        # Variant 1: Remove all whitespace, keep separators
        variant1 = re.sub(r'\s+', '', text)
        variants.append(variant1)
        
        # Variant 2: Normalize separators (spaces/hyphens)
        variant2 = re.sub(r'\s*-\s*', '-', text)  # Normalize around hyphens
        variant2 = re.sub(r'\s+', '', variant2)   # Remove remaining spaces
        variants.append(variant2)
        
        # Variant 3: Strip leading/trailing whitespace only
        variant3 = text.strip()
        variants.append(variant3)
        
        # Variant 4: Replace multiple spaces with single space
        variant4 = re.sub(r'\s+', ' ', text.strip())
        variants.append(variant4)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen and variant != text:  # Don't include original
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def apply_xml_replacement(self, docx_path: Path, pattern_matches: List[Dict], 
                            mapping_dict: Dict[str, str]) -> Tuple[Path, Dict[str, Any]]:
        """
        Apply replacements directly to DOCX XML for patterns that span multiple tags.
        
        Args:
            docx_path: Path to DOCX file
            pattern_matches: List of pattern matches from enhanced pattern matcher
            mapping_dict: Dictionary of old -> new text mappings
            
        Returns:
            Tuple of (output_path, processing_summary)
        """
        # Prepare output path
        processed_dir = Path("./processed")
        processed_dir.mkdir(exist_ok=True)
        output_path = processed_dir / f"{docx_path.stem}_processed.docx"
        
        # Copy original file to output location
        shutil.copy2(docx_path, output_path)
        
        replacements_made = 0
        processing_summary = {
            "file_path": str(docx_path),
            "output_path": str(output_path),
            "processing_time": datetime.now().isoformat(),
            "text_mode": self.text_mode,
            "total_matches": len(pattern_matches),
            "replacements_made": 0,
            "mapping_hits": 0,
            "mapping_misses": 0,
            "processing_log": [],
            "errors": []
        }
        
        try:
            # Process XML with formatting preservation
            with zipfile.ZipFile(output_path, 'r') as docx_zip:
                # Read document XML
                doc_xml = docx_zip.read('word/document.xml')
                root = etree.fromstring(doc_xml)
                
                # Sort matches by position (reverse order for safe replacement)
                sorted_matches = sorted(pattern_matches, key=lambda x: x['start_pos'], reverse=True)
                
                for match_data in sorted_matches:
                    matched_text = match_data['match_text']
                    print(f"🔍 DEBUG: Processing match: '{matched_text}'")
                    replacement_text = self.find_mapping_match(matched_text, mapping_dict)
                    print(f"🔍 DEBUG: Mapping result: '{replacement_text}'")
                    
                    if replacement_text:
                        # Apply simple direct replacement for now
                        try:
                            # Find all text elements containing the matched text
                            for text_elem in root.xpath('.//w:t', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                                if text_elem.text and matched_text in text_elem.text:
                                    # Replace the text directly
                                    if self.text_mode == "replace":
                                        text_elem.text = text_elem.text.replace(matched_text, replacement_text)
                                    else:  # append mode
                                        text_elem.text = text_elem.text.replace(matched_text, f"{matched_text} {replacement_text}")
                                    
                                    replacements_made += 1
                                    self.processing_log.append(
                                        f"Applied direct replacement: '{matched_text}' -> '{replacement_text}' (mode: {self.text_mode})"
                                    )
                                    print(f"✅ SUCCESS: Replaced '{matched_text}' with '{replacement_text}'")
                                    break
                        except Exception as e:
                            self.processing_log.append(
                                f"Failed to apply direct replacement for: '{matched_text}': {e}"
                            )
                    else:
                        # No mapping found - log for debugging
                        self.processing_log.append(
                            f"No mapping found for pattern match: '{matched_text}'"
                        )
                
                # Write back the modified XML with proper ZIP handling
                if replacements_made > 0:
                    self._write_modified_docx(output_path, root)
                
        except Exception as e:
            processing_summary["errors"].append(f"XML processing error: {e}")
        
        # Update summary with formatting preservation info
        formatting_summary = self.formatting_preserver.get_processing_summary()
        processing_summary.update({
            "replacements_made": replacements_made,
            "formatting_preserved": self.formatting_preserved,
            "mapping_hits": self.mapping_hits,
            "mapping_misses": self.mapping_misses,
            "processing_log": self.processing_log,
            "formatting_summary": formatting_summary
        })
        
        return output_path, processing_summary
    
    def _write_modified_docx(self, output_path: Path, root: etree.Element):
        """
        Write the modified XML back to the DOCX file.
        
        Args:
            output_path: Path to the output DOCX file
            root: Modified document root element
        """
        try:
            import tempfile
            
            # Create a temporary file for the new DOCX
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            # Copy the original DOCX to temp location
            shutil.copy2(output_path, temp_path)
            
            # Update the document.xml in the temp DOCX
            with zipfile.ZipFile(temp_path, 'a') as temp_zip:
                # Remove the old document.xml
                # Note: Python's zipfile doesn't support direct deletion,
                # so we create a new ZIP with all files except document.xml
                pass
            
            # Create new DOCX with updated document.xml
            with zipfile.ZipFile(output_path, 'r') as source_zip:
                with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as target_zip:
                    # Copy all files except document.xml
                    for item in source_zip.infolist():
                        if item.filename != 'word/document.xml':
                            data = source_zip.read(item.filename)
                            target_zip.writestr(item, data)
                    
                    # Write the modified document.xml
                    modified_xml = etree.tostring(root, encoding='utf-8', xml_declaration=True)
                    target_zip.writestr('word/document.xml', modified_xml)
            
            # Replace the original with the modified version
            shutil.move(temp_path, output_path)
            
            self.processing_log.append(f"Successfully wrote modified DOCX to {output_path}")
            
        except Exception as e:
            self.processing_log.append(f"Error writing modified DOCX: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
    
    def process_with_enhanced_patterns(self, docx_path: Path, patterns_path: Path, 
                                     mapping_path: Path) -> Dict[str, Any]:
        """
        Main processing function that integrates enhanced pattern matching with mapping and replacement.
        
        Args:
            docx_path: Path to DOCX file to process
            patterns_path: Path to patterns.json file
            mapping_path: Path to mapping.json file
            
        Returns:
            Processing summary dictionary
        """
        self.processing_log.append(f"Starting enhanced processing of {docx_path}")
        
        # Load patterns and mapping
        patterns = self.load_patterns_from_json(patterns_path)
        if not patterns:
            return {
                "error": "No patterns loaded",
                "processing_log": self.processing_log
            }
        
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_dict = json.load(f)
        except Exception as e:
            return {
                "error": f"Failed to load mapping: {e}",
                "processing_log": self.processing_log
            }
        
        # Use enhanced pattern matching from Task 2.1
        pattern_results = find_patterns_across_split_tags(docx_path, patterns)
        
        if pattern_results['total_matches'] == 0:
            self.processing_log.append("No pattern matches found")
            return {
                "message": "No patterns matched",
                "pattern_results": pattern_results,
                "processing_log": self.processing_log
            }
        
        self.processing_log.append(f"Found {pattern_results['total_matches']} pattern matches")
        
        # Apply replacements
        output_path, processing_summary = self.apply_xml_replacement(
            docx_path, pattern_results['matches'], mapping_dict
        )
        
        # Combine results
        final_summary = {
            **processing_summary,
            "pattern_results": pattern_results,
            "patterns_used": patterns,
            "mapping_entries": len(mapping_dict)
        }
        
        return final_summary


def process_docx(
    docx_path: Path,
    mapping_dict: Dict[str, str],
    regex_patterns: list,
    text_mode: str = "replace",
    ocr_mode: str = "replace",
    ocr_engine: str = "easyocr",
    confidence_min: float = 0.4,
    gpu: bool = True,
    process_images: bool = True
) -> Dict[str, Any]:
    """
    Process DOCX files using enhanced pattern matching and OCR-based image processing.
    Supports both text replacement and image text replacement with three OCR modes.
    Now includes comprehensive enhanced logging and reporting capabilities.

    Args:
        docx_path (Path): Path to DOCX file.
        mapping_dict (Dict[str, str]): Mapping of old -> new text.
        regex_patterns (list): List of regex pattern strings.
        text_mode (str): Text processing mode - "replace" or "append".
        ocr_mode (str): OCR processing mode - "replace", "append", or "append-image".
        ocr_engine (str): OCR engine to use - "easyocr" or "tesseract".
        confidence_min (float): Minimum OCR confidence threshold.
        gpu (bool): Enable GPU for OCR processing.
        process_images (bool): Whether to process images with OCR.

    Returns:
        Dict[str, Any]: Processing summary with results and statistics.
    """
    # Initialize enhanced logging system
    log_file = Path("./logs") / f"{docx_path.stem}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = SharedUtilities.setup_detailed_logger(
        name=f"docx_processor_{docx_path.stem}",
        log_file=str(log_file),
        level=20  # INFO level
    )
    
    # Initialize enhanced report generator
    report_generator = EnhancedReportGenerator(output_dir=Path("./reports"))
    
    # Start processing timer
    start_time = time.time()
    
    logger.info(f"Starting DOCX processing: {docx_path}")
    SharedUtilities.log_processing_phase(
        logger, "Initialization", files=1, matches=0, time=0.0
    )
    
    print(f"Processing {docx_path} with enhanced pattern matching and OCR (mode: {ocr_mode})...")
    
    # Initialize processors
    text_processor = EnhancedMappingProcessor(text_mode=text_mode)
    ocr_processor = None
    
    if process_images:
        try:
            # Initialize enhanced OCR components
            ocr_engine_instance = EnhancedOCREngine(
                primary_engine=ocr_engine,
                confidence_threshold=confidence_min,
                gpu_enabled=gpu
            )
            image_detector = ComprehensiveImageDetector()
            ocr_processor = OCRTextReplacementProcessor(
                ocr_engine=ocr_engine_instance,
                image_detector=image_detector,
                confidence_threshold=confidence_min
            )
            print(f"Enhanced OCR processor initialized: {ocr_engine} engine, {ocr_mode} mode")
        except Exception as e:
            print(f"⚠️  OCR processor initialization failed: {e}")
            print("Continuing with text-only processing...")
    
    # Combined processing results with enhanced logging structure
    combined_results = {
        "body_replacements": 0,
        "image_replacements": 0,
        "images_processed": 0,
        "font_adjustments": 0,
        "errors": [],
        "warnings": [],
        "text_processing": {},
        "ocr_processing": {},
        "processing_log": [],
        "output_path": "",
        "processing_time": datetime.now().isoformat(),
        "matches_detail": [],  # Enhanced logging for detailed match tracking
        "file_size_mb": docx_path.stat().st_size / (1024 * 1024),
        "success": False
    }
    
    try:
        # Phase 1: Text Processing with Enhanced Pattern Matching
        phase1_start = time.time()
        logger.info("Phase 1: Starting text processing with enhanced pattern matching")
        print("Phase 1: Processing text with enhanced pattern matching...")
        
        pattern_results = find_patterns_across_split_tags(docx_path, regex_patterns)
        
        if pattern_results['total_matches'] > 0:
            logger.info(f"Found {pattern_results['total_matches']} text pattern matches")
            print(f"Found {pattern_results['total_matches']} text pattern matches")
            
            # Apply XML-level text replacements
            output_path, text_summary = text_processor.apply_xml_replacement(
                docx_path, pattern_results['matches'], mapping_dict
            )
            
            # Log detailed match information
            for match in pattern_results['matches']:
                original_text = match.get('matched_text', '').strip()
                
                # Skip empty matches
                if not original_text:
                    continue
                    
                replacement_text = mapping_dict.get(original_text, original_text)
                location = f"paragraph_{match.get('paragraph_index', 'unknown')}"
                
                # Only log if there's an actual replacement
                if replacement_text != original_text:
                    # Log the match/replacement
                    SharedUtilities.log_match_replacement(
                        logger, original_text, replacement_text, text_mode, location
                    )
                    
                    # Add to detailed matches for reporting
                    combined_results["matches_detail"].append({
                        'original_text': original_text,
                        'replacement_text': replacement_text,
                        'mode': text_mode,
                        'location': location,
                        'phase': 'text_processing',
                        'confidence_score': None  # Not applicable for text processing
                    })
            
            # Log font information if available
            formatting_summary = text_summary.get("formatting_summary", {})
            for font_info in formatting_summary.get('fonts_used', []):
                SharedUtilities.log_font_info(
                    logger, 
                    family=font_info.get('family'),
                    size=font_info.get('size'),
                    color=font_info.get('color'),
                    bold=font_info.get('bold', False),
                    italic=font_info.get('italic', False)
                )
            
            combined_results["body_replacements"] = text_summary["replacements_made"]
            combined_results["font_adjustments"] = text_summary.get("formatting_preserved", 0)
            combined_results["text_processing"] = {
                "pattern_results": pattern_results,
                "mapping_hits": text_processor.mapping_hits,
                "mapping_misses": text_processor.mapping_misses,
                "formatting_preserved": text_summary.get("formatting_preserved", 0),
                "formatting_summary": text_summary.get("formatting_summary", {}),
                "processing_log": text_processor.processing_log
            }
            combined_results["output_path"] = str(output_path)
            combined_results["processing_log"].extend(text_processor.processing_log)
            
            phase1_time = time.time() - phase1_start
            SharedUtilities.log_processing_phase(
                logger, "Phase 1 - Text Processing", files=1, 
                matches=pattern_results['total_matches'], time=phase1_time
            )
            
            logger.info(f"Text processing complete: {combined_results['body_replacements']} replacements made")
            print(f"Text processing complete: {combined_results['body_replacements']} replacements made")
        else:
            print("No text pattern matches found")
            combined_results["warnings"].append("No text patterns matched in document")
            # Still create output path for potential OCR processing
            processed_dir = Path("./processed")
            processed_dir.mkdir(exist_ok=True)
            output_path = processed_dir / f"{docx_path.stem}_processed.docx"
            shutil.copy2(docx_path, output_path)
            combined_results["output_path"] = str(output_path)
        
        # Phase 2: OCR-based Image Processing
        if process_images and ocr_processor:
            phase2_start = time.time()
            logger.info(f"Phase 2: Starting OCR image processing ({ocr_mode} mode)")
            print(f"Phase 2: Processing images with OCR ({ocr_mode} mode)...")
            
            # Use the output from text processing as input for OCR processing
            input_for_ocr = Path(combined_results["output_path"]) if combined_results["output_path"] else docx_path
            
            ocr_summary = ocr_processor.process_docx_images(
                input_for_ocr, regex_patterns, mapping_dict
            )
            
            # Log detailed OCR match information
            if "processing_log" in ocr_summary:
                for ocr_match in ocr_summary.get("ocr_matches", []):
                    original_text = ocr_match.get('original_text', '')
                    replacement_text = ocr_match.get('replacement_text', '')
                    confidence = ocr_match.get('confidence', 0.0)
                    location = f"image_{ocr_match.get('image_index', 'unknown')}"
                    
                    # Log the OCR match/replacement
                    SharedUtilities.log_match_replacement(
                        logger, original_text, replacement_text, ocr_mode, location, confidence
                    )
                    
                    # Log bounding box information if available
                    bbox = ocr_match.get('bounding_box', {})
                    if bbox:
                        SharedUtilities.log_bounding_box(
                            logger, 
                            x=bbox.get('x', 0), y=bbox.get('y', 0),
                            width=bbox.get('width', 0), height=bbox.get('height', 0),
                            rotation=bbox.get('rotation', 0.0), confidence=confidence
                        )
                    
                    # Add to detailed matches for reporting
                    combined_results["matches_detail"].append({
                        'original_text': original_text,
                        'replacement_text': replacement_text,
                        'mode': ocr_mode,
                        'location': location,
                        'phase': 'ocr_processing',
                        'confidence_score': confidence,
                        'bounding_box': bbox
                    })
            
            combined_results["image_replacements"] = ocr_summary["replacements_made"]
            combined_results["images_processed"] = ocr_summary["images_processed"]
            combined_results["ocr_replacements"] = ocr_summary["replacements_made"]  # For reporting consistency
            combined_results["ocr_processing"] = ocr_summary
            # Safely extend processing log from OCR summary
            if "processing_log" in ocr_summary:
                combined_results["processing_log"].extend(ocr_summary["processing_log"])
            
            # Update output path if OCR processing created a new file
            if "output_path" in ocr_summary:
                combined_results["output_path"] = ocr_summary["output_path"]
            
            phase2_time = time.time() - phase2_start
            SharedUtilities.log_processing_phase(
                logger, "Phase 2 - OCR Processing", files=1,
                matches=ocr_summary.get("replacements_made", 0), time=phase2_time
            )
            
            logger.info(f"OCR processing complete: {combined_results['image_replacements']} replacements in {combined_results['images_processed']} images")
            print(f"OCR processing complete: {combined_results['image_replacements']} replacements in {combined_results['images_processed']} images")
        else:
            print("Skipping OCR processing (disabled or unavailable)")
            combined_results["ocr_processing"] = {"message": "OCR processing skipped"}
        
        # Phase 3: Advanced Document Elements Processing
        print("Phase 3: Processing advanced document elements...")
        
        # Task 3.1: Textboxes & Callout Boxes Enhancement
        textbox_processor = EnhancedTextboxOCRProcessor()
        textbox_replacements, textbox_results = textbox_processor.process_all_textboxes(docx_path, mapping_dict, regex_patterns)
        
        # Task 3.2: Headers, Footers & Document Sections
        sections_processor = DocumentSectionsProcessor()
        sections_results = sections_processor.process_all_sections(docx_path, regex_patterns, mapping_dict)
        
        # Update processing summary with Phase 3 results
        combined_results["phase3_enabled"] = True
        combined_results["textbox_replacements"] = textbox_replacements
        combined_results["textbox_font_adjustments"] = textbox_results.get("font_adjustments", 0)
        combined_results["textbox_shape_resizes"] = textbox_results.get("shape_resizes", 0)
        combined_results["textbox_overflow_detections"] = textbox_results.get("overflow_detections", 0)
        combined_results["sections_total_replacements"] = sections_results.get("total_replacements", 0)
        combined_results["sections_processed"] = sections_results.get("processed_sections", 0)
        combined_results["header_replacements"] = sum(r.get("replacements", 0) for r in sections_results.get("header_results", {}).values())
        combined_results["footer_replacements"] = sum(r.get("replacements", 0) for r in sections_results.get("footer_results", {}).values())
        
        print(f"Phase 3 processing complete: {textbox_replacements} textbox replacements, {sections_results.get('total_replacements', 0)} section replacements")
        
        # Final summary with enhanced logging and reporting
        total_processing_time = time.time() - start_time
        total_replacements = combined_results["body_replacements"] + combined_results["image_replacements"] + combined_results["textbox_replacements"] + combined_results["sections_total_replacements"]
        
        # Mark processing as successful
        combined_results["success"] = True
        combined_results["total_replacements"] = total_replacements
        combined_results["processing_time"] = total_processing_time
        
        # Create detailed processing log for enhanced reporting
        detailed_log = SharedUtilities.create_detailed_processing_log(
            file_path=str(docx_path),
            matches=combined_results["matches_detail"],
            statistics={
                'file_size_mb': combined_results["file_size_mb"],
                'total_matches': len(combined_results["matches_detail"]),
                'text_replacements': combined_results["body_replacements"],
                'ocr_replacements': combined_results.get("ocr_replacements", 0),
                'textbox_replacements': combined_results["textbox_replacements"],
                'section_replacements': combined_results["sections_total_replacements"],
                'processing_time': total_processing_time,
                'formatting_preserved': combined_results["font_adjustments"],
                'errors': combined_results["errors"],
                'warnings': combined_results["warnings"],
                'success': True
            }
        )
        
        # Add to report generator for comprehensive reporting
        report_generator.add_processing_log(detailed_log)
        
        # Generate comprehensive reports
        try:
            report_paths = {
                'json': report_generator.export_to_enhanced_json(),
                'html': report_generator.export_to_html(),
                'csv': report_generator.export_to_csv()
            }
            logger.info(f"Enhanced reports generated: {list(report_paths.keys())}")
            combined_results["report_paths"] = {k: str(v) for k, v in report_paths.items()}
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            combined_results["warnings"].append(f"Report generation failed: {e}")
        
        # Final logging
        SharedUtilities.log_processing_phase(
            logger, "Complete Processing", files=1, 
            matches=total_replacements, time=total_processing_time
        )
        
        logger.info(f"Processing completed successfully in {total_processing_time:.2f}s")
        logger.info(f"Total replacements: {total_replacements} across all phases")
        logger.info(f"Output file: {combined_results['output_path']}")
        
        print(f"\n✅ Processing complete:")
        print(f"   Text replacements: {combined_results['body_replacements']}")
        print(f"   Image replacements: {combined_results['image_replacements']}")
        print(f"   Textbox replacements: {combined_results['textbox_replacements']}")
        print(f"   Section replacements: {combined_results['sections_total_replacements']}")
        print(f"   Images processed: {combined_results['images_processed']}")
        print(f"   Total replacements: {total_replacements}")
        print(f"   Processing time: {total_processing_time:.2f}s")
        print(f"   Output: {combined_results['output_path']}")
        if "report_paths" in combined_results:
            print(f"   Reports: {', '.join(combined_results['report_paths'].keys())}")
        
        return combined_results
        
    except Exception as e:
        total_processing_time = time.time() - start_time
        error_msg = f"Processing failed: {e}"
        
        # Enhanced error logging
        logger.error(error_msg)
        logger.error(f"Processing failed after {total_processing_time:.2f}s")
        
        print(f"❌ {error_msg}")
        combined_results["errors"].append(error_msg)
        combined_results["processing_log"].append(error_msg)
        combined_results["success"] = False
        combined_results["processing_time"] = total_processing_time
        
        # Create detailed error log for reporting
        try:
            detailed_log = SharedUtilities.create_detailed_processing_log(
                file_path=str(docx_path),
                matches=combined_results["matches_detail"],
                statistics={
                    'file_size_mb': combined_results["file_size_mb"],
                    'total_matches': len(combined_results["matches_detail"]),
                    'text_replacements': combined_results["body_replacements"],
                    'ocr_replacements': combined_results.get("ocr_replacements", 0),
                    'textbox_replacements': combined_results.get("textbox_replacements", 0),
                    'section_replacements': combined_results.get("sections_total_replacements", 0),
                    'processing_time': total_processing_time,
                    'formatting_preserved': combined_results["font_adjustments"],
                    'errors': combined_results["errors"],
                    'warnings': combined_results["warnings"],
                    'success': False
                }
            )
            
            # Add to report generator even for failed processing
            report_generator.add_processing_log(detailed_log)
            
            # Generate error reports
            try:
                report_paths = {
                    'json': report_generator.export_to_enhanced_json(),
                    'html': report_generator.export_to_html(),
                    'csv': report_generator.export_to_csv()
                }
                combined_results["report_paths"] = {k: str(v) for k, v in report_paths.items()}
                logger.info(f"Error reports generated: {list(report_paths.keys())}")
            except Exception as report_error:
                logger.warning(f"Error report generation failed: {report_error}")
        except Exception as log_error:
            logger.warning(f"Error logging failed: {log_error}")
        return combined_results