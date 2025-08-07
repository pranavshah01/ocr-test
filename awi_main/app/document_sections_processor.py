"""
Document Sections Processor for Task 3.2: Headers, Footers & Document Sections.
Extends processing to all document sections with formatting preservation.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from lxml import etree
import zipfile
import logging
import re
from datetime import datetime
from .formatting_preservation import FormattingPreserver
from .shared_constants import XML_NAMESPACES, SharedUtilities

logger = logging.getLogger(__name__)


class DocumentSectionsProcessor:
    """
    Enhanced processor for headers, footers, and all document sections.
    Implements Task 3.2: Process text in headers/footers, handle sections consistently, preserve formatting.
    """
    
    def __init__(self):
        self.namespaces = XML_NAMESPACES
        self.formatting_preserver = FormattingPreserver()
        self.processed_sections = 0
        self.header_replacements = 0
        self.footer_replacements = 0
        self.section_replacements = 0
    
    def extract_document_parts(self, docx_path: Path) -> Dict[str, Any]:
        """
        Extract all document parts including headers, footers, and main document.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Dictionary containing all document parts and their XML content
        """
        parts = {
            'main_document': None,
            'headers': {},
            'footers': {},
            'relationships': {},
            'errors': []
        }
        
        try:
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                # Extract main document
                try:
                    main_doc_xml = docx_zip.read('word/document.xml')
                    parts['main_document'] = etree.fromstring(main_doc_xml)
                    logger.debug("Extracted main document XML")
                except Exception as e:
                    parts['errors'].append(f"Error reading main document: {e}")
                
                # Extract relationships to find headers and footers
                try:
                    rels_xml = docx_zip.read('word/_rels/document.xml.rels')
                    rels_root = etree.fromstring(rels_xml)
                    
                    for rel in rels_root.findall('.//r:Relationship', namespaces=self.namespaces):
                        rel_id = rel.get('Id')
                        rel_type = rel.get('Type')
                        target = rel.get('Target')
                        
                        parts['relationships'][rel_id] = {
                            'type': rel_type,
                            'target': target
                        }
                        
                        # Extract headers
                        if 'header' in rel_type.lower():
                            try:
                                header_xml = docx_zip.read(f'word/{target}')
                                parts['headers'][rel_id] = {
                                    'xml': etree.fromstring(header_xml),
                                    'target': target,
                                    'type': 'header'
                                }
                                logger.debug(f"Extracted header: {target}")
                            except Exception as e:
                                parts['errors'].append(f"Error reading header {target}: {e}")
                        
                        # Extract footers
                        elif 'footer' in rel_type.lower():
                            try:
                                footer_xml = docx_zip.read(f'word/{target}')
                                parts['footers'][rel_id] = {
                                    'xml': etree.fromstring(footer_xml),
                                    'target': target,
                                    'type': 'footer'
                                }
                                logger.debug(f"Extracted footer: {target}")
                            except Exception as e:
                                parts['errors'].append(f"Error reading footer {target}: {e}")
                
                except Exception as e:
                    parts['errors'].append(f"Error reading relationships: {e}")
        
        except Exception as e:
            parts['errors'].append(f"Error opening DOCX file: {e}")
        
        return parts
    
    def process_section_text(self, section_xml: etree.Element, section_type: str, 
                           patterns: List[str], mapping_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Process text in a document section (header, footer, or main document).
        
        Args:
            section_xml: XML element for the section
            section_type: Type of section ('header', 'footer', 'main')
            patterns: List of regex patterns to match
            mapping_dict: Dictionary of text replacements
            
        Returns:
            Dictionary containing processing results
        """
        results = {
            'section_type': section_type,
            'replacements': 0,
            'processed_paragraphs': 0,
            'formatting_preserved': 0,
            'errors': []
        }
        
        try:
            # Find all paragraphs in the section
            paragraphs = section_xml.findall('.//w:p', namespaces=self.namespaces)
            
            for para in paragraphs:
                results['processed_paragraphs'] += 1
                
                # Extract text runs from the paragraph
                runs = para.findall('.//w:r', namespaces=self.namespaces)
                
                if not runs:
                    continue
                
                # Reconstruct text across runs for pattern matching
                full_text = ""
                text_fragments = []
                
                for run in runs:
                    text_elements = run.findall('.//w:t', namespaces=self.namespaces)
                    for text_elem in text_elements:
                        if text_elem.text:
                            start_pos = len(full_text)
                            full_text += text_elem.text
                            end_pos = len(full_text)
                            text_fragments.append({
                                'text': text_elem.text,
                                'element': text_elem,
                                'run': run,
                                'start_pos': start_pos,
                                'end_pos': end_pos
                            })
                
                if not full_text.strip():
                    continue
                
                # Apply pattern matching
                for pattern in patterns:
                    try:
                        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
                        
                        for match in matches:
                            matched_text = match.group().strip()
                            
                            # Look up replacement in mapping
                            replacement = None
                            for key, value in mapping_dict.items():
                                if key.strip().lower() == matched_text.lower():
                                    replacement = value
                                    break
                            
                            if replacement:
                                # Find which text fragments contain this match
                                match_start = match.start()
                                match_end = match.end()
                                
                                affected_fragments = []
                                for fragment in text_fragments:
                                    if (fragment['start_pos'] < match_end and 
                                        fragment['end_pos'] > match_start):
                                        affected_fragments.append(fragment)
                                
                                if affected_fragments:
                                    # Perform replacement with formatting preservation
                                    success = self.replace_text_with_formatting(
                                        affected_fragments, matched_text, replacement, 
                                        match_start, match_end
                                    )
                                    
                                    if success:
                                        results['replacements'] += 1
                                        results['formatting_preserved'] += 1
                                        logger.debug(f"Replaced '{matched_text}' with '{replacement}' in {section_type}")
                    
                    except re.error as e:
                        error_msg = f"Invalid regex pattern '{pattern}': {e}"
                        logger.warning(error_msg)
                        results['errors'].append(error_msg)
        
        except Exception as e:
            error_msg = f"Error processing {section_type} section: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def replace_text_with_formatting(self, fragments: List[Dict], old_text: str, 
                                   new_text: str, match_start: int, match_end: int) -> bool:
        """
        Replace text while preserving formatting across multiple fragments.
        
        Args:
            fragments: List of text fragments that contain the match
            old_text: Text to replace
            new_text: Replacement text
            match_start: Start position of match in full text
            match_end: End position of match in full text
            
        Returns:
            True if replacement was successful
        """
        try:
            if not fragments:
                return False
            
            # Find the primary fragment (contains most of the match)
            primary_fragment = None
            for fragment in fragments:
                fragment_match_start = max(0, match_start - fragment['start_pos'])
                fragment_match_end = min(len(fragment['text']), match_end - fragment['start_pos'])
                
                if fragment_match_end > fragment_match_start:
                    primary_fragment = fragment
                    break
            
            if not primary_fragment:
                return False
            
            # Extract formatting from the primary fragment's run
            run_formatting = self.formatting_preserver.extract_run_formatting(primary_fragment['run'])
            para_formatting = self.formatting_preserver.extract_paragraph_formatting(
                primary_fragment['run'].getparent()
            )
            
            # Calculate the replacement within the primary fragment
            fragment_start = match_start - primary_fragment['start_pos']
            fragment_end = match_end - primary_fragment['start_pos']
            
            # Ensure bounds are within the fragment text
            fragment_start = max(0, fragment_start)
            fragment_end = min(len(primary_fragment['text']), fragment_end)
            
            if fragment_end <= fragment_start:
                return False
            
            # Perform the replacement
            original_text = primary_fragment['text']
            before_match = original_text[:fragment_start]
            after_match = original_text[fragment_end:]
            new_fragment_text = before_match + new_text + after_match
            
            # Update the text element
            primary_fragment['element'].text = new_fragment_text
            
            # Preserve formatting by ensuring the run has proper formatting
            if run_formatting or para_formatting:
                self.formatting_preserver.apply_run_formatting(primary_fragment['run'], run_formatting)
            
            # Clear text from other fragments that were part of the match
            for fragment in fragments:
                if fragment != primary_fragment:
                    # Check if this fragment was entirely within the match
                    if (fragment['start_pos'] >= match_start and 
                        fragment['end_pos'] <= match_end):
                        fragment['element'].text = ""
            
            return True
            
        except Exception as e:
            logger.error(f"Error replacing text with formatting: {e}")
            return False
    
    def process_all_sections(self, docx_path: Path, patterns: List[str], 
                           mapping_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Process all document sections including headers, footers, and main document.
        
        Args:
            docx_path: Path to DOCX file
            patterns: List of regex patterns to match
            mapping_dict: Dictionary of text replacements
            
        Returns:
            Dictionary containing comprehensive processing results
        """
        logger.info(f"Processing all document sections in {docx_path}")
        
        results = {
            'total_replacements': 0,
            'header_results': {},
            'footer_results': {},
            'main_document_results': {},
            'processed_sections': 0,
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Extract all document parts
            parts = self.extract_document_parts(docx_path)
            results['errors'].extend(parts['errors'])
            
            # Process main document
            if parts['main_document'] is not None:
                main_results = self.process_section_text(
                    parts['main_document'], 'main', patterns, mapping_dict
                )
                results['main_document_results'] = main_results
                results['total_replacements'] += main_results['replacements']
                results['processed_sections'] += 1
                self.section_replacements += main_results['replacements']
            
            # Process headers
            for header_id, header_data in parts['headers'].items():
                header_results = self.process_section_text(
                    header_data['xml'], 'header', patterns, mapping_dict
                )
                results['header_results'][header_id] = header_results
                results['total_replacements'] += header_results['replacements']
                results['processed_sections'] += 1
                self.header_replacements += header_results['replacements']
            
            # Process footers
            for footer_id, footer_data in parts['footers'].items():
                footer_results = self.process_section_text(
                    footer_data['xml'], 'footer', patterns, mapping_dict
                )
                results['footer_results'][footer_id] = footer_results
                results['total_replacements'] += footer_results['replacements']
                results['processed_sections'] += 1
                self.footer_replacements += footer_results['replacements']
            
            self.processed_sections = results['processed_sections']
            
            logger.info(f"Document sections processing completed: {results['total_replacements']} total replacements "
                       f"across {results['processed_sections']} sections")
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing document sections: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def write_modified_sections(self, docx_path: Path, output_path: Path, 
                              modified_parts: Dict[str, Any]) -> bool:
        """
        Write modified document sections back to a new DOCX file.
        
        Args:
            docx_path: Original DOCX file path
            output_path: Output DOCX file path
            modified_parts: Dictionary containing modified XML parts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Copy original DOCX to output location
            import shutil
            shutil.copy2(docx_path, output_path)
            
            # Update the copied DOCX with modified sections
            with zipfile.ZipFile(output_path, 'a') as docx_zip:
                # Update main document if modified
                if modified_parts.get('main_document') is not None:
                    main_xml = etree.tostring(modified_parts['main_document'], 
                                            encoding='utf-8', xml_declaration=True)
                    docx_zip.writestr('word/document.xml', main_xml)
                
                # Update headers if modified
                for header_id, header_data in modified_parts.get('headers', {}).items():
                    if header_data.get('xml') is not None:
                        header_xml = etree.tostring(header_data['xml'], 
                                                  encoding='utf-8', xml_declaration=True)
                        docx_zip.writestr(f"word/{header_data['target']}", header_xml)
                
                # Update footers if modified
                for footer_id, footer_data in modified_parts.get('footers', {}).items():
                    if footer_data.get('xml') is not None:
                        footer_xml = etree.tostring(footer_data['xml'], 
                                                  encoding='utf-8', xml_declaration=True)
                        docx_zip.writestr(f"word/{footer_data['target']}", footer_xml)
            
            logger.info(f"Modified document sections written to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing modified sections: {e}")
            return False


def process_document_sections(docx_path: Path, patterns: List[str], 
                            mapping_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Process all document sections with enhanced header/footer support.
    Implements Task 3.2: Headers, footers, and document sections processing.
    
    Args:
        docx_path: Path to DOCX file
        patterns: List of regex patterns to match
        mapping_dict: Dictionary of text replacements
        
    Returns:
        Dictionary containing comprehensive processing results
    """
    processor = DocumentSectionsProcessor()
    return processor.process_all_sections(docx_path, patterns, mapping_dict)
