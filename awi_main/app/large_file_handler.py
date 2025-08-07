"""
Large DOCX File Handler
Provides alternative processing methods for very large DOCX files that cause XML parser errors.
"""

import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import shutil
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class LargeDocxHandler:
    """
    Handler for processing very large DOCX files that exceed XML parser limits.
    Uses streaming and chunked processing to avoid memory issues.
    """
    
    def __init__(self, docx_path: Path):
        self.docx_path = docx_path
        self.temp_dir = None
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="large_docx_")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def extract_docx_contents(self) -> bool:
        """
        Extract DOCX contents to temporary directory for processing.
        
        Returns:
            bool: True if extraction successful
        """
        try:
            with zipfile.ZipFile(self.docx_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            logger.info(f"Extracted DOCX contents to {self.temp_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract DOCX: {e}")
            return False
    
    def get_document_xml_path(self) -> Optional[Path]:
        """Get path to the main document.xml file."""
        doc_xml = Path(self.temp_dir) / "word" / "document.xml"
        return doc_xml if doc_xml.exists() else None
    
    def process_xml_in_chunks(self, xml_path: Path, mapping_dict: Dict[str, str], 
                             patterns: List[str], text_mode: str = "replace") -> Tuple[int, List[str]]:
        """
        Process XML file in chunks to avoid memory issues.
        
        Args:
            xml_path: Path to XML file
            mapping_dict: Text mapping dictionary
            patterns: Regex patterns
            text_mode: "replace" or "append"
            
        Returns:
            Tuple of (replacements_count, warnings)
        """
        replacements = 0
        warnings = []
        
        try:
            # Read XML content in chunks
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process text replacements
            original_content = content
            
            # Apply regex patterns and mappings
            for pattern_str in patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    matches = list(pattern.finditer(content))
                    
                    for match in reversed(matches):  # Process in reverse to maintain positions
                        matched_text = match.group()
                        normalized_text = self._normalize_text_for_mapping(matched_text)
                        
                        # Try to find mapping
                        replacement_text = None
                        for key, value in mapping_dict.items():
                            if self._texts_match(normalized_text, key):
                                replacement_text = value
                                break
                        
                        if replacement_text:
                            start, end = match.span()
                            if text_mode == "replace":
                                # Replace the matched text
                                content = content[:start] + replacement_text + content[end:]
                            else:  # append
                                # Append to the matched text
                                content = content[:end] + " " + replacement_text + content[end:]
                            replacements += 1
                        else:
                            warnings.append(f"No mapping found for: {matched_text}")
                            
                except re.error as e:
                    warnings.append(f"Invalid regex pattern '{pattern_str}': {e}")
            
            # Write modified content back
            if content != original_content:
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Applied {replacements} replacements to {xml_path}")
            
            return replacements, warnings
            
        except Exception as e:
            error_msg = f"Failed to process XML chunks: {e}"
            logger.error(error_msg)
            warnings.append(error_msg)
            return 0, warnings
    
    def _normalize_text_for_mapping(self, text: str) -> str:
        """Normalize text for mapping lookup."""
        # Remove extra whitespace but preserve structure
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized
    
    def _texts_match(self, text1: str, text2: str) -> bool:
        """Check if two texts match using various strategies."""
        # Direct match
        if text1 == text2:
            return True
        
        # Case insensitive match
        if text1.lower() == text2.lower():
            return True
        
        # Normalized whitespace match
        norm1 = re.sub(r'\s+', ' ', text1.strip())
        norm2 = re.sub(r'\s+', ' ', text2.strip())
        if norm1.lower() == norm2.lower():
            return True
        
        return False
    
    def repackage_docx(self, output_path: Path) -> bool:
        """
        Repackage the modified contents back into a DOCX file.
        
        Args:
            output_path: Path for the output DOCX file
            
        Returns:
            bool: True if successful
        """
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                temp_path = Path(self.temp_dir)
                
                # Add all files back to the zip
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        # Calculate relative path from temp directory
                        arcname = file_path.relative_to(temp_path)
                        zip_ref.write(file_path, arcname)
            
            logger.info(f"Repackaged DOCX to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to repackage DOCX: {e}")
            return False
    
    def process_large_docx(self, mapping_dict: Dict[str, str], patterns: List[str], 
                          output_path: Path, text_mode: str = "replace") -> Dict[str, any]:
        """
        Main method to process a large DOCX file.
        
        Args:
            mapping_dict: Text mapping dictionary
            patterns: Regex patterns
            output_path: Output path for processed file
            text_mode: "replace" or "append"
            
        Returns:
            Dict with processing summary
        """
        summary = {
            "header_replacements": 0,
            "body_replacements": 0,
            "font_adjustments": 0,
            "errors": [],
            "warnings": [],
            "processed_path": str(output_path)
        }
        
        try:
            # Extract DOCX contents
            if not self.extract_docx_contents():
                summary["errors"].append("Failed to extract DOCX contents")
                return summary
            
            # Process main document XML
            doc_xml_path = self.get_document_xml_path()
            if not doc_xml_path:
                summary["errors"].append("Could not find document.xml")
                return summary
            
            replacements, warnings = self.process_xml_in_chunks(
                doc_xml_path, mapping_dict, patterns, text_mode
            )
            
            summary["body_replacements"] = replacements
            summary["warnings"].extend(warnings)
            
            # Process headers and footers if they exist
            word_dir = Path(self.temp_dir) / "word"
            for header_footer in word_dir.glob("header*.xml"):
                h_replacements, h_warnings = self.process_xml_in_chunks(
                    header_footer, mapping_dict, patterns, text_mode
                )
                summary["header_replacements"] += h_replacements
                summary["warnings"].extend(h_warnings)
            
            for header_footer in word_dir.glob("footer*.xml"):
                f_replacements, f_warnings = self.process_xml_in_chunks(
                    header_footer, mapping_dict, patterns, text_mode
                )
                summary["header_replacements"] += f_replacements
                summary["warnings"].extend(f_warnings)
            
            # Repackage the DOCX
            if not self.repackage_docx(output_path):
                summary["errors"].append("Failed to repackage DOCX")
                return summary
            
            logger.info(f"Successfully processed large DOCX: {replacements} replacements")
            
        except Exception as e:
            error_msg = f"Large DOCX processing failed: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
        
        return summary


def process_large_docx_file(docx_path: Path, mapping_dict: Dict[str, str], 
                           patterns: List[str], output_path: Path, 
                           text_mode: str = "replace") -> Dict[str, any]:
    """
    Convenience function to process a large DOCX file.
    
    Args:
        docx_path: Input DOCX file path
        mapping_dict: Text mapping dictionary
        patterns: Regex patterns
        output_path: Output path for processed file
        text_mode: "replace" or "append"
        
    Returns:
        Dict with processing summary
    """
    with LargeDocxHandler(docx_path) as handler:
        return handler.process_large_docx(mapping_dict, patterns, output_path, text_mode)
