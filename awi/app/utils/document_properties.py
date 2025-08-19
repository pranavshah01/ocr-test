"""
Document properties extraction utility.
Extracts metadata like page count, word count, etc. from DOCX files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Any
import zipfile
import logging

logger = logging.getLogger(__name__)

class DocumentPropertiesExtractor:
    """Extracts document properties from DOCX files."""
    
    @staticmethod
    def extract_properties(docx_path: Path) -> Dict[str, Any]:
        """
        Extract document properties from a DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Dictionary containing document properties
        """
        properties = {
            'pages': None,
            'words': None,
            'characters': None,
            'paragraphs': None,
            'lines': None,
            'application': None,
            'template': None,
            'creator': None,
            'last_modified_by': None,
            'revision': None,
            'created': None,
            'modified': None
        }
        
        try:
            with zipfile.ZipFile(docx_path, 'r') as zip_file:
                # Extract app.xml properties
                if 'docProps/app.xml' in zip_file.namelist():
                    app_props = DocumentPropertiesExtractor._parse_app_properties(zip_file.read('docProps/app.xml'))
                    properties.update(app_props)
                
                # Extract core.xml properties
                if 'docProps/core.xml' in zip_file.namelist():
                    core_props = DocumentPropertiesExtractor._parse_core_properties(zip_file.read('docProps/core.xml'))
                    properties.update(core_props)
                
                # If page count is not available, try to estimate from document structure
                if properties['pages'] is None:
                    properties['pages'] = DocumentPropertiesExtractor._estimate_page_count(zip_file)
                
        except Exception as e:
            logger.warning(f"Failed to extract document properties from {docx_path}: {e}")
        
        return properties
    
    @staticmethod
    def _parse_app_properties(app_xml_content: bytes) -> Dict[str, Any]:
        """Parse app.xml properties."""
        properties = {}
        
        try:
            root = ET.fromstring(app_xml_content)
            
            # Define namespace
            ns = {'ep': 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'}
            
            # Extract properties
            properties['pages'] = DocumentPropertiesExtractor._safe_get_int(root, './/ep:Pages', ns)
            properties['words'] = DocumentPropertiesExtractor._safe_get_int(root, './/ep:Words', ns)
            properties['characters'] = DocumentPropertiesExtractor._safe_get_int(root, './/ep:Characters', ns)
            properties['paragraphs'] = DocumentPropertiesExtractor._safe_get_int(root, './/ep:Paragraphs', ns)
            properties['lines'] = DocumentPropertiesExtractor._safe_get_int(root, './/ep:Lines', ns)
            properties['application'] = DocumentPropertiesExtractor._safe_get_text(root, './/ep:Application', ns)
            properties['template'] = DocumentPropertiesExtractor._safe_get_text(root, './/ep:Template', ns)
            
        except Exception as e:
            logger.warning(f"Failed to parse app.xml properties: {e}")
        
        return properties
    
    @staticmethod
    def _parse_core_properties(core_xml_content: bytes) -> Dict[str, Any]:
        """Parse core.xml properties."""
        properties = {}
        
        try:
            root = ET.fromstring(core_xml_content)
            
            # Define namespaces
            ns = {
                'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
                'dc': 'http://purl.org/dc/elements/1.1/',
                'dcterms': 'http://purl.org/dc/terms/'
            }
            
            # Extract properties
            properties['creator'] = DocumentPropertiesExtractor._safe_get_text(root, './/dc:creator', ns)
            properties['last_modified_by'] = DocumentPropertiesExtractor._safe_get_text(root, './/cp:lastModifiedBy', ns)
            properties['revision'] = DocumentPropertiesExtractor._safe_get_int(root, './/cp:revision', ns)
            properties['created'] = DocumentPropertiesExtractor._safe_get_text(root, './/dcterms:created', ns)
            properties['modified'] = DocumentPropertiesExtractor._safe_get_text(root, './/dcterms:modified', ns)
            
        except Exception as e:
            logger.warning(f"Failed to parse core.xml properties: {e}")
        
        return properties
    
    @staticmethod
    def _estimate_page_count(zip_file: zipfile.ZipFile) -> Optional[int]:
        """
        Estimate page count from document structure when not available in properties.
        
        Args:
            zip_file: Open ZIP file containing the DOCX
            
        Returns:
            Estimated page count or None if cannot estimate
        """
        try:
            # Count section breaks in document.xml
            if 'word/document.xml' in zip_file.namelist():
                doc_content = zip_file.read('word/document.xml').decode('utf-8')
                
                # Count section breaks (w:sectPr elements)
                section_count = doc_content.count('<w:sectPr')
                
                # Count page breaks
                page_break_count = doc_content.count('<w:br w:type="page"')
                
                # Estimate based on content length and structure
                # This is a rough estimation - actual page count may vary
                estimated_pages = max(1, section_count + page_break_count + 1)
                
                logger.info(f"Estimated page count: {estimated_pages} (sections: {section_count}, page breaks: {page_break_count})")
                return estimated_pages
                
        except Exception as e:
            logger.warning(f"Failed to estimate page count: {e}")
        
        return None
    
    @staticmethod
    def _safe_get_int(element: ET.Element, xpath: str, namespaces: Dict[str, str]) -> Optional[int]:
        """Safely get integer value from XML element."""
        try:
            found = element.find(xpath, namespaces)
            if found is not None and found.text:
                return int(found.text)
        except (ValueError, TypeError):
            pass
        return None
    
    @staticmethod
    def _safe_get_text(element: ET.Element, xpath: str, namespaces: Dict[str, str]) -> Optional[str]:
        """Safely get text value from XML element."""
        try:
            found = element.find(xpath, namespaces)
            if found is not None and found.text:
                return found.text.strip()
        except (ValueError, TypeError):
            pass
        return None

def extract_document_properties(docx_path: Path) -> Dict[str, Any]:
    """
    Convenience function to extract document properties.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        Dictionary containing document properties
    """
    return DocumentPropertiesExtractor.extract_properties(docx_path)
