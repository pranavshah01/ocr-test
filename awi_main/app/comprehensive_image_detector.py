#!/usr/bin/env python3
"""
Task 4.2: Comprehensive Image Detection for DOCX files.
Uses ZIP extraction to find ALL images including inline, floating, headers, footers, shapes, and backgrounds.
"""

import zipfile
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
from io import BytesIO
import logging

try:
    from .shared_constants import SharedUtilities
except ImportError:
    from shared_constants import SharedUtilities

logger = logging.getLogger(__name__)


class ComprehensiveImageDetector:
    """
    Comprehensive image detection for DOCX files using ZIP extraction.
    Finds ALL images including those missed by inline_shapes approach.
    """
    
    def __init__(self):
        self.supported_image_types = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
        self.image_locations = {
            'word/media/': 'main_media',
            'word/embeddings/': 'embedded_objects', 
            'word/theme/': 'theme_images',
            'docProps/': 'document_properties',
            'customXml/': 'custom_xml_images'
        }
        self.xml_image_references = {}
        
    def detect_all_images(self, docx_path: Path) -> Dict[str, Any]:
        """
        Detect all images in a DOCX file using comprehensive ZIP extraction.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Dictionary containing all detected images with metadata
        """
        results = {
            'total_images': 0,
            'images_by_location': {},
            'images_by_type': {},
            'image_details': [],
            'xml_references': {},
            'processing_errors': []
        }
        
        try:
            # Extract images using ZIP method
            zip_images = self._extract_images_from_zip(docx_path)
            
            # Analyze XML references for context
            xml_references = self._analyze_xml_references(docx_path)
            
            # Combine results
            results['total_images'] = len(zip_images)
            results['images_by_location'] = self._categorize_by_location(zip_images)
            results['images_by_type'] = self._categorize_by_type(zip_images)
            results['image_details'] = zip_images
            results['xml_references'] = xml_references
            
            logger.info(f"Comprehensive image detection found {results['total_images']} images in {docx_path.name}")
            
        except Exception as e:
            error_msg = f"Error in comprehensive image detection: {e}"
            logger.error(error_msg)
            results['processing_errors'].append(error_msg)
        
        return results
    
    def _extract_images_from_zip(self, docx_path: Path) -> List[Dict[str, Any]]:
        """
        Extract all images from DOCX ZIP archive.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            List of image dictionaries with metadata
        """
        images = []
        
        try:
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                # Get all files in the ZIP
                all_files = docx_zip.namelist()
                
                # Find all image files
                for file_path in all_files:
                    if self._is_image_file(file_path):
                        try:
                            image_data = self._extract_image_data(docx_zip, file_path)
                            if image_data:
                                images.append(image_data)
                        except Exception as e:
                            logger.warning(f"Failed to extract image {file_path}: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to open DOCX as ZIP: {e}")
            
        return images
    
    def _is_image_file(self, file_path: str) -> bool:
        """
        Check if a file path represents an image file.
        
        Args:
            file_path: File path within ZIP
            
        Returns:
            True if file is an image
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_image_types
    
    def _extract_image_data(self, docx_zip: zipfile.ZipFile, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract image data and metadata from ZIP file.
        
        Args:
            docx_zip: Open ZIP file object
            file_path: Path to image within ZIP
            
        Returns:
            Dictionary with image data and metadata
        """
        try:
            # Read image bytes
            image_bytes = docx_zip.read(file_path)
            
            # Load image for metadata
            image = Image.open(BytesIO(image_bytes))
            
            # Determine location category
            location_category = self._determine_location_category(file_path)
            
            # Extract filename and extension
            filename = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()
            
            image_data = {
                'zip_path': file_path,
                'filename': filename,
                'file_extension': file_ext,
                'location_category': location_category,
                'size_bytes': len(image_bytes),
                'dimensions': image.size,
                'format': image.format,
                'mode': image.mode,
                'image_bytes': image_bytes,
                'pil_image': image.copy(),
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            return image_data
            
        except Exception as e:
            logger.warning(f"Failed to process image {file_path}: {e}")
            return None
    
    def _determine_location_category(self, file_path: str) -> str:
        """
        Determine the category/location of an image based on its ZIP path.
        
        Args:
            file_path: Path within ZIP file
            
        Returns:
            Category string
        """
        for location_prefix, category in self.image_locations.items():
            if file_path.startswith(location_prefix):
                return category
        
        # Default category for unrecognized locations
        return 'other'
    
    def _categorize_by_location(self, images: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize images by their location within the DOCX structure.
        
        Args:
            images: List of image dictionaries
            
        Returns:
            Dictionary mapping location categories to counts
        """
        location_counts = {}
        
        for image in images:
            category = image.get('location_category', 'unknown')
            location_counts[category] = location_counts.get(category, 0) + 1
        
        return location_counts
    
    def _categorize_by_type(self, images: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize images by their file type/format.
        
        Args:
            images: List of image dictionaries
            
        Returns:
            Dictionary mapping file types to counts
        """
        type_counts = {}
        
        for image in images:
            file_ext = image.get('file_extension', 'unknown')
            type_counts[file_ext] = type_counts.get(file_ext, 0) + 1
        
        return type_counts
    
    def _analyze_xml_references(self, docx_path: Path) -> Dict[str, Any]:
        """
        Analyze XML files to find image references and context.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Dictionary with XML reference analysis
        """
        xml_analysis = {
            'document_xml_refs': [],
            'header_footer_refs': [],
            'relationship_refs': [],
            'shape_refs': [],
            'background_refs': []
        }
        
        try:
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                # Analyze main document XML
                if 'word/document.xml' in docx_zip.namelist():
                    doc_xml = docx_zip.read('word/document.xml').decode('utf-8')
                    xml_analysis['document_xml_refs'] = self._find_image_references_in_xml(doc_xml)
                
                # Analyze header/footer XML files
                for file_path in docx_zip.namelist():
                    if file_path.startswith('word/header') or file_path.startswith('word/footer'):
                        if file_path.endswith('.xml'):
                            xml_content = docx_zip.read(file_path).decode('utf-8')
                            refs = self._find_image_references_in_xml(xml_content)
                            if refs:
                                xml_analysis['header_footer_refs'].append({
                                    'file': file_path,
                                    'references': refs
                                })
                
                # Analyze relationships
                if 'word/_rels/document.xml.rels' in docx_zip.namelist():
                    rels_xml = docx_zip.read('word/_rels/document.xml.rels').decode('utf-8')
                    xml_analysis['relationship_refs'] = self._find_relationship_references(rels_xml)
                
        except Exception as e:
            logger.warning(f"Error analyzing XML references: {e}")
        
        return xml_analysis
    
    def _find_image_references_in_xml(self, xml_content: str) -> List[Dict[str, str]]:
        """
        Find image references in XML content.
        
        Args:
            xml_content: XML content as string
            
        Returns:
            List of image reference dictionaries
        """
        references = []
        
        # Common image reference patterns in DOCX XML
        patterns = [
            r'<a:blip\s+r:embed="([^"]+)"',  # Embedded images
            r'<a:blip\s+r:link="([^"]+)"',   # Linked images
            r'<v:imagedata\s+r:id="([^"]+)"', # VML images
            r'<w:drawing>.*?r:embed="([^"]+)".*?</w:drawing>',  # Drawing objects
            r'<pic:blipFill>.*?r:embed="([^"]+)".*?</pic:blipFill>'  # Picture fills
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, xml_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': 'image_reference',
                    'relationship_id': match,
                    'pattern': pattern
                })
        
        return references
    
    def _find_relationship_references(self, rels_xml: str) -> List[Dict[str, str]]:
        """
        Find relationship references to images.
        
        Args:
            rels_xml: Relationships XML content
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Pattern to find image relationships
        pattern = r'<Relationship\s+Id="([^"]+)"\s+Type="[^"]*image[^"]*"\s+Target="([^"]+)"'
        matches = re.findall(pattern, rels_xml, re.IGNORECASE)
        
        for rel_id, target in matches:
            relationships.append({
                'relationship_id': rel_id,
                'target': target,
                'type': 'image_relationship'
            })
        
        return relationships
    
    def get_images_for_ocr_processing(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter and prepare images for OCR processing.
        
        Args:
            detection_results: Results from detect_all_images()
            
        Returns:
            List of images ready for OCR processing
        """
        ocr_ready_images = []
        
        for image_data in detection_results.get('image_details', []):
            # Skip very small images (likely icons or decorative elements)
            width, height = image_data.get('dimensions', (0, 0))
            if width < 50 or height < 50:
                continue
            
            # Skip images without transparency that are likely photos
            if not image_data.get('has_transparency', False) and width * height > 100000:
                # Large images without transparency are likely photos, not text
                continue
            
            # Prepare for OCR
            ocr_image = {
                'image_id': f"{image_data['location_category']}_{image_data['filename']}",
                'pil_image': image_data['pil_image'],
                'original_path': image_data['zip_path'],
                'dimensions': image_data['dimensions'],
                'location_category': image_data['location_category'],
                'filename': image_data['filename']
            }
            
            ocr_ready_images.append(ocr_image)
        
        return ocr_ready_images


def detect_all_docx_images(docx_path: Path) -> Dict[str, Any]:
    """
    Convenience function to detect all images in a DOCX file.
    
    Args:
        docx_path: Path to DOCX file
        
    Returns:
        Comprehensive image detection results
    """
    detector = ComprehensiveImageDetector()
    return detector.detect_all_images(docx_path)


def get_ocr_ready_images(docx_path: Path) -> List[Dict[str, Any]]:
    """
    Convenience function to get images ready for OCR processing.
    
    Args:
        docx_path: Path to DOCX file
        
    Returns:
        List of images prepared for OCR
    """
    detector = ComprehensiveImageDetector()
    detection_results = detector.detect_all_images(docx_path)
    return detector.get_images_for_ocr_processing(detection_results)


if __name__ == "__main__":
    # Test the comprehensive image detector
    import sys
    
    if len(sys.argv) > 1:
        docx_file = Path(sys.argv[1])
        if docx_file.exists():
            results = detect_all_docx_images(docx_file)
            
            print(f"=== COMPREHENSIVE IMAGE DETECTION RESULTS ===")
            print(f"DOCX File: {docx_file.name}")
            print(f"Total Images Found: {results['total_images']}")
            print(f"\nImages by Location:")
            for location, count in results['images_by_location'].items():
                print(f"  {location}: {count}")
            print(f"\nImages by Type:")
            for img_type, count in results['images_by_type'].items():
                print(f"  {img_type}: {count}")
            
            ocr_images = get_ocr_ready_images(docx_file)
            print(f"\nOCR-Ready Images: {len(ocr_images)}")
            
        else:
            print(f"File not found: {docx_file}")
    else:
        print("Usage: python comprehensive_image_detector.py <docx_file>")
