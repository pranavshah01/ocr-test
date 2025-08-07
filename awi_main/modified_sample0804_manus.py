
import os
import sys
import json
import logging
import tempfile
import zipfile
# Conditional import: prefer lxml, fallback to xml.etree.ElementTree
try:
    import lxml.etree as LET
    ET = LET
    LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML_AVAILABLE = False
from pathlib import Path
import shutil
from typing import List, Dict, Any, Tuple

if not LXML_AVAILABLE:
    from xml.etree.ElementTree import Element
else:
    Element = ET.Element

ET.register_namespace('xml', 'http://www.w3.org/XML/1998/namespace')
ET.register_namespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import math


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        self.tesseract_available = False
        
        # Initialize OCR engines (only Tesseract for reliability)
        try:
            import pytesseract
            # Cross-platform Tesseract path detection
            import platform
            import shutil
            
            # Try to find tesseract in PATH first
            tesseract_cmd = shutil.which('tesseract')
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            elif platform.system() == 'Windows':
                # Common Windows installation paths
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', ''))
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
            # On macOS/Linux, tesseract is usually in PATH or standard locations
            
            self.tesseract_available = True
            logger.info("Tesseract is available")
        except ImportError:
            logger.warning("Tesseract not available")
            self.tesseract_available = False
        
        if not self.tesseract_available:
            logger.warning("No OCR engines available - image processing will be limited")

    def detect_text_in_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect text in image using Tesseract OCR"""
        results = []
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return results
        
        # Use Tesseract for OCR
        if self.tesseract_available:
            try:
                import pytesseract
                # Convert to RGB for Tesseract
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tesseract_data = pytesseract.image_to_data(
                    image_rgb,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'
                )
                for i in range(len(tesseract_data['text'])):
                    text = tesseract_data['text'][i].strip()
                    if text and tesseract_data['conf'][i] > 60:
                        bbox = {
                            'x': tesseract_data['left'][i],
                            'y': tesseract_data['top'][i],
                            'width': tesseract_data['width'][i],
                            'height': tesseract_data['height'][i],
                            'text': text,
                            'confidence': tesseract_data['conf'][i] / 100.0,
                            'points': None
                        }
                        results.append(bbox)
                logger.info(f"Tesseract found {len(results)} text regions")
            except Exception as e:
                logger.error(f"Tesseract failed: {e}")
        
        return results

    def replace_text_in_image(self, image_bytes: bytes, mapping: Dict[str, str], file_extension: str = '.png') -> Tuple[bytes, bool]:
        """Replace text in image and return new image bytes"""
        try:
            # Detect text in image
            text_regions = self.detect_text_in_image(image_bytes)
            
            if not text_regions:
                return image_bytes, False
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            draw = ImageDraw.Draw(image)
            
            # Try to find a suitable font (cross-platform)
            try:
                import platform
                if platform.system() == 'Windows':
                    font_paths = ['arial.ttf', 'C:/Windows/Fonts/arial.ttf']
                elif platform.system() == 'Darwin':  # macOS
                    font_paths = ['/System/Library/Fonts/Arial.ttf', '/Library/Fonts/Arial.ttf']
                else:  # Linux
                    font_paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
                
                base_font = None
                for font_path in font_paths:
                    try:
                        base_font = ImageFont.truetype(font_path, 12)
                        break
                    except:
                        continue
                
                if base_font is None:
                    base_font = ImageFont.load_default()
            except:
                try:
                    base_font = ImageFont.load_default()
                except:
                    base_font = None
            
            replacements_made = False
            
            # Apply text replacements to detected regions
            for region in text_regions:
                region_text = region['text']
                logger.debug(f"OCR detected text: '{region_text}' at position ({region['x']}, {region['y']})")
                
                replacement_found = False
                for old_text, new_text in mapping.items():
                    if old_text in region_text:
                        # Find position of the specific pattern within the text
                        pattern_start = region_text.find(old_text)
                        if pattern_start == -1:
                            continue
                        
                        # Calculate position and size of the specific pattern more precisely
                        total_chars = len(region_text)
                        char_width = region['width'] / total_chars
                        
                        # Calculate the exact width of the matched pattern
                        # Instead of using average char width for the entire text, use it only for the pattern
                        pattern_width = len(old_text) * char_width
                        
                        # Fine-tune starting position with slight adjustment to better match original text
                        position_adjust_x = 1  # Slight adjustment right (reduced from 2)
                        position_adjust_y = -1  # Slight adjustment upward
                        start_x = region['x'] + pattern_start * char_width + position_adjust_x
                        end_x = start_x + pattern_width
                        y = region['y'] + position_adjust_y  # Shift slightly upward
                        height = region['height']
                        
                        # Create white rectangle with adaptive padding based on the font size
                        # Smaller padding for horizontal edges to avoid overwriting adjacent text
                        h_padding = max(1, min(2, height * 0.05))  # Horizontal padding: between 1-2px, adaptive to text height
                        v_padding = max(1, min(2, height * 0.1))   # Vertical padding: slightly larger
                        
                        # Draw tighter white rectangle over the exact pattern only
                        draw.rectangle([start_x - h_padding, y - v_padding, end_x + h_padding, y + height + v_padding], fill='white')
                        
                        # Draw new text with adjusted font size to match original width
                        if base_font:
                            # Find optimal font size to fit the pattern width
                            font_size = 1
                            best_font = base_font
                            
                            # Find max font size that fits the pattern width (with padding consideration)
                            available_width = pattern_width - (2 * h_padding)
                            for size in range(8, 100):  # Try sizes from 8 to 100
                                try:
                                    if platform.system() == 'Windows':
                                        test_font = ImageFont.truetype('arial.ttf', size)
                                    elif platform.system() == 'Darwin':
                                        test_font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', size)
                                    else:
                                        test_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
                                    
                                    # Measure new text width
                                    test_width = test_font.getlength(new_text)
                                    if test_width <= available_width:
                                        best_font = test_font
                                        font_size = size
                                    else:
                                        break  # Stop when we exceed width
                                except:
                                    continue
                            
                            # Get text dimensions for better positioning
                            text_bbox = draw.textbbox((0, 0), new_text, font=best_font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            # Position text to align with the original text baseline
                            # Use the original region's top position for better alignment
                            text_y = y + (height - text_height) // 2
                            text_x = start_x + (pattern_width - text_width) // 2
                            
                            draw.text((text_x, text_y), new_text, font=best_font, fill='black')
                        else:
                            # Fallback without font
                            draw.text((start_x, y), new_text, fill='black')
                        
                        replacements_made = True
                        logger.info(f"Replaced '{old_text}' with '{new_text}' in image")
            
            if replacements_made:
                # Convert back to bytes
                output_buffer = io.BytesIO()
                # Use appropriate format based on file extension
                if file_extension.lower() in ['.jpg', '.jpeg']:
                    save_format = 'JPEG'
                elif file_extension.lower() == '.bmp':
                    save_format = 'BMP'
                else:
                    save_format = 'PNG'  # Default to PNG for .png and others
                image.save(output_buffer, format=save_format)
                return output_buffer.getvalue(), True
            
            return image_bytes, False
        except Exception as e:
            logger.error(f"Error replacing text in image: {e}")
            return image_bytes, False

            
def load_mapping(file_path: str) -> Dict[str, str]:
    """Load mapping from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            logger.info(f"Successfully loaded mapping file with {len(mapping)} entries")
            return mapping
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in mapping file: {file_path}")
        sys.exit(1)

def replace_text_in_hyperlink(hyperlink: Element, old_text: str, new_text: str, ns, W_NAMESPACE):
    """
    Replace text within a hyperlink element while preserving the hyperlink structure.
    """
    # Find all text elements within the hyperlink
    text_elements = hyperlink.findall('.//w:t', ns)
    
    # Concatenate all text to find the full hyperlink text
    full_text = ''.join([t_el.text or '' for t_el in text_elements])
    
    if old_text in full_text:
        # Replace the text
        updated_text = full_text.replace(old_text, new_text)
        
        # Clear all existing text elements
        for t_el in text_elements:
            t_el.text = ''
        
        # Set the new text in the first text element
        if text_elements:
            first_text = text_elements[0]
            # Preserve leading/trailing spaces per Word spec
            if updated_text.startswith(' ') or updated_text.endswith(' '):
                first_text.set('xml:space', 'preserve')
            else:
                if 'xml:space' in first_text.attrib:
                    del first_text.attrib['xml:space']
            first_text.text = updated_text
            
            # Clear other text elements
            for t_el in text_elements[1:]:
                t_el.text = ''

def set_paragraph_text(p: Element, runs, new_text, ns, W_NAMESPACE):
    """
    Replace the entire paragraph text with new_text in the first run.
    All other runs are cleared (runs themselves removed from the paragraph).
    """
    if not runs:
        return
    first_run = runs[0]
    t_first = first_run.find('w:t', ns)
    if t_first is None:
        t_first = ET.SubElement(first_run, f'{{{W_NAMESPACE}}}t')
    # Preserve leading/trailing spaces per Word spec
    if new_text.startswith(' ') or new_text.endswith(' '):
        t_first.set('xml:space', 'preserve')
    else:
        if 'xml:space' in t_first.attrib:
            del t_first.attrib['xml:space']
    t_first.text = new_text
    # Remove runs themselves (not just <w:t>) from the paragraph except first
    for r in runs[1:]:
        p.remove(r)

def adjust_font_size(runs, orig_len, new_len, p, ns, W_NAMESPACE):
    """
    Adjust font size for a list of runs and also the paragraph-level rPr so that increased text length fits the shape.
    Uses sqrt scaling for area-proportional fit.
    """
    if new_len > orig_len and orig_len > 0:
        ratio = orig_len / new_len
        scale = math.sqrt(ratio)
        min_font = 16  # 8 pt (Word uses half-points)
        # Determine the original size (prefer largest among runs, fallback to 28)
        orig_sz = None
        for r in runs:
            rPr = r.find('w:rPr', ns)
            if rPr is not None:
                sz_elem = rPr.find('w:sz', ns)
                if sz_elem is not None:
                    try:
                        sz = int(sz_elem.get(f'{{{W_NAMESPACE}}}val'))
                        if orig_sz is None or sz > orig_sz:
                            orig_sz = sz
                    except Exception:
                        continue
        if orig_sz is None:
            # Try paragraph-level <w:pPr>/<w:rPr>/<w:sz>
            pPr = p.find('w:pPr', ns)
            if pPr is not None:
                rPr_p = pPr.find('w:rPr', ns)
                if rPr_p is not None:
                    sz_elem = rPr_p.find('w:sz', ns)
                    if sz_elem is not None:
                        try:
                            sz = int(sz_elem.get(f'{{{W_NAMESPACE}}}val'))
                            orig_sz = sz
                        except Exception:
                            pass
        if orig_sz is None:
            orig_sz = 28  # default to 14 pt if not specified

        new_sz = int(max(orig_sz * scale, min_font))
        if new_sz > min_font:
            new_sz -= 1  # apply −1 logic so, e.g., 26→25
        if new_sz < orig_sz:
            # Set for all runs
            for r in runs:
                rPr = r.find('w:rPr', ns)
                if rPr is None:
                    rPr = ET.SubElement(r, f'{{{W_NAMESPACE}}}rPr')
                sz_elem = rPr.find('w:sz', ns)
                if sz_elem is None:
                    sz_elem = ET.SubElement(rPr, f'{{{W_NAMESPACE}}}sz')
                sz_elem.set(f'{{{W_NAMESPACE}}}val', str(new_sz))
                szcs = rPr.find('w:szCs', ns)
                if szcs is None:
                    szcs = ET.SubElement(rPr, f'{{{W_NAMESPACE}}}szCs')
                szcs.set(f'{{{W_NAMESPACE}}}val', str(new_sz))
            # Set at paragraph-level <w:pPr>/<w:rPr>
            pPr = p.find('w:pPr', ns)
            if pPr is None:
                pPr = ET.SubElement(p, f'{{{W_NAMESPACE}}}pPr')
            rPr_p = pPr.find('w:rPr', ns)
            if rPr_p is None:
                rPr_p = ET.SubElement(pPr, f'{{{W_NAMESPACE}}}rPr')
            sz_elem_p = rPr_p.find('w:sz', ns)
            if sz_elem_p is None:
                sz_elem_p = ET.SubElement(rPr_p, f'{{{W_NAMESPACE}}}sz')
            sz_elem_p.set(f'{{{W_NAMESPACE}}}val', str(new_sz))
            szcs_p = rPr_p.find('w:szCs', ns)
            if szcs_p is None:
                szcs_p = ET.SubElement(rPr_p, f'{{{W_NAMESPACE}}}szCs')
            szcs_p.set(f'{{{W_NAMESPACE}}}val', str(new_sz))

def update_docx_structure_for_dual_images(tmpdir_path: Path, modified_images: Dict[str, str], ns, W_NAMESPACE):
    """
    Update DOCX structure to include modified images below originals.
    This function finds image references in the document and inserts modified versions below them.
    """
    try:
        # Update relationships to include new images
        rels_file = tmpdir_path / 'word' / '_rels' / 'document.xml.rels'
        if rels_file.exists():
            if LXML_AVAILABLE:
                from lxml import etree
                rels_tree = etree.parse(str(rels_file))
                rels_root = rels_tree.getroot()
            else:
                rels_tree = ET.parse(str(rels_file))
                rels_root = rels_tree.getroot()
            
            # Find the highest relationship ID
            max_id = 0
            rel_namespace = 'http://schemas.openxmlformats.org/package/2006/relationships'
            rels_ns = {'r': rel_namespace}
            
            for rel in rels_root.findall('r:Relationship', rels_ns):
                rel_id = rel.get('Id', '')
                if rel_id.startswith('rId'):
                    try:
                        num = int(rel_id[3:])
                        max_id = max(max_id, num)
                    except ValueError:
                        continue
            
            # Add new relationships for modified images
            new_relationships = {}
            for original_name, modified_name in modified_images.items():
                max_id += 1
                new_rel_id = f'rId{max_id}'
                
                # Create new relationship element
                new_rel = ET.SubElement(rels_root, f'{{{rel_namespace}}}Relationship')
                new_rel.set('Id', new_rel_id)
                new_rel.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
                new_rel.set('Target', f'media/{modified_name}')
                
                new_relationships[original_name] = new_rel_id
                logger.info(f"Added relationship {new_rel_id} for {modified_name}")
            
            # Save updated relationships
            if LXML_AVAILABLE:
                rels_tree.write(str(rels_file), pretty_print=False, xml_declaration=True, encoding='UTF-8')
            else:
                rels_tree.write(str(rels_file), encoding='UTF-8', xml_declaration=True)
            
            # Update document.xml to include modified images below originals
            update_document_with_dual_images(tmpdir_path, new_relationships, ns, W_NAMESPACE)
            
    except Exception as e:
        logger.error(f"Error updating DOCX structure for dual images: {e}")

def update_document_with_dual_images(tmpdir_path: Path, new_relationships: Dict[str, str], ns, W_NAMESPACE):
    """
    Update the main document to insert modified images below originals.
    """
    try:
        doc_file = tmpdir_path / 'word' / 'document.xml'
        if not doc_file.exists():
            return
        
        if LXML_AVAILABLE:
            from lxml import etree
            tree = etree.parse(str(doc_file))
            root = tree.getroot()
        else:
            tree = ET.parse(str(doc_file))
            root = tree.getroot()
        
        # Find all drawing elements (images)
        drawing_elements = root.findall('.//w:drawing', ns)
        
        # Process each drawing element
        insertions_made = 0
        for drawing in drawing_elements:
            # Find the relationship ID for this image
            blip_elements = drawing.findall('.//a:blip', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
            
            for blip in blip_elements:
                embed_id = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if embed_id:
                    # Check if we have a modified version of this image
                    original_image_name = find_image_name_by_rel_id(tmpdir_path, embed_id)
                    if original_image_name and original_image_name in new_relationships:
                        # Create a duplicate of the drawing element for the modified image
                        modified_drawing = create_modified_image_element(drawing, new_relationships[original_image_name])
                        
                        # Insert the modified image after the original
                        parent = drawing.getparent()
                        if parent is not None:
                            # Find the paragraph containing this drawing
                            para = drawing
                            while para is not None and para.tag != f'{{{W_NAMESPACE}}}p':
                                para = para.getparent()
                            
                            if para is not None:
                                # Create a new paragraph for the modified image
                                new_para = ET.Element(f'{{{W_NAMESPACE}}}p')
                                new_run = ET.SubElement(new_para, f'{{{W_NAMESPACE}}}r')
                                new_run.append(modified_drawing)
                                
                                # Insert after the current paragraph
                                parent_of_para = para.getparent()
                                if parent_of_para is not None:
                                    para_index = list(parent_of_para).index(para)
                                    parent_of_para.insert(para_index + 1, new_para)
                                    insertions_made += 1
                                    logger.info(f"Inserted modified image below original for {original_image_name}")
        
        if insertions_made > 0:
            # Save the updated document
            if LXML_AVAILABLE:
                tree.write(str(doc_file), pretty_print=False, xml_declaration=True, encoding='UTF-8')
            else:
                tree.write(str(doc_file), encoding='UTF-8', xml_declaration=True)
            
            logger.info(f"Successfully inserted {insertions_made} modified images into document")
        
    except Exception as e:
        logger.error(f"Error updating document with dual images: {e}")

def find_image_name_by_rel_id(tmpdir_path: Path, rel_id: str) -> str:
    """
    Find the image filename by relationship ID.
    """
    try:
        rels_file = tmpdir_path / 'word' / '_rels' / 'document.xml.rels'
        if not rels_file.exists():
            return None
        
        if LXML_AVAILABLE:
            from lxml import etree
            rels_tree = etree.parse(str(rels_file))
            rels_root = rels_tree.getroot()
        else:
            rels_tree = ET.parse(str(rels_file))
            rels_root = rels_tree.getroot()
        
        rel_namespace = 'http://schemas.openxmlformats.org/package/2006/relationships'
        rels_ns = {'r': rel_namespace}
        
        for rel in rels_root.findall('r:Relationship', rels_ns):
            if rel.get('Id') == rel_id:
                target = rel.get('Target', '')
                if target.startswith('media/'):
                    return target[6:]  # Remove 'media/' prefix
        
        return None
    except Exception as e:
        logger.error(f"Error finding image name for rel ID {rel_id}: {e}")
        return None

def create_modified_image_element(original_drawing, new_rel_id: str):
    """
    Create a copy of the drawing element with updated relationship ID.
    """
    try:
        # Deep copy the drawing element
        if LXML_AVAILABLE:
            from lxml import etree
            modified_drawing = etree.fromstring(etree.tostring(original_drawing))
        else:
            # For ElementTree, we need to manually copy
            modified_drawing = copy_element_tree(original_drawing)
        
        # Update the relationship ID in the copy
        blip_elements = modified_drawing.findall('.//a:blip', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
        for blip in blip_elements:
            blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed', new_rel_id)
        
        return modified_drawing
    except Exception as e:
        logger.error(f"Error creating modified image element: {e}")
        return None

def copy_element_tree(element):
    """
    Recursively copy an ElementTree element (for when lxml is not available).
    """
    new_element = ET.Element(element.tag, element.attrib)
    new_element.text = element.text
    new_element.tail = element.tail
    
    for child in element:
        new_element.append(copy_element_tree(child))
    
    return new_element

def replace_patterns_in_docx(input_path: str, output_path: str, mapping: Dict[str, str], ocr_processor: OCRProcessor) -> Dict[str, int]:
    """Process DOCX file with text and image replacements"""
    stats = {
        'text_matches': 0,
        'text_replacements': 0,
        'images_processed': 0,
        'image_matches': 0,
        'image_replacements': 0
    }

    # Namespaces
    W_NAMESPACE = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    ns = {'w': W_NAMESPACE}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Unzip DOCX
        with zipfile.ZipFile(input_path, 'r') as zin:
            zin.extractall(tmpdir)

        # Process text in all XML parts
        xml_rel_paths = ['word/document.xml'] + \
                       [f"word/header{i}.xml" for i in range(1, 4)] + \
                       [f"word/footer{i}.xml" for i in range(1, 4)]

        for rel in xml_rel_paths:
            xml_file = tmpdir_path / rel
            if not xml_file.exists():
                continue

            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()

                # Only process paragraphs within text boxes (w:txbxContent)
                textboxes = root.findall('.//w:txbxContent', ns)
                if textboxes:
                    paragraphs = []
                    for tbx in textboxes:
                        paragraphs.extend(tbx.findall('.//w:p', ns))
                else:
                    paragraphs = root.findall('.//w:p', ns)  # fallback (shouldn't happen in normal flow)

                for p in paragraphs:
                    # Enhanced text processing to handle hyperlinks
                    runs = p.findall('w:r', ns)
                    hyperlinks = p.findall('.//w:hyperlink', ns)
                    
                    # Gather full paragraph text (concatenate all <w:t> in .//w:t)
                    para_text = ''.join([t_el.text or '' for t_el in p.findall('.//w:t', ns)])
                    replaced = False
                    
                    for old_text, new_text in mapping.items():
                        if old_text in para_text:
                            # Calculate new paragraph text
                            new_para_text = para_text.replace(old_text, f"{old_text} {new_text}")
                            
                            # Check if the text is within a hyperlink
                            text_in_hyperlink = False
                            for hyperlink in hyperlinks:
                                hyperlink_text = ''.join([t_el.text or '' for t_el in hyperlink.findall('.//w:t', ns)])
                                if old_text in hyperlink_text:
                                    # Replace text within hyperlink
                                    replace_text_in_hyperlink(hyperlink, old_text, new_para_text, ns, W_NAMESPACE)
                                    text_in_hyperlink = True
                                    logger.info(f"Replaced '{old_text}' with '{new_text}' in hyperlink text")
                                    break
                            
                            # If not in hyperlink, use standard replacement
                            if not text_in_hyperlink:
                                # Set new paragraph text in FIRST run, clear others
                                set_paragraph_text(p, runs, new_para_text, ns, W_NAMESPACE)
                                logger.info(f"Replaced '{old_text}' with '{new_text}' in paragraph text")
                            
                            stats['text_matches'] += 1
                            stats['text_replacements'] += 1
                            
                            # Adjust font size if new content is longer
                            adjust_font_size(
                                runs if not text_in_hyperlink else hyperlinks[0].findall('.//w:r', ns),
                                len(para_text),
                                len(new_para_text),
                                p,
                                ns,
                                W_NAMESPACE
                            )
                            replaced = True
                            break  # Only do one mapping per paragraph
                    # If not replaced, do nothing

                # Save modified XML
                if LXML_AVAILABLE:
                    tree.write(str(xml_file), pretty_print=False, xml_declaration=True, encoding='UTF-8')
                else:
                    tree.write(str(xml_file), encoding='UTF-8', xml_declaration=True)

            except Exception as e:
                logger.error(f"Error processing XML part {rel}: {e}")

        # Process images - create dual images (original + modified)
        media_dir = tmpdir_path / 'word' / 'media'
        modified_images = {}  # Track which images were modified
        
        if media_dir.exists():
            for img_file in media_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        stats['images_processed'] += 1

                        # Read image
                        with open(img_file, 'rb') as f:
                            img_bytes = f.read()

                        # Process image with OCR
                        new_img_bytes, replaced = ocr_processor.replace_text_in_image(img_bytes, mapping, img_file.suffix.lower())

                        if replaced:
                            stats['image_matches'] += 1
                            stats['image_replacements'] += 1

                            # Keep original image unchanged
                            # Create new modified image with suffix
                            original_name = img_file.stem
                            extension = img_file.suffix
                            modified_name = f"{original_name}_modified{extension}"
                            modified_path = media_dir / modified_name
                            
                            # Save modified image as new file
                            with open(modified_path, 'wb') as f:
                                f.write(new_img_bytes)
                            
                            # Track the relationship for DOCX structure updates
                            modified_images[img_file.name] = modified_name
                            logger.info(f"Created modified image: {modified_name}")

                    except Exception as e:
                        logger.error(f"Error processing image {img_file.name}: {e}")
        
        # Update DOCX structure to include modified images below originals
        if modified_images:
            update_docx_structure_for_dual_images(tmpdir_path, modified_images, ns, W_NAMESPACE)

        # Repack DOCX
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for folder, _, files in os.walk(tmpdir):
                for fname in files:
                    full = Path(folder) / fname
                    # Only skip .docx files in the root temp dir, not embedded objects
                    if full.suffix.lower() == '.docx' and full.parent == tmpdir_path:
                        continue
                    arc = full.relative_to(tmpdir_path).as_posix()
                    zout.write(str(full), arc)

    return stats

def main(input_path: str, mapping_path: str, output_path: str):
    """Main processing function"""
    try:
        logger.info(f"Starting processing for {input_path}")
        
        # Load mapping
        mapping = load_mapping(mapping_path)
        
        # Initialize OCR processor
        ocr_processor = OCRProcessor()
        
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Process based on file type
        if input_path.endswith('.docx'):
            # Process DOCX file
            stats = replace_patterns_in_docx(input_path, output_path, mapping, ocr_processor)
            
            logger.info(f"Processing completed:")
            logger.info(f"  Text matches: {stats['text_matches']}")
            logger.info(f"  Text replacements: {stats['text_replacements']}")
            logger.info(f"  Images processed: {stats['images_processed']}")
            logger.info(f"  Image matches: {stats['image_matches']}")
            logger.info(f"  Image replacements: {stats['image_replacements']}")
            
            logger.info(f"Successfully saved processed document to {output_path}")
            
        elif input_path.endswith('.doc'):
            logger.error("DOC file processing not supported. Please convert to DOCX first.")
            sys.exit(1)
        else:
            logger.error(f"Unsupported file format: {input_path}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_app.py <input_file>")
        print("Example: python ocr_app.py testimages_092051.docx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    mapping_file = 'mapping.json'
    output_file = 'processed_' + os.path.basename(input_file)
    
    main(input_file, mapping_file, output_file)