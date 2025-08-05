
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

    def replace_text_in_image(self, image_bytes: bytes, mapping: Dict[str, str], original_ext: str) -> Tuple[bytes, bool]:
        """Replace text in image and return new image bytes, preserving original format"""
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
                font_size = 12
                if platform.system() == 'Windows':
                    font_paths = ['arial.ttf', 'C:/Windows/Fonts/arial.ttf']
                elif platform.system() == 'Darwin':  # macOS
                    font_paths = ['/System/Library/Fonts/Arial.ttf', '/Library/Fonts/Arial.ttf']
                else:  # Linux
                    font_paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
                
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            replacements_made = False
            
            # Check each text region for matches
            for region in text_regions:
                text = region['text']
                for old_text, new_text in mapping.items():
                    if old_text in text:
                        # Calculate position and size
                        x = region['x']
                        y = region['y']
                        width = region['width']
                        height = region['height']
                        
                        # Create white rectangle to cover original text
                        draw.rectangle([x, y, x + width, y + height], fill='white')
                        
                        # Draw new text
                        if font:
                            # Calculate font size to fit the region
                            font_size = min(width // len(new_text), height) if len(new_text) > 0 else height
                            font_size = max(font_size, 8)  # Minimum font size
                            try:
                                # Try to create font with calculated size
                                if platform.system() == 'Windows':
                                    new_font = ImageFont.truetype('arial.ttf', font_size)
                                elif platform.system() == 'Darwin':
                                    new_font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', font_size)
                                else:
                                    new_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
                            except:
                                new_font = font
                            
                            # Center text in region
                            text_bbox = draw.textbbox((0, 0), new_text, font=new_font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            text_x = x + (width - text_width) // 2
                            text_y = y + (height - text_height) // 2
                            
                            draw.text((text_x, text_y), new_text, font=new_font, fill='black')
                        else:
                            # Fallback without font
                            draw.text((x, y), new_text, fill='black')
                        
                        replacements_made = True
                        logger.info(f"Replaced '{old_text}' with '{new_text}' in image")
                        break
            
            if replacements_made:
                # Convert back to bytes
                output_buffer = io.BytesIO()
                ext_to_format = {
                    '.png': 'PNG',
                    '.jpg': 'JPEG',
                    '.jpeg': 'JPEG',
                    '.bmp': 'BMP'
                }
                fmt = ext_to_format.get(original_ext, 'PNG')
                image.save(output_buffer, format=fmt)
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
                    runs = p.findall('w:r', ns)
                    # Gather full paragraph text (concatenate all <w:t> in .//w:t)
                    para_text = ''.join([t_el.text or '' for t_el in p.findall('.//w:t', ns)])
                    replaced = False
                    for old_text, new_text in mapping.items():
                        if old_text in para_text:
                            # Calculate new paragraph text
                            new_para_text = para_text.replace(old_text, f"{old_text} {new_text}")
                            # Set new paragraph text in FIRST run, clear others
                            set_paragraph_text(p, runs, new_para_text, ns, W_NAMESPACE)
                            stats['text_matches'] += 1
                            stats['text_replacements'] += 1
                            logger.info(f"Replaced '{old_text}' with '{new_text}' in paragraph text")
                            # Adjust font size if new content is longer
                            adjust_font_size(
                                runs,
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

        # Process images
        media_dir = tmpdir_path / 'word' / 'media'
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

                            # Save modified image
                            with open(img_file, 'wb') as f:
                                f.write(new_img_bytes)

                    except Exception as e:
                        logger.error(f"Error processing image {img_file.name}: {e}")

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