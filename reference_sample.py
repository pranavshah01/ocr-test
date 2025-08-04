import os
import sys
import json
import logging
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import re
import time


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

    def replace_text_in_image(self, image_bytes: bytes, mapping: Dict[str, str]) -> Tuple[bytes, bool]:
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
                image.save(output_buffer, format='PNG')
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
                
                # Process all paragraphs
                for p in root.findall('.//w:p', ns):
                    for r in p.findall('w:r', ns):
                        t = r.find('w:t', ns)
                        if t is not None and t.text:
                            orig_text = t.text
                            for old_text, new_text in mapping.items():
                                if old_text in orig_text:
                                    # Replace text
                                    new_content = orig_text.replace(old_text, f"{old_text} {new_text}")
                                    t.text = new_content
                                    stats['text_matches'] += 1
                                    stats['text_replacements'] += 1
                                    logger.info(f"Replaced '{old_text}' with '{new_text}' in text")
                
                # Save modified XML
                tree.write(str(xml_file), encoding='utf-8', xml_declaration=True)
                
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
                        new_img_bytes, replaced = ocr_processor.replace_text_in_image(img_bytes, mapping)
                        
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
                    arc = full.relative_to(tmpdir_path)
                    zout.write(str(full), str(arc))
    
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