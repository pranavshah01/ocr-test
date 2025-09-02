import re
import os
import logging
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from PIL import Image, ImageOps

from config import EASYOCR_LANGUAGES, EASYOCR_LANGUAGES_ALT, TESSERACT_LANGUAGES

logger = logging.getLogger(__name__)

# --- Image Extraction ---

def extract_images_with_mapping(docx_path: Path, temp_base_dir: Path) -> List[Dict[str, Any]]:
    """
    Extracts images from a DOCX file by parsing document.xml and its relationships.

    This method finds image references ('blips') in the document's XML, maps them
    to their media files via relationship IDs (rId), and extracts them to a
    temporary directory.

    Args:
        docx_path: The path to the .docx file.
        temp_base_dir: The base temporary directory to extract images into.

    Returns:
        A list of dictionaries, where each dictionary contains information
        about an extracted image, including its temporary path and original location.
    """
    image_info_list = []
    logger.info(f"Starting XML-based image extraction for: {docx_path.name}")

    try:
        # Open the .docx as a zip archive
        with zipfile.ZipFile(docx_path, "r") as docx:
            # Namespace mappings are crucial for parsing DOCX XML
            ns = {
                "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
                "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"
            }

            # Read relationships file to map rId to actual media file
            try:
                rels_xml = docx.read("word/_rels/document.xml.rels")
                rels_tree = ET.fromstring(rels_xml)
                rels_map = {
                    rel.attrib["Id"]: rel.attrib["Target"]
                    for rel in rels_tree.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
                }
            except KeyError:
                logger.warning(f"Could not find 'word/_rels/document.xml.rels' in {docx_path.name}. No images will be extracted.")
                return []


            # Read the main document content
            document_xml = docx.read("word/document.xml")
            doc_tree = ET.fromstring(document_xml)

            # Find all image references (<a:blip>) in the document
            for i, blip in enumerate(doc_tree.findall(".//a:blip", ns)):
                rid = blip.attrib.get(f"{{{ns['r']}}}embed")
                if not rid or rid not in rels_map:
                    continue

                # Get the path to the image within the DOCX archive
                img_target_path = rels_map[rid]
                img_full_path = f"word/{img_target_path}".replace("\\", "/")
                img_name = os.path.basename(img_full_path)

                # Create a unique temporary directory for this image
                temp_img_dir = temp_base_dir / f"image_{i}_{rid}"
                temp_img_dir.mkdir(parents=True, exist_ok=True)
                out_path = temp_img_dir / img_name

                try:
                    # Extract the image data and write it to the temporary file
                    with open(out_path, "wb") as f:
                        f.write(docx.read(img_full_path))

                    if not is_valid_image_file(out_path):
                        logger.warning(f"Skipping invalid image file from DOCX: {img_name}")
                        shutil.rmtree(temp_img_dir)
                        continue

                    # Store information about the extracted image
                    image_info = {
                        "rId": rid,
                        "media_filename": img_name,
                        "docx_path": img_full_path,
                        "temp_path": out_path,
                        "temp_dir": temp_img_dir,
                        "location": f"image_{i}_{rid}" # Unique location identifier
                    }
                    image_info_list.append(image_info)
                    logger.debug(f"Extracted '{img_name}' (rId={rid}) to '{out_path}'")

                except KeyError:
                    logger.warning(f"Image path '{img_full_path}' referenced in XML not found in DOCX archive.")
                except Exception as e:
                    logger.error(f"Failed to extract or save image {img_name}: {e}")
                    shutil.rmtree(temp_img_dir)

        logger.info(f"XML-based extraction found {len(image_info_list)} images in {docx_path.name}")
        return image_info_list

    except Exception as e:
        logger.error(f"Fatal error during XML-based image extraction for {docx_path.name}: {e}")
        return []


# --- Image Validation and Preprocessing ---

def is_valid_image_file(image_path: Path) -> bool:
    """Validate that a file is a valid, non-empty image."""
    try:
        if not image_path.exists() or image_path.stat().st_size == 0:
            return False
        with Image.open(image_path) as img:
            return img.size[0] > 0 and img.size[1] > 0
    except Exception:
        return False

def preprocess_image(image_path: Path) -> Optional[Image.Image]:
    """Preprocess an image for better OCR results."""
    try:
        with Image.open(image_path) as img:
            logger.debug(f"Original image mode: {img.mode}, size: {img.size}")
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def get_bounding_box_xywh(bbox: List) -> Optional[Tuple[int, int, int, int]]:
    """Normalize a variety of bbox formats to (x, y, w, h).

    Accepts:
    - A list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (EasyOCR/Tesseract style)
    - A flat list/tuple [x1, y1, x2, y2]
    Returns None if it cannot be parsed.
    """
    try:
        if not bbox:
            return None
        # Corner points case
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            return (x, y, w, h)
        # Flat [x1,y1,x2,y2]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x = int(min(x1, x2))
            y = int(min(y1, y2))
            w = int(abs(x2 - x1))
            h = int(abs(y2 - y1))
            return (x, y, w, h)
    except Exception:
        return None
    return None


def calculate_sub_bounding_box(
    full_bbox_xywh: Tuple[int, int, int, int],
    full_text: str,
    sub_text_match: Tuple[str, int, int]
) -> Tuple[int, int, int, int]:
    """Estimate a sub-bounding-box for substring [start:end] inside the OCR block.

    Approximates proportional width based on character indices.
    """
    try:
        x, y, w, h = full_bbox_xywh
        text_len = len(full_text or "")
        if text_len <= 0:
            return full_bbox_xywh
        _, start_char, end_char = sub_text_match
        if start_char < 0:
            start_char = 0
        if end_char < start_char:
            end_char = start_char
        new_x = x + int((start_char / text_len) * w)
        new_w = max(1, int(((end_char - start_char) / max(1, text_len)) * w))
        return (new_x, y, new_w, h)
    except Exception:
        return full_bbox_xywh


def format_bounding_box(bbox: List, rotation_angle: int = 0, image_size: tuple = None) -> str:
    """Format bounding box coordinates for display in reports as x=..,y=..,w=..,h=.."""
    xywh = get_bounding_box_xywh(bbox)
    if not xywh:
        return "N/A"
    x, y, w, h = xywh
    base = f"x={x}, y={y}, w={w}, h={h}"
    if rotation_angle and rotation_angle != 0:
        return f"{base} (rotated {rotation_angle}°)"
    return base


# --- OCR Management ---

class OCRResult:
    """A standardized class for holding OCR results from any engine."""
    def __init__(self, text: str, confidence: float = 0.0, bounding_box: Optional[List] = None, source_engine: str = "unknown"):
        self.text = text
        self.confidence = confidence
        self.bounding_box = bounding_box or [0, 0, 0, 0]
        self.source_engine = source_engine

class BasicOCRManager:
    """Basic OCR manager as a fallback."""
    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
    def process_image(self, image_path: Path) -> List[OCRResult]:
        logger.warning("Basic OCR manager does not perform OCR.")
        return []
    def process_image_with_image(self, image_path: Path, preprocessed_img: Image.Image) -> List[OCRResult]:
        logger.warning("Basic OCR manager does not perform OCR.")
        return []

class EasyOCRManager:
    """Manager for the EasyOCR engine."""
    def __init__(self, reader, confidence_threshold: float):
        self.reader = reader
        self.confidence_threshold = confidence_threshold

    def process_image_with_image(self, image_path: Path, preprocessed_img: Image.Image) -> List[OCRResult]:
        try:
            import numpy as np
            img_array = np.array(preprocessed_img)
            results = self.reader.readtext(img_array)
            return [
                OCRResult(text, conf, bbox, "easyocr")
                for bbox, text, conf in results
                if conf >= self.confidence_threshold
            ]
        except Exception as e:
            logger.error(f"EasyOCR error for {image_path.name}: {e}")
            return []

class TesseractManager:
    """Manager for the Tesseract OCR engine."""
    def __init__(self, pytesseract, confidence_threshold: float):
        self.pytesseract = pytesseract
        self.confidence_threshold = confidence_threshold

    def process_image_with_image(self, image_path: Path, preprocessed_img: Image.Image) -> List[OCRResult]:
        try:
            config = f'--oem 3 --psm 6 -l {"+".join(TESSERACT_LANGUAGES)}'
            data = self.pytesseract.image_to_data(preprocessed_img, config=config, output_type=self.pytesseract.Output.DICT)
            results = []
            for i in range(len(data['text'])):
                conf = float(data['conf'][i])
                text = data['text'][i].strip()
                if text and conf > (self.confidence_threshold * 100):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    results.append(OCRResult(text, conf / 100.0, bbox, "tesseract"))
            return results
        except Exception as e:
            logger.error(f"Tesseract error for {image_path.name}: {e}")
            return []


class HybridOCRManager:
    """Combines multiple OCR engines and intelligently merges results."""
    def __init__(self, confidence_threshold: float, use_gpu: bool):
        self.confidence_threshold = confidence_threshold
        self.easyocr_manager = self._init_easyocr(use_gpu)
        self.tesseract_manager = self._init_tesseract()

    def _init_easyocr(self, use_gpu: bool):
        try:
            import easyocr
            langs = EASYOCR_LANGUAGES
            try:
                return EasyOCRManager(easyocr.Reader(langs, gpu=use_gpu), self.confidence_threshold)
            except Exception:
                logger.warning(f"EasyOCR failed with {langs}, trying alternative {EASYOCR_LANGUAGES_ALT}")
                return EasyOCRManager(easyocr.Reader(EASYOCR_LANGUAGES_ALT, gpu=use_gpu), self.confidence_threshold)
        except ImportError:
            logger.warning("EasyOCR not installed, HybridManager will not use it.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR in HybridManager: {e}")
            return None

    def _init_tesseract(self):
        try:
            import pytesseract
            return TesseractManager(pytesseract, self.confidence_threshold)
        except ImportError:
            logger.warning("Pytesseract not installed, HybridManager will not use it.")
            return None

    def process_image_with_image(self, image_path: Path, preprocessed_img: Image.Image) -> List[OCRResult]:
        all_results = []
        if self.easyocr_manager:
            all_results.extend(self.easyocr_manager.process_image_with_image(image_path, preprocessed_img))
        if self.tesseract_manager:
            all_results.extend(self.tesseract_manager.process_image_with_image(image_path, preprocessed_img))

        # Basic deduplication for now; more advanced merging can be added here.
        seen_texts = set()
        unique_results = []
        for res in sorted(all_results, key=lambda x: x.confidence, reverse=True):
            if res.text not in seen_texts:
                unique_results.append(res)
                seen_texts.add(res.text)
        return unique_results
