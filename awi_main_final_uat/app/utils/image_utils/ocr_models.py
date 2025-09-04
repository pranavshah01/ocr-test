from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bounding_box
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
        }


@dataclass
class HybridOCRResult:
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    source_engine: str
    easyocr_result: Optional[OCRResult] = None
    tesseract_result: Optional[OCRResult] = None
    selection_reason: str = ""
    conflict_resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bounding_box
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
            'source_engine': self.source_engine,
            'easyocr_result': self.easyocr_result.to_dict() if self.easyocr_result else None,
            'tesseract_result': self.tesseract_result.to_dict() if self.tesseract_result else None,
            'selection_reason': self.selection_reason,
            'conflict_resolved': self.conflict_resolved
        }


@dataclass
class OCRMatch:
    ocr_result: OCRResult
    pattern: str
    replacement_text: str
    image_path: Path
    processing_mode: str
    extracted_pattern_text: str = ""
    wipe_boundaries: Optional[Tuple[int, int]] = None
    calculated_text_boundary: Optional[Tuple[int, int, int, int]] = None
    wipe_area_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ocr_result': self.ocr_result.to_dict(),
            'pattern': self.pattern,
            'replacement_text': self.replacement_text,
            'image_path': str(self.image_path),
            'processing_mode': self.processing_mode,
            'extracted_pattern_text': self.extracted_pattern_text,
            'wipe_boundaries': self.wipe_boundaries,
            'calculated_text_boundary': self.calculated_text_boundary,
            'wipe_area_info': self.wipe_area_info
        }


def create_ocr_result(text: str, confidence: float, bbox: Tuple[int, int, int, int]) -> OCRResult:
    return OCRResult(text=text, confidence=confidence, bounding_box=bbox)


def create_hybrid_ocr_result(text: str, confidence: float, bbox: Tuple[int, int, int, int],
                             source_engine: str, easyocr_result: Optional[OCRResult] = None,
                             tesseract_result: Optional[OCRResult] = None, 
                             selection_reason: str = "", conflict_resolved: bool = False) -> HybridOCRResult:
    return HybridOCRResult(
        text=text,
        confidence=confidence,
        bounding_box=bbox,
        source_engine=source_engine,
        easyocr_result=easyocr_result,
        tesseract_result=tesseract_result,
        selection_reason=selection_reason,
        conflict_resolved=conflict_resolved
    )
