import sys
from pathlib import Path
import logging

from app.utils.image_utils.hybrid_ocr_manager import HybridOCRManager

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(image_path: str):
    p = Path(image_path)
    if not p.exists():
        print(f"Image not found: {p}")
        sys.exit(1)

    # Lower threshold slightly to see borderline detections; disable GPU for simplicity
    mgr = HybridOCRManager(confidence_threshold=0.3, use_gpu=False)
    results = mgr.process_hybrid(p)
    print(f"Detections: {len(results)}")
    for i, r in enumerate(results):
        engine = getattr(r, 'source_engine', getattr(r, '_engine', 'hybrid'))
        reason = getattr(r, 'selection_reason', '')
        print(f"{i}: text={r.text!r}, conf={r.confidence:.3f}, bbox={r.bounding_box}, engine={engine} {reason}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m app.tools.ocr_single_image /absolute/path/to/image")
        sys.exit(2)
    main(sys.argv[1])
