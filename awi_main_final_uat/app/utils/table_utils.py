
import logging
from typing import List, Tuple

from docx.table import Table, _Cell

logger = logging.getLogger(__name__)

def should_process_cell(cell: _Cell, row_idx: int, cell_idx: int) -> bool:
    try:
        tc = cell._tc
        if hasattr(tc, 'tcPr') and tc.tcPr is not None:

            vmerge_elem = tc.tcPr.find('.//w:vMerge', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
            if vmerge_elem is not None:

                if row_idx == 0:

                    return True
                else:

                    return False


        return True
    except Exception as e:
        logger.debug(f"Error checking cell processing: {e}")
        return True

def get_table_cells_to_process(table: Table) -> List[Tuple[int, int, _Cell]]:
    cells_to_process = []

    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if should_process_cell(cell, row_idx, cell_idx):
                cells_to_process.append((row_idx, cell_idx, cell))
                logger.debug(f"Will process cell at row {row_idx}, cell {cell_idx}")
            else:
                logger.debug(f"Skipping vMerge continue cell at row {row_idx}, cell {cell_idx}")

    return cells_to_process

def extract_text_from_cell(cell: _Cell, row_idx: int, cell_idx: int) -> str:
    if not should_process_cell(cell, row_idx, cell_idx):
        return ""

    text_parts = []
    for paragraph in cell.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())

    return " ".join(text_parts)