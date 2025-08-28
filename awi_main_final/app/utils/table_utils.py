"""
Table processing utilities for handling vMerge and other table-specific features.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from docx.table import Table, _Cell

logger = logging.getLogger(__name__)

def should_process_cell(cell: _Cell, row_idx: int, cell_idx: int) -> bool:
    """
    Determine if a cell should be processed based on its position and vMerge status.
    
    This function uses a row-based approach to handle vMerge correctly.
    For the first row, cells with vMerge are restart cells and should be processed.
    For subsequent rows, cells with vMerge are continue cells and should be skipped.
    
    Args:
        cell: The cell to check
        row_idx: The row index (0-based)
        cell_idx: The cell index (0-based)
        
    Returns:
        True if the cell should be processed, False if it should be skipped
    """
    try:
        tc = cell._tc
        if hasattr(tc, 'tcPr') and tc.tcPr is not None:
            # Check if this cell has a vMerge element
            vmerge_elem = tc.tcPr.find('.//w:vMerge', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
            if vmerge_elem is not None:
                # This cell has vMerge
                if row_idx == 0:
                    # First row: vMerge cells are restart cells, should be processed
                    return True
                else:
                    # Subsequent rows: vMerge cells are continue cells, should be skipped
                    return False
        
        # No vMerge: always process
        return True
    except Exception as e:
        logger.debug(f"Error checking cell processing: {e}")
        return True

def get_table_cells_to_process(table: Table) -> List[Tuple[int, int, _Cell]]:
    """
    Get a list of cells that should be processed, considering vMerge.
    
    Args:
        table: The table to process
        
    Returns:
        List of tuples containing (row_index, cell_index, cell) for cells that should be processed
    """
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
    """
    Extract all text from a cell, considering its vMerge status.
    
    Args:
        cell: The cell to extract text from
        row_idx: The row index (0-based)
        cell_idx: The cell index (0-based)
        
    Returns:
        Combined text from all paragraphs in the cell
    """
    if not should_process_cell(cell, row_idx, cell_idx):
        return ""
    
    text_parts = []
    for paragraph in cell.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())
    
    return " ".join(text_parts)
