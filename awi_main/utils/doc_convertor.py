"""Document conversion utilities.

Handles conversion between supported document formats.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

def convert_doc_to_docx(doc_path: str) -> str:
    """
    Convert a .doc file to .docx using LibreOffice CLI.

    Args:
        doc_path (str): Path to the input .doc file.

    Returns:
        str: Path to the converted .docx file.

    Raises:
        ValueError: If input path does not end with .doc.
        RuntimeError: If conversion fails.
    """
    if not doc_path.lower().endswith('.doc') or doc_path.lower().endswith('.docx'):
        raise ValueError("Input file must have a .doc extension (not .docx)")

    input_dir = os.path.dirname(os.path.abspath(doc_path))
    base_name = os.path.splitext(os.path.basename(doc_path))[0]
    output_path = os.path.join(input_dir, f"{base_name}.docx")

    cmd = [
        'soffice',
        '--headless',
        '--convert-to', 'docx',
        '--outdir', input_dir,
        doc_path
    ]

    logger.info(f"Converting {doc_path} to {output_path} using LibreOffice CLI...")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False
        )
        if result.returncode != 0:
            logger.error(f"LibreOffice conversion failed: {result.stderr.strip()}")
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr.strip()}")

        logger.info(f"Conversion successful: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Exception during document conversion: {e}")
        raise