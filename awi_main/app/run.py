"""
CLI entrypoint for OCR DOCX text replacer.
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List
from functools import partial
import json

from loguru import logger

from . import doc_converter, docx_processor, report, diff_generator, text_extractor
from .comprehensive_image_detector import ComprehensiveImageDetector
from .enhanced_textbox_ocr_processor import EnhancedTextboxOCRProcessor
from .config import OCRConfig
from .shared_constants import SharedUtilities


def process_file(file_path: Path, config_data: dict, mapping_dict, patterns_list):
    """
    Processes a single DOCX file through the conversion, replacement, and reporting pipeline.
    """
    config = OCRConfig(**config_data)
    per_file_log = SharedUtilities.setup_logger(f"{file_path.name}_processor")
    per_file_log.info(f"Processing {file_path} ...")

    try:
        docx_path = doc_converter.convert(file_path)
        # Enhanced processing pipeline now handles all components (text, OCR, textboxes, sections)
        summary = docx_processor.process_docx(
            docx_path, 
            mapping_dict, 
            patterns_list,
            text_mode=config.text_mode,
            ocr_mode=config.ocr_mode,
            ocr_engine=config.ocr_engine,
            confidence_min=config.confidence_min,
            gpu=config.gpu,
            process_images=True
        )
        
        # Add file information to summary
        summary.update({
            "file": str(file_path.name)
        })

        report.write_report(summary, config.reports_dir)

        try:
            text_result = text_extractor.create_combined_text_file(
                docx_path=docx_path,
                ocr_dir=config.ocr_dir,
                file_name=file_path.stem,
                ocr_engine=config.ocr_engine,
                gpu=config.gpu,
                confidence_min=config.confidence_min
            )
            summary["combined_text_file"] = text_result.get("text_file")
            summary["combined_json_file"] = text_result.get("json_file")
            per_file_log.info(f"Created combined text files: {summary['combined_text_file']}")
        except Exception as text_err:
            per_file_log.warning(f"Combined text extraction failed for {file_path}: {text_err}")

        try:
            diff_generator.generate_html_diff(file_path, docx_path, config.reports_dir)
        except Exception as diff_err:
            per_file_log.warning(f"Diff generation failed for {file_path}: {diff_err}")

        per_file_log.info(f"Finished {file_path}")

    except Exception as e:
        per_file_log.error(f"Error processing {file_path}: {e}")

    return file_path


def main():
    """
    Command-line interface for the OCR DOCX Text Replacer.
    Discovers input files, loads patterns, and processes each document in parallel.
    """
    config = OCRConfig.from_cli_args()
    logger = SharedUtilities.setup_logger("ocr_docx_main")
    logger.info("Starting OCR DOCX replacement CLI...")

    source_path = Path.cwd() / "source_documents"
    
    # Use enhanced file discovery with error handling
    try:
        input_files = doc_converter.discover_and_convert_files(
            source_path, 
            recursive=True,  # Enable recursive search
            extensions=None  # Use default ['.doc', '.docx']
        )
        logger.info(f"Successfully processed {len(input_files)} files: {[f.name for f in input_files]}")
    except FileNotFoundError:
        logger.error(f"Source directory not found: {source_path}")
        logger.info("Please ensure the 'source_documents' directory exists and contains .doc or .docx files")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"File processing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during file discovery/conversion: {e}")
        sys.exit(1)

    try:
        with open(config.mapping_path, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mapping file: {e}")
        sys.exit(2)

    try:
        with open(config.patterns_path, "r", encoding="utf-8") as f:
            patterns_data = json.load(f)
            
            # Handle both list format and object format
            if isinstance(patterns_data, list):
                patterns_list = patterns_data
            elif isinstance(patterns_data, dict):
                # Extract pattern values, excluding metadata
                patterns_list = [
                    value for key, value in patterns_data.items() 
                    if not key.startswith('_') and isinstance(value, str)
                ]
                logger.info(f"Loaded {len(patterns_list)} patterns from object format")
            else:
                raise ValueError("Patterns file must contain either a list of regex strings or an object with pattern values.")
                
            if not patterns_list:
                raise ValueError("No valid patterns found in patterns file.")
                
    except Exception as e:
        logger.error(f"Failed to load patterns file: {e}")
        sys.exit(2)

    config_data = config.__dict__
    n_workers = config.workers
    logger.info(f"Using {n_workers} workers.")

    process_file_fn = partial(
        process_file,
        config_data=config_data,
        mapping_dict=mapping_dict,
        patterns_list=patterns_list
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(process_file_fn, input_files), total=len(input_files)))

    report.write_master_report(config.reports_dir)


if __name__ == "__main__":
    main()
