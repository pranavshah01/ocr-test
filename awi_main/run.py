"""Entry point for the AWI system.

This script launches the main application workflow.
"""

import argparse
import logging
from document_processor import process_documents

def main():
    parser = argparse.ArgumentParser(description="AWI Document Processor")
    parser.add_argument(
        '--input', '-i', nargs='+', required=True,
        help="Input file path(s) to process"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    input_files = args.input
    logging.info(f"AWI startup. Received input files: {input_files}")

    process_documents(input_files)

if __name__ == '__main__':
    main()