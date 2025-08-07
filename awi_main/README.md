# OCR Replacer Usage

## Installation

1. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Install system dependencies:**

   - **Tesseract OCR:**
     - macOS: `brew install tesseract`
     - Ubuntu: `sudo apt-get install tesseract-ocr`
     - Windows: [Download installer](https://github.com/tesseract-ocr/tesseract)

   - **LibreOffice (for .doc conversion on Mac/Linux):**
     - macOS: `brew install --cask libreoffice`
     - Ubuntu: `sudo apt-get install libreoffice`
     - Windows: (not required, uses win32com)

   - **Optional (for unoconv):**
     - `sudo apt-get install unoconv` (Linux)

   - **GPU OCR:**
     - Ensure CUDA drivers and PyTorch installed for EasyOCR GPU support.

## Running the CLI

Place your `.doc` and `.docx` files in the current directory.

```
python -m ocr_replacer.cli [--patterns patterns.json] [--ocr-engine easyocr|tesseract] [--workers N] [--ocr-mode overlay|duplicate] [--gpu|--no-gpu]
```

Outputs are written to:
- `./processed` (processed docs)
- `./reports` (JSON reports & HTML diffs)
- `./logs` (logs per file and all.log)

## Performance Tuning

- Default workers: min(available CPUs – 2, 8)
- Use `--workers N` to override.
- For GPU OCR: use `--gpu` (default if GPU is detected and supported).
- For CPU OCR: use `--no-gpu`.

## GPU OCR Tips

- EasyOCR will use CUDA GPU if available; else fall back to CPU.
- For best performance, ensure NVIDIA drivers and CUDA toolkit are installed.
- You can check GPU recognition in logs.

## Note

- All regex patterns and replacement mappings must be provided in `patterns.json` in the working directory.
- For additional options, use `python -m app.run --help`