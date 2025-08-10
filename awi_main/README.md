# Document Processing Pipeline

A comprehensive, cross-platform document processing system that automatically processes Microsoft Word documents (.doc/.docx) by applying text replacements, handling graphics/textboxes, and performing OCR-based image text replacement.

## 🚀 Features

### Core Functionality
- **Text Processing**: Pattern matching and replacement in document body text with font preservation
- **Graphics Processing**: Textbox and callout processing with font normalization and overflow handling
- **Image Processing**: OCR-based text detection and replacement with GPU acceleration
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **Parallel Processing**: Multi-threaded document processing for improved performance
- **Comprehensive Reporting**: JSON and HTML reports with detailed processing statistics

### Advanced Features
- **GPU Acceleration**: EasyOCR with CUDA/MPS support for faster image processing
- **Font Preservation**: Maintains original font attributes during text replacement
- **Overflow Detection**: Automatic font resizing for textbox content
- **Multiple Processing Modes**: Append or replace modes for text and OCR processing
- **Error Handling**: Graceful degradation with detailed error reporting
- **Document Integrity**: Validation to ensure processed documents remain uncorrupted

## 📋 Requirements

### System Requirements
- Python 3.10+
- macOS, Windows, or Linux
- 4GB+ RAM recommended
- GPU (optional, for accelerated OCR processing)

### Dependencies
All dependencies are managed through `requirements.txt` and installed automatically with uv:
- `python-docx` - Word document processing
- `easyocr` - OCR text extraction with GPU support
- `pytesseract` - Alternative OCR engine
- `Pillow` - Image processing
- `opencv-python` - Computer vision operations
- `torch` - GPU acceleration support

## 🛠 Installation

1. **Clone or download the project**
2. **Navigate to the project directory**
   ```bash
   cd awi_main
   ```

3. **Set up virtual environment with uv**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

4. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage
```bash
python -m app.run
```

### Advanced Usage
```bash
# Process with specific modes
python -m app.run --text-mode replace --ocr-mode append

# Enable GPU acceleration
python -m app.run --gpu

# Adjust parallel processing
python -m app.run --workers 4

# Verbose logging
python -m app.run --verbose

# Custom directories
python -m app.run --source-dir /path/to/docs --output-dir /path/to/output
```

### Command Line Options
- `--text-mode {append,replace}` - Text processing mode
- `--ocr-mode {replace,append}` - OCR processing mode  
- `--gpu` / `--no-gpu` - Enable/disable GPU acceleration
- `--workers N` - Number of parallel workers
- `--confidence-min X` - Minimum OCR confidence threshold
- `--verbose` - Enable detailed logging
- `--patterns FILE` - Custom patterns JSON file
- `--mapping FILE` - Custom mappings JSON file

## 📁 Project Structure

```
awi_main/
├── config.py                    # Main configuration
├── patterns.json               # Regex patterns for matching
├── mapping.json                # Text replacement mappings
├── requirements.txt            # Python dependencies
├── validate_system.py          # System validation script
├── app/
│   ├── run.py                  # Main entry point
│   ├── core/                   # Core processing modules
│   │   ├── document_processor.py
│   │   ├── parallel_manager.py
│   │   └── models.py
│   ├── processors/             # Content processors
│   │   ├── text_processor.py
│   │   ├── graphics_processor.py
│   │   └── image_processor.py
│   ├── converters/             # Format converters
│   │   └── doc_converter.py
│   └── utils/                  # Utilities
│       ├── shared_constants.py
│       ├── platform_utils.py
│       ├── docx_utils.py
│       └── report_generator.py
├── source_documents/           # Input documents (.doc/.docx)
├── processed/                  # Output processed documents
├── reports/                    # Processing reports (JSON/HTML)
└── logs/                      # System logs
```

## 🔧 Configuration

### Pattern Matching
Edit `patterns.json` to define regex patterns for text matching:
```json
{
  "pattern_name": "(?<!\\w)77[\\s-]*[0-9]{3}[\\s-]*[0-9]{6}[\\s-]*[0-9]{3}(?!\\w)"
}
```

### Text Mappings
Edit `mapping.json` to define text replacements:
```json
{
  "77-110-810096-000": "4022-110-810096-000",
  "77-130-120541-001": "4022-172-240862-36"
}
```

## 📊 Processing Modes

### Text Processing Modes
- **Append Mode**: Adds replacement text after original text
  - `"77-110-810096-000"` → `"77-110-810096-000 4022-110-810096-000"`
- **Replace Mode**: Replaces original text with replacement text
  - `"77-110-810096-000"` → `"4022-110-810096-000"`

### OCR Processing Modes
- **Replace Mode**: Overlays white rectangle and renders new text
- **Append Mode**: Shows both original and replacement text

## 📈 Performance

### Parallel Processing
- Automatic worker count detection (CPU cores - 2, max 8)
- Thread-based parallelism for optimal resource sharing
- Error isolation prevents single document failures from affecting batch

### GPU Acceleration
- **CUDA**: NVIDIA GPU support
- **MPS**: Apple Silicon GPU support  
- **CPU Fallback**: Automatic fallback when GPU unavailable

## 📋 Validation

Run the system validation script to verify functionality:
```bash
python validate_system.py
```

This validates:
- ✅ Configuration loading
- ✅ Document processing
- ✅ Report generation  
- ✅ GPU functionality

## 🐛 Troubleshooting

### Common Issues

1. **No conversion tools detected**
   - Install LibreOffice for .doc file support
   - Only affects .doc files; .docx files work without conversion tools

2. **GPU not available**
   - System automatically falls back to CPU processing
   - Install appropriate GPU drivers for acceleration

3. **OCR confidence too low**
   - Adjust `--confidence-min` parameter
   - Use higher quality images for better OCR results

4. **Font issues in textboxes**
   - Graphics processor automatically normalizes fonts
   - Overflow detection adjusts font sizes as needed

### Log Files
Check `logs/processing.log` for detailed processing information and error messages.

## 🎯 Validation Results

The system has been validated with the following test results:
- ✅ **Configuration**: Patterns and mappings loaded correctly
- ✅ **Document Processing**: Text replacements working correctly
- ✅ **Report Generation**: JSON and HTML reports generated successfully
- ✅ **GPU Functionality**: MPS GPU acceleration working on Apple Silicon

## 🏗 Architecture

The system follows a modular pipeline architecture:

1. **Document Discovery**: Automatic file discovery and filtering
2. **Format Conversion**: .doc to .docx conversion when needed
3. **Parallel Processing**: Multi-threaded document processing
4. **Content Processing**: Text, graphics, and image processing in sequence
5. **Report Generation**: Comprehensive JSON and HTML reporting
6. **Validation**: Document integrity and processing validation

## 📝 License

This project is part of the Document Processing Pipeline implementation.

## 🤝 Contributing

The system is designed to be modular and extensible. Key extension points:
- Add new OCR engines in `image_processor.py`
- Add new document formats in `doc_converter.py`
- Add new processing modes in individual processors
- Extend reporting formats in `report_generator.py`