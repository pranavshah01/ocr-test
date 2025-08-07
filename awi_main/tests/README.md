# OCR DOCX Text Replacement Utility - Test Suite

This directory contains comprehensive tests for all implemented features of the OCR DOCX Text Replacement Utility.

## Test Files

### `test_pattern_matching.py`
**Task 2.1: Enhanced Pattern Matching**
- Tests pattern detection across split XML tags
- Validates Unicode handling and normalization
- Tests XML structure preservation
- Verifies no false positives or missed matches

### `test_enhanced_mapping.py`
**Task 2.2: Mapping & Replacement Logic**
- Tests enhanced mapping processor functionality
- Validates pattern-to-mapping integration
- Tests both "replace" and "append" modes
- Verifies exact match processing only

### `test_formatting_preservation.py`
**Task 2.3: Formatting Preservation**
- Tests font family, size, color preservation
- Validates text style preservation (bold, italic, underline, etc.)
- Tests paragraph alignment and spacing
- Verifies complex nested formatting handling

### `test_ocr_modes.py`
**OCR Modes Implementation**
- Tests three OCR modes: replace, append, append-image
- Validates OCR engine integration (EasyOCR/Tesseract)
- Tests image text replacement functionality
- Verifies configuration validation

## Running Tests

### Run All Tests
```bash
# From the awi_main directory
cd /Users/pranavshah/pranav_projects/new-ocr-project/awi_main
python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# Pattern matching tests
python tests/test_pattern_matching.py

# Enhanced mapping tests
python tests/test_enhanced_mapping.py

# Formatting preservation tests
python tests/test_formatting_preservation.py

# OCR modes tests
python tests/test_ocr_modes.py
```

## Test Requirements

### Dependencies
- Python 3.8+
- lxml
- pathlib
- json
- re
- datetime

### Optional Dependencies (for full OCR testing)
- easyocr
- pytesseract
- opencv-python
- pillow
- numpy

### Test Data
Some tests require:
- `patterns.json` - Regex patterns for matching
- `mapping.json` - Text replacement mappings
- `test_documents/sample_document.docx` - Sample DOCX file (optional)

## Test Results

The test suite validates:

✅ **Enhanced Pattern Matching (Task 2.1)**
- Cross-tag pattern detection
- Unicode normalization
- XML structure preservation

✅ **Mapping & Replacement Logic (Task 2.2)**
- Pattern-to-mapping integration
- Replace and append modes
- Exact match processing

✅ **Formatting Preservation (Task 2.3)**
- Font properties preservation
- Text style maintenance
- Paragraph formatting
- Complex nested formatting

✅ **OCR Modes Implementation**
- Three OCR processing modes
- Engine integration and fallback
- Image text replacement

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the `awi_main` directory
   - Check that the `app` directory is accessible

2. **Missing Dependencies**
   - Install required packages: `pip install lxml`
   - For OCR tests: `pip install easyocr pytesseract opencv-python pillow numpy`

3. **Test File Not Found**
   - Some tests gracefully handle missing test documents
   - Check that `patterns.json` and `mapping.json` exist in the root directory

4. **OCR Engine Issues**
   - EasyOCR requires internet connection for first-time model download
   - Tesseract requires system installation
   - Tests will skip OCR functionality if engines are unavailable

## Test Coverage

The test suite covers:
- ✅ Configuration validation
- ✅ Pattern matching algorithms
- ✅ Mapping and replacement logic
- ✅ Formatting preservation
- ✅ OCR integration
- ✅ Error handling and edge cases
- ✅ Cross-platform compatibility

## Contributing

When adding new features:
1. Create corresponding test files in this directory
2. Follow the naming convention: `test_<feature_name>.py`
3. Update this README with test descriptions
4. Ensure tests are included in `run_all_tests.py`
