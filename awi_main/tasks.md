# OCR-Based DOCX Text Replacement Utility - Implementation Tasks

## 📋 Project Overview
Build a Python-based utility that processes `.docx` and `.doc` files, finds patterns defined in `patterns.json`, maps them to replacements from `mapping.json`, and replaces or appends text in body, headers, footers, textboxes, callout boxes, and images while preserving formatting and layout fidelity.

## 🎯 Key Requirements
- **Modes**: Only `text-mode` (append/replace) and `ocr-mode` (append/replace)
- **File Types**: Auto-convert `.doc` to `.docx`
- **Processing Areas**: Body, headers, footers, textboxes, callouts, images
- **OCR**: EasyOCR primary, Tesseract fallback
- **Output**: Modified files to `/processed/` with detailed logging
- **Cross-Platform**: Windows and macOS support

---

## 📝 Task Breakdown

### **Phase 1: Core Configuration & Setup**

#### Task 1.1: Update Configuration System ⭐ HIGH PRIORITY ✅ COMPLETED
- **File**: `app/config.py`
- **Description**: Modify configuration to enforce only required modes
- **Requirements**:
  - Remove unused OCR modes (overlay, duplicate, append2row)
  - Keep only `text-mode` and `ocr-mode` with values: `append`, `replace`
  - Add validation for required parameters
  - Update CLI argument parsing
- **Acceptance Criteria**:
  - Configuration only accepts `text-mode: [append|replace]`
  - Configuration only accepts `ocr-mode: [append|replace]`
  - Invalid modes throw clear error messages
  - All existing functionality preserved

#### Task 1.2: Enhance File Discovery & Conversion ✅ COMPLETED
- **File**: `app/doc_converter.py`
- **Description**: Improve .doc to .docx auto-conversion and file discovery
- **Requirements**:
  - Robust .doc to .docx conversion with error handling
  - Batch file discovery with proper filtering
  - Cross-platform path handling using `pathlib.Path`
  - Support for nested directory scanning
- **Acceptance Criteria**:
  - All .doc files automatically converted to .docx
  - Conversion preserves original formatting
  - Failed conversions logged with details
  - Supports Windows and macOS file paths

### **Phase 2: Text Processing Engine**

#### Task 2.1: Pattern Matching Across Split Tags ⭐ CRITICAL ✅ COMPLETED
- **File**: `app/text_extractor.py`
- **Description**: Reconstruct text across multiple `<w:t>` tags for accurate pattern matching
- **Requirements**:
  - Combine text from split `<w:t>` tags before pattern matching
  - Implement text normalization for consistent matching
  - Handle Unicode and multi-language boundaries
  - Preserve original tag structure for replacement
- **Acceptance Criteria**:
  - Patterns spanning multiple tags are detected
  - Unicode characters handled correctly
  - Original XML structure maintained
  - No false positives or missed matches

#### Task 2.2: Mapping & Replacement Logic ⭐ CRITICAL ✅ COMPLETED
- **File**: `app/docx_processor.py`
- **Description**: Implement exact pattern matching and mapping-based replacements
- **Requirements**:
  - Use `patterns.json` for regex pattern matching
  - Map matches using `mapping.json` lookup
  - Implement replace mode: replace `from_text` with `to_text`
  - Implement append mode: insert `to_text` after `from_text` with space
  - Handle exact matches only (no partial matches)
- **Acceptance Criteria**:
  - All patterns from `patterns.json` are matched correctly
  - Mappings from `mapping.json` applied accurately
  - Replace mode completely substitutes text
  - Append mode adds text with proper spacing
  - Only exact matches are processed

#### Task 2.3: Formatting Preservation ⭐ CRITICAL ✅ COMPLETED
- **File**: `app/docx_processor.py`
- **Description**: Ensure all formatting is preserved during text replacement
- **Requirements**:
  - Preserve font family, size, color, style
  - Maintain text alignment and spacing
  - Keep paragraph and section formatting
  - Handle complex nested formatting
- **Acceptance Criteria**:
  - Replaced text matches original formatting exactly
  - No formatting corruption or loss
  - Complex formatting (bold, italic, underline) preserved
  - Paragraph alignment maintained

### **Phase 3: Advanced Document Elements**

#### Task 3.1: Textboxes & Callout Boxes Enhancement ⭐ CRITICAL ✅ COMPLETED
- **File**: `app/image_callouts.py`
- **Description**: Improve handling of textboxes and callout boxes
- **Requirements**:
  - Normalize all font sizes to smallest in shape
  - Detect text overflow after replacement
  - Resize shapes to fit new text
  - Maintain layout, size, and alignment
- **Acceptance Criteria**:
  - Font sizes normalized within each shape
  - Text overflow handled gracefully
  - Shapes resize appropriately
  - Original layout preserved

#### Task 3.2: Headers, Footers & Document Sections ⭐ CRITICAL ✅ COMPLETED
- **File**: `app/text_extractor.py`, `app/docx_processor.py`
- **Description**: Extend processing to all document sections
- **Requirements**:
  - Process text in headers and footers
  - Handle different document sections consistently
  - Preserve section-specific formatting
  - Support multiple header/footer types
- **Acceptance Criteria**:
  - Headers and footers processed correctly
  - Section formatting maintained
  - All document parts included in processing
  - No section-specific errors

### **Phase 4: OCR Integration**

#### Task 4.1: OCR Engine Enhancement **[CRITICAL - COMPLETED]** ✅
**Priority:** Critical
**Status:** Completed
**Dependencies:** Phase 3 completion

**Objective:** Replace legacy OCR functions with enhanced OCR engine supporting EasyOCR/Tesseract fallback, confidence filtering, position/orientation extraction, and GPU acceleration.

**Requirements:**
- EasyOCR works as primary OCR engine with proper initialization and error handling
- Tesseract fallback functions properly when primary engine fails
- Position and orientation data captured for each text detection (angle, coordinates, dimensions)
- Confidence filtering prevents low-quality matches from being processed
- GPU acceleration utilized when available for improved performance

**Implementation Details:**
- Create `EnhancedOCREngine` class replacing legacy functions
- Support both EasyOCR and Tesseract with automatic fallback
- Extract text position, bounding box, orientation, and dimensions
- Implement confidence-based filtering with configurable thresholds
- Add GPU support detection and utilization
- Maintain backward compatibility with existing OCR calls

**Acceptance Criteria:**
- [x] EasyOCR works as primary OCR engine
- [x] Tesseract fallback functions properly
- [x] Position and orientation data captured
- [x] Confidence filtering prevents low-quality matches
- [x] GPU acceleration utilized when available

**Completion Summary:**
- ✅ Implemented `EnhancedOCREngine` class with comprehensive OCR functionality
- ✅ EasyOCR primary engine with Tesseract fallback logic and automatic engine detection
- ✅ GPU acceleration support with graceful CPU fallback
- ✅ Image preprocessing pipeline with 7 enhancement techniques (grayscale, contrast, sharp, inverted, etc.)
- ✅ **ENHANCED: Multi-angle rotation preprocessing** with 9 rotation angles (90°, 180°, 270°, ±15°, ±30°, ±45°)
- ✅ **ENHANCED: Smart orientation detection** with `detect_text_orientation()` method for optimization
- ✅ **ENHANCED: 16 image variants generated** per image for maximum text detection coverage
- ✅ Text orientation calculation and dimension extraction methods
- ✅ Confidence-based filtering with configurable thresholds
- ✅ Comprehensive statistics tracking and error handling
- ✅ Backward compatibility with legacy functions and convenience factory methods
- ✅ Full test suite validation with all acceptance criteria met

#### Task 4.2: Comprehensive Image Detection **[CRITICAL - COMPLETED]** ✅
**Priority:** Critical
**Status:** Completed
**Dependencies:** Task 4.1 completion

**Objective:** Replace limited inline_shapes approach with comprehensive ZIP extraction to find ALL images in DOCX files.

**Requirements:**
- Use ZIP extraction method to scan `word/media/` directory
- Process all image types: .png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp
- Handle inline images, floating images, headers/footers, text boxes, shapes
- Replace current `inline_shapes` only approach

**Acceptance Criteria:**
- [x] All images in document detected (not just inline)
- [x] ZIP extraction method implemented
- [x] Floating images and shapes processed
- [x] Header/footer images included
- [x] Background images handled

**Completion Summary:**
- ✅ Implemented `ComprehensiveImageDetector` class with ZIP extraction method
- ✅ **ZIP-based detection** finds ALL images in DOCX files (5 images detected in test)
- ✅ **Multi-location support**: word/media/, word/embeddings/, word/theme/, docProps/, customXml/
- ✅ **Advanced categorization**: Location-based (main_media, embedded_objects, theme_images) and type-based (.png, .jpg)
- ✅ **XML reference analysis** for image context and relationship mapping
- ✅ **OCR-ready image filtering** with size and transparency-based filtering
- ✅ **Comprehensive format support**: .png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp
- ✅ **Error handling and logging** for robust processing
- ✅ **Convenience functions**: detect_all_docx_images() and get_ocr_ready_images()
- ✅ Full test suite validation with all acceptance criteria met

#### Task 4.3: OCR Text Replacement/Append Logic **[CRITICAL - COMPLETED]** ✅
**Priority:** Critical
**Status:** Completed
**Dependencies:** Tasks 4.1 and 4.2 completion

**Objective:** Implement comprehensive OCR-based text replacement with OpenCV rendering supporting all three OCR modes.

**Requirements:**
- **Replace mode**: Replace from_text with to_text at exact position, preserving other text
- **Append mode**: Append to_text to from_text in two lines at same location
- **Append-image mode**: Create new image with replaced text and append after original
- Use OpenCV for high-quality text removal and rendering
- Preserve font size, color, and orientation
- Handle rotated text correctly

**Acceptance Criteria:**
- [x] Text replaced at exact original position
- [x] Append mode creates text in two lines
- [x] Append-image mode creates new images
- [x] OpenCV rendering preserves visual fidelity
- [x] Font properties maintained
- [x] Rotated text handled correctly

**Completion Summary:**
- ✅ Implemented `OCRTextReplacementProcessor` class with comprehensive OCR text replacement
- ✅ **Three OCR modes fully implemented**:
  - **Replace**: Text replaced at exact position with OpenCV inpainting (cv2.inpaint)
  - **Append**: Original + new text in two lines at same location with proper spacing
  - **Append-image**: New images created with replaced text for DOCX insertion
- ✅ **Advanced text processing**: Mapping and regex pattern support, confidence filtering, text normalization
- ✅ **Multiple bounding box format support**: EasyOCR, Tesseract coordinate systems handled
- ✅ **OpenCV rendering pipeline**: High-quality text removal and professional text rendering
- ✅ **Font system**: Cross-platform font fallbacks, size scaling based on original text bounds
- ✅ **Full integration**: Enhanced OCR engine + comprehensive image detector + text replacement
- ✅ **Processing statistics**: Comprehensive error handling and performance monitoring
- ✅ **Convenience functions**: process_docx_with_ocr() for easy DOCX processing
- ✅ Full test suite validation with all three modes tested and acceptance criteria met

### **Phase 5: Output & Reporting**

#### Task 5.1: Enhanced Logging System
- **File**: `app/logger.py`
- **Description**: Implement comprehensive per-file logging
- **Requirements**:
  - Log match text and replacement for each change
  - Include location, font info, bounding boxes (for images)
  - Track processing mode used
  - Generate structured log format
  - Support different log levels
- **Acceptance Criteria**:
  - Each file has detailed processing log
  - All matches and replacements recorded
  - Image bounding boxes logged
  - Processing modes tracked
  - Logs are human-readable and structured

#### Task 5.2: Report Generation
- **File**: `app/report.py`
- **Description**: Generate comprehensive processing reports
- **Requirements**:
  - Summary of all processed files
  - Statistics on matches and replacements
  - OCR confidence scores and image processing details
  - Visual diff generation for changes
  - Export to multiple formats (JSON, HTML, CSV)
- **Acceptance Criteria**:
  - Comprehensive processing summary
  - Detailed statistics provided
  - OCR metrics included
  - Visual diffs generated
  - Multiple export formats supported

### **Phase 6: Performance & Scalability**

#### Task 6.1: Large File Handling
- **File**: `app/large_file_handler.py`
- **Description**: Optimize for memory efficiency and large documents
- **Requirements**:
  - Implement streaming processing for very large documents
  - Add progress tracking and error recovery
  - Memory usage monitoring and optimization
  - Chunked processing for massive files
- **Acceptance Criteria**:
  - Large files (>100MB) processed efficiently
  - Memory usage stays within reasonable bounds
  - Progress tracking provides user feedback
  - Error recovery prevents data loss

#### Task 6.2: Parallel Processing Enhancement
- **File**: `app/run.py`
- **Description**: Improve multi-threading and worker management
- **Requirements**:
  - Optimize worker allocation based on system resources
  - Implement proper cross-platform multiprocessing
  - Add load balancing for different file sizes
  - Error isolation between workers
- **Acceptance Criteria**:
  - Optimal worker count automatically determined
  - Cross-platform multiprocessing works
  - Large files don't block small file processing
  - Worker errors don't crash entire process

### **Phase 7: Testing & Validation**

#### Task 7.1: Cross-Platform Testing
- **Description**: Ensure functionality across Windows and macOS
- **Requirements**:
  - Test file path handling on both platforms
  - Validate OCR engines on different systems
  - Test with various document types and sizes
  - Performance benchmarking
- **Acceptance Criteria**:
  - Identical functionality on Windows and macOS
  - All file paths resolve correctly
  - OCR engines work on both platforms
  - Performance meets requirements

#### Task 7.2: Integration Testing
- **Description**: End-to-end testing with real documents
- **Requirements**:
  - Test with sample documents from requirements
  - Validate output against expected results
  - Performance testing with large document sets
  - Edge case testing (corrupted files, unusual formats)
- **Acceptance Criteria**:
  - All sample documents process correctly
  - Output matches expected results
  - Performance meets scalability requirements
  - Edge cases handled gracefully

---

## 🚀 Implementation Priority

### **Critical Path (Week 1)**
1. Task 1.1: Update Configuration System
2. Task 2.1: Pattern Matching Across Split Tags
3. Task 4.2: Comprehensive Image Detection

### **High Priority (Week 2)**
4. Task 2.2: Mapping & Replacement Logic
5. Task 4.1: OCR Engine Enhancement
6. Task 4.3: OCR Text Replacement/Append

### **Medium Priority (Week 3)**
7. Task 2.3: Formatting Preservation
8. Task 3.1: Textboxes & Callout Boxes Enhancement
9. Task 5.1: Enhanced Logging System

### **Lower Priority (Week 4)**
10. Task 3.2: Headers, Footers & Document Sections
11. Task 5.2: Report Generation
12. Task 6.1: Large File Handling
13. Task 6.2: Parallel Processing Enhancement

### **Final Phase (Week 5)**
14. Task 7.1: Cross-Platform Testing
15. Task 7.2: Integration Testing

---

## 📊 Success Metrics

- **Functionality**: All patterns from `patterns.json` detected and replaced correctly
- **Coverage**: All document elements (body, headers, footers, textboxes, images) processed
- **Accuracy**: OCR confidence > 70% for text replacements
- **Performance**: Process 100+ page documents in < 5 minutes
- **Reliability**: 99%+ success rate on valid input documents
- **Cross-Platform**: Identical results on Windows and macOS

---

## 🔧 Technical Notes

### **Key Dependencies**
- `python-docx`: Document manipulation
- `EasyOCR`: Primary OCR engine
- `pytesseract`: Fallback OCR engine  
- `OpenCV`: Image processing and text rendering
- `lxml`: XML parsing for advanced document manipulation

### **Architecture Decisions**
- **Modular Design**: Each phase maps to specific modules
- **Configuration-Driven**: Single source of truth in `config.py`
- **Error Handling**: Graceful degradation with detailed logging
- **Memory Efficiency**: Streaming for large files

### **Testing Strategy**
- **Unit Tests**: Each module tested independently
- **Integration Tests**: End-to-end document processing
- **Performance Tests**: Large file and batch processing
- **Cross-Platform Tests**: Windows and macOS validation
