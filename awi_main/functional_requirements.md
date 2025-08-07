# OCR-Based DOCX Text Replacement Utility

## 🧠 Objective

Build a Python-based utility that:
- Processes `.docx` and `.doc` files
- Finds patterns defined in `patterns.json` and maps them to replacements from `mapping.json`
- Replaces or appends text in body, headers, footers, textboxes, callout boxes, and images
- Preserves formatting, layout, and location fidelity in all cases
- Handles both traditional text and text embedded in images via OCR

## 📂 Inputs

### 1. `mapping.json`
```json
{
  "77-abc-def-ghi": "Revised Text 1",
  "77-xyz-123-qwe": "Updated Term 2"
}
```

### 2. `patterns.json`
```json
{
  "pattern_77": "77-[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+"
}
```

### 3. `text_mode` (CLI/config param)
- `"append"` → append `to_text` after `from_text`
- `"replace"` → replace `from_text` with `to_text`

### 4. `.docx` or `.doc` Files
- `.doc` files must be **auto-converted** to `.docx`

## 🗃️ Processing Flow

### 1. Text Detection & Replacement

#### a. Pattern Matching
- Reconstruct text across split `<w:t>` tags
- Normalize and match patterns defined in `patterns.json`
- Extract matches and map using `mapping.json`

#### b. Multi-language Boundary Handling
- Support for any Unicode character before/after `from_text`
- Only replace exact matches

#### c. Replacement Logic
- `"replace"`: Replace `from_text` with `to_text`
- `"append"`: Insert `to_text` after `from_text` with space

#### d. Formatting Preservation
- Match: Font family, size, color, style, alignment

### 2. Textboxes and Callout Boxes

- Normalize all font sizes to smallest in shape
- Resize to fit if text overflows after replacement
- Maintain layout, size, alignment

### 3. OCR-Based Image Text Handling

#### a. OCR Engine
- Use EasyOCR, fallback to Tesseract

#### b. Match Criteria
- Use `patterns.json` to detect text
- Extract position, size, orientation

#### c. Replacement Rules

| Mode | Behavior |
|------|----------|
| replace | Replace text at exact location |
| append | Duplicate image below and apply change there |

#### d. Rendering
- Use OpenCV to draw text at correct position
- Preserve font size, color, orientation

## 💾 Output

- Modified files saved to `/processed/`
- File names include `_processed`

## 📑 Logging

Log for each file includes:
- Match text and replacement
- Location, font info, bounding boxes (if image)
- Mode used

## ⚠️ Constraints

- Patterns may span multiple `<w:t>` tags
- Images may include rotated text
- Tool must work cross-platform
- Must support large files

## 🧪 Edge Case Example

```xml
<w:t>编号</w:t>
<w:t>77-ab</w:t><w:t>c-def</w:t><w:t>-ghi!</w:t>
```
Detected Match: `77-abc-def-ghi` → `"My Replacement Text"`

Output:
```xml
<w:t>编号</w:t><w:t>My Replacement Text</w:t><w:t>!</w:t>
```


## ✅ Final Deliverables

- CLI tool with pattern + mapping support
- Image OCR and rendering support
- Full formatting preservation
- Logging for traceability