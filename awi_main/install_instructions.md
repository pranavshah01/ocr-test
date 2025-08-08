# Installation & Verification Guide for awi_main

Follow these instructions to set up your development environment for the `awi_main` project.

---

## 1. Create and Activate a Python Virtual Environment

**On macOS & Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

## 2. Install Project Dependencies

All required dependencies are listed in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify Python Package Imports

Open a Python REPL and run:

```python
import docx
from PIL import Image
import cv2
import easyocr
import pytesseract
import regex
import jsonschema

# The following are standard library modules (should always succeed):
import json
import pathlib

print("All dependencies imported successfully!")
```

If any import fails, ensure you've activated your virtual environment and installed all requirements.

---

## 4. Install LibreOffice CLI

LibreOffice CLI is required for document conversion (e.g., `.docx` to `.pdf`).

**Detect your OS (optional):**
```python
import platform
print(platform.system())
```

### Installation Instructions:

#### **Windows**
- Download LibreOffice from: https://www.libreoffice.org/download/download/
- Ensure `soffice` is added to your PATH (typically found in `C:\Program Files\LibreOffice\program\soffice.exe`).

#### **macOS**
```bash
brew install --cask libreoffice
```
- If you don't have Homebrew: https://brew.sh
- After installation, add `soffice` to your PATH if needed.

#### **Linux (Debian/Ubuntu)**
```bash
sudo apt-get update
sudo apt-get install libreoffice
```

#### **Linux (Fedora/RHEL/CentOS)**
```bash
sudo dnf install libreoffice
```
or
```bash
sudo yum install libreoffice
```

---

## 5. Test OCR Functionality

Below is a Python snippet demonstrating OCR using both EasyOCR and pytesseract on a sample image.

```python
# Replace 'sample.jpg' with the path to your image file
IMAGE_PATH = 'sample.jpg'

# EasyOCR Example
import easyocr
reader = easyocr.Reader(['en'])
results = reader.readtext(IMAGE_PATH)
print("EasyOCR Results:")
for bbox, text, conf in results:
    print(f"Text: {text} (Confidence: {conf:.2f})")

# pytesseract Example
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open(IMAGE_PATH))
print("pytesseract Results:")
print(text)
```

> **Note:** The above assumes `sample.jpg` exists in your working directory. Replace with your image as needed.

---

## Troubleshooting

- If `pytesseract` fails, ensure Tesseract OCR is installed on your system and available in your PATH.
  - **macOS:** `brew install tesseract`
  - **Ubuntu:** `sudo apt-get install tesseract-ocr`
  - **Windows:** Download from https://github.com/tesseract-ocr/tesseract

---

You're now ready to use the awi_main project!