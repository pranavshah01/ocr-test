#!/usr/bin/env python3
"""
Simple diagnostic script to identify import issues.
"""

import sys
from pathlib import Path

print("🔍 Starting Import Diagnostics")
print("=" * 40)

# Add app to path
app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(app_path))
print(f"✅ Added to path: {app_path}")

# Test imports one by one
imports_to_test = [
    ("shared_constants", "SharedUtilities"),
    ("report", "EnhancedReportGenerator"),
    ("text_extractor", "EnhancedPatternMatcher"),
    ("ocr_engine", "EnhancedOCREngine"),
    ("comprehensive_image_detector", "ComprehensiveImageDetector"),
    ("ocr_text_replacement", "OCRTextReplacementProcessor"),
    ("enhanced_textbox_ocr_processor", "EnhancedTextboxOCRProcessor"),
    ("formatting_preservation", "FormattingPreserver"),
    ("document_sections_processor", "DocumentSectionsProcessor"),
    ("docx_processor", "process_docx")
]

successful_imports = []
failed_imports = []

for module_name, class_name in imports_to_test:
    try:
        print(f"Testing import: {module_name}.{class_name}...")
        module = __import__(module_name, fromlist=[class_name])
        getattr(module, class_name)
        print(f"  ✅ SUCCESS: {module_name}.{class_name}")
        successful_imports.append((module_name, class_name))
    except Exception as e:
        print(f"  ❌ FAILED: {module_name}.{class_name} - {e}")
        failed_imports.append((module_name, class_name, str(e)))

print("\n" + "=" * 40)
print("📊 IMPORT SUMMARY")
print(f"✅ Successful: {len(successful_imports)}")
print(f"❌ Failed: {len(failed_imports)}")

if failed_imports:
    print("\n🚨 FAILED IMPORTS:")
    for module_name, class_name, error in failed_imports:
        print(f"  - {module_name}.{class_name}: {error}")

if successful_imports:
    print("\n✅ SUCCESSFUL IMPORTS:")
    for module_name, class_name in successful_imports:
        print(f"  - {module_name}.{class_name}")

print("\n🔍 Testing basic functionality...")
try:
    if ("shared_constants", "SharedUtilities") in successful_imports:
        from shared_constants import SharedUtilities
        logger = SharedUtilities.setup_detailed_logger("diagnostic")
        print("✅ Enhanced logging setup works")
    
    if ("report", "EnhancedReportGenerator") in successful_imports:
        from report import EnhancedReportGenerator
        generator = EnhancedReportGenerator()
        print("✅ Report generator creation works")
        
except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")

print("\n🎯 DIAGNOSTIC COMPLETE")
