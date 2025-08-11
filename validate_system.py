#!/usr/bin/env python3
"""
System validation script for the Document Processing Pipeline.
Validates core functionality and generates a validation report.
"""

import sys
import json
from pathlib import Path
from docx import Document

def validate_processed_document():
    """Validate that document processing worked correctly."""
    print("=== Document Processing Validation ===")
    
    # Check if processed document exists
    processed_path = Path("processed/test_document_processed.docx")
    if not processed_path.exists():
        print("❌ Processed document not found")
        return False
    
    print("✅ Processed document exists")
    
    # Check if document can be opened
    try:
        doc = Document(processed_path)
        print("✅ Processed document can be opened")
    except Exception as e:
        print(f"❌ Cannot open processed document: {e}")
        return False
    
    # Check document content
    content = ""
    for paragraph in doc.paragraphs:
        content += paragraph.text + " "
    
    # Check if replacements were made
    if "4022-110-810096-000" in content:
        print("✅ Text replacement found: 77-110-810096-000 → 4022-110-810096-000")
    else:
        print("❌ Expected text replacement not found")
        return False
    
    if "4022-172-240862-36" in content:
        print("✅ Text replacement found: 77-130-120541-001 → 4022-172-240862-36")
    else:
        print("❌ Expected text replacement not found")
        return False
    
    return True

def validate_reports():
    """Validate that reports were generated correctly."""
    print("\n=== Report Generation Validation ===")
    
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("❌ Reports directory not found")
        return False
    
    # Find latest batch report
    json_reports = list(reports_dir.glob("batch_summary_*.json"))
    if not json_reports:
        print("❌ No batch summary JSON reports found")
        return False
    
    latest_json = max(json_reports, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found batch summary JSON report: {latest_json.name}")
    
    # Validate JSON report content
    try:
        with open(latest_json, 'r') as f:
            report_data = json.load(f)
        
        if report_data.get('report_type') == 'batch_summary':
            print("✅ JSON report has correct type")
        else:
            print("❌ JSON report has incorrect type")
            return False
        
        batch_summary = report_data.get('batch_summary', {})
        if batch_summary.get('successful_files', 0) > 0:
            print(f"✅ Batch processing successful: {batch_summary.get('successful_files')} files")
        else:
            print("❌ No successful files in batch processing")
            return False
        
        aggregate_stats = report_data.get('aggregate_statistics', {})
        if aggregate_stats.get('total_text_matches', 0) > 0:
            print(f"✅ Text matches found: {aggregate_stats.get('total_text_matches')}")
        else:
            print("❌ No text matches found in aggregate statistics")
            return False
        
    except Exception as e:
        print(f"❌ Error reading JSON report: {e}")
        return False
    
    # Check for HTML report
    html_reports = list(reports_dir.glob("batch_summary_*.html"))
    if html_reports:
        print("✅ HTML batch summary report found")
    else:
        print("❌ No HTML batch summary reports found")
        return False
    
    return True

def validate_configuration():
    """Validate configuration and system setup."""
    print("\n=== Configuration Validation ===")
    
    # Check patterns.json
    patterns_path = Path("patterns.json")
    if patterns_path.exists():
        try:
            with open(patterns_path, 'r') as f:
                patterns = json.load(f)
            print(f"✅ Patterns file loaded: {len([k for k in patterns.keys() if not k.startswith('_')])} patterns")
        except Exception as e:
            print(f"❌ Error loading patterns: {e}")
            return False
    else:
        print("❌ Patterns file not found")
        return False
    
    # Check mapping.json
    mapping_path = Path("mapping.json")
    if mapping_path.exists():
        try:
            with open(mapping_path, 'r') as f:
                mappings = json.load(f)
            print(f"✅ Mappings file loaded: {len(mappings)} mappings")
        except Exception as e:
            print(f"❌ Error loading mappings: {e}")
            return False
    else:
        print("❌ Mappings file not found")
        return False
    
    return True

def validate_gpu_functionality():
    """Validate GPU functionality."""
    print("\n=== GPU Functionality Validation ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("✅ CUDA GPU available")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS GPU available")
        else:
            print("ℹ️  No GPU available, using CPU")
        
        # Test EasyOCR initialization
        import easyocr
        reader = easyocr.Reader(['en'], gpu=torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
        print("✅ EasyOCR initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU/OCR validation failed: {e}")
        return False

def main():
    """Run complete system validation."""
    print("Document Processing Pipeline - System Validation")
    print("=" * 50)
    
    validation_results = []
    
    # Run validation tests
    validation_results.append(("Configuration", validate_configuration()))
    validation_results.append(("Document Processing", validate_processed_document()))
    validation_results.append(("Report Generation", validate_reports()))
    validation_results.append(("GPU Functionality", validate_gpu_functionality()))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(validation_results)
    
    for test_name, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! System is working correctly.")
        return 0
    else:
        print("⚠️  Some validation tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())