#!/usr/bin/env python3
"""
Simple test for enhanced logging and reporting flow.
"""

import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_enhanced_flow():
    """Test the enhanced logging and reporting flow with a simple example."""
    print("🧪 Testing Enhanced Logging and Reporting Flow")
    print("=" * 50)
    
    try:
        # Import modules
        from app.docx_processor import process_docx
        from app.shared_constants import SharedUtilities
        from app.report import EnhancedReportGenerator
        
        print("✅ Modules imported successfully")
        
        # Test file paths
        test_docx = Path("source_documents/testimages_092051.docx")
        mapping_file = Path("mapping.json")
        patterns_file = Path("patterns.json")
        
        # Check if test files exist
        if not test_docx.exists():
            print(f"❌ Test file not found: {test_docx}")
            return False
        
        print(f"✅ Test file found: {test_docx}")
        
        # Load mapping and patterns
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
            # Handle object format patterns
            if isinstance(patterns_data, dict):
                patterns_list = [
                    value for key, value in patterns_data.items() 
                    if not key.startswith('_') and isinstance(value, str)
                ]
            else:
                patterns_list = patterns_data
        
        print(f"✅ Configuration loaded: {len(mapping_dict)} mappings, {len(patterns_list)} patterns")
        
        # Test enhanced logging setup
        print("\n🔧 Testing Enhanced Logging Setup...")
        logger = SharedUtilities.setup_detailed_logger(
            name="test_flow",
            log_file="./logs/test_flow.log"
        )
        logger.info("Enhanced logging test started")
        print("✅ Enhanced logging initialized")
        
        # Test report generator
        print("📊 Testing Enhanced Report Generator...")
        report_generator = EnhancedReportGenerator(output_dir=Path("./reports"))
        print("✅ Enhanced report generator initialized")
        
        # Test processing with limited scope (text only, no OCR to avoid hanging)
        print("\n🚀 Testing Processing Pipeline...")
        try:
            results = process_docx(
                test_docx,
                mapping_dict,
                patterns_list,
                text_mode="replace",
                ocr_mode="replace",
                ocr_engine="easyocr",
                confidence_min=0.4,
                gpu=False,  # Disable GPU to avoid hanging
                process_images=False  # Disable OCR processing for this test
            )
            
            print("✅ Processing completed successfully!")
            print(f"   Results: {results.get('success', False)}")
            print(f"   Text replacements: {results.get('body_replacements', 0)}")
            print(f"   Processing time: {results.get('processing_time', 0):.2f}s")
            print(f"   Reports generated: {list(results.get('report_paths', {}).keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_flow()
    if success:
        print("\n🎉 Enhanced logging and reporting flow test PASSED!")
    else:
        print("\n💥 Enhanced logging and reporting flow test FAILED!")
    
    sys.exit(0 if success else 1)
