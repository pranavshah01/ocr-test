#!/usr/bin/env python3
"""
Comprehensive Test Runner for OCR DOCX Text Replacement Utility.
Runs all tests in the tests folder and provides a summary report.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util


def run_test_file(test_file: Path) -> tuple[bool, str, str]:
    """
    Run a single test file and capture its output.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out after 5 minutes"
    except Exception as e:
        return False, "", f"Error running test: {e}"


def discover_test_files() -> list[Path]:
    """
    Discover all test files in the tests directory.
    
    Returns:
        List of test file paths
    """
    tests_dir = Path(__file__).parent
    test_files = []
    
    # Find all Python files starting with 'test_'
    for test_file in tests_dir.glob("test_*.py"):
        if test_file.name != "test_runner.py":  # Exclude this file
            test_files.append(test_file)
    
    return sorted(test_files)


def print_test_header():
    """Print the test suite header."""
    print("=" * 80)
    print("OCR DOCX TEXT REPLACEMENT UTILITY - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_test_summary(results: list[tuple[str, bool, str, str]]):
    """
    Print the test summary.
    
    Args:
        results: List of (test_name, success, stdout, stderr) tuples
    """
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, success, stdout, stderr in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
        
        if success:
            passed += 1
        else:
            failed += 1
            if stderr:
                print(f"    Error: {stderr[:100]}{'...' if len(stderr) > 100 else ''}")
    
    total = passed + failed
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The OCR DOCX Text Replacement Utility is working correctly.")
        print("\nImplemented Features:")
        print("  ✓ Enhanced pattern matching across split XML tags")
        print("  ✓ Mapping and replacement logic (replace/append modes)")
        print("  ✓ Comprehensive formatting preservation")
        print("  ✓ Three OCR modes for image text replacement")
        print("  ✓ Cross-platform compatibility")
    else:
        print(f"⚠️  {failed} test(s) failed. Check the detailed output above.")
    
    print(f"\nTest run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Run all tests and provide a comprehensive report."""
    print_test_header()
    
    # Discover test files
    test_files = discover_test_files()
    
    if not test_files:
        print("No test files found in the tests directory.")
        return 1
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    print()
    
    # Run all tests
    results = []
    
    for test_file in test_files:
        test_name = test_file.stem.replace("test_", "").replace("_", " ").title()
        print(f"Running {test_name}...")
        
        success, stdout, stderr = run_test_file(test_file)
        results.append((test_name, success, stdout, stderr))
        
        if success:
            print(f"  ✓ {test_name} completed successfully")
        else:
            print(f"  ✗ {test_name} failed")
            if stderr:
                print(f"    Error: {stderr[:200]}{'...' if len(stderr) > 200 else ''}")
        print()
    
    # Print detailed output for failed tests
    failed_tests = [(name, stdout, stderr) for name, success, stdout, stderr in results if not success]
    
    if failed_tests:
        print("\n" + "=" * 80)
        print("DETAILED OUTPUT FOR FAILED TESTS")
        print("=" * 80)
        
        for test_name, stdout, stderr in failed_tests:
            print(f"\n--- {test_name} ---")
            if stdout:
                print("STDOUT:")
                print(stdout)
            if stderr:
                print("STDERR:")
                print(stderr)
            print("-" * 40)
    
    # Print summary
    print_test_summary(results)
    
    # Return appropriate exit code
    all_passed = all(success for _, success, _, _ in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
