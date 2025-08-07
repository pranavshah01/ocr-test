from pathlib import Path
import json
import numpy as np
import html
from datetime import datetime
from typing import Dict, Any, List, Optional
import statistics as stats

from .shared_constants import SharedUtilities

def sanitize_for_json(obj):
    """Convert any non-JSON serializable objects to JSON serializable types."""
    # Handle NumPy integers (including int64)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    # Handle NumPy floats
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    # Handle NumPy booleans
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle NumPy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle other NumPy scalars with .item() method
    elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        return obj.item()
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    # Handle lists recursively
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    # Handle tuples recursively
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    # Handle sets
    elif isinstance(obj, set):
        return [sanitize_for_json(item) for item in obj]
    # Return as-is for native Python types
    else:
        return obj

def write_report(summary: dict, reports_dir: Path):
    """Append or create a JSON file per document in reports_dir/{stem}.json."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)
    file_stem = summary.get("file", "report")
    out_path = reports_dir / f"{Path(file_stem).stem}.json"
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {out_path}: {e}")
            print(f"Creating backup and starting fresh...")
            # Create backup of corrupted file
            backup_path = out_path.with_suffix('.json.backup')
            out_path.rename(backup_path)
            old = []
        except Exception as e:
            print(f"Error reading {out_path}: {e}")
            old = []
    else:
        old = []
    # Sanitize the summary data before adding it
    sanitized_summary = sanitize_for_json(summary)
    old.append(sanitized_summary)
    try:
        # Sanitize the entire data structure before JSON serialization
        sanitized_data = sanitize_for_json(old)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing JSON report to {out_path}: {e}")
        # Try to write a minimal report instead
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([{"file": summary.get("file", "unknown"), "error": str(e)}], f, indent=2)
        except Exception:
            print(f"Failed to write even minimal report to {out_path}")

def write_master_report(reports_dir: Path):
    """Summarize all JSON reports in reports_dir into summary.json."""
    reports_dir = Path(reports_dir)
    summary_path = reports_dir / "summary.json"
    all_reports = []
    for file in reports_dir.glob("*.json"):
        if file.name == "summary.json":
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and single dict formats
                if isinstance(data, list):
                    all_reports.extend(data)
                elif isinstance(data, dict):
                    all_reports.append(data)
        except Exception:
            continue
    
    # Filter out non-dict entries to avoid AttributeError
    valid_reports = [r for r in all_reports if isinstance(r, dict)]
    
    master = {
        "documents": len(valid_reports),
        "total_header_replacements": sum(r.get("header_replacements", 0) for r in valid_reports),
        "total_body_replacements": sum(r.get("body_replacements", 0) for r in valid_reports),
        "total_callout_replacements": sum(r.get("callout_replacements", 0) for r in valid_reports),
        "total_ocr_replacements": sum(r.get("ocr_replacements", 0) for r in valid_reports),
        "total_text_replacements": sum(r.get("text_replacements", 0) for r in valid_reports),
        "total_textbox_replacements": sum(r.get("textbox_replacements", 0) for r in valid_reports),
        "total_section_replacements": sum(r.get("section_replacements", 0) for r in valid_reports),
        "files": [r.get("file") for r in valid_reports if r.get("file")],
        "errors": [r.get("errors", []) for r in valid_reports if r.get("errors")],
        "success_rate": (len([r for r in valid_reports if r.get("success", False)]) / max(len(valid_reports), 1)) * 100,
        "total_processing_time": sum(r.get("processing_time", 0) for r in valid_reports if isinstance(r.get("processing_time"), (int, float))),
        "invalid_entries_skipped": len(all_reports) - len(valid_reports)
    }
    # Sanitize the master report data before JSON serialization
    sanitized_master = sanitize_for_json(master)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_master, f, indent=2, ensure_ascii=False)


# Enhanced Reporting Classes for Task 5.2
class EnhancedReportGenerator:
    """
    Enhanced report generator for comprehensive processing reports.
    
    Implements Task 5.2 requirements:
    - Summary of all processed files
    - Statistics on matches and replacements
    - OCR confidence scores and image processing details
    - Visual diff generation for changes
    - Export to multiple formats (JSON, HTML, CSV)
    """
    
    def __init__(self, output_dir: Path = None, filename: str = None):
        """
        Initialize enhanced report generator.
        
        Args:
            output_dir: Directory to save reports (default: ./reports)
            filename: Base filename for reports (without extension)
        """
        self.output_dir = output_dir or Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.processing_logs: List[Dict[str, Any]] = []
        self.generation_time = datetime.now()
    
    def add_processing_log(self, log_data: Dict[str, Any]):
        """
        Add a processing log to the report data.
        
        Args:
            log_data: Processing log dictionary from enhanced logging
        """
        self.processing_logs.append(sanitize_for_json(log_data))
    
    def generate_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive processing statistics.
        
        Returns:
            Complete statistics structure
        """
        if not self.processing_logs:
            return {"message": "No processing logs available"}
        
        # Basic metrics
        total_files = len(self.processing_logs)
        successful_files = len([log for log in self.processing_logs if log.get('success', False)])
        failed_files = total_files - successful_files
        
        # Replacement metrics
        total_matches = sum(log.get('processing_summary', {}).get('total_matches', 0) for log in self.processing_logs)
        text_replacements = sum(log.get('processing_summary', {}).get('text_replacements', 0) for log in self.processing_logs)
        ocr_replacements = sum(log.get('processing_summary', {}).get('ocr_replacements', 0) for log in self.processing_logs)
        textbox_replacements = sum(log.get('processing_summary', {}).get('textbox_replacements', 0) for log in self.processing_logs)
        section_replacements = sum(log.get('processing_summary', {}).get('section_replacements', 0) for log in self.processing_logs)
        
        # Processing time metrics
        processing_times = [log.get('processing_summary', {}).get('processing_time', 0.0) for log in self.processing_logs]
        processing_times = [t for t in processing_times if t > 0]
        
        # OCR confidence metrics and text data
        ocr_scores = []
        ocr_text_data = {}
        
        for log in self.processing_logs:
            # Extract OCR confidence scores from matches
            matches_detail = log.get('matches_detail', [])
            for match in matches_detail:
                if match.get('confidence_score') is not None:
                    ocr_scores.append(match['confidence_score'])
            
            # Extract OCR text data if available
            if 'ocr_text_data' in log and isinstance(log['ocr_text_data'], dict):
                ocr_data = log['ocr_text_data']
                file_info = log.get('file_info', {})
                file_path = file_info.get('path', 'unknown')
                ocr_text_data[file_path] = ocr_data
                
                # Also extract confidence scores from OCR blocks
                if 'images' in ocr_data:
                    for image in ocr_data['images']:
                        if 'ocr_blocks' in image:
                            for block in image['ocr_blocks']:
                                if 'confidence' in block:
                                    ocr_scores.append(block['confidence'])
        
        # Error and warning metrics
        total_errors = sum(len(log.get('errors', [])) for log in self.processing_logs)
        total_warnings = sum(len(log.get('warnings', [])) for log in self.processing_logs)
        
        return {
            'report_metadata': {
                'generated_at': self.generation_time.isoformat(),
                'report_version': '2.0',
                'generator': 'Enhanced Report Generator v2.0',
                'total_files_analyzed': total_files
            },
            'processing_metrics': {
                'total_files_processed': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'success_rate': (successful_files / max(total_files, 1)) * 100,
                'total_processing_time': sum(processing_times),
                'average_processing_time': stats.mean(processing_times) if processing_times else 0.0,
                'total_matches': total_matches,
                'total_replacements': text_replacements + ocr_replacements + textbox_replacements + section_replacements,
                'text_replacements': text_replacements,
                'ocr_replacements': ocr_replacements,
                'textbox_replacements': textbox_replacements,
                'section_replacements': section_replacements,
                'errors_encountered': total_errors,
                'warnings_generated': total_warnings
            },
            'ocr_analysis': self._generate_ocr_analysis(ocr_scores),
            'ocr_text_data': ocr_text_data,
            'performance_analysis': self._generate_performance_analysis(processing_times),
            'error_analysis': self._generate_error_analysis(),
            'file_processing_details': self.processing_logs
        }
    
    def _generate_ocr_analysis(self, ocr_scores: List[float]) -> Dict[str, Any]:
        """Generate OCR-specific analysis."""
        if not ocr_scores:
            return {
                'ocr_enabled': False,
                'message': 'No OCR processing detected in analyzed files'
            }
        
        return {
            'ocr_enabled': True,
            'total_ocr_detections': len(ocr_scores),
            'average_confidence': stats.mean(ocr_scores),
            'confidence_distribution': {
                'high_confidence': len([s for s in ocr_scores if s >= 0.8]),
                'medium_confidence': len([s for s in ocr_scores if 0.5 <= s < 0.8]),
                'low_confidence': len([s for s in ocr_scores if s < 0.5])
            },
            'confidence_statistics': {
                'min': min(ocr_scores),
                'max': max(ocr_scores),
                'median': stats.median(ocr_scores),
                'std_deviation': stats.stdev(ocr_scores) if len(ocr_scores) > 1 else 0.0
            }
        }
    
    def _generate_performance_analysis(self, processing_times: List[float]) -> Dict[str, Any]:
        """Generate performance analysis."""
        if not processing_times:
            return {'message': 'No processing time data available'}
        
        return {
            'processing_time_analysis': {
                'total_time': sum(processing_times),
                'average_time': stats.mean(processing_times),
                'median_time': stats.median(processing_times),
                'fastest_file': min(processing_times),
                'slowest_file': max(processing_times),
                'std_deviation': stats.stdev(processing_times) if len(processing_times) > 1 else 0.0
            },
            'throughput_metrics': {
                'files_per_second': (
                    len(processing_times) / sum(processing_times)
                    if sum(processing_times) > 0 else 0
                ),
                'total_matches_per_second': (
                    sum(log.get('processing_summary', {}).get('total_matches', 0) for log in self.processing_logs) / sum(processing_times)
                    if sum(processing_times) > 0 else 0
                )
            }
        }
    
    def _generate_error_analysis(self) -> Dict[str, Any]:
        """Generate error and warning analysis."""
        all_errors = []
        all_warnings = []
        
        for log in self.processing_logs:
            all_errors.extend(log.get('errors', []))
            all_warnings.extend(log.get('warnings', []))
        
        return {
            'error_summary': {
                'total_errors': len(all_errors),
                'files_with_errors': len([log for log in self.processing_logs if log.get('errors')]),
                'error_rate': (
                    len([log for log in self.processing_logs if log.get('errors')]) / 
                    max(len(self.processing_logs), 1) * 100
                )
            },
            'warning_summary': {
                'total_warnings': len(all_warnings),
                'files_with_warnings': len([log for log in self.processing_logs if log.get('warnings')]),
                'warning_rate': (
                    len([log for log in self.processing_logs if log.get('warnings')]) / 
                    max(len(self.processing_logs), 1) * 100
                )
            },
            'recent_errors': all_errors[-10:] if all_errors else [],
            'recent_warnings': all_warnings[-10:] if all_warnings else []
        }
    
    def export_to_enhanced_json(self, filename: str = None) -> Path:
        """
        Export enhanced report to JSON format.
        
        Args:
            filename: Custom filename (default: auto-generated)
            
        Returns:
            Path to exported JSON file
        """
        if filename is None:
            if self.filename:
                timestamp = self.generation_time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.filename}_report_{timestamp}.json"
            else:
                timestamp = self.generation_time.strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_processing_report_{timestamp}.json"
        
        json_path = self.output_dir / filename
        report_data = self.generate_comprehensive_statistics()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        return json_path
    
    def export_to_html(self, filename: str = None) -> Path:
        """
        Export report to HTML format with visual styling.
        
        Args:
            filename: Custom filename (default: auto-generated)
            
        Returns:
            Path to exported HTML file
        """
        if filename is None:
            if self.filename:
                timestamp = self.generation_time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.filename}_report_{timestamp}.html"
            else:
                timestamp = self.generation_time.strftime("%Y%m%d_%H%M%S")
                filename = f"processing_report_{timestamp}.html"
        
        html_path = self.output_dir / filename
        report_data = self.generate_comprehensive_statistics()
        
        html_content = self._generate_html_report(report_data)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    

    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content with enhanced styling."""
        metadata = report_data['report_metadata']
        metrics = report_data['processing_metrics']
        ocr_analysis = report_data.get('ocr_analysis', {})
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR DOCX Processing Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #007acc 0%, #0056b3 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #007acc; transition: transform 0.2s; }}
        .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2.2em; font-weight: bold; color: #007acc; margin-bottom: 5px; }}
        .metric-label {{ color: #666; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; margin-bottom: 20px; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        .progress-bar {{ width: 100%; height: 25px; background: #e9ecef; border-radius: 12px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #007acc, #0056b3); transition: width 0.5s ease; border-radius: 12px; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; border-radius: 8px; overflow: hidden; }}
        .stats-table th {{ background: #007acc; color: white; padding: 15px; text-align: left; }}
        .stats-table td {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
        .stats-table tr:hover {{ background: #f8f9fa; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OCR DOCX Processing Report</h1>
            <p>Generated on {metadata['generated_at'][:19].replace('T', ' ')}</p>
        </div>
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_files_processed']}</div>
                    <div class="metric-label">Files Processed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value success">{metrics['successful_files']}</div>
                    <div class="metric-label">Successful</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_replacements']}</div>
                    <div class="metric-label">Total Replacements</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['success_rate']:.1f}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Summary</h2>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {metrics['success_rate']:.1f}%"></div>
                </div>
                <p>Success Rate: {metrics['success_rate']:.1f}% ({metrics['successful_files']}/{metrics['total_files_processed']} files)</p>
            </div>
            
            <div class="section">
                <h2>Replacement Statistics</h2>
                <table class="stats-table">
                    <tr><th>Type</th><th>Count</th><th>Percentage</th></tr>
                    <tr><td>Text Replacements</td><td>{metrics['text_replacements']}</td><td>{(metrics['text_replacements']/max(metrics['total_replacements'],1)*100):.1f}%</td></tr>
                    <tr><td>OCR Replacements</td><td>{metrics['ocr_replacements']}</td><td>{(metrics['ocr_replacements']/max(metrics['total_replacements'],1)*100):.1f}%</td></tr>
                    <tr><td>Textbox Replacements</td><td>{metrics['textbox_replacements']}</td><td>{(metrics['textbox_replacements']/max(metrics['total_replacements'],1)*100):.1f}%</td></tr>
                    <tr><td>Section Replacements</td><td>{metrics['section_replacements']}</td><td>{(metrics['section_replacements']/max(metrics['total_replacements'],1)*100):.1f}%</td></tr>
                </table>
            </div>"""
        
        # Add OCR analysis if available
        if ocr_analysis.get('ocr_enabled'):
            html_content += f"""
            <div class="section">
                <h2>OCR Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{ocr_analysis['total_ocr_detections']}</div>
                        <div class="metric-label">OCR Detections</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{ocr_analysis['average_confidence']:.2f}</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value success">{ocr_analysis['confidence_distribution']['high_confidence']}</div>
                        <div class="metric-label">High Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value warning">{ocr_analysis['confidence_distribution']['medium_confidence']}</div>
                        <div class="metric-label">Medium Confidence</div>
                    </div>
                </div>
            </div>"""
        
        # Add OCR text section
        html_content += self._generate_ocr_text_section(report_data)
        
        html_content += f"""
        </div>
        <div class="footer">
            <p>Report generated by Enhanced OCR DOCX Text Replacement Utility v2.0</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_ocr_text_section(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML section for OCR text data."""
        ocr_text_data = report_data.get('ocr_text_data', {})
        
        if not ocr_text_data:
            return ""
        
        html_section = """
            <div class="section">
                <h2>OCR Text Data</h2>"""
        
        for file_path, ocr_data in ocr_text_data.items():
            if isinstance(ocr_data, dict) and 'images' in ocr_data:
                html_section += f"""
                <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3 style="color: #007acc; margin-top: 0;">File: {file_path.split('/')[-1]}</h3>
                    <p><strong>Total Images:</strong> {ocr_data.get('total_images', 0)}</p>
                    <p><strong>Orientations Found:</strong> {', '.join(ocr_data.get('orientations_found', []))}</p>"""
                
                for i, image in enumerate(ocr_data.get('images', [])):
                    html_section += f"""
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <h4>Image {i+1}</h4>
                        <p><strong>Size:</strong> {image.get('image_size', {}).get('width', 0)}x{image.get('image_size', {}).get('height', 0)}</p>
                        <p><strong>Orientations:</strong> {', '.join(image.get('orientations_detected', []))}</p>"""
                    
                    # Add text by orientation
                    for orientation, text_data in image.get('ocr_text_by_orientation', {}).items():
                        html_section += f"""
                        <div style="margin: 5px 0; padding: 8px; background: white; border-left: 3px solid #007acc;">
                            <p><strong>{orientation}:</strong> {text_data.get('text', '')}</p>
                            <p style="font-size: 0.9em; color: #666;">Blocks: {text_data.get('blocks_count', 0)}, Confidence: {text_data.get('average_confidence', 0):.2f}</p>
                        </div>"""
                    
                    html_section += """
                    </div>"""
                
                html_section += """
                </div>"""
        
        html_section += """
            </div>"""
        
        return html_section


# Enhanced convenience functions
def create_enhanced_report_generator(output_dir: Path = None) -> EnhancedReportGenerator:
    """
    Create an enhanced report generator instance.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        EnhancedReportGenerator instance
    """
    return EnhancedReportGenerator(output_dir)


def generate_comprehensive_report(processing_logs: List[Dict[str, Any]], output_dir: Path = None) -> Dict[str, Path]:
    """
    Generate comprehensive reports in all formats from processing logs.
    
    Args:
        processing_logs: List of processing log dictionaries
        output_dir: Directory to save reports
        
    Returns:
        Dictionary mapping format names to file paths
    """
    generator = EnhancedReportGenerator(output_dir)
    
    for log in processing_logs:
        generator.add_processing_log(log)
    
    return {
        'json': generator.export_to_enhanced_json(),
        'html': generator.export_to_html()
    }