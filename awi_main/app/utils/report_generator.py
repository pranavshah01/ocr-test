"""
Report generator for comprehensive processing reports.
Generates both JSON and HTML reports for individual documents and batch summaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..core.models import ProcessingResult, BatchProcessingResult, ProcessingStatistics
from ..utils.platform_utils import PathManager

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive processing reports in JSON and HTML formats."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        PathManager.ensure_directory(output_dir)
        
        logger.info(f"Report generator initialized with output directory: {output_dir}")
    
    def generate_document_reports(self, result: ProcessingResult) -> Dict[str, Path]:
        """
        Generate individual document reports (JSON and HTML).
        
        Args:
            result: Processing result for a single document
            
        Returns:
            Dictionary with paths to generated reports
        """
        report_paths = {}
        
        try:
            # Generate base filename
            doc_name = result.input_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{doc_name}_report_{timestamp}"
            
            # Generate JSON report
            json_path = self._generate_document_json_report(result, base_name)
            if json_path:
                report_paths['json'] = json_path
            
            # Generate HTML report
            html_path = self._generate_document_html_report(result, base_name)
            if html_path:
                report_paths['html'] = html_path
            
            logger.info(f"Generated document reports for {result.input_path.name}")
            
        except Exception as e:
            logger.error(f"Error generating document reports: {e}")
        
        return report_paths
    
    def generate_batch_reports(self, batch_result: BatchProcessingResult) -> Dict[str, Path]:
        """
        Generate batch summary reports (JSON and HTML).
        
        Args:
            batch_result: Batch processing result
            
        Returns:
            Dictionary with paths to generated reports
        """
        report_paths = {}
        
        try:
            # Generate base filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"batch_summary_{timestamp}"
            
            # Generate JSON report
            json_path = self._generate_batch_json_report(batch_result, base_name)
            if json_path:
                report_paths['json'] = json_path
            
            # Generate HTML report
            html_path = self._generate_batch_html_report(batch_result, base_name)
            if html_path:
                report_paths['html'] = html_path
            
            logger.info(f"Generated batch summary reports for {batch_result.total_files} files")
            
        except Exception as e:
            logger.error(f"Error generating batch reports: {e}")
        
        return report_paths
    
    def _generate_document_json_report(self, result: ProcessingResult, base_name: str) -> Optional[Path]:
        """Generate JSON report for individual document."""
        try:
            report_data = {
                'report_type': 'document',
                'generated_at': datetime.now().isoformat(),
                'document_info': {
                    'input_path': str(result.input_path),
                    'output_path': str(result.output_path) if result.output_path else None,
                    'processing_time': result.processing_time,
                    'success': result.success,
                    'error_message': result.error_message
                },
                'processing_summary': {
                    'total_matches': result.total_matches,
                    'text_matches': result.text_matches,
                    'graphics_matches': result.graphics_matches,
                    'image_matches': result.image_matches
                },
                'detailed_results': result.processing_log.to_dict() if result.processing_log else None
            }
            
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating document JSON report: {e}")
            return None
    
    def _generate_document_html_report(self, result: ProcessingResult, base_name: str) -> Optional[Path]:
        """Generate HTML report for individual document."""
        try:
            html_content = self._create_document_html_content(result)
            
            html_path = self.output_dir / f"{base_name}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating document HTML report: {e}")
            return None
    
    def _generate_batch_json_report(self, batch_result: BatchProcessingResult, base_name: str) -> Optional[Path]:
        """Generate JSON report for batch processing."""
        try:
            # Calculate aggregate statistics
            statistics = ProcessingStatistics()
            for result in batch_result.results:
                statistics.add_result(result)
            
            report_data = {
                'report_type': 'batch_summary',
                'generated_at': datetime.now().isoformat(),
                'batch_summary': {
                    'total_files': batch_result.total_files,
                    'successful_files': batch_result.successful_files,
                    'failed_files': batch_result.failed_files,
                    'success_rate': batch_result.success_rate,
                    'total_processing_time': batch_result.processing_time,
                    'average_processing_time': batch_result.processing_time / batch_result.total_files if batch_result.total_files > 0 else 0.0
                },
                'aggregate_statistics': statistics.to_dict(),
                'individual_results': [result.to_dict() for result in batch_result.results]
            }
            
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating batch JSON report: {e}")
            return None
    
    def _generate_batch_html_report(self, batch_result: BatchProcessingResult, base_name: str) -> Optional[Path]:
        """Generate HTML report for batch processing."""
        try:
            html_content = self._create_batch_html_content(batch_result)
            
            html_path = self.output_dir / f"{base_name}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating batch HTML report: {e}")
            return None
    
    def _create_document_html_content(self, result: ProcessingResult) -> str:
        """Create HTML content for document report."""
        status_color = "green" if result.success else "red"
        status_text = "SUCCESS" if result.success else "FAILED"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Processing Report - {result.input_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
        .status {{ font-weight: bold; color: {status_color}; }}
        .section {{ margin: 20px 0; }}
        .section h3 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .match-table {{ font-size: 14px; }}
        .error {{ color: red; }}
        .warning {{ color: orange; }}
        .success {{ color: green; }}
        .stats {{ display: flex; justify-content: space-around; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .stat-item {{ text-align: center; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Document Processing Report</h1>
            <p><strong>Document:</strong> {result.input_path.name}</p>
            <p><strong>Status:</strong> <span class="status">{status_text}</span></p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Processing Summary</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{result.total_matches}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{result.text_matches}</div>
                    <div class="stat-label">Text Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{result.graphics_matches}</div>
                    <div class="stat-label">Graphics Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{result.image_matches}</div>
                    <div class="stat-label">Image Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{result.processing_time:.2f}s</div>
                    <div class="stat-label">Processing Time</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>File Information</h3>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Input Path</td><td>{result.input_path}</td></tr>
                <tr><td>Output Path</td><td>{result.output_path or 'N/A'}</td></tr>
                <tr><td>Processing Time</td><td>{result.processing_time:.3f} seconds</td></tr>
                <tr><td>Success</td><td>{'Yes' if result.success else 'No'}</td></tr>
                {f'<tr><td>Error Message</td><td class="error">{result.error_message}</td></tr>' if result.error_message else ''}
            </table>
        </div>
"""
        
        # Add detailed match information if available
        if result.processing_log:
            html += self._create_matches_html_section(result.processing_log)
        
        html += """
    </div>
</body>
</html>"""
        
        return html
    
    def _create_batch_html_content(self, batch_result: BatchProcessingResult) -> str:
        """Create HTML content for batch report."""
        statistics = ProcessingStatistics()
        for result in batch_result.results:
            statistics.add_result(result)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Processing Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .section h3 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .stats {{ display: flex; justify-content: space-around; background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .stat-item {{ text-align: center; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #ddd; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: #4CAF50; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Batch Processing Summary Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Files Processed:</strong> {batch_result.total_files}</p>
        </div>
        
        <div class="section">
            <h3>Overall Statistics</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{batch_result.total_files}</div>
                    <div class="stat-label">Total Files</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.successful_files}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.failed_files}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.processing_time:.1f}s</div>
                    <div class="stat-label">Total Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{statistics.average_processing_time:.2f}s</div>
                    <div class="stat-label">Avg Time/File</div>
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <p><strong>Success Rate:</strong></p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {batch_result.success_rate}%;"></div>
                </div>
                <p style="text-align: center; margin-top: 5px;">{batch_result.success_rate:.1f}%</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Match Statistics</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{statistics.total_matches}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{statistics.total_text_matches}</div>
                    <div class="stat-label">Text Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{statistics.total_graphics_matches}</div>
                    <div class="stat-label">Graphics Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{statistics.total_image_matches}</div>
                    <div class="stat-label">Image Matches</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Individual File Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Status</th>
                        <th>Processing Time</th>
                        <th>Total Matches</th>
                        <th>Text</th>
                        <th>Graphics</th>
                        <th>Images</th>
                        <th>Error Message</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for result in batch_result.results:
            status_class = "success" if result.success else "error"
            status_text = "SUCCESS" if result.success else "FAILED"
            
            html += f"""
                    <tr>
                        <td>{result.input_path.name}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{result.processing_time:.2f}s</td>
                        <td>{result.total_matches}</td>
                        <td>{result.text_matches}</td>
                        <td>{result.graphics_matches}</td>
                        <td>{result.image_matches}</td>
                        <td class="error">{result.error_message or ''}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _create_matches_html_section(self, processing_log) -> str:
        """Create HTML section for detailed match information."""
        html = ""
        
        # Text matches
        if processing_log.text_matches:
            html += """
        <div class="section">
            <h3>Text Matches</h3>
            <table class="match-table">
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Original Text</th>
                        <th>Replacement</th>
                        <th>Location</th>
                        <th>Position</th>
                    </tr>
                </thead>
                <tbody>
"""
            for match in processing_log.text_matches:
                html += f"""
                    <tr>
                        <td>{match.pattern}</td>
                        <td>{match.original_text}</td>
                        <td>{match.replacement_text}</td>
                        <td>{match.location}</td>
                        <td>{match.position}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Graphics matches
        if processing_log.graphics_matches:
            html += """
        <div class="section">
            <h3>Graphics Matches</h3>
            <table class="match-table">
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Original Text</th>
                        <th>Replacement</th>
                        <th>Location</th>
                        <th>Font Info</th>
                    </tr>
                </thead>
                <tbody>
"""
            for match in processing_log.graphics_matches:
                font_info = f"Size: {match.font_info.get('size', 'N/A')}" if match.font_info else "N/A"
                html += f"""
                    <tr>
                        <td>{match.pattern}</td>
                        <td>{match.original_text}</td>
                        <td>{match.replacement_text}</td>
                        <td>{match.location}</td>
                        <td>{font_info}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Image matches
        if processing_log.image_matches:
            html += """
        <div class="section">
            <h3>Image Matches</h3>
            <table class="match-table">
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Original Text</th>
                        <th>Replacement</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                        <th>Mode</th>
                    </tr>
                </thead>
                <tbody>
"""
            for match in processing_log.image_matches:
                bbox = match.ocr_result.bounding_box
                bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                html += f"""
                    <tr>
                        <td>{match.pattern}</td>
                        <td>{match.ocr_result.text}</td>
                        <td>{match.replacement_text}</td>
                        <td>{match.ocr_result.confidence:.2f}</td>
                        <td>{bbox_str}</td>
                        <td>{match.processing_mode}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        return html

def create_report_generator(output_dir: Path) -> ReportGenerator:
    """
    Factory function to create a ReportGenerator instance.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        ReportGenerator instance
    """
    return ReportGenerator(output_dir)