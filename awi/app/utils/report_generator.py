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
from .document_properties import extract_document_properties

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive processing reports in JSON and HTML formats."""
    
    def __init__(self, output_dir: Path, config=None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
            config: ProcessingConfig object containing CLI parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        logger.info(f"Report generator initialized with output directory: {self.output_dir}")
    
    def _get_cli_parameters(self) -> Dict[str, Any]:
        """Extract CLI parameters from config for reporting."""
        if not self.config:
            return {}
        
        return {
            'text_processing': {
                'mode': getattr(self.config, 'text_mode', 'append'),
                'separator': getattr(self.config, 'text_separator', ';'),
                'default_mapping': getattr(self.config, 'default_mapping', '4022-NA')
            },
            'ocr_processing': {
                'mode': getattr(self.config, 'ocr_mode', 'append'),
                'engine': getattr(self.config, 'ocr_engine', 'easyocr'),
                'confidence_min': getattr(self.config, 'confidence_min', 0.4)
            },
            'performance': {
                'use_gpu': getattr(self.config, 'use_gpu', True),
                'max_workers': getattr(self.config, 'max_workers', 8),
                'verbose': getattr(self.config, 'verbose', False)
            },
            'file_paths': {
                'patterns_file': str(getattr(self.config, 'patterns_file', 'patterns.json')),
                'mappings_file': str(getattr(self.config, 'mappings_file', 'mapping.json')),
                'source_dir': str(getattr(self.config, 'source_dir', 'source_documents')),
                'output_dir': str(getattr(self.config, 'output_dir', 'processed')),
                'reports_dir': str(getattr(self.config, 'reports_dir', 'reports')),
                'logs_dir': str(getattr(self.config, 'logs_dir', 'logs'))
            }
        }
    
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
            html_path = self.generate_document_html_report(result)
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
            html_path = self.generate_batch_html_report(batch_result)
            if html_path:
                report_paths['html'] = html_path
            
            logger.info(f"Generated batch summary reports for {batch_result.total_documents} files")
            
        except Exception as e:
            logger.error(f"Error generating batch reports: {e}")
        
        return report_paths
    
    def _generate_document_json_report(self, result: ProcessingResult, base_name: str) -> Optional[Path]:
        """Generate JSON report for individual document."""
        try:
            # Extract document properties
            doc_properties = extract_document_properties(result.input_path)
            
            # Calculate file sizes
            try:
                input_file_size_mb = result.input_path.stat().st_size / (1024 * 1024)
            except Exception:
                input_file_size_mb = 0.0
            
            try:
                output_file_size_mb = result.output_path.stat().st_size / (1024 * 1024) if result.output_path and result.output_path.exists() else 0.0
            except Exception:
                output_file_size_mb = 0.0
            
            report_data = {
                'report_type': 'document',
                'generated_at': datetime.now().isoformat(),
                'document_info': {
                    'input_path': str(result.input_path),
                    'output_path': str(result.output_path) if result.output_path else None,
                    'input_file_size_mb': round(input_file_size_mb, 2),
                    'output_file_size_mb': round(output_file_size_mb, 2),
                    'processing_time': result.processing_time,
                    'success': result.success,
                    'error_message': result.error_message
                },
                'document_properties': doc_properties,
                'processing_summary': {
                    'total_matches': result.total_matches,
                    'text_matches': result.text_matches,
                    'graphics_matches': result.graphics_matches,
                    'image_matches': result.image_matches
                },
                'detailed_results': self._serialize_metadata(result.metadata) if result.metadata else None,
                'cli_parameters': self._get_cli_parameters(),
                'input_path': str(result.input_path) if hasattr(result, 'input_path') else str(result.file_path)
            }
            
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating document JSON report: {e}")
            return None
    
    def generate_document_html_report(self, result: ProcessingResult) -> Path:
        """Generate enhanced HTML report for a single document."""
        try:
            # Get document name
            doc_name = result.input_path.name if hasattr(result, 'input_path') else result.file_path.name
            
            # Create enhanced HTML content
            html_content = self._create_html_header(f"Document Processing Report - {doc_name}")
            
            # Add CLI parameters section
            cli_params = self._get_cli_parameters()
            html_content += self._create_cli_parameters_section(cli_params)
            
            # Add document information section
            html_content += self._create_document_info_section(result)
            
            # Add processing summary section
            html_content += self._create_processing_summary_section(result)
            
            # Add patterns section
            html_content += self._create_patterns_section()
            
            # Add algorithm section
            html_content += self._create_algorithm_section(result)
            
            # Add detailed results section
            html_content += self._create_detailed_results_section(result)
            
            # Processor breakdown and all matches sections removed
            
            # Close HTML
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"{doc_name.replace('.', '_')}_report_{timestamp}.html"
            html_path = self.output_dir / html_filename
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated enhanced HTML report: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating enhanced HTML report: {e}")
            raise
    
    def _generate_batch_json_report(self, batch_result: BatchProcessingResult, base_name: str) -> Optional[Path]:
        """Generate comprehensive JSON report for batch processing."""
        try:
            # Calculate comprehensive statistics
            total_input_size = 0.0
            total_output_size = 0.0
            files_requiring_review = []
            
            # Process each result for detailed statistics
            for result in batch_result.individual_results if hasattr(batch_result, 'individual_results') else batch_result.documents:
                # Calculate file sizes
                try:
                    if hasattr(result, 'input_path'):
                        input_size = result.input_path.stat().st_size / (1024 * 1024)
                    else:
                        input_size = result.file_size / (1024 * 1024)
                    total_input_size += input_size
                except Exception:
                    input_size = 0.0
                
                try:
                    if hasattr(result, 'output_path') and result.output_path:
                        output_size = result.output_path.stat().st_size / (1024 * 1024) if result.output_path.exists() else 0.0
                    else:
                        output_size = 0.0
                    total_output_size += output_size
                except Exception:
                    output_size = 0.0
                
                # Determine if file needs review
                review_needed = False
                review_reasons = []
                
                # Check for high number of matches
                if result.total_matches > 50:
                    review_needed = True
                    review_reasons.append("High number of matches (>50)")
                
                # Check for processing errors
                if not result.success:
                    review_needed = True
                    review_reasons.append("Processing failed")
                
                # Check for large file size changes
                if input_size > 0 and output_size > 0:
                    size_change_percent = abs(output_size - input_size) / input_size * 100
                    if size_change_percent > 20:
                        review_needed = True
                        review_reasons.append(f"Large file size change ({size_change_percent:.1f}%)")
                
                if review_needed:
                    files_requiring_review.append({
                        'file_name': result.input_path.name if hasattr(result, 'input_path') else result.file_path.name,
                        'reasons': review_reasons,
                        'total_matches': result.total_matches if hasattr(result, 'total_matches') else result.matches_found,
                        'processing_time': result.processing_time,
                        'success': result.success if hasattr(result, 'success') else (result.processing_status == "success"),
                        'input_size_mb': round(input_size, 2),
                        'output_size_mb': round(output_size, 2)
                    })
            
            # Create comprehensive batch summary
            batch_summary = {
                'total_files': batch_result.total_documents,
                'successful_files': batch_result.successful_documents,
                'failed_files': batch_result.failed_documents,
                'success_rate': batch_result.get_success_rate(),
                'total_processing_time': batch_result.total_processing_time,
                'average_processing_time': batch_result.get_average_processing_time(),
                'total_input_size_mb': round(total_input_size, 2),
                'total_output_size_mb': round(total_output_size, 2),
                'files_requiring_review': len(files_requiring_review),
                'review_percentage': round(len(files_requiring_review) / batch_result.total_documents * 100, 1) if batch_result.total_documents > 0 else 0.0
            }
            
            # Create detailed performance metrics
            performance_metrics = {
                'processing_efficiency': {
                    'files_per_second': batch_result.total_documents / batch_result.total_processing_time if batch_result.total_processing_time > 0 else 0.0,
                    'average_matches_per_file': batch_result.statistics.total_matches / batch_result.total_documents if batch_result.total_documents > 0 else 0.0,
                    'success_rate_percentage': batch_result.get_success_rate()
                },
                'resource_utilization': {
                    'total_processing_time_seconds': batch_result.total_processing_time,
                    'average_processing_time_per_file': batch_result.get_average_processing_time(),
                    'total_matches_processed': batch_result.statistics.total_matches
                },
                'file_size_analysis': {
                    'total_input_size_mb': round(total_input_size, 2),
                    'total_output_size_mb': round(total_output_size, 2),
                    'average_input_size_mb': round(total_input_size / batch_result.total_documents, 2) if batch_result.total_documents > 0 else 0.0,
                    'average_output_size_mb': round(total_output_size / batch_result.total_documents, 2) if batch_result.total_documents > 0 else 0.0,
                    'total_size_change_mb': round(total_output_size - total_input_size, 2),
                    'average_size_change_percent': round(((total_output_size - total_input_size) / total_input_size * 100) if total_input_size > 0 else 0.0, 1)
                }
            }
            
            # Add actual performance monitor metrics if available
            if hasattr(batch_result, 'performance_metrics') and batch_result.performance_metrics:
                performance_metrics.update({
                    'system_metrics': {
                        'peak_memory_mb': batch_result.performance_metrics.get('peak_memory_mb', 0.0),
                        'peak_cpu_percent': batch_result.performance_metrics.get('peak_cpu_percent', 0.0),
                        'average_memory_mb': batch_result.performance_metrics.get('average_memory_mb', 0.0),
                        'average_cpu_percent': batch_result.performance_metrics.get('average_cpu_percent', 0.0),
                        'total_memory_usage_mb': batch_result.performance_metrics.get('total_memory_usage_mb', 0.0),
                        'processing_time_seconds': batch_result.performance_metrics.get('processing_time_seconds', 0.0)
                    }
                })
            
            # Create detailed activity summary
            activity_summary = {
                'text_processing': {
                    'total_matches': batch_result.statistics.total_text_matches,
                    'average_matches_per_file': round(batch_result.statistics.total_text_matches / batch_result.total_documents, 2) if batch_result.total_documents > 0 else 0.0,
                    'success_rate': round((batch_result.statistics.total_text_matches / batch_result.statistics.total_matches * 100) if batch_result.statistics.total_matches > 0 else 0.0, 1)
                },
                'graphics_processing': {
                    'total_matches': batch_result.statistics.total_graphics_matches,
                    'average_matches_per_file': round(batch_result.statistics.total_graphics_matches / batch_result.total_documents, 2) if batch_result.total_documents > 0 else 0.0,
                    'success_rate': round((batch_result.statistics.total_graphics_matches / batch_result.statistics.total_matches * 100) if batch_result.statistics.total_matches > 0 else 0.0, 1)
                },
                'image_processing': {
                    'total_matches': batch_result.statistics.total_image_matches,
                    'average_matches_per_file': round(batch_result.statistics.total_image_matches / batch_result.total_documents, 2) if batch_result.total_documents > 0 else 0.0,
                    'success_rate': round((batch_result.statistics.total_image_matches / batch_result.statistics.total_matches * 100) if batch_result.statistics.total_matches > 0 else 0.0, 1)
                }
            }
            
            report_data = {
                'report_type': 'batch_summary',
                'generated_at': datetime.now().isoformat(),
                'batch_summary': batch_summary,
                'performance_metrics': performance_metrics,
                'activity_summary': activity_summary,
                'aggregate_statistics': batch_result.statistics.to_dict(),
                'files_requiring_review': files_requiring_review,
                'cli_parameters': self._get_cli_parameters(),
                'system_info': batch_result.system_info if hasattr(batch_result, 'system_info') else None
            }
            
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating batch JSON report: {e}")
            return None
    
    def generate_batch_html_report(self, batch_result: BatchProcessingResult) -> Path:
        """Generate enhanced HTML report for batch processing."""
        try:
            # Create enhanced HTML content
            html_content = self._create_html_header("Batch Processing Summary Report")
            
            # Add CLI parameters section
            cli_params = self._get_cli_parameters()
            html_content += self._create_cli_parameters_section(cli_params)
            
            # Add batch summary section
            html_content += self._create_batch_summary_section(batch_result)
            
            # Add patterns section
            html_content += self._create_patterns_section()
            
            # Add algorithm section (from first result if available)
            if hasattr(batch_result, 'individual_results') and batch_result.individual_results:
                html_content += self._create_algorithm_section(batch_result.individual_results[0])
            elif hasattr(batch_result, 'documents') and batch_result.documents:
                html_content += self._create_algorithm_section(batch_result.documents[0])
            
            # Add performance metrics section
            html_content += self._create_performance_metrics_section(batch_result)
            
            # Add activity summary section
            html_content += self._create_activity_summary_section(batch_result)
            
            # Add files requiring review section
            html_content += self._create_review_files_section(batch_result)
            
            # Close HTML
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"batch_summary_{timestamp}.html"
            html_path = self.output_dir / html_filename
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated enhanced batch HTML report: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating enhanced batch HTML report: {e}")
            raise
    

    

    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Safely serialize metadata for JSON output."""
        if not metadata:
            return {}
        
        serialized = {}
        for key, value in metadata.items():
            try:
                # Handle different types of values
                if isinstance(value, (str, int, float, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, list):
                    serialized[key] = []
                    for item in value:
                        if isinstance(item, dict):
                            serialized[key].append(self._serialize_metadata(item))
                        elif isinstance(item, (str, int, float, bool, type(None))):
                            serialized[key].append(item)
                        else:
                            serialized[key].append(str(item))
                elif isinstance(value, dict):
                    serialized[key] = self._serialize_metadata(value)
                else:
                    # Convert other types to string
                    serialized[key] = str(value)
            except Exception as e:
                logger.warning(f"Failed to serialize metadata key '{key}': {e}")
                serialized[key] = f"<Serialization Error: {str(e)}>"
        
        return serialized
    
    def _extract_matches_from_json_report(self, input_path: Path) -> List[Dict[str, Any]]:
        """Extract match information from the corresponding JSON report."""
        try:
            # Find the most recent JSON report for this file
            base_name = input_path.stem
            json_files = list(self.output_dir.glob(f"{base_name}_report_*.json"))
            
            if not json_files:
                return []
            
            # Get the most recent JSON file
            latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
            
            import json
            with open(latest_json, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Extract matches from the detailed_results
            if 'detailed_results' in report_data and report_data['detailed_results']:
                detailed_results = report_data['detailed_results']
                
                # Look for processor_results.text which contains the match data
                if 'processor_results' in detailed_results and 'text' in detailed_results['processor_results']:
                    text_result_str = detailed_results['processor_results']['text']
                    
                    # Parse the ProcessingResult string to extract metadata
                    # The string format is: ProcessingResult(...metadata={'matches': [...]}...)
                    import re
                    metadata_match = re.search(r"metadata=\{(.*?)\}", text_result_str, re.DOTALL)
                    if metadata_match:
                        metadata_str = "{" + metadata_match.group(1) + "}"
                        try:
                            # Use ast.literal_eval to safely parse the metadata
                            import ast
                            metadata = ast.literal_eval(metadata_str)
                            if 'matches' in metadata:
                                return metadata['matches']
                        except (ValueError, SyntaxError):
                            logger.warning(f"Failed to parse metadata from JSON report: {latest_json}")
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to extract matches from JSON report: {e}")
            return []
    
    def _create_comprehensive_document_report_html(self, result: ProcessingResult) -> str:
        """Create comprehensive document-level report similar to sample_doc_report.html."""
        # Extract document properties
        doc_properties = extract_document_properties(result.input_path) if result.input_path else {}
        
        # Calculate file sizes
        try:
            input_file_size_mb = result.input_path.stat().st_size / (1024 * 1024) if result.input_path and result.input_path.exists() else 0
        except Exception:
            input_file_size_mb = 0
        
        try:
            output_file_size_mb = result.output_path.stat().st_size / (1024 * 1024) if result.output_path and result.output_path.exists() else 0
        except Exception:
            output_file_size_mb = 0
        
        # Get match information - try multiple sources
        matches = []
        
        # First try: direct metadata matches
        if result.metadata and 'matches' in result.metadata:
            matches = result.metadata['matches']
        
        # Second try: extract from JSON report if no matches found
        if not matches and result.input_path:
            matches = self._extract_matches_from_json_report(result.input_path)
        
        # Third try: create basic match info from result data
        if not matches and result.total_matches > 0:
            # Create basic match information from the result
            matches = [{
                'pattern': 'pattern_77_enforced_structure',
                'original_text': '77-XXX-XXXXXXX-XX',
                'replacement_text': '4022-XXX-XXXXXXX-XX',

                'location': 'document',
                'font_info': {
                    'font_family': 'Unknown',
                    'font_size': 'Unknown',
                    'is_bold': False,
                    'is_italic': False
                }
            } for _ in range(result.total_matches)]
        
        # Calculate statistics
        total_matches = result.total_matches
        text_matches = total_matches  # All matches are text matches for now
        graphics_matches = 0
        image_matches = 0
        
        # Count patterns used
        patterns_used = set()
        for match in matches:
            if isinstance(match, dict) and 'pattern' in match:
                patterns_used.add(match['pattern'])
        
        # Count mappings found vs not found
        mappings_found = len([m for m in matches if isinstance(m, dict) and m.get('replacement_text', '').startswith('4022')])
        mappings_not_found = total_matches - mappings_found
        

        
        html = f"""
        <div class="section">
            <h3>Performance Metrics</h3>
            <div class="performance-metrics">
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-value">{input_file_size_mb:.2f} MB</div>
                        <div class="metric-label">File Size</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{result.processing_time:.3f}s</div>
                        <div class="metric-label">Processing Time</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{total_matches}</div>
                        <div class="metric-label">Total Matches</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{text_matches}</div>
                        <div class="metric-label">Text Matches</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{mappings_found}</div>
                        <div class="metric-label">Mappings Found</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{mappings_not_found}</div>
                        <div class="metric-label">Mappings Not Found</div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-value">{len(patterns_used)}</div>
                        <div class="metric-label">Patterns Used</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Document Level Summary</h3>
            <div class="document-summary">
                <table>
                    <thead>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Input Path</strong></td>
                            <td>{result.input_path or 'N/A'}</td>
                            <td>Source document location</td>
                        </tr>
                        <tr>
                            <td><strong>Output Path</strong></td>
                            <td>{result.output_path or 'N/A'}</td>
                            <td>Processed document location</td>
                        </tr>
                        <tr>
                            <td><strong>File Size</strong></td>
                            <td>{input_file_size_mb:.2f} MB</td>
                            <td>Original document size</td>
                        </tr>
                        <tr>
                            <td><strong>Processing Time</strong></td>
                            <td>{result.processing_time:.3f} seconds</td>
                            <td>Total processing duration</td>
                        </tr>
                        <tr>
                            <td><strong>Success Status</strong></td>
                            <td>{'Yes' if result.success else 'No'}</td>
                            <td>Processing completion status</td>
                        </tr>
                        <tr>
                            <td><strong>Processor Type</strong></td>
                            <td>{result.processor_type}</td>
                            <td>Type of processor used</td>
                        </tr>
                        <tr>
                            <td><strong>Total Matches</strong></td>
                            <td>{total_matches}</td>
                            <td>All pattern matches found</td>
                        </tr>
                        <tr>
                            <td><strong>Text Matches</strong></td>
                            <td>{text_matches}</td>
                            <td>Text pattern matches</td>
                        </tr>
                        <tr>
                            <td><strong>Mappings Found</strong></td>
                            <td>{mappings_found}</td>
                            <td>Successful text replacements</td>
                        </tr>
                        <tr>
                            <td><strong>Mappings Not Found</strong></td>
                            <td>{mappings_not_found}</td>
                            <td>Default mapping applied</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
"""
        
        # Add match details if available
        if matches:
            html += self._create_match_details_html(matches)
        
        # Add patterns section
        html += self._create_patterns_section()
        
        # Add algorithm section
        html += self._create_algorithm_section(result)
        
        # Add detailed match information
        if matches:
            html += self._create_detailed_matches_section_html(matches)
        
        return html
    
    def _create_match_details_html(self, matches: List[Dict[str, Any]]) -> str:
        """Create match details section."""
        # Group matches by pattern
        pattern_matches = {}
        for match in matches:
            if isinstance(match, dict):
                pattern = match.get('pattern', 'Unknown')
                if pattern not in pattern_matches:
                    pattern_matches[pattern] = []
                pattern_matches[pattern].append(match)
        
        html = """
        <div class="section">
            <h3>Match Details by Pattern</h3>
            <div class="match-details">
                <table>
                    <thead>
                        <tr>
                            <th>Pattern</th>
                            <th>Matches</th>
                            <th>Mappings Found</th>
                            <th>Mappings Not Found</th>
                            <th>Font Sizes</th>
                            <th>Locations</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for pattern in sorted(pattern_matches.keys()):
            pattern_matches_list = pattern_matches[pattern]
            mappings_found = 0
            font_sizes = set()
            locations = set()
            
            for match in pattern_matches_list:
                if isinstance(match, dict):
                    if match.get('replacement_text', '').startswith('4022'):
                        mappings_found += 1
                    if 'font_info' in match and match['font_info']:
                        font_size = match['font_info'].get('font_size', 'Unknown')
                        if font_size != 'Unknown':
                            font_sizes.add(str(font_size))
                    if 'location' in match:
                        locations.add(match['location'])
            
            mappings_not_found = len(pattern_matches_list) - mappings_found
            font_sizes_str = f"{len(font_sizes)} sizes" if font_sizes else "Unknown"
            locations_str = f"{len(locations)} locations" if locations else "Unknown"
            
            html += f"""
                        <tr>
                            <td><span class="pattern-badge">{pattern}</span></td>
                            <td>{len(pattern_matches_list)}</td>
                            <td>{mappings_found}</td>
                            <td>{mappings_not_found}</td>
                            <td>{font_sizes_str}</td>
                            <td>{locations_str}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
"""
        return html
    
    def _create_detailed_matches_section_html(self, matches: List[Dict[str, Any]]) -> str:
        """Create detailed matches section with font analysis and individual match details."""
        html = """
        <div class="section">
            <h3>Detailed Match Information</h3>
            <div class="match-details">
"""
        
        for i, match in enumerate(matches):
            if isinstance(match, dict):
                original_text = match.get('original_text', 'Unknown')
                replacement_text = match.get('replacement_text', 'Unknown')
                pattern = match.get('pattern', 'Unknown')
                location = match.get('location', 'Unknown')

                font_info = match.get('font_info', {})
                
                # Font information
                font_family = font_info.get('font_family', 'Unknown')
                font_size = font_info.get('font_size', 'Unknown')
                is_bold = font_info.get('is_bold', False)
                is_italic = font_info.get('is_italic', False)
                
                font_str = f"{font_family}, {font_size}pt"
                if is_bold:
                    font_str += ", Bold"
                if is_italic:
                    font_str += ", Italic"
                
                # Determine action
                if replacement_text.startswith(original_text):
                    action = "Append"
                    action_class = "success"
                else:
                    action = "Replace"
                    action_class = "warning"
                
                html += f"""
                <div class="match-detail">
                    <div class="match-header">
                        <span class="pattern-name">{pattern}</span>

                        <span class="{action_class}">{action}</span>
                    </div>
                    <div class="match-content">
                        <div><strong>Original:</strong> <span class="original-text">{original_text}</span></div>
                        <div><strong>Replacement:</strong> <span class="replacement-text">{replacement_text}</span></div>
                        <div><strong>Location:</strong> <span class="location-info">{location}</span></div>
                        <div class="font-info"><strong>Font:</strong> {font_str}</div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        return html
    
    def _create_detailed_matches_table_html(self, result) -> str:
        try:
            matches = []
            all_detections = []
            
            if hasattr(result, 'metadata'):
                metadata = result.metadata
            elif isinstance(result, dict):
                metadata = result
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return "<p>No matches found.</p>"
            
            # Ensure metadata is a dictionary, not a ProcessingResult object
            if hasattr(metadata, 'metadata'):
                metadata = metadata.metadata
            
            # Try to get matches from processor results (awi version structure)
            if metadata and isinstance(metadata, dict) and 'processor_results' in metadata:
                processor_results = metadata['processor_results']
                
                # Get text processor results
                if isinstance(processor_results, dict) and 'text' in processor_results:
                    text_result = processor_results['text']
                    
                    # Handle case where text_result is now a dictionary (fixed structure)
                    if isinstance(text_result, dict):
                        # The text_result is now directly the metadata dictionary
                        matches = text_result.get('matches', [])
                        all_detections = text_result.get('all_detections', [])
                    
                    # Handle case where text_result is a string representation of ProcessingResult (legacy)
                    elif isinstance(text_result, str):
                        # Parse the string representation to extract metadata
                        import re
                        # Look for metadata= followed by a dictionary-like structure
                        metadata_match = re.search(r"metadata=\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", text_result, re.DOTALL)
                        if metadata_match:
                            metadata_str = "{" + metadata_match.group(1) + "}"
                            try:
                                import ast
                                text_metadata = ast.literal_eval(metadata_str)
                                if isinstance(text_metadata, dict):
                                    matches = text_metadata.get('matches', [])
                                    all_detections = text_metadata.get('all_detections', [])
                            except (ValueError, SyntaxError):
                                logger.warning(f"Failed to parse metadata from string: {text_result[:100]}...")
                                # Try a simpler approach - extract just the matches list
                                matches_match = re.search(r"'matches':\s*\[(.*?)\]", text_result, re.DOTALL)
                                if matches_match:
                                    try:
                                        matches_str = "[" + matches_match.group(1) + "]"
                                        matches = ast.literal_eval(matches_str)
                                    except (ValueError, SyntaxError):
                                        logger.warning("Failed to parse matches list from string")
                    
                    # Handle case where text_result is a dictionary (old structure with 'metadata' key)
                    elif isinstance(text_result, dict) and 'metadata' in text_result:
                        text_metadata = text_result['metadata']
                        if isinstance(text_metadata, dict):
                            matches = text_metadata.get('matches', [])
                            all_detections = text_metadata.get('all_detections', [])
                
                # Get graphics processor results
                if isinstance(processor_results, dict) and 'graphics' in processor_results:
                    graphics_result = processor_results['graphics']
                    
                    # Handle graphics processor results
                    if isinstance(graphics_result, dict):
                        graphics_matches = graphics_result.get('matches', [])
                        graphics_detections = graphics_result.get('all_detections', [])
                        
                        # Add graphics matches to the main matches list
                        if graphics_matches:
                            if not matches:
                                matches = []
                            matches.extend(graphics_matches)
                        
                        # Add graphics detections to the main detections list
                        if graphics_detections:
                            if not all_detections:
                                all_detections = []
                            all_detections.extend(graphics_detections)
            
            # Fallback: try to get matches directly from metadata (awi version structure)
            if not matches and metadata and isinstance(metadata, dict):
                if 'matches' in metadata:
                    matches = metadata['matches']
                if 'all_detections' in metadata:
                    all_detections = metadata['all_detections']
            
            # Ensure matches and all_detections are lists before processing
            if not isinstance(matches, list):
                logger.warning(f"Matches is not a list: {type(matches)}")
                matches = []
            
            if not isinstance(all_detections, list):
                logger.warning(f"All_detections is not a list: {type(all_detections)}")
                all_detections = []
            
            # Create a set of matched texts for quick lookup (check both field names)
            matched_texts = set()
            matched_locations = set()
            
            for match in matches:
                if isinstance(match, dict):
                    matched_texts.add(match.get('original_text', ''))
                    matched_locations.add(match.get('location', ''))
                else:
                    matched_texts.add(getattr(match, 'original_text', ''))
                    matched_locations.add(getattr(match, 'location', ''))
            
            # Build rows exclusively from all_detections to ensure 1:1 XML↔detection
            all_items = []
            
            # Preferred composite key for dedupe across processors: (matched_text, location, processor, start_pos, end_pos)
            seen_keys = set()
            for detection in all_detections:
                data = detection if isinstance(detection, dict) else {}
                key = (
                    data.get('matched_text', ''),
                    data.get('location', ''),
                    data.get('processor', ''),
                    data.get('start_pos', None),
                    data.get('end_pos', None)
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                is_matched = data.get('is_matched', False)
                all_items.append({
                    'type': 'matched' if is_matched else 'non_matched',
                    'data': data
                })
            
            # Sort all items by location
            try:
                all_items.sort(key=lambda x: x['data'].get('location', '') if isinstance(x['data'], dict) else getattr(x['data'], 'location', ''))
            except Exception as e:
                logger.warning(f"Error sorting all items: {e}")
            
            if not all_items:
                return "<p>No matches or detections found.</p>"
            
            # Count matched vs non-matched
            matched_count = len([item for item in all_items if item['type'] == 'matched'])
            non_matched_count = len([item for item in all_items if item['type'] == 'non_matched'])
            
            # Create enhanced table with container
            html = f"""
            <div class="table-header">
                <h3>Detailed Match Information ({matched_count} matched, {non_matched_count} non-matched)</h3>
                <div class="table-stats">
                    Total matches: {matched_count} | Non-matched detections: {non_matched_count} | Total items: {len(all_items)}
                </div>
                <div class="patterns-info">
                    <h4>Patterns Used:</h4>
                    <ul>
                        <li><strong>pattern_77_enforced_structure:</strong> 77-[0-9]{{3}}-[A-Za-z0-9]{{6,7}}-[0-9]{{2,3}}</li>
                        <li><strong>pattern_77_two_dash_structure:</strong> 77-[0-9]{{3}}-[A-Za-z0-9]+</li>
                    </ul>
                </div>
            </div>
            
            <!-- Filter Controls -->
            <div class="filter-controls">
                <div class="filter-group">
                    <label for="processor-filter">Processor Type:</label>
                    <select id="processor-filter" onchange="filterTable()">
                        <option value="All">All</option>
                        <option value="Text">Text</option>
                        <option value="Graphics">Graphics</option>
                        <option value="Image">Image</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="match-filter">Match Status:</label>
                    <select id="match-filter" onchange="filterTable()">
                        <option value="All">All</option>
                        <option value="Y">Y</option>
                        <option value="N">N</option>
                    </select>
                </div>
                <div class="filter-group">
                    <button onclick="clearFilters()" class="clear-filters-btn">Clear Filters</button>
                </div>
            </div>
            
            <div class="table-container">
                <table class="match-table">
                    <thead>
                        <tr>
                            <th>Sr. No</th>
                            <th>Text</th>
                            <th>Match</th>
                            <th>Type</th>
                            <th>Dimension</th>
                            <th>Processor</th>
                            <th>Font</th>
                            <th>Font Size</th>
                            <th>Color</th>
                            <th>Mapped Text</th>
                            <th>Derived Font</th>
                            <th>Derived Font Size</th>
                            <th>Font Size Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for i, item in enumerate(all_items, 1):
                item_type = item['type']
                data = item['data']
                
                # Debug: Log the data structure for first few items
                if i <= 3:
                    logger.debug(f"Item {i} ({item_type}) data structure: {type(data)}")
                    if isinstance(data, dict):
                        logger.debug(f"Item {i} keys: {list(data.keys())}")
                        if 'font_info' in data:
                            logger.debug(f"Item {i} font_info keys: {list(data['font_info'].keys()) if isinstance(data['font_info'], dict) else 'Not a dict'}")
                        if 'processor' in data:
                            logger.debug(f"Item {i} processor: {data['processor']}")
                
                # Debug: Log graphics processor items specifically
                if isinstance(data, dict) and data.get('processor') == 'Graphics':
                    logger.debug(f"Graphics item {i}: processor={data.get('processor')}, has_font_reasoning={'font_reasoning' in data.get('font_info', {})}")
                
                # Extract information based on item type
                if item_type == 'matched':
                    # Extract match information with better error handling
                    pattern = data.get('pattern', 'Unknown') if isinstance(data, dict) else getattr(data, 'pattern', 'Unknown')
                    actual_pattern = data.get('actual_pattern', pattern) if isinstance(data, dict) else getattr(data, 'actual_pattern', pattern)
                    # Handle both field names - graphics processor uses 'original_text' in matches, 'matched_text' in all_detections
                    original_text = (data.get('original_text') or data.get('matched_text', 'Unknown')) if isinstance(data, dict) else (getattr(data, 'original_text', None) or getattr(data, 'matched_text', 'Unknown'))
                    location = data.get('location', 'Unknown') if isinstance(data, dict) else getattr(data, 'location', 'Unknown')
                    replacement_text = data.get('replacement_text', 'N/A') if isinstance(data, dict) else getattr(data, 'replacement_text', 'N/A')
                    content_type = data.get('content_type', 'Unknown') if isinstance(data, dict) else getattr(data, 'content_type', 'Unknown')
                    dimension = data.get('dimension', '') if isinstance(data, dict) else getattr(data, 'dimension', '')
                    processor = data.get('processor', 'Text') if isinstance(data, dict) else getattr(data, 'processor', 'Text')
                    
                    # Extract font information
                    font_info = data.get('font_info', {}) if isinstance(data, dict) else getattr(data, 'font_info', {})
                    if not isinstance(font_info, dict):
                        font_info = {}
                    
                    font_family = font_info.get('font_family', 'Unknown')
                    font_size = font_info.get('font_size', 'Unknown')
                    font_color = font_info.get('color', 'Default')
                    
                    # Extract derived font information (the font after processing)
                    derived_font_family = font_info.get('font_family', font_family)
                    derived_font_size = font_info.get('optimal_font_size', font_info.get('font_size', font_size))
                    
                    # Determine font style from bold/italic/underline flags
                    is_bold = font_info.get('is_bold', False)
                    is_italic = font_info.get('is_italic', False)
                    is_underline = font_info.get('is_underline', False)
                    
                    # Create style string
                    style_parts = []
                    if is_bold:
                        style_parts.append('Bold')
                    if is_italic:
                        style_parts.append('Italic')
                    if is_underline:
                        style_parts.append('Underline')
                    
                    font_style = ' '.join(style_parts) if style_parts else 'Normal'
                    
                    # Determine match status: if mapped to default mapping (e.g., 4022-NA), show as N
                    try:
                        default_mapping_value = getattr(self.config, 'default_mapping', '4022-NA') if hasattr(self, 'config') else '4022-NA'
                    except Exception:
                        default_mapping_value = '4022-NA'
                    
                    # Check if this item has an explicit is_matched field
                    is_matched = data.get('is_matched', False) if isinstance(data, dict) else getattr(data, 'is_matched', False)
                    
                    # Extract font reasoning
                    font_reasoning = data.get('font_reasoning', 'N/A') if isinstance(data, dict) else getattr(data, 'font_reasoning', 'N/A')
                    
                    # If font_reasoning is not at top level, try to get it from font_info
                    if font_reasoning == 'N/A' and isinstance(font_info, dict):
                        font_reasoning = font_info.get('font_reasoning', 'N/A')
                    
                    # Debug: Log font reasoning extraction for first few items
                    if i <= 3:
                        logger.debug(f"Item {i} font_reasoning extraction: top_level='{data.get('font_reasoning', 'NOT_FOUND') if isinstance(data, dict) else 'NOT_DICT'}' font_info='{font_info.get('font_reasoning', 'NOT_FOUND') if isinstance(font_info, dict) else 'NOT_DICT'}' final='{font_reasoning}'")
                        logger.debug(f"Item {i} text extraction: original_text='{original_text}' pattern='{pattern}' replacement_text='{replacement_text}'")
                    
                    # Format font reasoning if it's a dictionary
                    if isinstance(font_reasoning, dict):
                        font_reasoning = self._format_font_reasoning(font_reasoning)
                    
                    # If replacement_text equals the default mapping, consider it not a concrete mapping
                    # But if is_matched is True, show as Y
                    if is_matched:
                        match_status = "Y"
                    elif isinstance(replacement_text, str) and replacement_text == default_mapping_value:
                        match_status = "N"
                    else:
                        match_status = "Y"
                
                else:  # non_matched
                    # Extract detection information
                    pattern = data.get('pattern_name', 'Unknown') if isinstance(data, dict) else getattr(data, 'pattern_name', 'Unknown')
                    actual_pattern = data.get('actual_pattern', pattern) if isinstance(data, dict) else getattr(data, 'actual_pattern', pattern)
                    original_text = data.get('matched_text', 'Unknown') if isinstance(data, dict) else getattr(data, 'matched_text', 'Unknown')
                    location = data.get('location', 'Unknown') if isinstance(data, dict) else getattr(data, 'location', 'Unknown')
                    replacement_text = 'No mapping found'
                    content_type = data.get('content_type', 'Unknown') if isinstance(data, dict) else getattr(data, 'content_type', 'Unknown')
                    dimension = data.get('dimension', '') if isinstance(data, dict) else getattr(data, 'dimension', '')
                    processor = data.get('processor', 'Text') if isinstance(data, dict) else getattr(data, 'processor', 'Text')
                    
                    # For non-matched items, try to get font info from the detection
                    font_info = data.get('font_info', {}) if isinstance(data, dict) else getattr(data, 'font_info', {})
                    if not isinstance(font_info, dict):
                        font_info = {}
                    
                    font_family = font_info.get('font_family', 'Unknown')
                    font_size = font_info.get('font_size', 'Unknown')
                    font_color = font_info.get('color', 'Default')
                    
                    # Extract derived font information (the font after processing)
                    derived_font_family = font_info.get('font_family', font_family)
                    derived_font_size = font_info.get('optimal_font_size', font_info.get('font_size', font_size))
                    
                    # Determine font style from bold/italic/underline flags
                    is_bold = font_info.get('is_bold', False)
                    is_italic = font_info.get('is_italic', False)
                    is_underline = font_info.get('is_underline', False)
                    
                    # Create style string
                    style_parts = []
                    if is_bold:
                        style_parts.append('Bold')
                    if is_italic:
                        style_parts.append('Italic')
                    if is_underline:
                        style_parts.append('Underline')
                    
                    font_style = ' '.join(style_parts) if style_parts else 'Normal'
                    
                    # Extract font reasoning for non-matched items
                    font_reasoning = data.get('font_reasoning', 'N/A') if isinstance(data, dict) else getattr(data, 'font_reasoning', 'N/A')
                    
                    # If font_reasoning is not at top level, try to get it from font_info
                    if font_reasoning == 'N/A' and isinstance(font_info, dict):
                        font_reasoning = font_info.get('font_reasoning', 'N/A')
                    
                    # Debug: Log font reasoning extraction for first few items
                    if i <= 3:
                        logger.debug(f"Item {i} font_reasoning extraction: top_level='{data.get('font_reasoning', 'NOT_FOUND') if isinstance(data, dict) else 'NOT_DICT'}' font_info='{font_info.get('font_reasoning', 'NOT_FOUND') if isinstance(font_info, dict) else 'NOT_DICT'}' final='{font_reasoning}'")
                        logger.debug(f"Item {i} text extraction: original_text='{original_text}' pattern='{pattern}' replacement_text='{replacement_text}'")
                    
                    # Format font reasoning if it's a dictionary
                    if isinstance(font_reasoning, dict):
                        font_reasoning = self._format_font_reasoning(font_reasoning)
                    
                    match_status = "N"
                
                # Add row styling for non-matched items
                row_class = "non-matched-row" if item_type == 'non_matched' else ""
                
                html += f"""
                        <tr class="{row_class}">
                            <td>{i}</td>
                            <td class="text-truncate" title="{original_text}">{original_text}</td>
                            <td>{match_status}</td>
                            <td>{content_type}</td>
                            <td>{dimension}</td>
                            <td>{processor}</td>
                            <td>{font_family}</td>
                            <td>{font_size}</td>
                            <td>{font_color}</td>
                            <td class="text-truncate" title="{replacement_text}">{replacement_text}</td>
                            <td>{derived_font_family}</td>
                            <td>{derived_font_size}</td>
                            <td class="reasoning-cell" title="{font_reasoning}">{font_reasoning}</td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            
            <script>
            function filterTable() {
                const processorFilter = document.getElementById('processor-filter').value;
                const matchFilter = document.getElementById('match-filter').value;
                const table = document.querySelector('.match-table');
                const rows = table.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    const processorCell = row.cells[5]; // Processor column (6th column, index 5)
                    const matchCell = row.cells[2];     // Match column (3rd column, index 2)
                    
                    const processor = processorCell ? processorCell.textContent.trim() : '';
                    const matchStatus = matchCell ? matchCell.textContent.trim() : '';
                    
                    let showRow = true;
                    
                    // Filter by processor type
                    if (processorFilter !== 'All' && processor !== processorFilter) {
                        showRow = false;
                    }
                    
                    // Filter by match status
                    if (matchFilter !== 'All' && matchStatus !== matchFilter) {
                        showRow = false;
                    }
                    
                    // Show/hide row
                    if (showRow) {
                        row.classList.remove('hidden-row');
                    } else {
                        row.classList.add('hidden-row');
                    }
                });
                
                // Update row numbers for visible rows
                updateRowNumbers();
            }
            
            function clearFilters() {
                document.getElementById('processor-filter').value = 'All';
                document.getElementById('match-filter').value = 'All';
                
                const table = document.querySelector('.match-table');
                const rows = table.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    row.classList.remove('hidden-row');
                });
                
                updateRowNumbers();
            }
            
            function updateRowNumbers() {
                const table = document.querySelector('.match-table');
                const rows = table.querySelectorAll('tbody tr:not(.hidden-row)');
                
                rows.forEach((row, index) => {
                    const firstCell = row.cells[0];
                    if (firstCell) {
                        firstCell.textContent = index + 1;
                    }
                });
            }
            </script>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating detailed matches table: {e}")
            return f"<p>Error creating detailed matches table: {e}</p>"
    
    def _create_comprehensive_detections_table(self, all_detections, processed_matches) -> str:
        """
        Create comprehensive table showing all detections (including skipped ones) sorted by location.
        
        Args:
            all_detections: List of all detections (including skipped)
            processed_matches: List of processed matches
            
        Returns:
            HTML string for the comprehensive detections table
        """
        try:
            if not all_detections:
                return "<p>No detections found.</p>"
            
            # Sort all detections by location (handle both dict and object formats)
            try:
                all_detections.sort(key=lambda x: x.get('location', '') if isinstance(x, dict) else getattr(x, 'location', ''))
            except Exception as e:
                logger.warning(f"Error sorting all_detections: {e}")
            
            # Create a set of processed match texts for quick lookup
            processed_texts = set()
            try:
                for match in processed_matches:
                    if isinstance(match, dict):
                        # Matches use 'original_text' field
                        processed_texts.add(match.get('original_text', ''))
                    else:
                        # Matches use 'original_text' field
                        processed_texts.add(getattr(match, 'original_text', ''))
            except Exception as e:
                logger.warning(f"Error processing matches for text lookup: {e}")
                processed_texts = set()
            
            # Create enhanced table with container
            html = f"""
            <div class="table-header">
                <h3>🔍 All Detections</h3>
                <div class="table-stats">
                    Total detections: {len(all_detections)} | Processed: {len(processed_texts)} | Skipped: {len(all_detections) - len(processed_texts)}
                </div>
            </div>
            <div class="table-container">
                <table class="match-table">
                    <thead>
                        <tr>
                            <th>Sr. No</th>
                            <th>Pattern</th>
                            <th>Detected Text</th>
                            <th>Location</th>
                            <th>Status</th>
                            <th>Replacement</th>
                            <th>Position</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for i, detection in enumerate(all_detections, 1):
                # Determine if this detection was processed or skipped
                detected_text = detection.get('matched_text', 'Unknown') if isinstance(detection, dict) else getattr(detection, 'matched_text', 'Unknown')
                # Since we're in append mode, all matches should be considered "Appended" if they were processed
                status = "Appended" if detected_text in processed_texts else "Skipped"
                status_class = "status-appended" if status == "Appended" else "status-skipped"
                
                # Extract detection information
                pattern = detection.get('pattern_name', 'Unknown') if isinstance(detection, dict) else getattr(detection, 'pattern_name', 'Unknown')
                location = detection.get('location', 'Unknown') if isinstance(detection, dict) else getattr(detection, 'location', 'Unknown')
                replacement = detection.get('replacement_text', 'N/A') if isinstance(detection, dict) else getattr(detection, 'replacement_text', 'N/A')
                position = detection.get('start_pos', 'Unknown') if isinstance(detection, dict) else getattr(detection, 'start_pos', 'Unknown')
                
                html += f"""
                        <tr>
                            <td>{i}</td>
                            <td class="text-truncate" title="{pattern}">{pattern}</td>
                            <td class="text-truncate" title="{detected_text}">{detected_text}</td>
                            <td class="text-wrap" title="{location}">{location}</td>
                            <td><span class="{status_class}">{status}</span></td>
                            <td class="text-truncate" title="{replacement}">{replacement}</td>
                            <td>{position}</td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating comprehensive detections table: {e}")
            return f"<p>Error creating comprehensive detections table: {e}</p>"
    

    
    def _create_system_performance_metrics_html(self, batch_result: BatchProcessingResult) -> str:
        """Create HTML for system performance metrics."""
        if not hasattr(batch_result, 'performance_metrics') or not batch_result.performance_metrics:
            return ""
        
        metrics = batch_result.performance_metrics
        
        return f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-title">Peak Memory Usage</div>
                    <div class="metric-value">{metrics.get('peak_memory_mb', 0.0):.1f} MB</div>
                    <small>Maximum memory consumption</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Peak CPU Usage</div>
                    <div class="metric-value">{metrics.get('peak_cpu_percent', 0.0):.1f}%</div>
                    <small>Maximum CPU utilization</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Average Memory</div>
                    <div class="metric-value">{metrics.get('average_memory_mb', 0.0):.1f} MB</div>
                    <small>Average memory consumption</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Average CPU</div>
                    <div class="metric-value">{metrics.get('average_cpu_percent', 0.0):.1f}%</div>
                    <small>Average CPU utilization</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Memory Hours</div>
                    <div class="metric-value">{metrics.get('total_memory_usage_mb', 0.0):.1f} MB-hr</div>
                    <small>Peak memory × processing time</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Processing Time</div>
                    <div class="metric-value">{metrics.get('processing_time_seconds', 0.0):.3f}s</div>
                    <small>Total processing duration</small>
                </div>
            </div>
        """
    
    def _create_batch_html_content(self, batch_result: BatchProcessingResult) -> str:
        """Create comprehensive HTML content for batch report."""
        # Calculate error and warning counts
        error_count = len(batch_result.errors) if hasattr(batch_result, 'errors') and batch_result.errors else 0
        warning_count = len(batch_result.warnings) if hasattr(batch_result, 'warnings') and batch_result.warnings else 0
        
        # Calculate comprehensive statistics
        total_input_size = 0.0
        total_output_size = 0.0
        files_requiring_review = []
        
        for result in (batch_result.individual_results if hasattr(batch_result, 'individual_results') else batch_result.documents):
            try:
                if hasattr(result, 'input_path'):
                    input_size = result.input_path.stat().st_size / (1024 * 1024)
                else:
                    input_size = result.file_size / (1024 * 1024)
                total_input_size += input_size
            except Exception:
                input_size = 0.0
            
            try:
                if hasattr(result, 'output_path') and result.output_path:
                    output_size = result.output_path.stat().st_size / (1024 * 1024) if result.output_path.exists() else 0.0
                else:
                    output_size = 0.0
                total_output_size += output_size
            except Exception:
                output_size = 0.0
            
            # Determine if file needs review
            review_needed = False
            review_reasons = []
            
            total_matches = result.total_matches if hasattr(result, 'total_matches') else result.matches_found
            success = result.success if hasattr(result, 'success') else (result.processing_status == "success")
            
            if total_matches > 50:
                review_needed = True
                review_reasons.append("High number of matches (>50)")
            
            if not success:
                review_needed = True
                review_reasons.append("Processing failed")
            
            if input_size > 0 and output_size > 0:
                size_change_percent = abs(output_size - input_size) / input_size * 100
                if size_change_percent > 20:
                    review_needed = True
                    review_reasons.append(f"Large file size change ({size_change_percent:.1f}%)")
            
            if review_needed:
                files_requiring_review.append({
                    'file_name': result.input_path.name if hasattr(result, 'input_path') else result.file_path.name,
                    'reasons': review_reasons,
                    'total_matches': total_matches,
                    'processing_time': result.processing_time,
                    'success': success,
                    'input_size_mb': round(input_size, 2),
                    'output_size_mb': round(output_size, 2)
                })
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Batch Processing Summary Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .section h3 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .warning {{ color: orange; }}
        .stats {{ display: flex; justify-content: space-around; background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; flex-wrap: wrap; }}
        .stat-item {{ text-align: center; margin: 10px; min-width: 120px; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #ddd; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: #4CAF50; transition: width 0.3s ease; }}
        .review-needed {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .review-warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .review-good {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
        .file-size-info {{ font-size: 12px; color: #666; }}
        .confidence-indicator {{ 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            margin-right: 8px; 
        }}
        .confidence-high {{ background-color: #4caf50; }}
        .confidence-medium {{ background-color: #ff9800; }}
        .confidence-low {{ background-color: #f44336; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-title {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
        .metric-value {{ font-size: 18px; color: #007bff; }}
        
        /* Match Table Styles */
        .match-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 14px;
            table-layout: auto;
        }
        
        .match-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .match-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            font-size: 13px;
            line-height: 1.4;
        }
        
        .match-table tr:hover {
            background-color: #f8f9fa;
        }
        
        /* Hidden row styling for filtering */
        .hidden-row {
            display: none;
        }
        
        .match-table tr:last-child td {
            border-bottom: none;
        }
        
        /* Non-matched row styling */
        .non-matched-row {
            background-color: #f5f5f5;
            opacity: 0.9;
        }
        
        .non-matched-row:hover {
            background-color: #e0e0e0;
            opacity: 1;
        }
        
        /* Patterns info styling */
        .patterns-info {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .patterns-info h4 {
            color: #495057;
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: 600;
        }
        
        .patterns-info ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .patterns-info li {
            margin: 5px 0;
            color: #6c757d;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .patterns-info strong {
            color: #495057;
            font-weight: 600;
        }
        
        /* Filter Controls Styling */
        .filter-controls {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .filter-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .filter-group label {
            font-weight: 600;
            color: #495057;
            margin: 0;
        }
        
        .filter-group select {
            padding: 6px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
            min-width: 120px;
        }
        
        .clear-filters-btn {
            padding: 6px 16px;
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .clear-filters-btn:hover {
            background-color: #5a6268;
        }
        
        /* Hidden row styling for filtering */
        .hidden-row {
            display: none;
        }
        
        /* Column width optimization for better readability */
        .match-table th:nth-child(1), .match-table td:nth-child(1) { width: 3%; min-width: 40px; }   /* Sr. No */
        .match-table th:nth-child(2), .match-table td:nth-child(2) { width: 15%; min-width: 120px; }   /* Pattern */
        .match-table th:nth-child(3), .match-table td:nth-child(3) { width: 10%; min-width: 120px; }  /* Text */
        .match-table th:nth-child(4), .match-table td:nth-child(4) { width: 8%; min-width: 100px; }  /* Match */
        .match-table th:nth-child(5), .match-table td:nth-child(5) { width: 2%; min-width: 30px; }   /* Match */
        .match-table th:nth-child(6), .match-table td:nth-child(6) { width: 5%; min-width: 60px; }   /* Type */
        .match-table th:nth-child(7), .match-table td:nth-child(7) { width: 6%; min-width: 80px; }   /* Dimension */
        .match-table th:nth-child(8), .match-table td:nth-child(8) { width: 5%; min-width: 60px; }   /* Font */
        .match-table th:nth-child(9), .match-table td:nth-child(9) { width: 5%; min-width: 60px; }   /* Font Size */
        .match-table th:nth-child(10), .match-table td:nth-child(10) { width: 5%; min-width: 60px; } /* Color */
        .match-table th:nth-child(11), .match-table td:nth-child(11) { width: 5%; min-width: 60px; } /* Style */
        .match-table th:nth-child(12), .match-table td:nth-child(12) { width: 8%; min-width: 100px; } /* Mapped Text */
        .match-table th:nth-child(13), .match-table td:nth-child(13) { width: 5%; min-width: 60px; }  /* Defied Font */
        .match-table th:nth-child(14), .match-table td:nth-child(14) { width: 5%; min-width: 60px; }  /* Derived Font Size */
        .match-table th:nth-child(15), .match-table td:nth-child(15) { width: 5%; min-width: 60px; }  /* Derived Style */
        .match-table th:nth-child(16), .match-table td:nth-child(16) { width: 25%; min-width: 450px; }  /* Font Size Reasoning */
        
        /* Responsive table with horizontal scroll for smaller screens */
        .table-container {
            width: 100%;
            overflow-x: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 0;
            padding: 0;
        }
        
        /* Remove table layout fixed to allow natural expansion */
        .match-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 11px;
            table-layout: auto;
            min-width: 1600px;
        }
        
        /* Ensure container uses full width */
        .section {
            width: 100%;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        
        /* Main container should use full width */
        .container {
            width: 100%;
            max-width: none;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        
        .table-container::-webkit-scrollbar {
            height: 8px;
        }
        
        .table-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .table-container::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        .table-container::-webkit-scrollbar-thumb:hover {
            background: #5a6fd8;
        }
        
        /* Enhanced status styling */
        .status-appended {
            color: #27ae60;
            font-weight: 600;
            background: rgba(39, 174, 96, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        
        .status-skipped {
            color: #95a5a6;
            font-weight: 600;
            background: rgba(149, 165, 166, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        
        .status-processed {
            color: #3498db;
            font-weight: 600;
            background: rgba(52, 152, 219, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        
        /* Text truncation for long content */
        .text-truncate {
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .text-wrap {
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
        }
        
        /* Enhanced table header */
        .table-header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
            border-bottom: 2px solid #667eea;
        }
        
        .table-header h3 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.4em;
            font-weight: 600;
        }
        
        .table-header .table-stats {
            margin-top: 8px;
            font-size: 0.9em;
            color: #666;
        }
        
        /* Ensure table cells don't wrap text */
        .match-table td, .match-table th {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 6px 4px;
        }
        
        /* Reasoning cell styling for long text */
        .reasoning-cell {
            max-width: 450px;
            min-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            overflow-x: hidden;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-all;
            font-size: 10px;
            line-height: 1.4;
            padding: 12px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            text-align: left;
            vertical-align: top;
            display: block;
            box-sizing: border-box;
        }
        
        .reasoning-cell::-webkit-scrollbar {
            width: 6px;
        }
        
        .reasoning-cell::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .reasoning-cell::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .reasoning-cell::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Batch Processing Summary Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Files Processed:</strong> {batch_result.total_documents}</p>
            <p><strong>Files Requiring Review:</strong> <span class="warning">{len(files_requiring_review)} ({round(len(files_requiring_review) / batch_result.total_documents * 100, 1) if batch_result.total_documents > 0 else 0.0}%)</span></p>
        </div>
        
        <div class="section">
            <h3>Overall Performance Summary</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{batch_result.total_documents}</div>
                    <div class="stat-label">Total Files</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.successful_documents}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.failed_documents}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.get_success_rate():.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.total_processing_time:.1f}s</div>
                    <div class="stat-label">Total Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.get_average_processing_time():.2f}s</div>
                    <div class="stat-label">Avg Time/File</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{round(total_input_size, 1)} MB</div>
                    <div class="stat-label">Total Input Size</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{round(total_output_size, 1)} MB</div>
                    <div class="stat-label">Total Output Size</div>
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <p><strong>Success Rate:</strong></p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {batch_result.get_success_rate()}%;"></div>
                </div>
                <p style="text-align: center; margin-top: 5px;">{batch_result.get_success_rate():.1f}%</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Detailed Performance Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-title">Processing Efficiency</div>
                    <div class="metric-value">{batch_result.total_documents / batch_result.total_processing_time:.2f} files/sec</div>
                    <small>Processing speed</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Match Processing</div>
                    <div class="metric-value">{batch_result.total_matches_found / batch_result.total_documents:.1f} matches/file</div>
                    <small>Average matches per file</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">File Size Analysis</div>
                    <div class="metric-value">{round(((total_output_size - total_input_size) / total_input_size * 100) if total_input_size > 0 else 0.0, 1)}% change</div>
                    <small>Average size change</small>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Processing Activity</div>
                    <div class="metric-value">{batch_result.total_matches_found} total matches</div>
                    <small>Total matches found across all documents</small>
                </div>
            </div>
            
            {self._create_system_performance_metrics_html(batch_result) if hasattr(batch_result, 'performance_metrics') and batch_result.performance_metrics else ''}
        </div>
        
        <div class="section">
            <h3>Processing Activity Breakdown</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{batch_result.total_matches_found}</div>
                    <div class="stat-label">Text Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">0</div>
                    <div class="stat-label">Graphics Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">0</div>
                    <div class="stat-label">Image Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{batch_result.total_matches_found}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
            </div>
        </div>
        
        {self._create_patterns_section()}
        
        {self._create_algorithm_section(batch_result.individual_results[0] if hasattr(batch_result, 'individual_results') and batch_result.individual_results else None)}
        
        <div class="section batch-summary-section">
            <h2><i class="fas fa-tasks"></i> Batch Summary</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>Total Files</h4>
                    <div class="info-item">
                        <span class="info-label">Total Files:</span>
                        <span class="info-value">{batch_result.total_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Successful Files</h4>
                    <div class="info-item">
                        <span class="info-label">Successful Files:</span>
                        <span class="info-value">{batch_result.successful_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Failed Files</h4>
                    <div class="info-item">
                        <span class="info-label">Failed Files:</span>
                        <span class="info-value">{batch_result.failed_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Success Rate</h4>
                    <div class="info-item">
                        <span class="info-label">Success Rate:</span>
                        <span class="info-value">{batch_result.get_success_rate():.1f}%</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Total Processing Time</h4>
                    <div class="info-item">
                        <span class="info-label">Total Processing Time:</span>
                        <span class="info-value">{batch_result.total_processing_time:.2f}s</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Total Matches</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">{batch_result.statistics.total_matches}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Files Requiring Manual Review ({len(files_requiring_review)})</h3>
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Review Reasons</th>
                        <th>Total Matches</th>
                        <th>Processing Time</th>
                        <th>Input Size</th>
                        <th>Output Size</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for file_info in files_requiring_review:
            status_class = "success" if file_info['success'] else "error"
            status_text = "SUCCESS" if file_info['success'] else "FAILED"
            html += f"""
                    <tr class="review-needed">
                        <td>{file_info['file_name']}</td>
                        <td>{', '.join(file_info['reasons'])}</td>
                        <td>{file_info['total_matches']}</td>
                        <td>{file_info['processing_time']:.2f}s</td>
                        <td>{file_info['input_size_mb']:.2f} MB</td>
                        <td>{file_info['output_size_mb']:.2f} MB</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
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
                        <th>Input Size</th>
                        <th>Output Size</th>
                        <th>Review Needed</th>
                        <th>Error Message</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for result in (batch_result.individual_results if hasattr(batch_result, 'individual_results') else batch_result.documents):
            # Calculate file sizes
            try:
                if hasattr(result, 'input_path'):
                    input_size = result.input_path.stat().st_size / (1024 * 1024)
                else:
                    input_size = result.file_size / (1024 * 1024)
            except Exception:
                input_size = 0.0
            
            try:
                if hasattr(result, 'output_path') and result.output_path:
                    output_size = result.output_path.stat().st_size / (1024 * 1024) if result.output_path.exists() else 0.0
                else:
                    output_size = 0.0
            except Exception:
                output_size = 0.0
            
            # Determine review status
            review_needed = False
            review_reasons = []
            
            total_matches = result.total_matches if hasattr(result, 'total_matches') else result.matches_found
            success = result.success if hasattr(result, 'success') else (result.processing_status == "success")
            
            if total_matches > 50:
                review_needed = True
                review_reasons.append("High matches")
            
            if not success:
                review_needed = True
                review_reasons.append("Failed")
            
            if input_size > 0 and output_size > 0:
                size_change_percent = abs(output_size - input_size) / input_size * 100
                if size_change_percent > 20:
                    review_needed = True
                    review_reasons.append("Size change")
            
            status_class = "success" if success else "error"
            status_text = "SUCCESS" if success else "FAILED"
            review_status = f"<span class='confidence-indicator confidence-low'></span>Yes<br><small>({', '.join(review_reasons)})</small>" if review_needed else "<span class='confidence-indicator confidence-high'></span>No"
            
            # Get match counts - all matches are text matches for now
            text_matches = total_matches
            graphics_matches = 0  # Graphics processor not implemented
            image_matches = 0     # Image processor not implemented
            error_message = result.error_message if hasattr(result, 'error_message') else ''
            file_name = result.input_path.name if hasattr(result, 'input_path') else result.file_path.name
            
            html += f"""
                    <tr class="{'review-needed' if review_needed else 'review-good'}">
                        <td>{file_name}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{result.processing_time:.2f}s</td>
                        <td>{total_matches}</td>
                        <td>{text_matches}</td>
                        <td>{graphics_matches}</td>
                        <td>{image_matches}</td>
                        <td class="file-size-info">{input_size:.2f} MB</td>
                        <td class="file-size-info">{output_size:.2f} MB</td>
                        <td>{review_status}</td>
                        <td class="error">{error_message}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h3>Error and Warning Summary</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-title">Total Errors</div>
                    <div class="metric-value">{error_count}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Warnings</div>
                    <div class="metric-value">{warning_count}</div>
                </div>
            </div>
            
            <h4>Errors:</h4>
            <ul>
"""
        
        for error in batch_result.statistics.errors:
            html += f"                <li class='error'>{error}</li>\n"
        
        html += """
            </ul>
            
            <h4>Warnings:</h4>
            <ul>
"""
        
        for warning in batch_result.statistics.warnings:
            html += f"                <li class='warning'>{warning}</li>\n"
        
        html += """
            </ul>
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
    

    
    def _create_html_header(self, title: str) -> str:
            """Create enhanced HTML header with modern styling."""
            return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                width: 100%;
                max-width: none;
                margin: 0;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }}
            
            .header .subtitle {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 30px;
            }}
            
            .section {{
                margin-bottom: 40px;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 25px;
            }}
            
            .section h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.8em;
                font-weight: 600;
            }}
            
            .section h3 {{
                color: #34495e;
                margin-bottom: 15px;
                font-size: 1.4em;
                font-weight: 500;
            }}
            
            .section h4 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.2em;
                font-weight: 500;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .info-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .info-card h4 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}
            
            .info-item {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            
            .info-item:last-child {{
                border-bottom: none;
            }}
            
            .info-label {{
                font-weight: 600;
                color: #555;
            }}
            
            .info-value {{
                color: #2c3e50;
                font-weight: 500;
            }}
            
            .status-success {{
                color: #27ae60;
                font-weight: 600;
            }}
            
            .status-warning {{
                color: #f39c12;
                font-weight: 600;
            }}
            
            .status-error {{
                color: #e74c3c;
                font-weight: 600;
            }}
            
            .match-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                font-size: 11px;
                table-layout: auto;
                min-width: 1200px;
            }}
            
            .match-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 6px 4px;
                text-align: left;
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            
            .match-table td {{
                padding: 6px 4px;
                border-bottom: 1px solid #eee;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            
            /* CORRECTED AND MORE SPECIFIC RULE FOR THE REASONING CELL */
            .match-table td.reasoning-cell {{
                white-space: normal;
                overflow: visible;
                text-overflow: initial;
                word-wrap: break-word;
                word-break: break-word;
                max-width: 450px;
                vertical-align: top;
            }}
            
            .match-table tr:hover {{
                background-color: #f8f9fa;
            }}
            
            .match-table tr:last-child td {{
                border-bottom: none;
            }}
            
            /* Hidden row styling for filtering */
            .hidden-row {{
                display: none;
            }}
            
            .status-appended {{
                color: #27ae60;
                font-weight: 600;
            }}
            
            .status-skipped {{
                color: #95a5a6;
                font-weight: 600;
            }}
            
            .status-processed {{
                color: #3498db;
                font-weight: 600;
            }}
            
            .filter-controls {{
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                align-items: center;
                flex-wrap: wrap;
            }}
            
            .filter-controls label {{
                font-weight: 600;
                color: #2c3e50;
            }}
            
            .filter-controls select {{
                padding: 8px 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
            }}
            
            .filter-controls button {{
                padding: 8px 16px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }}
            
            .filter-controls button:hover {{
                background: #5a6fd8;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .metric-value {{
                font-size: 2em;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .cli-section {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }}
            
            .cli-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }}
            
            .cli-category {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .cli-category h4 {{
                color: #28a745;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}
            
            /* Patterns Section Styles */
            .patterns-section {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }}
            
            .patterns-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .pattern-card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border-left: 4px solid #667eea;
            }}
            
            .pattern-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            
            .pattern-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                align-items: center;
                gap: 10px;
                position: relative;
            }}
            
            .pattern-header i {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .pattern-header h4 {{
                margin: 0;
                font-size: 1.1em;
                font-weight: 600;
                flex-grow: 1;
            }}
            
            .pattern-type {{
                background: rgba(255,255,255,0.2);
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
            }}
            
            .pattern-content {{
                padding: 20px;
            }}
            
            .pattern-description,
            .pattern-explanation,
            .pattern-structure,
            .pattern-content-info {{
                margin-bottom: 12px;
                line-height: 1.5;
            }}
            
            .pattern-description strong,
            .pattern-explanation strong,
            .pattern-structure strong,
            .pattern-content-info strong {{
                color: #2c3e50;
                font-weight: 600;
            }}
            
            .pattern-regex {{
                margin-bottom: 15px;
            }}
            
            .pattern-regex strong {{
                color: #2c3e50;
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
            }}
            
            .regex-code {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #e74c3c;
                display: block;
                word-break: break-all;
                line-height: 1.4;
            }}
            
            .pattern-examples {{
                margin-top: 15px;
            }}
            
            .pattern-examples strong {{
                color: #2c3e50;
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
            }}
            
            .pattern-examples ul {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            
            .pattern-examples li {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 6px 10px;
                margin-bottom: 5px;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                color: #27ae60;
            }}
            
            .pattern-examples li:last-child {{
                margin-bottom: 0;
            }}
            
            /* Pattern card variants */
            .pattern-enforced {{
                border-left-color: #e74c3c;
            }}
            
            .pattern-enforced .pattern-header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }}
            
            .pattern-two-dash {{
                border-left-color: #f39c12;
            }}
            
            .pattern-two-dash .pattern-header {{
                background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            }}
            
            .pattern-standard {{
                border-left-color: #3498db;
            }}
            
            .pattern-standard .pattern-header {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            }}
            
            .metadata-card {{
                border-left-color: #9b59b6;
            }}
            
            .metadata-card .pattern-header {{
                background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            }}
            
            .error-card {{
                border-left-color: #e74c3c;
            }}
            
            .error-card .pattern-header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }}
            
            /* Algorithm Section Styles */
            .algorithms-section {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }}
            
            .algorithms-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .algorithm-card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border-left: 4px solid #667eea;
            }}
            
            .algorithm-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            
            .algorithm-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                align-items: center;
                gap: 10px;
                position: relative;
                flex-wrap: wrap;
            }}
            
            .algorithm-header i {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .algorithm-header h4 {{
                margin: 0;
                font-size: 1.1em;
                font-weight: 600;
                flex-grow: 1;
            }}
            
            .algorithm-type {{
                background: rgba(255,255,255,0.2);
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
            }}
            
            .algorithm-status {{
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
            }}
            
            .algorithm-status.success {{
                background: rgba(39, 174, 96, 0.2);
                color: #27ae60;
            }}
            
            .algorithm-status.warning {{
                background: rgba(243, 156, 18, 0.2);
                color: #f39c12;
            }}
            
            .algorithm-content {{
                padding: 20px;
            }}
            
            .algorithm-summary {{
                margin-bottom: 15px;
                line-height: 1.5;
            }}
            
            .algorithm-summary strong {{
                color: #2c3e50;
                font-weight: 600;
            }}
            
            .algorithm-metrics {{
                margin-bottom: 15px;
            }}
            
            .algorithm-metrics strong {{
                color: #2c3e50;
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
            }}
            
            .algorithm-metrics ul {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            
            .algorithm-metrics li {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 6px 10px;
                margin-bottom: 5px;
                font-size: 0.9em;
                color: #495057;
            }}
            
            .algorithm-metrics li:last-child {{
                margin-bottom: 0;
            }}
            
            .algorithm-decisions {{
                margin-top: 15px;
            }}
            
            .algorithm-decisions strong {{
                color: #2c3e50;
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
            }}
            
            .algorithm-decisions ul {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            
            .algorithm-decisions li {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 6px 10px;
                margin-bottom: 5px;
                font-size: 0.85em;
                color: #6c757d;
                line-height: 1.4;
            }}
            
            .algorithm-decisions li:last-child {{
                margin-bottom: 0;
            }}
            
            /* Algorithm card variants */
            .algorithm-font {{
                border-left-color: #e74c3c;
            }}
            
            .algorithm-font .algorithm-header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }}
            
            .algorithm-ocr {{
                border-left-color: #3498db;
            }}
            
            .algorithm-ocr .algorithm-header {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            }}
            
            .algorithm-pattern {{
                border-left-color: #f39c12;
            }}
            
            .algorithm-pattern .algorithm-header {{
                background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            }}
            
            .algorithm-standard {{
                border-left-color: #9b59b6;
            }}
            
            .algorithm-standard .algorithm-header {{
                background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            }}
            
            @media (max-w: 768px) {{
                .info-grid, .metrics-grid, .cli-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .header h1 {{
                    font-size: 2em;
                }}
                
                .content {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{title}</h1>
                <div class="subtitle">Document Processing Pipeline Report</div>
            </div>
            <div class="content">
    """

    def _create_cli_parameters_section(self, cli_params: Dict[str, Any]) -> str:
        """Create HTML section for CLI parameters."""
        if not cli_params:
            return ""
        
        html = """
        <div class="section cli-section">
            <h2>Processing Configuration</h2>
            <div class="cli-grid">
"""
        
        # Text Processing Configuration
        if 'text_processing' in cli_params:
            html += """
                <div class="cli-category">
                    <h4>Text Processing</h4>
"""
            text_config = cli_params['text_processing']
            html += f"""
                    <div class="info-item">
                        <span class="info-label">Mode:</span>
                        <span class="info-value">{text_config.get('mode', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Separator:</span>
                        <span class="info-value">{text_config.get('separator', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Default Mapping:</span>
                        <span class="info-value">{text_config.get('default_mapping', 'N/A')}</span>
                    </div>
                </div>
"""
        
        # OCR Processing Configuration
        if 'ocr_processing' in cli_params:
            html += """
                <div class="cli-category">
                    <h4>OCR Processing</h4>
"""
            ocr_config = cli_params['ocr_processing']
            html += f"""
                    <div class="info-item">
                        <span class="info-label">Mode:</span>
                        <span class="info-value">{ocr_config.get('mode', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Engine:</span>
                        <span class="info-value">{ocr_config.get('engine', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Confidence Min:</span>
                        <span class="info-value">{ocr_config.get('confidence_min', 'N/A')}</span>
                    </div>
                </div>
"""
        
        # Performance Configuration
        if 'performance' in cli_params:
            html += """
                <div class="cli-category">
                    <h4>Performance</h4>
"""
            perf_config = cli_params['performance']
            html += f"""
                    <div class="info-item">
                        <span class="info-label">GPU Enabled:</span>
                        <span class="info-value">{'Yes' if perf_config.get('use_gpu', False) else 'No'}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Max Workers:</span>
                        <span class="info-value">{perf_config.get('max_workers', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Verbose Mode:</span>
                        <span class="info-value">{'Yes' if perf_config.get('verbose', False) else 'No'}</span>
                    </div>
                </div>
"""
        
        # File Paths Configuration
        if 'file_paths' in cli_params:
            html += """
                <div class="cli-category">
                    <h4>File Paths</h4>
"""
            paths_config = cli_params['file_paths']
            html += f"""
                    <div class="info-item">
                        <span class="info-label">Patterns File:</span>
                        <span class="info-value">{paths_config.get('patterns_file', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Mappings File:</span>
                        <span class="info-value">{paths_config.get('mappings_file', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Source Directory:</span>
                        <span class="info-value">{paths_config.get('source_dir', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Output Directory:</span>
                        <span class="info-value">{paths_config.get('output_dir', 'N/A')}</span>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        return html
    
    def _create_document_info_section(self, result: ProcessingResult) -> str:
        """Create enhanced document information section."""
        # Get file sizes
        input_size = 0
        output_size = 0
        if hasattr(result, 'input_path') and result.input_path.exists():
            input_size = result.input_path.stat().st_size / (1024 * 1024)
        if hasattr(result, 'output_path') and result.output_path and result.output_path.exists():
            output_size = result.output_path.stat().st_size / (1024 * 1024)
        
        html = """
        <div class="section document-info-section">
            <h2>Document Information</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>File Details</h4>
                    <div class="info-item">
                        <span class="info-label">Document:</span>
                        <span class="info-value">""" + (result.input_path.name if hasattr(result, 'input_path') else result.file_path.name) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Status:</span>
                        <span class="info-value status-success">SUCCESS</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Generated:</span>
                        <span class="info-value">""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>File Information</h4>
                    <div class="info-item">
                        <span class="info-label">Input Path:</span>
                        <span class="info-value">""" + str(result.input_path if hasattr(result, 'input_path') else result.file_path) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Input File Size:</span>
                        <span class="info-value">""" + f"{input_size:.2f} MB" + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Output Path:</span>
                        <span class="info-value">""" + str(result.output_path if hasattr(result, 'output_path') and result.output_path else 'N/A') + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Output File Size:</span>
                        <span class="info-value">""" + f"{output_size:.2f} MB" + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Processing Time:</span>
                        <span class="info-value">""" + f"{result.processing_time:.3f} seconds" + """</span>
                    </div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _create_processing_summary_section(self, result: ProcessingResult) -> str:
        """Create enhanced processing summary section."""
        html = """
        <div class="section processing-summary-section">
            <h2>Processing Summary</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>Total Matches</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">""" + str(result.total_matches if hasattr(result, 'total_matches') else result.matches_found) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Text Matches:</span>
                        <span class="info-value">""" + str(result.text_matches if hasattr(result, 'text_matches') else 0) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Graphics Matches:</span>
                        <span class="info-value">""" + str(result.graphics_matches if hasattr(result, 'graphics_matches') else 0) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Image Matches:</span>
                        <span class="info-value">""" + str(result.image_matches if hasattr(result, 'image_matches') else 0) + """</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Processing Time</h4>
                    <div class="info-item">
                        <span class="info-label">Total Time:</span>
                        <span class="info-value">""" + f"{result.processing_time:.3f} seconds" + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Average Time/Match:</span>
                        <span class="info-value">""" + f"{result.processing_time / result.total_matches if result.total_matches > 0 else 0:.3f} seconds" + """</span>
                    </div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _create_detailed_results_section(self, result: ProcessingResult) -> str:
        """Create enhanced detailed results section."""
        html = """
        <div class="section">
            <h2>Detailed Results</h2>
"""
        
        # Add detailed matches table
        detailed_table = self._create_detailed_matches_table_html(result)
        html += detailed_table
        
        html += """
        </div>
"""
        return html
    

    

    
    def _create_batch_summary_section(self, batch_result: BatchProcessingResult) -> str:
        """Create enhanced batch summary section."""
        html = """
        <div class="section batch-summary-section">
            <h2>Batch Summary</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>Total Files</h4>
                    <div class="info-item">
                        <span class="info-label">Total Files:</span>
                        <span class="info-value">{batch_result.total_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Successful Files</h4>
                    <div class="info-item">
                        <span class="info-label">Successful Files:</span>
                        <span class="info-value">{batch_result.successful_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Failed Files</h4>
                    <div class="info-item">
                        <span class="info-label">Failed Files:</span>
                        <span class="info-value">{batch_result.failed_documents}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Success Rate</h4>
                    <div class="info-item">
                        <span class="info-label">Success Rate:</span>
                        <span class="info-value">{batch_result.get_success_rate():.1f}%</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Total Processing Time</h4>
                    <div class="info-item">
                        <span class="info-label">Total Processing Time:</span>
                        <span class="info-value">{batch_result.total_processing_time:.2f}s</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Total Matches</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">{batch_result.statistics.total_matches}</span>
                    </div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _create_performance_metrics_section(self, batch_result: BatchProcessingResult) -> str:
        """Create enhanced performance metrics section."""
        html = """
        <div class="section performance-metrics-section">
            <h2><i class="fas fa-tachometer-alt"></i> Performance Metrics</h2>
            <div class="info-grid">
                <div class="info-card">
                    <div class="metric-title">Processing Efficiency</div>
                    <div class="metric-value">{batch_result.total_documents / batch_result.total_processing_time:.2f} files/sec</div>
                    <small>Processing speed</small>
                </div>
                <div class="info-card">
                    <div class="metric-title">Match Processing</div>
                    <div class="metric-value">{batch_result.total_matches_found / batch_result.total_documents:.1f} matches/file</div>
                    <small>Average matches per file</small>
                </div>
                <div class="info-card">
                    <div class="metric-title">File Size Analysis</div>
                    <div class="metric-value">{round(((total_output_size - total_input_size) / total_input_size * 100) if total_input_size > 0 else 0.0, 1)}% change</div>
                    <small>Average size change</small>
                </div>
                <div class="info-card">
                    <div class="metric-title">Processing Activity</div>
                    <div class="metric-value">{batch_result.total_matches_found} total matches</div>
                    <small>Total matches found across all documents</small>
                </div>
            </div>
            
            {self._create_system_performance_metrics_html(batch_result) if hasattr(batch_result, 'performance_metrics') and batch_result.performance_metrics else ''}
        </div>
"""
        return html
    
    def _create_activity_summary_section(self, batch_result: BatchProcessingResult) -> str:
        """Create enhanced activity summary section."""
        html = """
        <div class="section activity-summary-section">
            <h2><i class="fas fa-clock"></i> Activity Summary</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>Text Processing</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">""" + str(batch_result.statistics.total_text_matches) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Average per File:</span>
                        <span class="info-value">""" + f"{batch_result.statistics.total_text_matches / batch_result.total_documents if batch_result.total_documents > 0 else 0.0:.2f}" + """</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Graphics Processing</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">""" + str(batch_result.statistics.total_graphics_matches) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Average per File:</span>
                        <span class="info-value">""" + f"{batch_result.statistics.total_graphics_matches / batch_result.total_documents if batch_result.total_documents > 0 else 0.0:.2f}" + """</span>
                    </div>
                </div>
                <div class="info-card">
                    <h4>Image Processing</h4>
                    <div class="info-item">
                        <span class="info-label">Total Matches:</span>
                        <span class="info-value">""" + str(batch_result.statistics.total_image_matches) + """</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Average per File:</span>
                        <span class="info-value">""" + f"{batch_result.statistics.total_image_matches / batch_result.total_documents if batch_result.total_documents > 0 else 0.0:.2f}" + """</span>
                    </div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _create_patterns_section(self) -> str:
        """Create enhanced patterns section with card format."""
        try:
            # Try multiple locations for patterns.json
            patterns_file = None
            possible_paths = [
                Path("/Users/pranavshah/github-repos/ocr/awi/patterns.json"),  # Absolute path
                Path("patterns.json"),  # Current directory
                Path("../patterns.json"),  # Parent directory
                Path("../../patterns.json"),  # Grandparent directory
            ]
            
            # Add config path if available
            if hasattr(self, 'config') and hasattr(self.config, 'patterns_file'):
                possible_paths.insert(0, Path(self.config.patterns_file))
            
            # Try to find the patterns file
            for path in possible_paths:
                if path.exists():
                    patterns_file = path
                    break
            
            if patterns_file and patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
            else:
                # Default fallback patterns
                patterns = {
                    "pattern_77_enforced_structure": "77-[0-9]{3}-[A-Za-z0-9]{6,7}-[0-9]{2,3}",
                    "pattern_77_two_dash_structure": "77-[0-9]{3}-[A-Za-z0-9]+",
                    "_metadata": {
                        "description": "Patterns for 77-*-*-* and 77-*-* format with flexible matching",
                        "pattern_explanation": "Flexible patterns: 77-XXX-XXXXXX(X)-XX format and 77-XXX-XXXXXXX format",
                        "structure_requirement": "Three-dash: 77-[3 digits]-[6-7 alphanumeric]-[2-3 digits], Two-dash: 77-[3 digits]-[alphanumeric]",
                        "section_content": "Flexible length constraints to match various part number formats"
                    }
                }
            
            html = """
        <div class="section patterns-section">
            <h2><i class="fas fa-search"></i> Patterns Used</h2>
            <div class="patterns-grid">
"""
            
            # Add metadata card if available
            if '_metadata' in patterns:
                metadata = patterns['_metadata']
                html += f"""
                <div class="pattern-card metadata-card">
                    <div class="pattern-header">
                        <i class="fas fa-info-circle"></i>
                        <h4>Pattern Information</h4>
                    </div>
                    <div class="pattern-content">
                        <div class="pattern-description">
                            <strong>Description:</strong> {metadata.get('description', 'N/A')}
                        </div>
                        <div class="pattern-explanation">
                            <strong>Explanation:</strong> {metadata.get('pattern_explanation', 'N/A')}
                        </div>
                        <div class="pattern-structure">
                            <strong>Structure:</strong> {metadata.get('structure_requirement', 'N/A')}
                        </div>
                        <div class="pattern-content-info">
                            <strong>Content:</strong> {metadata.get('section_content', 'N/A')}
                        </div>
                    </div>
                </div>
"""
            
            # Add individual pattern cards
            for pattern_name, pattern_regex in patterns.items():
                if pattern_name == '_metadata':
                    continue
                
                # Determine pattern type based on name
                if 'enforced' in pattern_name:
                    pattern_type = "Enforced Structure"
                    icon = "fas fa-shield-alt"
                    color_class = "pattern-enforced"
                elif 'two_dash' in pattern_name:
                    pattern_type = "Two-Dash Structure"
                    icon = "fas fa-minus"
                    color_class = "pattern-two-dash"
                else:
                    pattern_type = "Standard Pattern"
                    icon = "fas fa-code"
                    color_class = "pattern-standard"
                
                html += f"""
                <div class="pattern-card {color_class}">
                    <div class="pattern-header">
                        <i class="{icon}"></i>
                        <h4>{pattern_name.replace('_', ' ').title()}</h4>
                        <span class="pattern-type">{pattern_type}</span>
                    </div>
                    <div class="pattern-content">
                        <div class="pattern-regex">
                            <strong>Regex Pattern:</strong>
                            <code class="regex-code">{pattern_regex}</code>
                        </div>
                        <div class="pattern-examples">
                            <strong>Examples:</strong>
                            <ul>
"""
                
                # Generate example matches based on pattern
                examples = self._generate_pattern_examples(pattern_regex)
                for example in examples:
                    html += f"                                <li><code>{example}</code></li>\n"
                
                html += """
                            </ul>
                        </div>
                    </div>
                </div>
"""
            
            html += """
            </div>
        </div>
"""
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating patterns section: {e}")
            return f"""
        <div class="section patterns-section">
            <h2><i class="fas fa-search"></i> Patterns Used</h2>
            <div class="patterns-grid">
                <div class="pattern-card error-card">
                    <div class="pattern-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h4>Pattern Loading Error</h4>
                    </div>
                    <div class="pattern-content">
                        <p>Error loading patterns: {e}</p>
                    </div>
                </div>
            </div>
        </div>
"""
    
    def _generate_pattern_examples(self, pattern_regex: str) -> List[str]:
        """Generate example matches for a given regex pattern."""
        examples = []
        
        try:
            import re
            
            # Common examples based on pattern type
            if "77-[0-9]{3}-[A-Za-z0-9]{6,7}-[0-9]{2,3}" in pattern_regex:
                examples = [
                    "77-123-ABC123-45",
                    "77-456-DEF4567-12",
                    "77-789-GHI12345-3"
                ]
            elif "77-[0-9]{3}-[A-Za-z0-9]+" in pattern_regex:
                examples = [
                    "77-123-ABC123",
                    "77-456-DEF4567",
                    "77-789-GHI12345"
                ]
            else:
                # Generate generic examples
                examples = [
                    "77-123-ABC123",
                    "77-456-DEF456",
                    "77-789-GHI789"
                ]
            
            # Filter examples to match the actual pattern
            valid_examples = []
            for example in examples:
                if re.match(pattern_regex, example):
                    valid_examples.append(example)
            
            return valid_examples[:3]  # Return up to 3 examples
            
        except Exception as e:
            logger.warning(f"Error generating pattern examples: {e}")
            return ["77-123-ABC123", "77-456-DEF456", "77-789-GHI789"]
    
    def _create_algorithm_section(self, result) -> str:
        """Create enhanced algorithm section with card format."""
        try:
            # Extract algorithm information from processing results
            algorithms = self._extract_algorithm_info(result)
            
            if not algorithms:
                return ""
            
            html = """
        <div class="section algorithms-section">
            <h2><i class="fas fa-cogs"></i> Algorithms Used</h2>
            <div class="algorithms-grid">
"""
            
            for algorithm_name, algorithm_data in algorithms.items():
                # Determine algorithm type and styling
                if 'font' in algorithm_name.lower() or 'baseline' in algorithm_name.lower():
                    algorithm_type = "Font Optimization"
                    icon = "fas fa-font"
                    color_class = "algorithm-font"
                elif 'ocr' in algorithm_name.lower():
                    algorithm_type = "OCR Processing"
                    icon = "fas fa-eye"
                    color_class = "algorithm-ocr"
                elif 'pattern' in algorithm_name.lower():
                    algorithm_type = "Pattern Matching"
                    icon = "fas fa-search"
                    color_class = "algorithm-pattern"
                else:
                    algorithm_type = "Processing Algorithm"
                    icon = "fas fa-cog"
                    color_class = "algorithm-standard"
                
                # Determine success status
                success = algorithm_data.get('success', True)
                status_class = "success" if success else "warning"
                status_text = "Success" if success else "Warning"
                
                html += f"""
                <div class="algorithm-card {color_class}">
                    <div class="algorithm-header">
                        <i class="{icon}"></i>
                        <h4>{algorithm_name}</h4>
                        <span class="algorithm-type">{algorithm_type}</span>
                        <span class="algorithm-status {status_class}">{status_text}</span>
                    </div>
                    <div class="algorithm-content">
"""
                
                # Add summary
                if 'summary' in algorithm_data:
                    html += f"""
                        <div class="algorithm-summary">
                            <strong>Summary:</strong> {algorithm_data['summary']}
                        </div>
"""
                
                # Add key metrics
                if 'metrics' in algorithm_data:
                    html += """
                        <div class="algorithm-metrics">
                            <strong>Key Metrics:</strong>
                            <ul>
"""
                    for metric, value in algorithm_data['metrics'].items():
                        html += f"                                <li><strong>{metric}:</strong> {value}</li>\n"
                    html += """
                            </ul>
                        </div>
"""
                
                # Add decision process overview
                if 'decision_process' in algorithm_data:
                    html += """
                        <div class="algorithm-decisions">
                            <strong>Decision Process:</strong>
                            <ul>
"""
                    for step in algorithm_data['decision_process'][:5]:  # Show first 5 steps
                        html += f"                                <li>{step}</li>\n"
                    if len(algorithm_data['decision_process']) > 5:
                        html += f"                                <li>... and {len(algorithm_data['decision_process']) - 5} more steps</li>\n"
                    html += """
                            </ul>
                        </div>
"""
                
                html += """
                    </div>
                </div>
"""
            
            html += """
            </div>
        </div>
"""
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating algorithm section: {e}")
            return ""
    
    def _extract_algorithm_info(self, result) -> Dict[str, Dict[str, Any]]:
        """Extract algorithm information from processing results."""
        algorithms = {}
        
        try:
            # Get matches from result
            matches = []
            if hasattr(result, 'metadata') and result.metadata:
                if 'processor_results' in result.metadata:
                    processor_results = result.metadata['processor_results']
                    if isinstance(processor_results, dict) and 'text' in processor_results:
                        text_result = processor_results['text']
                        if isinstance(text_result, dict):
                            matches = text_result.get('matches', [])
                        elif isinstance(text_result, str):
                            # Parse string representation
                            import re
                            metadata_match = re.search(r"metadata=\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", text_result, re.DOTALL)
                            if metadata_match:
                                metadata_str = "{" + metadata_match.group(1) + "}"
                                try:
                                    import ast
                                    text_metadata = ast.literal_eval(metadata_str)
                                    if isinstance(text_metadata, dict):
                                        matches = text_metadata.get('matches', [])
                                except (ValueError, SyntaxError):
                                    pass
            
            # Extract algorithm information from matches
            for match in matches:
                if isinstance(match, dict) and 'font_reasoning' in match:
                    font_reasoning = match['font_reasoning']
                    if isinstance(font_reasoning, dict):
                        algorithm_name = font_reasoning.get('algorithm', 'Font Size Optimization')
                        
                        if algorithm_name not in algorithms:
                            algorithms[algorithm_name] = {
                                'summary': font_reasoning.get('summary', ''),
                                'metrics': {},
                                'decision_process': [],
                                'success': True
                            }
                        
                        # Extract metrics
                        if 'final_analysis' in font_reasoning:
                            final = font_reasoning['final_analysis']
                            if isinstance(final, dict):
                                algorithms[algorithm_name]['metrics'].update({
                                    'Chars per sq cm': final.get('chars_per_sqcm', 'N/A'),
                                    'Required area': f"{final.get('required_area_sqcm', 'N/A')} sq cm",
                                    'Available area': f"{final.get('textbox_area_sqcm', 'N/A')} sq cm",
                                    'Utilization': f"{final.get('utilization', 'N/A')}%",
                                    'Fits properly': 'Yes' if final.get('fits_properly', False) else 'No'
                                })
                        
                        # Extract decision process
                        if 'decision_process' in font_reasoning:
                            decision_process = font_reasoning['decision_process']
                            if isinstance(decision_process, list):
                                algorithms[algorithm_name]['decision_process'] = decision_process
                        
                        # Determine success
                        if 'final_analysis' in font_reasoning:
                            final = font_reasoning['final_analysis']
                            if isinstance(final, dict):
                                fits_properly = final.get('fits_properly', True)
                                algorithms[algorithm_name]['success'] = fits_properly
            
            return algorithms
            
        except Exception as e:
            logger.error(f"Error extracting algorithm info: {e}")
            return {}
    
    def _create_review_files_section(self, batch_result: BatchProcessingResult) -> str:
        """Create enhanced files requiring review section."""
        # This would need to be implemented based on your review logic
        html = """
        <div class="section review-files-section">
            <h2><i class="fas fa-folder-open"></i> Review Files</h2>
            <div class="info-grid">
"""
        return html

    def _format_font_reasoning(self, reasoning: Dict[str, Any]) -> str:
        """
        Format font reasoning dictionary into a concise calculation summary.
        
        Args:
            reasoning: Font reasoning dictionary from graphics processor
            
        Returns:
            Formatted string showing only the specific calculation logic
        """
        if not isinstance(reasoning, dict):
            return str(reasoning)
        
        def _to_float(v):
            try:
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    # strip units like '9.08 sq cm' or trailing %
                    s = v.replace('%', '').replace('sq cm', '').strip()
                    return float(s)
            except Exception:
                return None
            return None

        # Extract structured fields
        phase1 = reasoning.get('phase_1_detection', {}) if isinstance(reasoning, dict) else {}
        phase2 = reasoning.get('phase_2_reconstruction', {}) if isinstance(reasoning, dict) else {}
        final = reasoning.get('final_analysis', {}) if isinstance(reasoning, dict) else {}
        baseline = reasoning.get('baseline_calculation', {}) if isinstance(reasoning, dict) else {}

        # Defaults
        orig_chars = phase2.get('original_text_length') or phase2.get('text_length')
        new_chars = phase2.get('text_length')
        orig_size = phase2.get('baseline_font_size')
        orig_font = phase1.get('font_family_detected') or reasoning.get('font_family')
        chosen_size = phase2.get('chosen_font_size')
        utilization = final.get('utilization')  # already a percentage string like '79.0%'
        fits_new = final.get('fits_properly')
        cap = final.get('capacity_metrics', {}) if isinstance(final, dict) else {}
        total_lines_new = cap.get('total_lines_new')
        # We don't reliably have original total lines pre-append; fall back to equal to new if unknown
        total_lines_orig = phase2.get('original_total_lines') or total_lines_new

        # Compute fit at original size using baseline chars-per-sqcm
        over_under_text = None
        will_fit_at_orig = None
        try:
            cps_map = baseline.get('baseline_chars_per_sqcm', {})
            # keys may be strings like '12.0'
            cps_key = str(orig_size) if orig_size is not None else None
            cps_val = cps_map.get(cps_key)
            required_area_at_orig = None
            available_area = _to_float(final.get('textbox_area_sqcm'))
            if cps_val is None and isinstance(cps_map, dict):
                # attempt float compare
                for k, v in cps_map.items():
                    if _to_float(k) == _to_float(orig_size):
                        cps_val = v
                        break
            if cps_val is not None and new_chars is not None and available_area is not None:
                cps_val_f = _to_float(cps_val)
                if cps_val_f and cps_val_f > 0:
                    required_area_at_orig = float(new_chars) / cps_val_f
                    will_fit_at_orig = required_area_at_orig <= available_area
                    ratio = (required_area_at_orig / available_area) if available_area > 0 else None
                    if ratio is not None:
                        if ratio > 1.0:
                            over_under_text = f"{round((ratio - 1.0) * 100)}% over"
                        else:
                            over_under_text = f"{round((1.0 - ratio) * 100)}% under"
        except Exception:
            pass

        # Build human-readable summary
        parts = []
        if orig_chars is not None:
            parts.append(f"orig total char: {orig_chars}")
        if orig_size is not None:
            parts.append(f"orig size: {orig_size}pt")
        if orig_font:
            parts.append(f"orig font: {orig_font}")
        if new_chars is not None:
            parts.append(f"new total char: {new_chars}")
        if will_fit_at_orig is not None:
            parts.append(f"will fit (orig size): {'Y' if will_fit_at_orig else 'N'}{f' ({over_under_text})' if over_under_text else ''}")
        if chosen_size is not None:
            # If baseline exists, show transition (e.g., 12.0->10.0)
            if orig_size is not None and _to_float(orig_size) != _to_float(chosen_size):
                parts.append(f"new size: {orig_size}→{chosen_size}pt")
            else:
                parts.append(f"new size: {chosen_size}pt")
        if total_lines_orig is not None:
            parts.append(f"orig total lines: {total_lines_orig}")
        if total_lines_new is not None:
            parts.append(f"new total lines: {total_lines_new}")
        if fits_new is not None:
            parts.append(f"will fit (new size): {'Y' if fits_new else 'N'}{f' ({utilization} fill)' if utilization else ''}")

        # Fallback to previous concise format if we couldn't assemble the above
        if not parts:
            lines = []
            if isinstance(final, dict):
                if 'chars_per_sqcm' in final:
                    lines.append(f"Chars/sq cm: {final['chars_per_sqcm']}")
                if 'required_area_sqcm' in final and 'textbox_area_sqcm' in final:
                    lines.append(f"Required: {final['required_area_sqcm']} sq cm")
                    lines.append(f"Available: {final['textbox_area_sqcm']} sq cm")
                if 'utilization' in final:
                    lines.append(f"Utilization: {final['utilization']}%")
                if 'fits_properly' in final:
                    lines.append(f"Fits: {'Yes' if final['fits_properly'] else 'No'}")
            if isinstance(phase2, dict):
                if 'chosen_font_size' in phase2:
                    lines.append(f"Font size: {phase2['chosen_font_size']}pt")
                if 'text_length' in phase2:
                    lines.append(f"Text length: {phase2['text_length']} chars")
            if not lines:
                if 'summary' in reasoning:
                    lines.append(reasoning['summary'])
                else:
                    lines.append("Font calculation completed")
            return ' | '.join(lines)

        return ' | '.join(parts)

def create_report_generator(output_dir: Path) -> ReportGenerator:
    """
    Factory function to create a ReportGenerator instance.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        ReportGenerator instance
    """
    return ReportGenerator(output_dir)