
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ..core.models import (
    BatchReport, ProcessingResult, CLIParameters,
    PerformanceMetrics, PatternInfo, ProcessingStatus
)


class ReportGenerator:

    def __init__(self, output_dir: Path, config: Any):
        self.output_dir = Path(output_dir)
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.cli_defaults = self._load_cli_defaults()

    def _create_cli_parameters(self) -> CLIParameters:
        return CLIParameters(
            text_mode=self.config.text_mode,
            text_separator=self.config.text_separator,
            default_mapping=self.config.default_mapping,
            ocr_mode=self.config.ocr_mode,
            ocr_engine=self.config.ocr_engine,
            use_gpu=self.config.use_gpu,
            gpu_device=self.config.gpu_device,
            gpu_available=self.config.gpu_available,
            max_workers=self.config.max_workers,
            confidence_min=self.config.confidence_min,
            verbose=self.config.verbose,
            patterns_file=str(self.config.patterns_file),
            mappings_file=str(self.config.mappings_file),
            source_dir=str(self.config.source_dir),
            output_dir=str(self.config.output_dir),
            reports_dir=str(self.config.reports_dir),
            logs_dir=str(self.config.logs_dir),
            suffix=self.config.suffix,
            processing_timeout=self.config.processing_timeout,
            max_retries=self.config.max_retries,
            min_file_size=self.config.min_file_size,
            max_file_size=self.config.max_file_size
        )

    def _load_cli_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        try:

            from config import get_default_config
            default_cfg = get_default_config()
            defaults = {
                "text_mode": default_cfg.text_mode,
                "text_separator": default_cfg.text_separator,
                "default_mapping": default_cfg.default_mapping,
                "ocr_mode": default_cfg.ocr_mode,
                "ocr_engine": default_cfg.ocr_engine,
                "use_gpu": default_cfg.use_gpu,
                "gpu_device": default_cfg.gpu_device,
                "gpu_available": default_cfg.gpu_available,
                "max_workers": default_cfg.max_workers,
                "confidence_min": default_cfg.confidence_min,
                "verbose": default_cfg.verbose,
                "patterns_file": str(default_cfg.patterns_file),
                "mappings_file": str(default_cfg.mappings_file),
                "source_dir": str(default_cfg.source_dir),
                "output_dir": str(default_cfg.output_dir),
                "reports_dir": str(default_cfg.reports_dir),
                "logs_dir": str(default_cfg.logs_dir),
                "suffix": default_cfg.suffix,
                "processing_timeout": default_cfg.processing_timeout,
                "max_retries": default_cfg.max_retries,
                "min_file_size": default_cfg.min_file_size,
                "max_file_size": default_cfg.max_file_size,
            }
        except Exception:

            defaults = {
                "text_mode": self.config.text_mode,
                "text_separator": self.config.text_separator,
                "default_mapping": self.config.default_mapping,
                "ocr_mode": self.config.ocr_mode,
                "ocr_engine": self.config.ocr_engine,
                "use_gpu": self.config.use_gpu,
                "gpu_device": self.config.gpu_device,
                "gpu_available": self.config.gpu_available,
                "max_workers": self.config.max_workers,
                "confidence_min": self.config.confidence_min,
                "verbose": self.config.verbose,
                "patterns_file": str(self.config.patterns_file),
                "mappings_file": str(self.config.mappings_file),
                "source_dir": str(self.config.source_dir),
                "output_dir": str(self.config.output_dir),
                "reports_dir": str(self.config.reports_dir),
                "logs_dir": str(self.config.logs_dir),
                "suffix": self.config.suffix,
                "processing_timeout": self.config.processing_timeout,
                "max_retries": self.config.max_retries,
                "min_file_size": self.config.min_file_size,
                "max_file_size": self.config.max_file_size,
            }
        return defaults

    def _cli_params_with_defaults(self, cli: CLIParameters) -> List[Dict[str, Any]]:
        mapping = [
            ("Text Mode", "text_mode"),
            ("Text Separator", "text_separator"),
            ("Default Mapping", "default_mapping"),
            ("OCR Mode", "ocr_mode"),
            ("OCR Engine", "ocr_engine"),
            ("Use GPU", "use_gpu"),
            ("GPU Device", "gpu_device"),
            ("GPU Available", "gpu_available"),
            ("Max Workers", "max_workers"),
            ("Confidence Min", "confidence_min"),
            ("Verbose", "verbose"),
            ("Patterns File", "patterns_file"),
            ("Mappings File", "mappings_file"),
            ("Source Directory", "source_dir"),
            ("Output Directory", "output_dir"),
            ("Reports Directory", "reports_dir"),
            ("Logs Directory", "logs_dir"),
            ("Suffix", "suffix"),
            ("Processing Timeout", "processing_timeout"),
            ("Max Retries", "max_retries"),
            ("Min File Size", "min_file_size"),
            ("Max File Size", "max_file_size"),
        ]

        items: List[Dict[str, Any]] = []
        for label, key in mapping:
            current_val = getattr(cli, key)
            default_val = self.cli_defaults.get(key, "")
            items.append({"label": label, "key": key, "value": current_val, "default": default_val})

        return items

    def _render_cli_params_grid(self, cli: CLIParameters) -> str:
        items = self._cli_params_with_defaults(cli)
        cards = ""
        for it in items:

            current_val = str(it['value'])
            default_val = str(it['default'])
            is_user_provided = current_val != default_val


            user_value_line = f'<div class="cli-value">{current_val}</div>' if is_user_provided else ''

            cards += f"""
            <div class="cli-card">
                <div class="cli-label">{it['label']}</div>
                {user_value_line}
                <div class="cli-default">Default: {it['default']}</div>
            </div>
            """
        return f'<div class="cli-grid">{cards}</div>'

    def _create_performance_metrics(self, batch_result: Any) -> PerformanceMetrics:

        if batch_result.performance_metrics:
            return batch_result.performance_metrics


        processing_times = [r.processing_time for r in batch_result.individual_results if hasattr(r, 'processing_time')]
        max_time = max(processing_times) if processing_times else 0.0

        file_sizes = [
            (
                (r.file_summary.doc_file_size or r.file_summary.docx_file_size)
                if r.file_summary else 0
            )
            for r in batch_result.individual_results
        ]
        total_size_bytes = sum(file_sizes)
        avg_size_mb = (total_size_bytes / max(1, batch_result.total_documents)) / (1024 * 1024) if batch_result.total_documents > 0 else 0.0

        skipped = 0

        return PerformanceMetrics(
            processing_time_seconds=batch_result.total_processing_time,
            total_files_processed=batch_result.total_documents,
            successful_files=batch_result.successful_documents,
            failed_files=batch_result.failed_documents,
            skipped_files=skipped,
            total_matches_found=batch_result.total_matches_found,
            average_processing_time_per_file=(
                batch_result.total_processing_time / batch_result.total_documents
                if batch_result.total_documents > 0 else 0.0
            ),
            max_processing_time_seconds=max_time,
            total_file_size_bytes=total_size_bytes,
            average_file_size_mb=avg_size_mb
        )

    def _render_metrics_grid_batch(self, perf: PerformanceMetrics) -> str:
        items = [
            ("Total Processing Time (s)", f"{perf.processing_time_seconds:.2f}"),
            ("Max Processing Time (s)", f"{perf.max_processing_time_seconds:.2f}"),
            ("Total Files", perf.total_files_processed),
            ("Successful Files", perf.successful_files),
            ("Error Files", perf.failed_files),
            ("Skipped Files", perf.skipped_files),
            ("Total File Size (MB)", f"{perf.total_file_size_bytes / (1024*1024):.2f}"),
            ("Avg File Size (MB)", f"{perf.average_file_size_mb:.2f}"),
            ("Avg Time per File (s)", f"{perf.average_processing_time_per_file:.2f}"),
            ("Total Matches Found", perf.total_matches_found),
        ]


        if perf.peak_memory_mb > 0:
            items.extend([
                ("Peak Memory (MB)", f"{perf.peak_memory_mb:.2f}"),
                ("Peak CPU (%)", f"{perf.peak_cpu_percent:.2f}"),
                ("Avg Memory (MB)", f"{perf.average_memory_mb:.2f}"),
                ("Avg CPU (%)", f"{perf.average_cpu_percent:.2f}"),
            ])

        if perf.success_rate_percent > 0:
            items.extend([
                ("Success Rate (%)", f"{perf.success_rate_percent:.2f}"),
                ("Throughput (files/s)", f"{perf.processing_throughput_files_per_second:.3f}"),
            ])

        if perf.gpu_available:
            items.extend([
                ("GPU Available", "Yes"),
                ("GPU Samples", perf.gpu_utilization_samples),
            ])

        cards = "".join([
            f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>'
            for k, v in items
        ])
        return f'<div class="metric-grid">{cards}</div>'

    def _render_metrics_grid_file(self, perf: PerformanceMetrics) -> str:
        items = [
            ("Processing Time (s)", f"{perf.processing_time_seconds:.2f}"),
            ("File Size (MB)", f"{perf.total_file_size_bytes / (1024*1024):.2f}"),
            ("Total Matches Found", perf.total_matches_found),
        ]


        if perf.peak_memory_mb > 0:
            items.extend([
                ("Peak Memory (MB)", f"{perf.peak_memory_mb:.2f}"),
                ("Peak CPU (%)", f"{perf.peak_cpu_percent:.2f}"),
            ])

        cards = "".join([
            f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>'
            for k, v in items
        ])
        return f'<div class="metric-grid">{cards}</div>'

    def _create_pattern_info(self) -> PatternInfo:
        patterns = {}
        total_patterns = 0
        try:
            if self.config.patterns_file.exists():
                with open(self.config.patterns_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)

                patterns = {k: v for k, v in raw.items() if not str(k).startswith("_")}
                total_patterns = len(patterns)
        except Exception:
            pass

        return PatternInfo(
            patterns_file=str(self.config.patterns_file),
            total_patterns=total_patterns,
            patterns=patterns
        )

    def _render_patterns_grid(self, pattern_info: PatternInfo) -> str:
        if not pattern_info or not pattern_info.patterns:
            return ""


        entries = [(k, v) for k, v in pattern_info.patterns.items() if not str(k).startswith("_")]
        if not entries:
            return ""

        cards = "".join([
            f'<div class="pattern-card"><div class="pattern-name">{name}</div><div class="pattern-regex"><code>{value}</code></div></div>'
            for name, value in entries
        ])
        return f'<div class="pattern-grid">{cards}</div>'


    def generate_batch_reports(self, batch_report: BatchReport) -> None:

        self._generate_batch_json_report(batch_report)


        self._generate_batch_html_report(batch_report)

    def _generate_batch_json_report(self, batch_report: BatchReport) -> None:
        filename = f"batch_summary_{self.timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_report.to_dict(), f, indent=2, default=str)

        print(f"Batch JSON report generated: {filepath}")

    def _generate_batch_html_report(self, batch_report: BatchReport) -> None:
        filename = f"batch_summary_{self.timestamp}.html"
        filepath = self.output_dir / filename

        html_content = self._create_batch_html_content(batch_report)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Batch HTML report generated: {filepath}")

    def _create_batch_html_content(self, batch_report: BatchReport) -> str:
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Processing Batch Report - {batch_report.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ width: 95%; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .cli-params {{ background-color: #f8f9fa; }}
        .performance {{ background-color: #e8f5e8; }}
        .patterns {{ background-color: #fff3cd; }}
        .summary-stats {{ background-color: #f8f9fa; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .success {{ color: green; }}
        .fail {{ color: red; }}
        .no-matches {{ color: orange; }}
        .cli-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 12px; }}
        .cli-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fff; min-height: 60px; display: flex; flex-direction: column; justify-content: space-between; }}
        .cli-label {{ font-weight: 600; font-size: 12px; color: #555; }}
        .cli-value {{ font-size: 14px; color: #111; margin-top: 4px; font-weight: 500; }}
        .cli-default {{ font-size: 12px; color: #888; margin-top: auto; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 12px; }}
        .metric-card {{ border: 1px solid #cde7cd; border-radius: 6px; padding: 8px; background: #f7fff7; }}
        .metric-label {{ font-weight: 600; font-size: 12px; color: #356; }}
        .metric-value {{ font-size: 16px; color: #143; margin-top: 6px; line-height: 1.2; word-break: break-word; overflow-wrap: anywhere; white-space: normal; }}
        .pattern-grid {{ display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 12px; overflow-x: auto; padding-bottom: 6px; }}
        .pattern-card {{ border: 1px solid #ffe8a3; border-radius: 8px; padding: 12px; background: #fffaf0; min-width: 260px; }}
        .pattern-name {{ font-weight: 600; font-size: 12px; color: #7a5; margin-bottom: 6px; }}
        .pattern-regex code {{ font-family: Menlo, Consolas, monospace; font-size: 12px; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Processing Batch Report</h1>
        <p><strong>Generated:</strong> {batch_report.timestamp}</p>

        <div class="section cli-params">
            <h2>CLI Parameters</h2>
            {self._render_cli_params_grid(batch_report.cli_parameters)}
        </div>

        <div class="section performance">
            <h2>Performance Metrics</h2>
            {self._render_metrics_grid_batch(batch_report.performance)}
        </div>

        <div class="section patterns">
            <h2>Patterns Information</h2>
            <div class="param-row">
                <span class="param-label">Total Patterns:</span>
                <span class="param-value">{batch_report.patterns.total_patterns}</span>
            </div>
            {self._render_patterns_grid(batch_report.patterns)}
        </div>

        <div class="section summary-stats">
            <h2>Summary Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Sr. No</th>
                        <th>Src Doc File Name</th>
                        <th>Src Doc File Size (MB)</th>
                        <th>Src Dox File Name</th>
                        <th>Src Dox File Size (MB)</th>
                        <th>Processed Docx File Name</th>
                        <th>Processed Docx File Size (MB)</th>
                        <th>Parser Used</th>
                        <th>Text Matches</th>
                        <th>Graphics Matches</th>
                        <th>Image Matches</th>
                        <th>Time (min)</th>
                        <th>Status</th>
                        <th>Failure Reason</th>
                    </tr>
                </thead>
                <tbody>
        """

        for i, result in enumerate(batch_report.file_reports, 1):

            status_class = "success" if result.status == ProcessingStatus.SUCCESS else "fail"
            text_status_class = "success" if result.total_text_matches > 0 else "no-matches"
            graphics_status_class = "success" if result.total_graphics_matches > 0 else "no-matches"
            image_status_class = "success" if result.total_image_matches > 0 else "no-matches"

            doc_mb = "" if not result.doc_file_size else f"{result.doc_file_size/(1024*1024):.2f}"
            docx_mb = "" if not result.docx_file_size else f"{result.docx_file_size/(1024*1024):.2f}"


            show_processed = result.status in (ProcessingStatus.PROCESSED, ProcessingStatus.SUCCESS)
            processed_name = result.processed_file_name if show_processed else ""
            processed_mb = (
                f"{result.processed_file_size/(1024*1024):.2f}"
                if show_processed and result.processed_file_size else ""
            )

            html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{result.doc_file_name}</td>
                        <td>{doc_mb}</td>
                        <td>{result.docx_file_name}</td>
                        <td>{docx_mb}</td>
                        <td>{processed_name}</td>
                        <td>{processed_mb}</td>
                        <td>{result.parser}</td>
                        <td class="{text_status_class}">{result.total_text_matches} | {"Success" if result.total_text_matches > 0 else "No Matches"}</td>
                        <td class="{graphics_status_class}">{result.total_graphics_matches} | {"Success" if result.total_graphics_matches > 0 else "No Matches"}</td>
                        <td class="{image_status_class}">{result.total_image_matches} | {"Success" if result.total_image_matches > 0 else "No Matches"}</td>
                        <td>{result.processing_time_minutes:.2f}</td>
                        <td class="{status_class}">{result.status.value}</td>
                        <td>{result.failure_reason}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """

        return html

    def _format_graphics_font_size(self, src_text_size: str, src_text_font: str) -> str:
        if not src_text_size:
            return src_text_font or ""


        if "|" in src_text_size and ("Seg" in src_text_size or "pt" in src_text_size):

            parts = src_text_size.split(" | ")


            overall_size = ""
            segments = []

            for part in parts:
                if part.startswith("Seg"):

                    segments.append(part)
                elif "pt" in part and not part.startswith("Seg"):

                    overall_size = part


            if segments:
                formatted_segments = "<br>".join(segments)
                if overall_size:
                    return f"{overall_size}<br>{formatted_segments}"
                else:
                    return formatted_segments
            else:

                return src_text_size.replace(" | ", "<br>")
        else:

            if src_text_size and src_text_font:
                return f"{src_text_size}pt {src_text_font}"
            else:
                return src_text_size or src_text_font or ""

    def generate_document_reports(self, result: ProcessingResult) -> None:

        if result.status == ProcessingStatus.PENDING:
            return


        source_name = result.doc_file_name or result.docx_file_name or "document"
        self._generate_file_json_report(result, source_name)
        self._generate_file_html_report(result, source_name)

    def _create_performance_metrics_from_result(self, result: ProcessingResult) -> PerformanceMetrics:

        if hasattr(result, 'performance') and result.performance:
            return result.performance

        file_size_bytes = (
            result.doc_file_size
            or result.docx_file_size
            or result.processed_file_size
            or 0
        )


        success_rate = 100.0 if result.success else 0.0


        throughput = 1.0 / result.processing_time if result.processing_time > 0 else 0.0

        return PerformanceMetrics(
            processing_time_seconds=result.processing_time,
            total_files_processed=1,
            successful_files=1 if result.success else 0,
            failed_files=0 if result.success else 1,
            total_matches_found=result.total_matches,
            average_processing_time_per_file=result.processing_time,
            max_processing_time_seconds=result.processing_time,
            total_file_size_bytes=file_size_bytes,
            average_file_size_mb=file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 0.0,
            success_rate_percent=success_rate,
            processing_throughput_files_per_second=throughput
        )

    def _generate_file_json_report(self, file_report: ProcessingResult, source_filename: str) -> None:

        safe_filename = "".join(c for c in source_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')

        filename = f"{safe_filename}_{self.timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(file_report.to_dict(), f, indent=2, default=str)

        print(f"File JSON report generated: {filepath}")

    def _generate_file_html_report(self, file_report: ProcessingResult, source_filename: str) -> None:

        safe_filename = "".join(c for c in source_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')

        filename = f"{safe_filename}_{self.timestamp}.html"
        filepath = self.output_dir / filename

        html_content = self._create_file_html_content(file_report, source_filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"File HTML report generated: {filepath}")

    def _create_file_html_content(self, file_report: ProcessingResult, source_filename: str) -> str:


        src_name = file_report.doc_file_name or file_report.docx_file_name or ""
        src_ext = (Path(src_name).suffix.lower() if src_name else "")
        is_doc = src_ext == ".doc"
        is_docx = src_ext == ".docx"
        src_size = file_report.doc_file_size or file_report.docx_file_size or 0
        processed_name = file_report.processed_file_name or ""
        processed_size = file_report.processed_file_size or 0


        doc_file_name = src_name if is_doc else "N/A"
        doc_file_size = src_size if is_doc else 0


        if is_docx:
            docx_file_name = src_name
            docx_file_size = src_size
        elif is_doc:
            try:
                docx_file_name = Path(src_name).with_suffix(".docx").name if src_name else "N/A"
            except Exception:
                docx_file_name = "N/A"

            docx_file_size = file_report.docx_file_size or 0
        else:
            docx_file_name = "N/A"
            docx_file_size = 0


        show_processed = file_report.status in (ProcessingStatus.PROCESSED, ProcessingStatus.SUCCESS)
        processed_name_disp = processed_name if show_processed else ""
        processed_size_disp = processed_size if show_processed else 0

        doc_mb = "" if not doc_file_size else f"{doc_file_size/(1024*1024):.2f}"
        docx_mb = "" if not docx_file_size else f"{docx_file_size/(1024*1024):.2f}"
        processed_mb = "" if not processed_size_disp else f"{processed_size_disp/(1024*1024):.2f}"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Processing Report - {source_filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ width: 95%; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .cli-params {{ background-color: #f8f9fa; }}
        .performance {{ background-color: #e8f5e8; }}
        .patterns {{ background-color: #fff3cd; }}
        .file-summary {{ background-color: #e3f2fd; }}
        .match-details {{ background-color: #f8f9fa; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .success {{ color: green; }}
        .fail {{ color: red; }}
        .cli-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 12px; }}
        .cli-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fff; min-height: 60px; display: flex; flex-direction: column; justify-content: space-between; }}
        .cli-label {{ font-weight: 600; font-size: 12px; color: #555; }}
        .cli-value {{ font-size: 14px; color: #111; margin-top: 4px; font-weight: 500; }}
        .cli-default {{ font-size: 12px; color: #888; margin-top: auto; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 12px; }}
        .metric-card {{ border: 1px solid #cde7cd; border-radius: 6px; padding: 8px; background: #f7fff7; }}
        .metric-label {{ font-weight: 600; font-size: 12px; color: #356; }}
        .metric-value {{ font-size: 16px; color: #143; margin-top: 6px; line-height: 1.2; word-break: break-word; overflow-wrap: anywhere; white-space: normal; }}
        .metric-card-compact .metric-value {{ font-size: 14px; }}
        .metric-grid-compact {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin-top: 12px; }}
        .metric-card-compact {{ border: 1px solid #cde7cd; border-radius: 6px; padding: 8px; background: #f7fff7; }}
        .pattern-grid {{ display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 12px; overflow-x: auto; padding-bottom: 6px; }}
        .pattern-card {{ border: 1px solid #ffe8a3; border-radius: 8px; padding: 12px; background: #fffaf0; min-width: 260px; }}
        .pattern-name {{ font-weight: 600; font-size: 12px; color: #7a5; margin-bottom: 6px; }}
        .pattern-regex code {{ font-family: Menlo, Consolas, monospace; font-size: 12px; color: #333; }}
        .match-y {{ background-color: #d4edda; }}
        .match-n {{ background-color: #f8d7da; }}
        .failure-reason {{
            width: 100%;
            min-height: 60px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            resize: vertical;
            background-color: #f8f9fa;
        }}
        .failure-reason:empty {{
            color: #999;
            font-style: italic;
        }}
        .failure-reason:empty::before {{
            content: "No failure reason provided";
        }}
        .text-column {{ max-width: 200px; word-wrap: break-word; }}
        .font-column {{ max-width: 300px; font-size: 10px; word-wrap: break-word; }}
        .dimension-column {{ max-width: 200px; word-wrap: break-word; font-size: 9px; }}

        .reasoning-column {{ max-width: 600px; word-wrap: break-word; }}
        .orig-id-column {{ max-width: 80px; word-wrap: break-word; font-size: 9px; }}
        .match-flag-column {{ max-width: 40px; text-align: center; }}
        .fallback-column {{ max-width: 40px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>File Processing Report</h1>
        <p><strong>Source File:</strong> {source_filename}</p>
        <p><strong>Generated:</strong> {file_report.timestamp}</p>

        <div class="section cli-params">
            <h2>CLI Parameters</h2>
            {self._render_cli_params_grid(file_report.cli_parameters)}
        </div>

        <div class="section patterns">
            <h2>Patterns Information</h2>
            <div class="param-row">
                <span class="param-label">Total Patterns:</span>
                <span class="param-value">{self._create_pattern_info().total_patterns}</span>
            </div>
            {self._render_patterns_grid(self._create_pattern_info())}
        </div>

        <div class="section performance">
            <h2>File Processing Summary & Performance Metrics</h2>
            <div class="metric-grid-compact">
                <div class="metric-card-compact"><div class="metric-label">Src Doc File Name</div><div class="metric-value">{doc_file_name}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Src Doc File Size (MB)</div><div class="metric-value">{doc_mb}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Src Docx File Name</div><div class="metric-value">{docx_file_name}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Src Docx File Size (MB)</div><div class="metric-value">{docx_mb}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Processed Docx File Name</div><div class="metric-value">{processed_name_disp}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Processed Docx File Size (MB)</div><div class="metric-value">{processed_mb}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Processing Time (s)</div><div class="metric-value">{file_report.processing_time:.2f}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Status</div><div class="metric-value">{file_report.status.value}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Parser</div><div class="metric-value">{file_report.parser}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Text Matches</div><div class="metric-value">{file_report.total_text_matches}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Graphics Matches</div><div class="metric-value">{file_report.total_graphics_matches}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Image Matches</div><div class="metric-value">{file_report.total_image_matches}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Peak Memory (MB)</div><div class="metric-value">{self._create_performance_metrics_from_result(file_report).peak_memory_mb:.2f}</div></div>
                <div class="metric-card-compact"><div class="metric-label">Peak CPU (%)</div><div class="metric-value">{self._create_performance_metrics_from_result(file_report).peak_cpu_percent:.2f}</div></div>
            </div>

            <div style="margin-top: 20px;">
                <h3>Failure Reason</h3>
                <textarea class="failure-reason" readonly>{file_report.error_message or ""}</textarea>
            </div>
        </div>

        <div class="section match-details">
            <h2>Match Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Sr. No</th>
                        <th>Type</th>
                        <th>Orig ID/Name</th>
                        <th>Src Text</th>
                        <th>Src Font & Size</th>
                        <th>Src Color</th>
                        <th>Src Dimension</th>
                        <th>Src Graphics Lines</th>
                        <th>Mapped Text</th>
                        <th>Mapped Font</th>
                        <th>Mapped Color</th>
                        <th>Mapped Size</th>
                        <th>Lines With Appended Text</th>
                        <th>Match Flag</th>
                        <th>Is Fallback</th>
                        <th>Reconstructed</th>
                        <th>Recon Bbox</th>
                        <th>Reasoning</th>
                    </tr>
                </thead>
                <tbody>
        """

        for match in file_report.match_details:
            match_class = "match-y" if match.match_flag.value == "Y" else "match-n"


            if match.type.value == "Text":

                src_font_size = f"{match.src_text_size}pt {match.src_text_font}" if match.src_text_size and match.src_text_font else f"{match.src_text_size or match.src_text_font or ''}"
            else:

                src_font_size = self._format_graphics_font_size(match.src_text_size, match.src_text_font)


            src_graphics_lines_val = match.src_graphics_lines if match.type.value == "Graphics" else ""
            lines_with_appended_text_val = match.lines_with_appended_text if match.type.value == "Graphics" else ""

            html += f"""
                    <tr class="{match_class}">
                        <td>{match.sr_no}</td>
                        <td>{match.type.value}</td>
                        <td class="orig-id-column">{match.orig_id_name}</td>
                        <td class="text-column">{match.src_text}</td>
                        <td class="font-column">{src_font_size}</td>
                        <td>{match.src_text_color}</td>
                        <td class="dimension-column">{match.src_dimension}</td>
                        <td>{src_graphics_lines_val}</td>
                        <td class="text-column">{match.mapped_text}</td>
                        <td class="font-column">{match.mapped_text_font}</td>
                        <td>{match.mapped_text_color}</td>
                        <td>{match.mapped_text_size}</td>
                        <td>{lines_with_appended_text_val}</td>
                        <td class="match-flag-column">{match.match_flag.value}</td>
                        <td class="fallback-column">{match.is_fallback.value}</td>
                        <td class="fallback-column">{'Y' if match.reconstructed else 'N'}</td>
                        <td class="dimension-column">{match.reconstruction_bbox_dimensions}</td>
                        <td class="reasoning-column">{match.reconstruction_reasoning or match.reasoning or ""}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """

        return html