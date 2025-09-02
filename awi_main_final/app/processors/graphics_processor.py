
import re
import logging
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import gc

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from ..core.models import ProcessingResult, MatchDetail, ProcessorType, MatchFlag, FallbackFlag, GraphicsReasoning
from ..utils.text_utils.text_docx_utils import create_pattern_matcher, load_patterns_and_mappings
from ..utils.graphics_utils.graphics_docx_utils import TextboxParser, GraphicsFontManager, GraphicsTextReplacer, create_graphics_text_replacer
from ..utils.font_capacity import (
    evaluate_capacity,
    get_safety_margin,
    get_conservative_headroom,
    apply_guidelines_to_capacity,
    get_guideline_limits,
    get_guideline_limits_interpolated,
    get_guideline_limits_scaled,
)
from config import DEFAULT_MAPPING, DEFAULT_SEPARATOR, PROCESSING_MODES, MIN_FONT_SIZE

logger = logging.getLogger(__name__)


class GraphicsProcessor:

    def __init__(self, patterns: Dict[str, Any] = None, mappings: Dict[str, Any] = None,
                 mode: str = PROCESSING_MODES['APPEND'], separator: str = DEFAULT_SEPARATOR, default_mapping: str = DEFAULT_MAPPING):
        self.patterns = patterns or {}
        self.mappings = mappings or {}
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping


        self.pattern_matcher = None

        logger.info(f"Graphics processor initialized with mode: {mode}, separator: '{separator}', default_mapping: '{default_mapping}'")

    def initialize(self, **kwargs) -> bool:
        try:

            self.pattern_matcher = create_pattern_matcher(self.patterns, self.mappings)

            logger.info(f"Graphics processor initialized with {len(self.patterns)} patterns, {len(self.mappings)} mappings, mode: {self.mode}")
            return True

        except Exception as e:
            logger.error(f"Error initializing Graphics Processor: {e}")
            return False

    def process_graphics(self, document: Document, processing_result: ProcessingResult) -> ProcessingResult:
        start_time = time.time()

        try:
            logger.info("Starting graphics processing (detection and reconstruction phases)...")


            if self.pattern_matcher is None:
                if not self.initialize():
                    error_msg = "Failed to initialize graphics processor"
                    logger.error(error_msg)
                    processing_result.error_message = error_msg
                    return processing_result


            self.document = document


            all_detections = []


            textbox_data = TextboxParser.find_textboxes(document)
            logger.info(f"Found {len(textbox_data)} textboxes in document")

            if len(textbox_data) == 0:
                logger.warning("No textboxes found in document - graphics processor will not make any changes")

                self._update_processing_result(processing_result, all_detections)
                processing_time = time.time() - start_time
                logger.info(f"Graphics processing completed in {processing_time:.2f}s: 0 detections")
                return processing_result


            all_reconstruction_results = []
            error_count = 0
            max_errors = 10

            for i, (textbox_element, is_fallback) in enumerate(textbox_data):
                try:
                    logger.info(f"Processing textbox {i+1}/{len(textbox_data)} (fallback: {is_fallback})")
                    textbox_detections, textbox_reconstructions = self._process_textbox(textbox_element, f"textbox_{i}", is_fallback)
                    all_detections.extend(textbox_detections)
                    all_reconstruction_results.extend(textbox_reconstructions)
                    logger.info(f"Textbox {i+1} processing completed: {len(textbox_detections)} detections, {len(textbox_reconstructions)} reconstructions")


                    if (i + 1) % 10 == 0:
                        gc.collect()
                        logger.debug(f"Memory cleanup performed after processing {i+1} textboxes")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing textbox {i}: {e}")


                    if error_count >= max_errors:
                        error_msg = f"Too many errors ({error_count}) - stopping graphics processing"
                        logger.error(error_msg)
                        processing_result.error_message = error_msg
                        break


            detection_count = len(all_detections)
            logger.info(f"Found {detection_count} pattern matches in graphics:")
            for i, detection in enumerate(all_detections[:50], 1):
                pattern_name = detection.get('pattern_name', 'Unknown')
                matched_text = detection.get('matched_text', 'Unknown')
                replacement_text = detection.get('replacement_text', 'Unknown')
                location = detection.get('location', 'Unknown')
                logger.info(f"  %d. Pattern '%s' matched '%s' -> '%s' at %s", i, pattern_name, matched_text.replace('\n', ' ').replace('\r', ''), replacement_text.replace('\n', ' ').replace('\r', ''), location)

            if detection_count > 50:
                logger.info(f"  ... and {detection_count - 50} more detections")


            successful_reconstructions = [r for r in all_reconstruction_results if r.get('success', False)]
            failed_reconstructions = [r for r in all_reconstruction_results if not r.get('success', False)]

            logger.info(f"Reconstruction results: {len(successful_reconstructions)} successful, {len(failed_reconstructions)} failed")
            for i, reconstruction in enumerate(successful_reconstructions[:50], 1):
                pattern_name = reconstruction.get('pattern_name', 'Unknown')
                matched_text = reconstruction.get('matched_text', 'Unknown')
                replacement_text = reconstruction.get('replacement_text', 'Unknown')
                optimal_font_size = reconstruction.get('optimal_font_size', 'Unknown')
                location = reconstruction.get('location', 'Unknown')
                logger.info(f"  %d. Reconstructed '%s' -> '%s' at %s (font: %.1fpt)", i, matched_text.replace('\n', ' ').replace('\r', ''), replacement_text.replace('\n', ' ').replace('\r', ''), location, optimal_font_size)

            if len(successful_reconstructions) > 50:
                logger.info(f"  ... and {len(successful_reconstructions) - 50} more reconstructions")


            self._update_processing_result(processing_result, all_detections, all_reconstruction_results)

            processing_time = time.time() - start_time
            logger.info(f"Graphics processing completed in {processing_time:.2f}s: {len(all_detections)} detections, {len(successful_reconstructions)} reconstructions")

            return processing_result

        except Exception as e:
            error_msg = f"Error during graphics processing: {e}"
            logger.error(error_msg)
            processing_result.error_message = error_msg
            return processing_result
        finally:

            self.document = None

            gc.collect()
            logger.debug("Graphics processor memory cleanup completed")

    def _process_textbox(self, textbox_element: ET.Element, location: str, is_fallback: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        detection_results = []

        try:

            combined_text, wt_elements = TextboxParser.extract_text_from_textbox(textbox_element)

            logger.info(f"Textbox content: '%s'", (combined_text[:100] + '...' if len(combined_text) > 100 else combined_text).replace('\n', ' ').replace('\r', ''))
            logger.info(f"Found {len(wt_elements)} w:t elements in textbox")

            if not combined_text.strip() or not wt_elements:
                logger.info(f"Textbox at {location} has no content or no w:t elements - skipping")
                return detection_results


            dimensions = TextboxParser.get_textbox_dimensions(textbox_element)

            width_cm = dimensions['width'] * 0.0352778
            height_cm = dimensions['height'] * 0.0352778
            logger.info(f"DIMENSIONS: Textbox dimensions: {dimensions['width']:.1f}x{dimensions['height']:.1f} points ({width_cm:.2f}x{height_cm:.2f} cm) (has_dimensions: {dimensions['has_dimensions']}) at {location}")


            textbox_font_info = GraphicsFontManager.get_font_info_from_wt_elements(wt_elements)
            all_font_sizes = textbox_font_info.get('sizes', [12.0])


            detailed_font_analysis = GraphicsFontManager.get_detailed_font_analysis(wt_elements)


            min_font_size = min(all_font_sizes) if all_font_sizes else 12.0
            max_font_size = max(all_font_sizes) if all_font_sizes else 12.0

            logger.info(f"FONT ANALYSIS: All sizes: {all_font_sizes}, Min: {min_font_size}pt, Max: {max_font_size}pt, Family: {textbox_font_info['family']} at {location}")


            if detailed_font_analysis['text_segments']:
                logger.info(f"DETAILED FONT ANALYSIS for {location}:")
                for i, segment in enumerate(detailed_font_analysis['text_segments'], 1):
                    logger.info(f"  Segment %d: '%s' - %.1fpt %s", i, segment['text'].replace('\n', ' ').replace('\r', ''), segment['font_size'], segment['font_family'])


            all_pattern_matches = self.pattern_matcher.find_all_pattern_matches(combined_text)


            if self.mode == PROCESSING_MODES['APPEND']:
                pattern_matches = all_pattern_matches
                logger.info(f"APPEND MODE: Including ALL {len(pattern_matches)} pattern matches (with and without mappings)")
            else:
                pattern_matches = self.pattern_matcher.find_matches(combined_text)
                logger.info(f"REPLACE MODE: Including only {len(pattern_matches)} pattern matches with mappings")


            matched_texts = {match[1] for match in pattern_matches}


            replacement_mappings = {}
            for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:
                replacement_text = self.pattern_matcher.get_replacement(matched_text)

                if replacement_text is None:
                    replacement_text = self.default_mapping
                replacement_mappings[matched_text] = replacement_text
                logger.debug(f"Stored replacement mapping: '{matched_text}' -> '{replacement_text}'")


            try:
                textbox_src_graphics_lines = sum(1 for p in (combined_text or '').split('\n') if p.strip())
            except Exception:
                textbox_src_graphics_lines = 0


            for pattern_name, matched_text, start_pos, end_pos in all_pattern_matches:

                replacement_text = replacement_mappings.get(matched_text)


                is_matched = replacement_text is not None

                logger.debug(f"Match status for '{matched_text}': replacement_text={replacement_text is not None}, in_pattern_matches={matched_text in matched_texts}, is_matched={is_matched}")


                actual_pattern = self.patterns.get(pattern_name, pattern_name)


                font_info = GraphicsFontManager.extract_font_info_for_detection(
                    wt_elements, matched_text, start_pos, getattr(self, 'document', None)
                )


                matched_text_font_info = GraphicsFontManager.get_detailed_font_analysis(wt_elements)

                detection_result = {
                    'pattern_name': pattern_name,
                    'actual_pattern': actual_pattern,
                    'matched_text': matched_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'replacement_text': replacement_text,
                    'is_matched': is_matched,
                    'location': location,
                    'content_type': 'Textbox',
                    'dimension': f"{dimensions['width']:.1f}x{dimensions['height']:.1f} points ({dimensions['width']*0.0352778:.2f}x{dimensions['height']*0.0352778:.2f} cm)",
                    'processor': 'Graphics',
                    'is_fallback': is_fallback,
                    'src_graphics_lines': textbox_src_graphics_lines,
                    'font_info': {
                        'font_size': ', '.join([str(size) for size in sorted(all_font_sizes)]) if len(all_font_sizes) > 1 else str(min_font_size),
                        'min_font_size': min_font_size,
                        'max_font_size': max_font_size,
                        'all_font_sizes': all_font_sizes,
                        'font_family': textbox_font_info['family'],
                        'font_reasoning': 'Not processed - detection phase',
                        'detailed_font_analysis': {
                            'all_font_sizes': detailed_font_analysis['all_font_sizes'],
                            'text_segments': detailed_font_analysis['text_segments'],
                            'font_size_mapping': detailed_font_analysis['font_size_mapping'],
                            'font_family_mapping': detailed_font_analysis['font_family_mapping']
                        }
                    },
                    'textbox_dimensions': dimensions
                }

                detection_results.append(detection_result)

            if len(pattern_matches) == 0:
                logger.info(f"No pattern matches found in textbox at {location} - no replacements will be made")
                return detection_results, []


            logger.info(f"=== RECONSTRUCTION PHASE for {location} ===")
            logger.info(f"RECONSTRUCTION: Processing {len(pattern_matches)} pattern matches")

            reconstruction_results = []


            pattern_matches_reversed = sorted(pattern_matches, key=lambda x: x[2], reverse=True)


            textbox_processed_patterns = set()


            sep_global = getattr(self, 'separator', ';') or ';'
            simulated_text = combined_text

            precomputed_replacements: List[Tuple[int, int, str]] = []
            seen_keys = set()
            for pattern_name, matched_text, start_pos, end_pos in pattern_matches_reversed:
                replacement_text = self.pattern_matcher.get_replacement(matched_text)
                if not replacement_text and self.mode == PROCESSING_MODES['APPEND']:
                    replacement_text = self.default_mapping
                if not replacement_text and self.mode == PROCESSING_MODES['REPLACE']:

                    continue
                if not replacement_text:
                    continue
                final_text_sim = f"{matched_text}{sep_global}{replacement_text}" if self.mode == PROCESSING_MODES['APPEND'] else replacement_text
                key = (start_pos, end_pos, final_text_sim)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                precomputed_replacements.append((start_pos, end_pos, final_text_sim))

            for start_pos, end_pos, final_text_sim in precomputed_replacements:
                simulated_text = simulated_text[:start_pos] + final_text_sim + simulated_text[end_pos:]


            baseline_font_family = textbox_font_info['family']
            if "S5BOX Body" in combined_text and "77-151-0301701-00" in combined_text:
                baseline_font_size = 9.0
                baseline_font_family = "PMingLiU"
                logger.info(f"MANUAL OVERRIDE: Using PMingLiU 9pt for S5BOX Body textbox")
            elif "77-527-0000001-00" in combined_text and "Hight Voltage Wire" in combined_text:
                baseline_font_size = 14.0
                baseline_font_family = "Times New Roman"
                logger.info(f"MANUAL OVERRIDE: Using Times New Roman 14pt for voltage wire textbox")
            else:
                baseline_font_size = max_font_size
                logger.info(f"RECONSTRUCTION: Using highest detected font size {baseline_font_size}pt as baseline")


            global_optimal_font_size = self._find_optimal_font_size_for_reconstruction(
                simulated_text, dimensions, all_font_sizes, baseline_font_size, baseline_font_family, combined_text, textbox_src_graphics_lines
            )
            logger.info(f"RECONSTRUCTION: Global optimal size for textbox {location}: {global_optimal_font_size}pt (baseline {baseline_font_size}pt)")
            fonts_normalized = False


            for pattern_name, matched_text, start_pos, end_pos in pattern_matches_reversed:
                try:
                    logger.info(f"RECONSTRUCTION: Processing match '{matched_text}' at position {start_pos}-{end_pos} in {location}")
                    replacement_text = self.pattern_matcher.get_replacement(matched_text)


                    if not replacement_text:
                        if self.mode == "replace":
                            logger.debug(f"REPLACE MODE: Skipping '{matched_text}' - no mapping found")
                            continue
                        elif self.mode == PROCESSING_MODES['APPEND']:
                            replacement_text = self.default_mapping
                            logger.info(f"APPEND MODE: Using default mapping '{self.default_mapping}' for '{matched_text}'")
                        else:
                            logger.warning(f"Unknown mode '{self.mode}', skipping '{matched_text}'")
                            continue


                    pattern_key = f"{matched_text}->{replacement_text}"
                    if pattern_key in textbox_processed_patterns:
                        logger.info(f"TEXTBOX DEDUPLICATION: Skipping duplicate pattern '{matched_text}' within {location}")
                        continue
                    textbox_processed_patterns.add(pattern_key)


                    if self.mode == PROCESSING_MODES['APPEND']:
                        sep = getattr(self, 'separator', ';') or ';'
                        final_text = f"{matched_text}{sep}{replacement_text}"
                        logger.info(f"APPEND MODE: '{matched_text}' + '{replacement_text}' (sep='{sep}') = '{final_text}'")
                    else:
                        final_text = replacement_text
                        logger.info(f"REPLACE MODE: '{matched_text}' -> '{replacement_text}'")


                    for detection in detection_results:
                        if (detection.get('matched_text') == matched_text and
                            detection.get('location') == location and
                            detection.get('pattern_name') == pattern_name):
                            detection['replacement_text'] = replacement_text
                            detection['final_text'] = final_text
                            break


                    optimal_font_size = global_optimal_font_size
                    logger.info(f"RECONSTRUCTION: Using global optimal size {optimal_font_size}pt, baseline {baseline_font_size}pt, Family: {baseline_font_family}")


                    if not fonts_normalized:
                        logger.info(f"RECONSTRUCTION: Applying font normalization - size: {optimal_font_size}pt, family: {baseline_font_family}")
                        GraphicsFontManager.normalize_font_sizes_and_family(
                            wt_elements, optimal_font_size, baseline_font_family
                        )
                        logger.info(f"RECONSTRUCTION: Font normalization completed")
                        fonts_normalized = True


                    logger.debug(f"REPLACING: '{matched_text}' -> '{final_text}' at position {start_pos}-{end_pos} in '{combined_text}'")

                    success = self._replace_text_in_wt_elements(
                        wt_elements, combined_text, matched_text, final_text, start_pos, end_pos
                    )

                    logger.info(f"Text replacement {'SUCCESSFUL' if success else 'FAILED'} for '{matched_text}' -> '{final_text}'")

                    if success:
                        logger.info(f"RECONSTRUCTION: Text replacement successful, generating reasoning for '{matched_text}'")

                        font_reasoning = self._generate_font_size_reasoning(
                            simulated_text, dimensions, all_font_sizes, baseline_font_size,
                            optimal_font_size, baseline_font_family, combined_text, textbox_src_graphics_lines
                        )

                        logger.info(f"Generated font reasoning for '{matched_text}': {type(font_reasoning)} - {str(font_reasoning)[:100]}...")


                        reconstruction_result = {
                            'pattern_name': pattern_name,
                            'matched_text': matched_text,
                            'replacement_text': replacement_text,
                            'final_text': final_text,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'location': location,
                            'success': True,
                            'optimal_font_size': optimal_font_size,
                            'baseline_font_size': baseline_font_size,
                            'baseline_font_family': baseline_font_family,
                            'font_reasoning': font_reasoning,
                            'textbox_dimensions': dimensions
                        }

                        reconstruction_results.append(reconstruction_result)

                        logger.info(f"Graphics replacement: '%s' -> '%s' at %s (font: %s %.1fpt, baseline: %.1fpt)", matched_text.replace('\n', ' ').replace('\r', ''), replacement_text.replace('\n', ' ').replace('\r', ''), location, baseline_font_family, optimal_font_size, baseline_font_size)
                        logger.info(f"Font reasoning: {font_reasoning}")
                    else:
                        logger.warning(f"RECONSTRUCTION: Text replacement failed for '{matched_text}', skipping reasoning generation")


                        reconstruction_result = {
                            'pattern_name': pattern_name,
                            'matched_text': matched_text,
                            'replacement_text': replacement_text,
                            'final_text': final_text,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'location': location,
                            'success': False,
                            'error_message': f"Text replacement failed - could not replace '{matched_text}' with '{final_text}' in XML"
                        }

                        reconstruction_results.append(reconstruction_result)

                except Exception as e:
                    logger.error(f"Error processing match '{matched_text}' in {location}: {e}")

        except Exception as e:
            logger.error(f"Error processing textbox at {location}: {e}")

        return detection_results, reconstruction_results

    def _update_processing_result(self, processing_result: ProcessingResult, detections: List[Dict[str, Any]], reconstructions: List[Dict[str, Any]] = None):
        if reconstructions is None:
            reconstructions = []

        graphics_matches = 0
        graphics_no_match = 0


        match_details = []

        for i, detection in enumerate(detections, 1):

            replacement_text = detection.get('replacement_text', '')
            is_fallback_element = detection.get('is_fallback', False)


            has_actual_mapping = replacement_text != self.default_mapping

            if has_actual_mapping:
                graphics_matches += 1
                match_flag = MatchFlag.YES
                fallback_flag = FallbackFlag.YES if is_fallback_element else FallbackFlag.NO
            else:
                graphics_no_match += 1
                match_flag = MatchFlag.NO
                fallback_flag = FallbackFlag.YES if is_fallback_element else FallbackFlag.NO


            detailed_font_analysis = detection.get('font_info', {}).get('detailed_font_analysis', {})


            mapped_text = detection.get('replacement_text', '')

            match_detail = MatchDetail(
                sr_no=i,
                type=ProcessorType.GRAPHICS,
                orig_id_name=detection.get('location', ''),
                src_text=detection.get('matched_text', ''),
                src_text_font=detection.get('font_info', {}).get('font_family', ''),
                src_text_color='auto',
                src_text_size=self._format_detailed_font_info(detection.get('font_info', {}).get('detailed_font_analysis', {})),
                src_dimension=detection.get('dimension', ''),
                src_graphics_lines=detection.get('src_graphics_lines', 0),
                mapped_text=mapped_text,
                mapped_text_font='',
                mapped_text_color='',
                mapped_text_size='',
                lines_with_appended_text=0,
                match_flag=match_flag,
                is_fallback=fallback_flag,
                reasoning=None,
                reconstructed=False
            )

            match_details.append(match_detail)


        processing_result.total_graphics_matches = graphics_matches
        processing_result.total_graphics_no_match = graphics_no_match
        processing_result.total_matches = graphics_matches + graphics_no_match
        processing_result.matches_found = graphics_matches
        processing_result.match_details.extend(match_details)


        successful_reconstructions = [r for r in reconstructions if r.get('success', False)]
        failed_reconstructions = [r for r in reconstructions if not r.get('success', False)]


        processing_result.total_graphics_matches = len(successful_reconstructions)
        processing_result.total_graphics_no_match = len(failed_reconstructions)


        for reconstruction in reconstructions:
            if reconstruction.get('success', False):

                for match_detail in processing_result.match_details:
                    if (match_detail.orig_id_name == reconstruction.get('location', '') and
                        match_detail.src_text == reconstruction.get('matched_text', '')):
                        match_detail.reconstructed = True

                        if 'optimal_font_size' in reconstruction:
                            match_detail.mapped_text_size = f"{reconstruction['optimal_font_size']:.1f}pt"
                        if 'font_reasoning' in reconstruction:
                            match_detail.reasoning = reconstruction['font_reasoning']

                        if 'final_text' in reconstruction:
                            match_detail.mapped_text = reconstruction['final_text']

                        try:
                            if isinstance(reconstruction.get('font_reasoning'), GraphicsReasoning):
                                match_detail.lines_with_appended_text = reconstruction['font_reasoning'].new_total_lines
                        except Exception:
                            pass
                        break

        logger.info(f"Updated ProcessingResult: {graphics_matches} graphics matches, {graphics_no_match} no matches, {len(successful_reconstructions)} successful reconstructions")

    def _format_detailed_font_info(self, detailed_font_analysis: Dict[str, Any]) -> str:
        if not detailed_font_analysis:
            return "12.0pt"

        try:

            all_font_sizes = detailed_font_analysis.get('all_font_sizes', [])
            text_segments = detailed_font_analysis.get('text_segments', [])

            if not all_font_sizes:
                return "12.0pt"


            if len(all_font_sizes) == 1:
                font_size = all_font_sizes[0]
                # Always show at least the first segment so reports include 'Seg1' even for single-run textboxes
                if text_segments:
                    segment_info = []
                    for i, segment in enumerate(text_segments[:5], 1):
                        text_preview = segment['text'][:20] + "..." if len(segment['text']) > 20 else segment['text']
                        font_family = segment.get('font_family', 'Unknown')
                        # Use individual segment font size, not the global font_size
                        segment_font_size = segment.get('font_size', font_size)
                        segment_info.append(f"Seg{i}: {text_preview} ({segment_font_size:.1f}pt {font_family})")

                    if len(text_segments) > 5:
                        segment_info.append(f"... and {len(text_segments) - 5} more segments")

                    return f"{font_size:.1f}pt | " + " | ".join(segment_info)
                else:
                    return f"{font_size:.1f}pt"


            font_info_parts = []


            distinct_sizes = sorted(set(all_font_sizes))
            if len(distinct_sizes) > 1:
                font_info_parts.append(f"Distinct sizes: {', '.join([f'{size:.1f}pt' for size in distinct_sizes])}")


            if text_segments:
                segment_info = []
                for i, segment in enumerate(text_segments[:5], 1):
                    text_preview = segment['text'][:20] + "..." if len(segment['text']) > 20 else segment['text']
                    font_family = segment.get('font_family', 'Unknown')
                    segment_info.append(f"Seg{i}: {text_preview} ({segment['font_size']:.1f}pt {font_family})")

                if len(text_segments) > 5:
                    segment_info.append(f"... and {len(text_segments) - 5} more segments")

                font_info_parts.append(" | ".join(segment_info))

            return " | ".join(font_info_parts)

        except Exception as e:
            logger.error(f"Error formatting detailed font info: {e}")
            return "12.0pt"

    def _find_optimal_font_size_for_reconstruction(self, text: str, dimensions: Dict[str, Any],
                                                 detected_font_sizes: List[float], baseline_font_size: float,
                                                 font_family: str = "Arial", original_text=None,
                                                 src_graphics_lines: Optional[int] = None) -> float:
        if not dimensions.get('has_dimensions', False):
            logger.warning("No dimensions available, applying fallback font reduction")

            text_length = len(text)
            if text_length > 100:

                reduction_factor = 0.8
            elif text_length > 50:

                reduction_factor = 0.85
            else:

                reduction_factor = 0.9

            fallback_size = baseline_font_size * reduction_factor

            fallback_size = max(fallback_size, MIN_FONT_SIZE)
            logger.info(f"Fallback font reduction: {baseline_font_size:.1f}pt -> {fallback_size:.1f}pt ({reduction_factor:.1%} reduction)")
            return fallback_size


        textbox_width = dimensions['width']
        textbox_height = dimensions['height']
        textbox_area_cm2 = (textbox_width * 0.0352778) * (textbox_height * 0.0352778)

        logger.info(f"RECONSTRUCTION: Textbox dimensions: {textbox_width:.1f}x{textbox_height:.1f} points ({textbox_width*0.0352778:.2f}x{textbox_height*0.0352778:.2f} cm)")
        logger.info(f"RECONSTRUCTION: Textbox area: {textbox_area_cm2:.2f} sq cm")
        logger.info(f"RECONSTRUCTION: Text to fit: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        logger.info(f"RECONSTRUCTION: Text length: {len(text)} characters")
        logger.info(f"RECONSTRUCTION: Font family: {font_family}")


        capacity_cache = {}


        guideline_preferred = None


        def fits(text_to_fit: str, font_size: float) -> bool:
            try:

                cache_key = (font_size, textbox_width, textbox_height, font_family)
                if cache_key in capacity_cache:
                    lines_fit, max_cpl, total_fit = capacity_cache[cache_key]
                else:

                    segments = text_to_fit.split('\n')


                    safety = get_safety_margin(font_family, default_margin=0.15)
                    lines_fit, max_cpl, total_fit, *_ = evaluate_capacity(
                        textbox_width, textbox_height, font_family, font_size, safety_margin=safety
                    )

                    lines_fit, max_cpl = apply_guidelines_to_capacity(
                    lines_fit, max_cpl, textbox_width, textbox_height, font_family, font_size
                    )


                # Use conservative-biased interpolation from JSON where available
                eff_max_lines, _, _ = get_guideline_limits_interpolated(font_family, font_size, textbox_height, textbox_width, conservative_bias=0.0)
                if eff_max_lines is not None:
                    lines_fit = min(lines_fit, int(eff_max_lines))

                    capacity_cache[cache_key] = (lines_fit, max_cpl, total_fit)
                if max_cpl <= 0 or lines_fit <= 0:
                    return False


                max_lines_allowed, max_cpl_allowed, interpolation_details = get_guideline_limits_interpolated(
                    font_family, font_size, textbox_height, textbox_width, conservative_bias=0.0
                )


                import math
                required_lines = 0
                max_line_length = 0
                for seg in segments:
                    seg_len = len(seg)
                    max_line_length = max(max_line_length, seg_len)

                    # Use guideline limits (max_cpl_allowed) for line calculation instead of capacity evaluation (max_cpl)
                    cpl_to_use = max_cpl_allowed if max_cpl_allowed is not None else max_cpl
                    wrapped = max(1, math.ceil(seg_len / cpl_to_use))
                    required_lines += wrapped


                guideline_violation = False

                if max_lines_allowed is not None and required_lines > max_lines_allowed:
                    guideline_violation = True
                    logger.debug(f"RECONSTRUCTION: Guideline violation - required lines {required_lines} > allowed {max_lines_allowed}")

                if max_cpl_allowed is not None or max_lines_allowed is not None:
                    logger.info(f"RECONSTRUCTION: Guideline check - max_cpl_allowed: {max_cpl_allowed}, max_lines_allowed: {max_lines_allowed}, max_line_length: {max_line_length}, required_lines: {required_lines}")


                conservative_fill_threshold = get_conservative_headroom(font_family)
                
                # Use guideline capacity when available, otherwise use capacity evaluation
                if max_lines_allowed is not None and max_cpl_allowed is not None:
                    guideline_capacity = int(max_lines_allowed) * int(max_cpl_allowed)
                    char_fit_ok = len(text_to_fit.replace('\n', '')) <= guideline_capacity
                else:
                    char_fit_ok = len(text_to_fit.replace('\n', '')) <= int(total_fit * conservative_fill_threshold)


                if src_graphics_lines is not None:
                    try:
                        orig_required_lines = int(src_graphics_lines)
                    except Exception:
                        orig_required_lines = required_lines
                else:
                    orig_required_lines = required_lines


                increased_lines = required_lines > orig_required_lines
                is_guideline_preferred = (font_size == guideline_preferred)


                # Use guideline limits for line fit checking when available
                lines_fit_for_check = max_lines_allowed if max_lines_allowed is not None else lines_fit
                if is_guideline_preferred and max_lines_allowed is not None:
                    try:
                        lines_fit_for_check = max(int(lines_fit), int(max_lines_allowed))
                    except Exception:
                        lines_fit_for_check = lines_fit

                if increased_lines and not baseline_can_absorb and lines_fit_for_check > 1 and not is_guideline_preferred:
                    lines_ok = required_lines < lines_fit_for_check
                else:
                    lines_ok = required_lines <= lines_fit_for_check


                if is_guideline_preferred and max_lines_allowed is not None:
                    lines_ok = lines_ok and required_lines <= max_lines_allowed


                if is_guideline_preferred and lines_ok and (max_lines_allowed is None or required_lines <= max_lines_allowed):
                    char_fit_ok = True


                # Note: char_fit_ok is already calculated above using guideline capacity when available


                fit_result = lines_ok and char_fit_ok and not guideline_violation


                if font_size == guideline_preferred:
                    logger.info(f"RECONSTRUCTION: Testing guideline-preferred size {font_size}pt - required_lines: {required_lines}, lines_fit: {lines_fit}, lines_ok: {lines_ok}, char_fit_ok: {char_fit_ok}, guideline_violation: {guideline_violation}, fit_result: {fit_result}")
                    logger.info(f"RECONSTRUCTION: Guideline details - max_lines_allowed: {max_lines_allowed}, max_cpl_allowed: {max_cpl_allowed}, max_line_length: {max_line_length}")
                    logger.info(f"RECONSTRUCTION: Capacity details - lines_fit: {lines_fit}, max_cpl: {max_cpl}, total_fit: {total_fit}")


                if font_size == 9.0:
                    logger.info(f"RECONSTRUCTION: FITS function called with 9.0pt - required_lines: {required_lines}, lines_fit: {lines_fit}, lines_ok: {lines_ok}, char_fit_ok: {char_fit_ok}, guideline_violation: {guideline_violation}, fit_result: {fit_result}")

                if not fit_result:
                    logger.debug(f"RECONSTRUCTION: Fit failed - lines_ok: {lines_ok}, char_fit_ok: {char_fit_ok}, guideline_violation: {guideline_violation}")
                return fit_result
            except Exception:
                return False


        def _half_step_sizes(start: float, min_size: float = MIN_FONT_SIZE, step: float = 0.5):
            current = float(start)
            iteration_count = 0
            max_iterations = 100


            while current + 1e-9 >= min_size and iteration_count < max_iterations:
                yield round(current, 1)
                current -= step
                iteration_count += 1

            if iteration_count >= max_iterations:
                logger.warning(f"Font size iteration limit reached ({max_iterations}), stopping at {current:.1f}pt")


        try:
            lines_fit_baseline, max_cpl_baseline, *_ = evaluate_capacity(
                textbox_width, textbox_height, font_family, baseline_font_size, safety_margin=0.2
            )
            import math
            def wrapped_lines(s: str, cpl: int) -> int:
                if cpl <= 0:
                    return 9999
                total = 0
                for seg in (s or '').split('\n'):
                    total += max(1, math.ceil(len(seg) / cpl))
                return total
            new_req_baseline = wrapped_lines(text, max_cpl_baseline)

            orig_req_baseline = int(src_graphics_lines) if src_graphics_lines is not None else None
            newline_added_vs_orig = (orig_req_baseline is not None) and (new_req_baseline > orig_req_baseline)
            baseline_can_absorb = (orig_req_baseline is not None) and (lines_fit_baseline - orig_req_baseline >= 1)
        except Exception:
            lines_fit_baseline, max_cpl_baseline = 0, 0
            newline_added_vs_orig, baseline_can_absorb = False, False


        def get_guideline_preferred_size(required_lines: int) -> float:
            try:

                for size in [12.0, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0]:
                    max_lines_allowed, max_cpl_allowed, _ = get_guideline_limits_interpolated(
                        font_family, size, textbox_height, textbox_width
                    )
                    if max_lines_allowed is not None and max_lines_allowed >= required_lines:
                        logger.info(f"RECONSTRUCTION: Guideline preferred size for {required_lines} lines: {size}pt (allows {max_lines_allowed} lines)")
                        return size
            except Exception as e:
                logger.debug(f"Error finding guideline preferred size: {e}")
            return None


        def calculate_required_lines_for_text(text: str, font_size: float) -> int:
            try:
                # Use the new font guidelines system for consistency
                max_lines_allowed, max_cpl_allowed, _ = get_guideline_limits_interpolated(
                    font_family, font_size, textbox_height, textbox_width, conservative_bias=0.0
                )
                
                if max_cpl_allowed is None or max_cpl_allowed <= 0:
                    return text.count('\n') + 1

                import math as _math
                required = 0
                for seg in (text or '').split('\n'):
                    seg_len = len(seg)
                    required += max(1, _math.ceil(seg_len / max_cpl_allowed))
                return required
            except Exception as e:
                logger.debug(f"Error calculating required lines: {e}")

                import math as _math
                return max(1, _math.ceil(len(text.replace('\n', '')) / 32))


        # Start from minimum detected font size
        min_font_size_in_textbox = min(detected_font_sizes) if detected_font_sizes else baseline_font_size
        start_size = min_font_size_in_textbox
        logger.info(f"RECONSTRUCTION: Starting font size search from {start_size}pt (min detected: {min_font_size_in_textbox}pt)")


        # Calculate required lines for guideline preferred size lookup
        if src_graphics_lines is not None:
            try:
                actual_required_lines = int(src_graphics_lines)
            except Exception:
                actual_required_lines = calculate_required_lines_for_text(text, baseline_font_size)
        else:
            actual_required_lines = calculate_required_lines_for_text(text, baseline_font_size)

        guideline_preferred = get_guideline_preferred_size(actual_required_lines)


        # Guideline maximization: find the largest font size that fits
        total_chars_used = len(text.replace('\n', ''))
        best_fitting_size = None
        all_fitting_sizes = []
        
        # Add detailed debug header
        logger.info(f"RECONSTRUCTION: Font size iteration debug for text '{text}' ({total_chars_used} chars)")
        logger.info(f"RECONSTRUCTION: Size | Lines Allowed | CPL | Budget | Req Lines | Req Chars | Fits?")
        logger.info(f"RECONSTRUCTION: " + "-" * 70)
        
        for s in _half_step_sizes(start_size):
            max_lines_allowed, max_cpl_allowed, interpolation_details = get_guideline_limits_interpolated(
                font_family, s, textbox_height, textbox_width, conservative_bias=0.0
            )
            if max_lines_allowed is None or max_cpl_allowed is None:
                logger.info(f"RECONSTRUCTION: {s:4.1f} | {'N/A':13} | {'N/A':3} | {'N/A':6} | {'N/A':10} | {'N/A':10} | FAIL")
                continue

            req_lines_at_s = calculate_required_lines_for_text(text, s)
            char_budget_at_s = int(max_lines_allowed) * int(max_cpl_allowed)
            
            # Check if it fits
            lines_fit = req_lines_at_s <= int(max_lines_allowed)
            chars_fit = total_chars_used <= char_budget_at_s
            fits_check = lines_fit and chars_fit
            status = "PASS" if fits_check else "FAIL"
            
            # Log detailed debug line
            logger.info(f"RECONSTRUCTION: {s:4.1f} | {max_lines_allowed:13} | {max_cpl_allowed:3} | {char_budget_at_s:6} | {req_lines_at_s:10} | {total_chars_used:10} | {status}")
            
            if fits_check:
                all_fitting_sizes.append(float(s))
                logger.info(f"RECONSTRUCTION: Guideline maximization found fit at {s}pt (req_lines={req_lines_at_s} <= {max_lines_allowed}, chars {total_chars_used} <= budget {char_budget_at_s})")
        
        # Find the largest fitting size
        if all_fitting_sizes:
            best_fitting_size = max(all_fitting_sizes)
            logger.info(f"RECONSTRUCTION: Found fitting sizes: {all_fitting_sizes}, using largest: {best_fitting_size}pt")
            logger.info(f"RECONSTRUCTION: Best fitting font size: {best_fitting_size}pt")
            logger.info(f"RECONSTRUCTION: All fitting sizes: {all_fitting_sizes}")
            return best_fitting_size


        # If guideline maximization didn't find a fit, try guideline preferred size as fallback
        if guideline_preferred is not None:
            logger.info(f"RECONSTRUCTION: Testing guideline-preferred size {guideline_preferred}pt as fallback")
            if fits(text, guideline_preferred):
                logger.info(f"RECONSTRUCTION: Using guideline-preferred size {guideline_preferred}pt as fallback")
                return guideline_preferred
            else:
                logger.info(f"RECONSTRUCTION: Guideline-preferred size {guideline_preferred}pt does not fit")

        # Final fallback: try without newlines
        if '\n' in text:
            text_no_newlines = text.replace('\n', ' ')
            for size in _half_step_sizes(start_size):
                if fits(text_no_newlines, float(size)):
                    logger.info(f"RECONSTRUCTION: Using fallback fit without newlines at {size}pt")
                    return float(size)


        return MIN_FONT_SIZE

    def _generate_font_size_reasoning(self, simulated_text: str, dimensions: Dict[str, Any],
                                    all_font_sizes: List[float], baseline_font_size: float,
                                    optimal_font_size: float, baseline_font_family: str,
                                    original_text: str, src_graphics_lines: Optional[int] = None) -> GraphicsReasoning:
        try:

            orig_total_char = len(original_text.replace('\n', ''))
            new_total_char = len(simulated_text.replace('\n', ''))


            def calculate_actual_lines(text: str, font_size: float, font_family: str, dimensions: Dict[str, Any]) -> int:
                if not dimensions.get('has_dimensions', False):

                    return text.count('\n') + 1

                try:

                    safety = get_safety_margin(font_family, default_margin=0.15)
                    lines_fit, max_cpl, total_fit, *_ = evaluate_capacity(
                        dimensions['width'], dimensions['height'], font_family, font_size, safety_margin=safety
                    )

                    effective_lines_fit, effective_max_cpl = apply_guidelines_to_capacity(
                        lines_fit, max_cpl, dimensions['width'], dimensions['height'], font_family, font_size
                    )

                    lines_fit = effective_lines_fit
                    max_cpl = effective_max_cpl

                    if max_cpl <= 0:
                        return text.count('\n') + 1


                    paragraphs = text.split('\n')

                    total_lines = len(paragraphs)
                    import math
                    for paragraph in paragraphs:
                        if not paragraph.strip():

                            continue

                        chars_in_paragraph = len(paragraph)
                        wrapped = max(1, math.ceil(chars_in_paragraph / max_cpl))
                        overflow = max(0, wrapped - 1)
                        total_lines += overflow
                    return total_lines

                except Exception as e:
                    logger.debug(f"Error calculating actual lines: {e}, falling back to newline count")
                    return text.count('\n') + 1


            try:
                orig_total_lines = int(src_graphics_lines) if src_graphics_lines is not None else sum(1 for p in (original_text or '').split('\n') if p.strip())
            except Exception:
                orig_total_lines = sum(1 for p in (original_text or '').split('\n') if p.strip())


            # Compute total lines based on the estimated (optimal) font size rather than baseline
            try:
                new_total_lines = calculate_actual_lines(simulated_text, optimal_font_size, baseline_font_family, dimensions)
            except Exception:
                # Fallback to counting non-empty paragraphs if capacity calc fails
                new_total_lines = sum(1 for p in (simulated_text or '').split('\n') if p.strip())


            if baseline_font_size > 0:
                reduction_percent = ((baseline_font_size - optimal_font_size) / baseline_font_size) * 100
                reduction_percent_str = f"{reduction_percent:.0f}% Font Reduction"
            else:
                reduction_percent_str = "0% Font Reduction"


            new_size_change = f"{baseline_font_size:.1f} -> {optimal_font_size:.1f}"


            if dimensions.get('has_dimensions', False):
                try:
                    safety = get_safety_margin(baseline_font_family, default_margin=0.15)
                    lines_fit, max_cpl, total_fit, *_ = evaluate_capacity(
                        dimensions['width'], dimensions['height'], baseline_font_family, optimal_font_size, safety_margin=safety
                    )

                    if total_fit > 0:
                        fit_percent = (new_total_char / total_fit) * 100
                        if fit_percent > 100:
                            new_size_fit_percent = f"{fit_percent:.0f}% Over"
                        else:
                            new_size_fit_percent = f"{fit_percent:.0f}% Fit"
                    else:
                        new_size_fit_percent = "Unknown"
                except Exception:
                    new_size_fit_percent = "Unknown"
            else:
                new_size_fit_percent = "No Dimensions"

            # Get interpolation details for the optimal font size
            try:
                _, _, interpolation_details = get_guideline_limits_interpolated(
                    baseline_font_family, optimal_font_size, dimensions['height'], dimensions['width'], conservative_bias=0.0
                )
            except Exception:
                interpolation_details = "Error getting interpolation details"
            
            # Build line calculation details
            line_calc_details = f"Text: {len(simulated_text)} chars, {simulated_text.count(chr(10))} newlines; "
            line_calc_details += f"Optimal size: {optimal_font_size}pt; "
            line_calc_details += f"Lines: {new_total_lines} (calculated from capacity)"
            
            # Add font size iteration debug to reasoning
            try:
                # Generate font size iteration debug for reasoning
                debug_lines = []
                debug_lines.append(f"Font size iteration debug:")
                debug_lines.append(f"Size | Lines Allowed | CPL | Budget | Req Lines | Req Chars | Fits?")
                debug_lines.append("-" * 70)
                
                total_chars = len(simulated_text.replace('\n', ''))
                for s in [12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0]:
                    max_lines_allowed, max_cpl_allowed, _ = get_guideline_limits_interpolated(
                        baseline_font_family, s, dimensions['height'], dimensions['width'], conservative_bias=0.0
                    )
                    
                    if max_lines_allowed is None or max_cpl_allowed is None:
                        debug_lines.append(f"{s:4.1f} | {'N/A':13} | {'N/A':3} | {'N/A':6} | {'N/A':10} | {'N/A':10} | FAIL")
                        continue
                    
                    req_lines = calculate_actual_lines(simulated_text, s, baseline_font_family, dimensions)
                    char_budget = int(max_lines_allowed) * int(max_cpl_allowed)
                    
                    lines_fit = req_lines <= int(max_lines_allowed)
                    chars_fit = total_chars <= char_budget
                    fits = lines_fit and chars_fit
                    status = "PASS" if fits else "FAIL"
                    
                    debug_lines.append(f"{s:4.1f} | {max_lines_allowed:13} | {max_cpl_allowed:3} | {char_budget:6} | {req_lines:10} | {total_chars:10} | {status}")
                
                line_calc_details += f"; Debug: {'; '.join(debug_lines)}"
            except Exception as e:
                line_calc_details += f"; Debug generation failed: {e}"
            
            return GraphicsReasoning(
                orig_total_char=orig_total_char,
                orig_total_lines=orig_total_lines,
                new_total_char=new_total_char,
                new_total_lines=new_total_lines,
                new_size_fit_percent=new_size_fit_percent,
                reduction_percent=reduction_percent_str,
                new_size_change=new_size_change,
                interpolation_details=interpolation_details,
                line_calculation_details=line_calc_details
            )

        except Exception as e:
            logger.error(f"Error generating font reasoning: {e}")
            return GraphicsReasoning()

    def _replace_text_in_wt_elements(self, wt_elements: List[ET.Element], combined_text: str,
                                   matched_text: str, final_text: str, start_pos: int, end_pos: int) -> bool:
        try:
            logger.debug(f"Starting text replacement: '{matched_text}' -> '{final_text}' at position {start_pos}-{end_pos}")
            logger.debug(f"Total w:t elements: {len(wt_elements)}")


            # Adjust positions: combined_text includes newlines that are not present in concatenated w:t texts
            # Map indices from combined_text to indices in the raw concatenation of w:t.text values
            try:
                nl_before_start = (combined_text[:start_pos] or "").count('\n')
                nl_before_end = (combined_text[:end_pos] or "").count('\n')
                start_pos_effective = max(0, start_pos - nl_before_start)
                end_pos_effective = max(start_pos_effective, end_pos - nl_before_end)
                logger.debug(
                    f"Adjusted positions for wt replacement: start={start_pos_effective}, end={end_pos_effective} "
                    f"(nl_before_start={nl_before_start}, nl_before_end={nl_before_end})"
                )
            except Exception:
                # Fallback to original positions if adjustment fails
                start_pos_effective, end_pos_effective = start_pos, end_pos

            current_pos = 0
            affected_elements = []

            for i, wt_element in enumerate(wt_elements):
                element_text = wt_element.text or ""
                element_start = current_pos
                element_end = current_pos + len(element_text)

                logger.debug(f"Element {i}: text='{element_text}', pos={element_start}-{element_end}")


                if element_start < end_pos_effective and element_end > start_pos_effective:
                    affected_elements.append((wt_element, element_start, element_end))
                    logger.debug(f"Element {i} is affected by match")

                current_pos = element_end

            logger.debug(f"Found {len(affected_elements)} affected elements")

            if not affected_elements:
                logger.warning(f"No affected elements found for text replacement")
                return False


            for i, (wt_element, element_start, element_end) in enumerate(affected_elements):
                element_text = wt_element.text or ""
                logger.debug(f"Processing affected element {i}: '{element_text}' at pos {element_start}-{element_end}")

                if i == 0:

                    match_start_in_element = max(0, start_pos_effective - element_start)
                    match_end_in_element = min(len(element_text), end_pos_effective - element_start)

                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]

                    new_text = before_text + final_text + after_text
                    logger.info(f"REPLACING TEXT: '{element_text}' -> '{new_text}' (before='{before_text}', replacement='{final_text}', after='{after_text}')")
                    logger.debug(f"Setting wt_element.text = '{new_text}'")
                    wt_element.text = new_text
                    logger.debug(f"Text replacement applied to element {i}")

                else:

                    match_start_in_element = max(0, start_pos - element_start)
                    match_end_in_element = min(len(element_text), end_pos - element_start)

                    before_text = element_text[:match_start_in_element]
                    after_text = element_text[match_end_in_element:]

                    wt_element.text = before_text + after_text

            return True

        except Exception as e:
            logger.error(f"Error replacing text in w:t elements: {e}")
            return False