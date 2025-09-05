
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple

from docx import Document

from config import DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE

logger = logging.getLogger(__name__)


XML_NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
}


class GraphicsFontManager:

    @staticmethod
    def get_font_info_from_wt_elements(wt_elements: List[ET.Element]) -> Dict[str, Any]:
        font_info = {
            'family': DEFAULT_FONT_FAMILY,
            'sizes': [],
            'colors': [],
            'styles': []
        }

        try:
            for wt_element in wt_elements:

                parent = wt_element.getparent()
                if parent is None:
                    continue


                for rpr in parent.iter():
                    if rpr.tag.endswith('}rPr'):
                        for rFonts in rpr.iter():
                            if rFonts.tag.endswith('}rFonts'):
                                font_family = rFonts.get('w:ascii') or rFonts.get('w:eastAsia') or rFonts.get('w:hAnsi')
                                if font_family and font_family not in font_info['family']:
                                    font_info['family'] = font_family
                                break


                        for sz in rpr.iter():
                            if sz.tag.endswith('}sz') or sz.tag.endswith('}szCs'):
                                size_attr = sz.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                                if size_attr:
                                    try:

                                        size_pt = float(size_attr) / 2.0
                                        if size_pt not in font_info['sizes']:
                                            font_info['sizes'].append(size_pt)
                                    except (ValueError, TypeError):
                                        continue


                        for color in rpr.iter():
                            if color.tag.endswith('}color'):
                                color_attr = color.get('w:val')
                                if color_attr and color_attr not in font_info['colors']:
                                    font_info['colors'].append(color_attr)


                        for b in rpr.iter():
                            if b.tag.endswith('}b'):
                                if 'bold' not in font_info['styles']:
                                    font_info['styles'].append('bold')
                        for i in rpr.iter():
                            if i.tag.endswith('}i'):
                                if 'italic' not in font_info['styles']:
                                    font_info['styles'].append('italic')
                        break


            if not font_info['sizes']:
                font_info['sizes'] = [DEFAULT_FONT_SIZE]

            logger.debug(f"Extracted font info: {font_info}")

        except Exception as e:
            logger.error(f"Error extracting font info: {e}")

        return font_info

    @staticmethod
    def get_detailed_font_analysis(wt_elements: List[ET.Element]) -> Dict[str, Any]:
        detailed_analysis = {
            'all_font_sizes': [],
            'font_size_mapping': {},
            'font_family_mapping': {},
            'text_segments': [],
            'min_font_size': DEFAULT_FONT_SIZE,
            'max_font_size': DEFAULT_FONT_SIZE,
            'primary_font_family': DEFAULT_FONT_FAMILY
        }

        try:
            current_pos = 0
            all_sizes = set()
            all_families = set()

            for wt_element in wt_elements:

                text_content = wt_element.text or ""
                if not text_content:
                    continue


                parent = wt_element.getparent()
                if parent is None:
                    continue


                font_size = DEFAULT_FONT_SIZE
                font_family = DEFAULT_FONT_FAMILY


                for rpr in parent.iter():
                    if rpr.tag.endswith('}rPr'):

                        for rFonts in rpr.iter():
                            if rFonts.tag.endswith('}rFonts'):
                                font_family = (rFonts.get('w:ascii') or
                                             rFonts.get('w:eastAsia') or
                                             rFonts.get('w:hAnsi') or
                                             DEFAULT_FONT_FAMILY)
                                break


                        for sz in rpr.iter():
                            if sz.tag.endswith('}sz') or sz.tag.endswith('}szCs'):
                                size_attr = sz.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                                if size_attr:
                                    try:

                                        font_size = float(size_attr) / 2.0
                                    except (ValueError, TypeError):
                                        font_size = DEFAULT_FONT_SIZE
                                break
                        break


                text_segment = {
                    'text': text_content,
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(text_content),
                    'font_size': font_size,
                    'font_family': font_family
                }
                detailed_analysis['text_segments'].append(text_segment)


                for i in range(len(text_content)):
                    pos = current_pos + i
                    detailed_analysis['font_size_mapping'][pos] = font_size
                    detailed_analysis['font_family_mapping'][pos] = font_family


                all_sizes.add(font_size)
                all_families.add(font_family)

                current_pos += len(text_content)


            detailed_analysis['all_font_sizes'] = sorted(list(all_sizes))
            detailed_analysis['min_font_size'] = min(all_sizes) if all_sizes else DEFAULT_FONT_SIZE
            detailed_analysis['max_font_size'] = max(all_sizes) if all_sizes else DEFAULT_FONT_SIZE
            detailed_analysis['primary_font_family'] = list(all_families)[0] if all_families else DEFAULT_FONT_FAMILY

            logger.debug(f"Detailed font analysis: {len(detailed_analysis['all_font_sizes'])} unique sizes, "
                        f"range {detailed_analysis['min_font_size']}-{detailed_analysis['max_font_size']}pt")

        except Exception as e:
            logger.error(f"Error in detailed font analysis: {e}")

        return detailed_analysis

    @staticmethod
    def extract_font_info_for_detection(wt_elements: List[ET.Element], matched_text: str,
                                      start_pos: int, document: Document = None) -> Dict[str, Any]:
        font_info = {
            'font_family': DEFAULT_FONT_FAMILY,
            'font_size': DEFAULT_FONT_SIZE,
            'font_color': 'auto',
            'font_style': 'normal'
        }

        try:

            general_font_info = GraphicsFontManager.get_font_info_from_wt_elements(wt_elements)

            if general_font_info['family']:
                font_info['font_family'] = general_font_info['family']

            if general_font_info['sizes']:

                font_info['font_size'] = general_font_info['sizes'][0]

            if general_font_info['colors']:
                font_info['font_color'] = general_font_info['colors'][0]

            if general_font_info['styles']:
                font_info['font_style'] = ', '.join(general_font_info['styles'])

            logger.debug(f"Font info for detection '{matched_text}': {font_info}")

        except Exception as e:
            logger.error(f"Error extracting font info for detection: {e}")

        return font_info

    @staticmethod
    def normalize_font_sizes(wt_elements: List[ET.Element], target_size: float):
        try:
            for wt_element in wt_elements:

                parent = wt_element.getparent()
                while parent is not None and not parent.tag.endswith('}r'):
                    parent = parent.getparent()

                if parent is not None:

                    GraphicsFontManager._ensure_paragraph_level_size(parent, target_size)

                    r_pr = parent.find('.//w:rPr', namespaces=XML_NAMESPACES)
                    if r_pr is None:


                        from lxml import etree
                        r_pr = etree.SubElement(parent, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
                        logger.debug(f"Created missing w:rPr element for text run to enable font size normalization")

                    GraphicsFontManager._ensure_rpr_before_text(parent, r_pr)


                    sz_element = r_pr.find('.//w:sz', namespaces=XML_NAMESPACES)
                    if sz_element is not None:
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:

                        from lxml import etree
                        sz_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz')
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))


                    sz_cs_element = r_pr.find('.//w:szCs', namespaces=XML_NAMESPACES)
                    if sz_cs_element is not None:
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:

                        from lxml import etree
                        sz_cs_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))

        except Exception as e:
            logger.error(f"Error normalizing font sizes: {e}")

    @staticmethod
    def normalize_font_sizes_and_family(wt_elements: List[ET.Element], target_size: float, target_family: str):
        try:
            for wt_element in wt_elements:

                parent = wt_element.getparent()
                while parent is not None and not parent.tag.endswith('}r'):
                    parent = parent.getparent()

                if parent is not None:

                    GraphicsFontManager._ensure_paragraph_level_size(parent, target_size)

                    r_pr = parent.find('.//w:rPr', namespaces=XML_NAMESPACES)
                    if r_pr is None:


                        from lxml import etree
                        r_pr = etree.SubElement(parent, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
                        logger.debug(f"Created missing w:rPr element for text run to enable font family and size normalization")

                    GraphicsFontManager._ensure_rpr_before_text(parent, r_pr)


                    r_fonts = r_pr.find('.//w:rFonts', namespaces=XML_NAMESPACES)
                    if r_fonts is not None:
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', target_family)
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', target_family)
                    else:

                        from lxml import etree
                        r_fonts = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', target_family)
                        r_fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', target_family)


                    sz_element = r_pr.find('.//w:sz', namespaces=XML_NAMESPACES)
                    if sz_element is not None:
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:

                        from lxml import etree
                        sz_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz')
                        sz_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))


                    sz_cs_element = r_pr.find('.//w:szCs', namespaces=XML_NAMESPACES)
                    if sz_cs_element is not None:
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
                    else:

                        from lxml import etree
                        sz_cs_element = etree.SubElement(r_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
                        sz_cs_element.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))

        except Exception as e:
            logger.error(f"Error normalizing font sizes and family: {e}")

    @staticmethod
    def _ensure_paragraph_level_size(run_element: ET.Element, target_size: float) -> None:
        try:

            p = run_element.getparent()
            while p is not None and not p.tag.endswith('}p'):
                p = p.getparent()
            if p is None:
                return

            p_pr = p.find('.//w:pPr', namespaces=XML_NAMESPACES)
            if p_pr is None:
                from lxml import etree
                p_pr = etree.SubElement(p, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')

            from lxml import etree
            p_rpr = p_pr.find('.//w:rPr', namespaces=XML_NAMESPACES)
            if p_rpr is None:
                p_rpr = etree.SubElement(p_pr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')

            szcs = p_rpr.find('.//w:szCs', namespaces=XML_NAMESPACES)
            if szcs is None:
                szcs = etree.SubElement(p_rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs')
            szcs.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(target_size * 2)))
        except Exception as e:
            logger.debug(f"Failed to ensure paragraph-level size: {e}")

    @staticmethod
    def _ensure_rpr_before_text(run_element: ET.Element, r_pr: ET.Element) -> None:
        try:

            t_elem = None
            for child in list(run_element):
                if child.tag.endswith('}t'):
                    t_elem = child
                    break
            if t_elem is None:
                return

            if run_element.index(r_pr) > run_element.index(t_elem):
                run_element.remove(r_pr)
                run_element.insert(run_element.index(t_elem), r_pr)
        except Exception as e:
            logger.debug(f"Failed to order rPr before text: {e}")


class GraphicsTextReplacer:

    def __init__(self, mode: str = "append", separator: str = ";", default_mapping: str = "4022-NA"):
        self.mode = mode
        self.separator = separator
        self.default_mapping = default_mapping

    def replace_text_in_runs(self, runs: List, original_text: str, replacement_text: str,
                           start_pos: int, end_pos: int) -> bool:
        try:

            text_span = GraphicsTextReconstructor.find_text_in_runs(runs, original_text, start_pos)
            if not text_span:
                return False

            span_start, span_end, affected_runs = text_span


            if affected_runs:
                first_run = affected_runs[0]
                run_text = first_run.text


                current_pos = 0
                for run in runs:
                    if run == first_run:
                        break
                    current_pos += len(run.text)

                match_start_in_run = span_start - current_pos
                match_end_in_run = min(len(run_text), span_end - current_pos)


                before_text = run_text[:match_start_in_run]
                after_text = run_text[match_end_in_run:]


                if self.mode == "append":
                    final_text = f"{original_text}{self.separator}{replacement_text}"
                else:
                    final_text = replacement_text

                first_run.text = before_text + final_text + after_text


                for run in affected_runs[1:]:
                    run.text = ""

                return True

        except Exception as e:
            logger.error(f"Error replacing text in runs: {e}")
            return False


class GraphicsTextReconstructor:

    @staticmethod
    def find_text_in_runs(runs: List, text: str, start_pos: int) -> Optional[Tuple[int, int, List]]:
        try:
            current_pos = 0
            affected_runs = []
            span_start = None
            span_end = None

            for run in runs:
                run_text = run.text or ""
                run_start = current_pos
                run_end = current_pos + len(run_text)


                if run_start <= start_pos < run_end:

                    span_start = start_pos
                    affected_runs.append(run)


                    remaining_text = text
                    remaining_start = start_pos

                    for next_run in runs[runs.index(run):]:
                        next_run_text = next_run.text or ""
                        next_run_start = remaining_start
                        next_run_end = remaining_start + len(next_run_text)


                        if remaining_text.startswith(next_run_text):

                            span_end = next_run_end
                            if next_run not in affected_runs:
                                affected_runs.append(next_run)
                            break
                        elif next_run_text.startswith(remaining_text):

                            span_end = next_run_start + len(remaining_text)
                            if next_run not in affected_runs:
                                affected_runs.append(next_run)
                            break
                        else:

                            if next_run not in affected_runs:
                                affected_runs.append(next_run)
                            remaining_text = remaining_text[len(next_run_text):]
                            remaining_start = next_run_end

                    if span_end is None:

                        return None

                    return span_start, span_end, affected_runs

                current_pos = run_end

            return None

        except Exception as e:
            logger.error(f"Error finding text in runs: {e}")
            return None


def create_graphics_text_replacer(mode: str = "append", separator: str = ";",
                                                                  default_mapping: str = "4022-NA") -> GraphicsTextReplacer:
    return GraphicsTextReplacer(mode, separator, default_mapping)


class TextboxParser:

    @staticmethod
    def find_textboxes(document: Document) -> List[Tuple[ET.Element, bool]]:
        textboxes = []

        try:

            doc_xml = document._element


            for drawing in doc_xml.iter():
                if drawing.tag.endswith('}drawing'):

                    for textbox in drawing.iter():
                        if textbox.tag.endswith('}txbx'):
                            textboxes.append((textbox, False))


            if not textboxes:
                for paragraph in doc_xml.iter():
                    if paragraph.tag.endswith('}p'):

                        p_pr = paragraph.find('.//w:pPr', namespaces=XML_NAMESPACES)
                        if p_pr is not None:

                            for child in p_pr:
                                if child.tag.endswith('}textboxTightWrap') or child.tag.endswith('}textbox'):
                                    textboxes.append((paragraph, True))
                                    break

            logger.info(f"Found {len(textboxes)} textboxes in document")

        except Exception as e:
            logger.error(f"Error finding textboxes: {e}")

        return textboxes

    @staticmethod
    def extract_text_from_textbox(textbox_element: ET.Element) -> Tuple[str, List[ET.Element]]:
        combined_text = ""
        wt_elements = []

        try:

            paragraphs_found = False
            paragraph_texts = []
            for para in textbox_element.iter():
                if para.tag.endswith('}p'):
                    paragraphs_found = True
                    parts = []
                    has_text_run = False
                    for node in para.iter():

                        if node.tag.endswith('}br') or node.tag.endswith('}cr'):
                            parts.append('\n')
                        elif node.tag.endswith('}t'):
                            text_val = (node.text or "")
                            if text_val.strip():
                                has_text_run = True
                            wt_elements.append(node)
                            parts.append(text_val)

                    if has_text_run:
                        paragraph_texts.append("".join(parts))

            if paragraphs_found:
                combined_text = "\n".join(paragraph_texts)
            else:

                for wt_element in textbox_element.iter():
                    if wt_element.tag.endswith('}t'):
                        text_content = wt_element.text or ""
                        combined_text += text_content
                        wt_elements.append(wt_element)

            logger.debug(f"Extracted {len(wt_elements)} w:t elements with {len(combined_text)} characters")

        except Exception as e:
            logger.error(f"Error extracting text from textbox: {e}")

        return combined_text, wt_elements

    @staticmethod
    def get_textbox_dimensions(textbox_element: ET.Element) -> Dict[str, Any]:
        dimensions = {
            'width': 0.0,
            'height': 0.0,
            'has_dimensions': False
        }

        try:

            for element in textbox_element.iter():
                if element.tag.endswith('}txbxContent'):

                    width_attr = element.get('w:width')
                    height_attr = element.get('w:height')

                    if width_attr and height_attr:
                        try:
                            dimensions['width'] = float(width_attr)
                            dimensions['height'] = float(height_attr)
                            dimensions['has_dimensions'] = True
                            logger.debug(f"Found dimensions from w:txbxContent: {dimensions['width']:.1f}x{dimensions['height']:.1f}")
                            break
                        except (ValueError, TypeError):
                            continue


                elif element.tag.endswith('}shape'):
                    width_attr = element.get('style')
                    if width_attr:

                        import re
                        width_match = re.search(r'width:\s*([\d.]+)pt', width_attr)
                        height_match = re.search(r'height:\s*([\d.]+)pt', width_attr)

                        if width_match and height_match:
                            dimensions['width'] = float(width_match.group(1))
                            dimensions['height'] = float(height_match.group(1))
                            dimensions['has_dimensions'] = True
                            logger.debug(f"Found dimensions from VML shape: {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
                            break

            if not dimensions['has_dimensions']:

                parent = textbox_element.getparent()
                while parent is not None:

                    for extent in parent.iter():
                        if extent.tag.endswith('}extent'):
                            cx = extent.get('cx')
                            cy = extent.get('cy')

                            if cx and cy:
                                try:

                                    dimensions['width'] = int(cx) / 914400 * 72
                                    dimensions['height'] = int(cy) / 914400 * 72
                                    dimensions['has_dimensions'] = True
                                    logger.debug(f"Found dimensions from w:drawing extent: {cx}x{cy} EMUs = {dimensions['width']:.1f}x{dimensions['height']:.1f} points")
                                    break
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Error converting dimensions: {e}")
                                    continue

                    if dimensions['has_dimensions']:
                        break

                    parent = parent.getparent()


            if not dimensions['has_dimensions']:
                combined_text, _ = TextboxParser.extract_text_from_textbox(textbox_element)
                if combined_text:

                    lines = combined_text.count('\n') + 1
                    dimensions['width'] = len(combined_text) * 6.0
                    dimensions['height'] = lines * 12.0
                    dimensions['has_dimensions'] = True
                    logger.debug(f"Estimated dimensions: {dimensions['width']:.1f}x{dimensions['height']:.1f} points")

        except Exception as e:
            logger.error(f"Error getting textbox dimensions: {e}")

        return dimensions