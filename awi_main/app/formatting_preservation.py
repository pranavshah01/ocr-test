#!/usr/bin/env python3
"""
Formatting Preservation Module for DOCX Text Replacement.
Handles preservation of font family, size, color, style, alignment, and complex nested formatting.
"""

from typing import Dict, Any, List, Optional, Tuple
from lxml import etree
import logging
from datetime import datetime
from .shared_constants import XML_NAMESPACES, SharedUtilities


class FormattingPreserver:
    """
    Handles preservation of all formatting attributes during text replacement in DOCX XML.
    Preserves font family, size, color, style, alignment, and complex nested formatting.
    """
    
    def __init__(self):
        self.processing_log = []
        self.formatting_preserved = 0
        self.formatting_errors = 0
        
        # DOCX XML namespaces
        self.namespaces = XML_NAMESPACES
    
    def extract_run_formatting(self, run_element: etree.Element) -> Dict[str, Any]:
        """
        Extract all formatting properties from a run element.
        
        Args:
            run_element: XML run element (<w:r>)
            
        Returns:
            Dictionary containing all formatting properties
        """
        formatting = {
            'font_family': None,
            'font_size': None,
            'font_color': None,
            'bold': False,
            'italic': False,
            'underline': None,
            'strike': False,
            'superscript': False,
            'subscript': False,
            'highlight': None,
            'spacing': None,
            'position': None,
            'kern': None,
            'lang': None,
            'style_id': None,
            'complex_formatting': {}
        }
        
        try:
            # Find run properties element
            rpr = run_element.find('.//w:rPr', self.namespaces)
            if rpr is None:
                return formatting
            
            # Font family
            font_elem = rpr.find('w:rFonts', self.namespaces)
            if font_elem is not None:
                formatting['font_family'] = font_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii')
            
            # Font size
            sz_elem = rpr.find('w:sz', self.namespaces)
            if sz_elem is not None:
                formatting['font_size'] = sz_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Font color
            color_elem = rpr.find('w:color', self.namespaces)
            if color_elem is not None:
                formatting['font_color'] = color_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Bold
            bold_elem = rpr.find('w:b', self.namespaces)
            formatting['bold'] = bold_elem is not None
            
            # Italic
            italic_elem = rpr.find('w:i', self.namespaces)
            formatting['italic'] = italic_elem is not None
            
            # Underline
            u_elem = rpr.find('w:u', self.namespaces)
            if u_elem is not None:
                formatting['underline'] = u_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', 'single')
            
            # Strike
            strike_elem = rpr.find('w:strike', self.namespaces)
            formatting['strike'] = strike_elem is not None
            
            # Superscript/Subscript
            vertAlign_elem = rpr.find('w:vertAlign', self.namespaces)
            if vertAlign_elem is not None:
                val = vertAlign_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                formatting['superscript'] = val == 'superscript'
                formatting['subscript'] = val == 'subscript'
            
            # Highlight
            highlight_elem = rpr.find('w:highlight', self.namespaces)
            if highlight_elem is not None:
                formatting['highlight'] = highlight_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Character spacing
            spacing_elem = rpr.find('w:spacing', self.namespaces)
            if spacing_elem is not None:
                formatting['spacing'] = spacing_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Position
            position_elem = rpr.find('w:position', self.namespaces)
            if position_elem is not None:
                formatting['position'] = position_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Kerning
            kern_elem = rpr.find('w:kern', self.namespaces)
            if kern_elem is not None:
                formatting['kern'] = kern_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Language
            lang_elem = rpr.find('w:lang', self.namespaces)
            if lang_elem is not None:
                formatting['lang'] = lang_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Style ID
            style_elem = rpr.find('w:rStyle', self.namespaces)
            if style_elem is not None:
                formatting['style_id'] = style_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Store complex formatting elements for exact replication
            formatting['complex_formatting']['rpr_xml'] = etree.tostring(rpr, encoding='unicode')
            
            self.processing_log.append(f"Extracted formatting: font={formatting['font_family']}, size={formatting['font_size']}, color={formatting['font_color']}")
            
        except Exception as e:
            self.formatting_errors += 1
            self.processing_log.append(f"Error extracting formatting: {e}")
        
        return formatting
    
    def extract_paragraph_formatting(self, paragraph_element: etree.Element) -> Dict[str, Any]:
        """
        Extract paragraph-level formatting properties.
        
        Args:
            paragraph_element: XML paragraph element (<w:p>)
            
        Returns:
            Dictionary containing paragraph formatting properties
        """
        formatting = {
            'alignment': None,
            'indent_left': None,
            'indent_right': None,
            'indent_first_line': None,
            'spacing_before': None,
            'spacing_after': None,
            'line_spacing': None,
            'style_id': None,
            'keep_next': False,
            'keep_lines': False,
            'page_break_before': False,
            'widow_control': True,
            'complex_formatting': {}
        }
        
        try:
            # Find paragraph properties element
            ppr = paragraph_element.find('w:pPr', self.namespaces)
            if ppr is None:
                return formatting
            
            # Alignment
            jc_elem = ppr.find('w:jc', self.namespaces)
            if jc_elem is not None:
                formatting['alignment'] = jc_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Indentation
            ind_elem = ppr.find('w:ind', self.namespaces)
            if ind_elem is not None:
                formatting['indent_left'] = ind_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}left')
                formatting['indent_right'] = ind_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}right')
                formatting['indent_first_line'] = ind_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}firstLine')
            
            # Spacing
            spacing_elem = ppr.find('w:spacing', self.namespaces)
            if spacing_elem is not None:
                formatting['spacing_before'] = spacing_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}before')
                formatting['spacing_after'] = spacing_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}after')
                formatting['line_spacing'] = spacing_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}line')
            
            # Paragraph style
            style_elem = ppr.find('w:pStyle', self.namespaces)
            if style_elem is not None:
                formatting['style_id'] = style_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            
            # Keep with next
            keep_next_elem = ppr.find('w:keepNext', self.namespaces)
            formatting['keep_next'] = keep_next_elem is not None
            
            # Keep lines together
            keep_lines_elem = ppr.find('w:keepLines', self.namespaces)
            formatting['keep_lines'] = keep_lines_elem is not None
            
            # Page break before
            page_break_elem = ppr.find('w:pageBreakBefore', self.namespaces)
            formatting['page_break_before'] = page_break_elem is not None
            
            # Widow control
            widow_elem = ppr.find('w:widowControl', self.namespaces)
            if widow_elem is not None:
                val = widow_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', '1')
                formatting['widow_control'] = val != '0'
            
            # Store complex formatting for exact replication
            formatting['complex_formatting']['ppr_xml'] = etree.tostring(ppr, encoding='unicode')
            
            self.processing_log.append(f"Extracted paragraph formatting: alignment={formatting['alignment']}, style={formatting['style_id']}")
            
        except Exception as e:
            self.formatting_errors += 1
            self.processing_log.append(f"Error extracting paragraph formatting: {e}")
        
        return formatting
    
    def create_formatted_run(self, text: str, run_formatting: Dict[str, Any]) -> etree.Element:
        """
        Create a new run element with preserved formatting.
        
        Args:
            text: Text content for the run
            run_formatting: Formatting properties to apply
            
        Returns:
            New run element with formatting applied
        """
        try:
            # Create new run element
            run = etree.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')
            
            # Create run properties if we have formatting
            if any(v for k, v in run_formatting.items() if k != 'complex_formatting' and v):
                rpr = etree.SubElement(run, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
                
                # Apply font family
                if run_formatting.get('font_family'):
                    fonts = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
                    fonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', run_formatting['font_family'])
                
                # Apply font size
                if run_formatting.get('font_size'):
                    sz = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz')
                    sz.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(run_formatting['font_size']))
                
                # Apply font color
                if run_formatting.get('font_color'):
                    color = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}color')
                    color.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', run_formatting['font_color'])
                
                # Apply bold
                if run_formatting.get('bold'):
                    etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b')
                
                # Apply italic
                if run_formatting.get('italic'):
                    etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}i')
                
                # Apply underline
                if run_formatting.get('underline'):
                    u = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}u')
                    u.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', run_formatting['underline'])
                
                # Apply strike
                if run_formatting.get('strike'):
                    etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}strike')
                
                # Apply superscript/subscript
                if run_formatting.get('superscript') or run_formatting.get('subscript'):
                    vertAlign = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}vertAlign')
                    val = 'superscript' if run_formatting.get('superscript') else 'subscript'
                    vertAlign.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', val)
                
                # Apply highlight
                if run_formatting.get('highlight'):
                    highlight = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}highlight')
                    highlight.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', run_formatting['highlight'])
                
                # Apply character spacing
                if run_formatting.get('spacing'):
                    spacing = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}spacing')
                    spacing.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(run_formatting['spacing']))
                
                # Apply position
                if run_formatting.get('position'):
                    position = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}position')
                    position.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(run_formatting['position']))
                
                # Apply kerning
                if run_formatting.get('kern'):
                    kern = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}kern')
                    kern.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(run_formatting['kern']))
                
                # Apply language
                if run_formatting.get('lang'):
                    lang = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}lang')
                    lang.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', run_formatting['lang'])
                
                # Apply style
                if run_formatting.get('style_id'):
                    style = etree.SubElement(rpr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rStyle')
                    style.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', run_formatting['style_id'])
            
            # Create text element
            t = etree.SubElement(run, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
            t.text = text
            
            # Preserve space if needed
            if text and (text.startswith(' ') or text.endswith(' ') or '  ' in text):
                t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            
            self.formatting_preserved += 1
            self.processing_log.append(f"Created formatted run with text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            return run
            
        except Exception as e:
            self.formatting_errors += 1
            self.processing_log.append(f"Error creating formatted run: {e}")
            # Return basic run without formatting as fallback
            run = etree.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')
            t = etree.SubElement(run, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
            t.text = text
            return run
    
    def apply_formatting_preserving_replacement(self, root: etree.Element, match_data: Dict, 
                                              replacement_text: str, text_mode: str) -> bool:
        """
        Apply text replacement while preserving all formatting.
        
        Args:
            root: Document root element
            match_data: Match data with fragment information
            replacement_text: Text to replace with
            text_mode: "replace" or "append"
            
        Returns:
            True if replacement was successful
        """
        try:
            # Get fragment information from match_data
            fragments = match_data.get('fragments', [])
            if not fragments:
                self.processing_log.append("No fragments found in match data")
                return False
            
            # Find the first and last fragments to determine replacement scope
            first_fragment = fragments[0]
            last_fragment = fragments[-1]
            
            # Locate the actual XML elements
            first_run = self._find_run_by_fragment(root, first_fragment)
            last_run = self._find_run_by_fragment(root, last_fragment)
            
            if first_run is None or last_run is None:
                self.processing_log.append("Could not locate runs for replacement")
                return False
            
            # Extract formatting from the first run (primary formatting)
            primary_formatting = self.extract_run_formatting(first_run)
            
            # Get parent paragraph for paragraph-level formatting
            paragraph = first_run.getparent()
            while paragraph is not None and paragraph.tag != '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p':
                paragraph = paragraph.getparent()
            
            paragraph_formatting = {}
            if paragraph is not None:
                paragraph_formatting = self.extract_paragraph_formatting(paragraph)
            
            # Apply replacement based on text_mode
            if text_mode == "replace":
                final_text = replacement_text
            else:  # append mode
                original_text = match_data.get('match_text', '')
                final_text = f"{original_text} {replacement_text}"
            
            # Create new formatted run(s)
            if text_mode == "append":
                # Create two runs for append mode
                original_run = self.create_formatted_run(match_data.get('match_text', ''), primary_formatting)
                space_run = self.create_formatted_run(" ", primary_formatting)
                replacement_run = self.create_formatted_run(replacement_text, primary_formatting)
                new_runs = [original_run, space_run, replacement_run]
            else:
                # Single run for replace mode
                new_runs = [self.create_formatted_run(final_text, primary_formatting)]
            
            # Remove old content from affected runs
            self._clear_matched_content(root, fragments)
            
            # Insert new formatted runs
            self._insert_formatted_runs(first_run, new_runs)
            
            self.processing_log.append(f"Successfully applied formatting-preserving replacement: '{match_data.get('match_text', '')}' -> '{final_text}'")
            return True
            
        except Exception as e:
            self.formatting_errors += 1
            self.processing_log.append(f"Error in formatting-preserving replacement: {e}")
            return False
    
    def _find_run_by_fragment(self, root: etree.Element, fragment: Dict) -> Optional[etree.Element]:
        """
        Find a run element by fragment information.
        
        Args:
            root: Document root element
            fragment: Fragment information with xpath or other identifiers
            
        Returns:
            Run element if found, None otherwise
        """
        try:
            # Use xpath if available
            if 'xpath' in fragment:
                runs = root.xpath(fragment['xpath'], namespaces=self.namespaces)
                if runs:
                    return runs[0]
            
            # Fallback: search by text content and position
            if 'text' in fragment and 'position' in fragment:
                all_runs = root.xpath('.//w:r', namespaces=self.namespaces)
                for run in all_runs:
                    t_elem = run.find('w:t', self.namespaces)
                    if t_elem is not None and fragment['text'] in (t_elem.text or ''):
                        return run
            
            return None
            
        except Exception as e:
            self.processing_log.append(f"Error finding run by fragment: {e}")
            return None
    
    def _clear_matched_content(self, root: etree.Element, fragments: List[Dict]):
        """
        Clear the matched content from the specified fragments.
        
        Args:
            root: Document root element
            fragments: List of fragment information
        """
        try:
            for fragment in fragments:
                run = self._find_run_by_fragment(root, fragment)
                if run is not None:
                    # Clear text content but preserve run structure
                    t_elem = run.find('w:t', self.namespaces)
                    if t_elem is not None:
                        # Remove only the matched portion
                        start_pos = fragment.get('start_pos', 0)
                        end_pos = fragment.get('end_pos', len(t_elem.text or ''))
                        original_text = t_elem.text or ''
                        
                        # Keep text before and after the match
                        new_text = original_text[:start_pos] + original_text[end_pos:]
                        t_elem.text = new_text if new_text else None
                        
                        # Remove empty text elements
                        if not new_text:
                            run.remove(t_elem)
            
        except Exception as e:
            self.processing_log.append(f"Error clearing matched content: {e}")
    
    def _insert_formatted_runs(self, reference_run: etree.Element, new_runs: List[etree.Element]):
        """
        Insert new formatted runs at the position of the reference run.
        
        Args:
            reference_run: Reference run element for positioning
            new_runs: List of new run elements to insert
        """
        try:
            parent = reference_run.getparent()
            if parent is None:
                return
            
            # Find the position of the reference run
            ref_index = list(parent).index(reference_run)
            
            # Insert new runs after the reference run
            for i, new_run in enumerate(new_runs):
                parent.insert(ref_index + 1 + i, new_run)
            
            self.processing_log.append(f"Inserted {len(new_runs)} formatted runs")
            
        except Exception as e:
            self.processing_log.append(f"Error inserting formatted runs: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of formatting preservation processing.
        
        Returns:
            Processing summary dictionary
        """
        return {
            "formatting_preserved": self.formatting_preserved,
            "formatting_errors": self.formatting_errors,
            "processing_log": self.processing_log,
            "processing_time": datetime.now().isoformat()
        }
    
    def apply_run_formatting(self, run_element: etree.Element, formatting: Dict[str, Any]) -> None:
        """
        Apply formatting properties to a run element.
        
        Args:
            run_element: XML run element (<w:r>)
            formatting: Dictionary containing formatting properties to apply
        """
        if not formatting or not run_element:
            return
        
        try:
            # Get or create run properties element
            rpr = run_element.find('.//w:rPr', self.namespaces)
            if rpr is None:
                rpr = etree.SubElement(run_element, f"{{{self.namespaces['w']}}}rPr")
            
            # Apply font family
            if formatting.get('font_family'):
                font_elem = rpr.find('.//w:rFonts', self.namespaces)
                if font_elem is None:
                    font_elem = etree.SubElement(rpr, f"{{{self.namespaces['w']}}}rFonts")
                font_elem.set(f"{{{self.namespaces['w']}}}ascii", formatting['font_family'])
                font_elem.set(f"{{{self.namespaces['w']}}}hAnsi", formatting['font_family'])
            
            # Apply font size
            if formatting.get('font_size'):
                size_elem = rpr.find('.//w:sz', self.namespaces)
                if size_elem is None:
                    size_elem = etree.SubElement(rpr, f"{{{self.namespaces['w']}}}sz")
                size_elem.set(f"{{{self.namespaces['w']}}}val", str(formatting['font_size']))
            
            # Apply bold
            if formatting.get('bold'):
                bold_elem = rpr.find('.//w:b', self.namespaces)
                if bold_elem is None:
                    etree.SubElement(rpr, f"{{{self.namespaces['w']}}}b")
            
            # Apply italic
            if formatting.get('italic'):
                italic_elem = rpr.find('.//w:i', self.namespaces)
                if italic_elem is None:
                    etree.SubElement(rpr, f"{{{self.namespaces['w']}}}i")
            
            # Apply underline
            if formatting.get('underline'):
                underline_elem = rpr.find('.//w:u', self.namespaces)
                if underline_elem is None:
                    underline_elem = etree.SubElement(rpr, f"{{{self.namespaces['w']}}}u")
                underline_elem.set(f"{{{self.namespaces['w']}}}val", formatting['underline'])
            
            # Apply font color
            if formatting.get('font_color'):
                color_elem = rpr.find('.//w:color', self.namespaces)
                if color_elem is None:
                    color_elem = etree.SubElement(rpr, f"{{{self.namespaces['w']}}}color")
                color_elem.set(f"{{{self.namespaces['w']}}}val", formatting['font_color'])
            
            self.formatting_preserved += 1
            self.processing_log.append(f"Applied formatting to run: {list(formatting.keys())}")
            
        except Exception as e:
            self.formatting_errors += 1
            self.processing_log.append(f"Error applying run formatting: {e}")
