"""
Shared pattern matching utilities for all processors.
Handles regex pattern matching and text mapping across image, text, and graphics processors.
Includes enhanced universal pattern matching for better detection.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class UniversalMatch:
    """Enhanced match with universal character support and context information."""
    pattern_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    preceding_context: str = ""
    following_context: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_name': self.pattern_name,
            'matched_text': self.matched_text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'preceding_context': self.preceding_context,
            'following_context': self.following_context,
            'confidence': self.confidence
        }

class PatternMatcher:
    """Enhanced pattern matcher with universal character support - shared across all processors."""
    
    def __init__(self, patterns: Dict[str, str], mappings: Dict[str, str], enhanced_mode: bool = True):
        """
        Initialize pattern matcher.
        
        Args:
            patterns: Dictionary of pattern names to regex patterns
            mappings: Dictionary of original text to replacement text
            enhanced_mode: Whether to use enhanced universal pattern matching
        """
        self.patterns = patterns
        self.mappings = mappings
        self.enhanced_mode = enhanced_mode
        self.compiled_patterns = {}
        self.enhanced_patterns = {}
        
        if enhanced_mode:
            # Create enhanced patterns and compile them
            self._create_enhanced_patterns()
        
        self._compile_patterns()
    
    def _create_enhanced_patterns(self):
        """Create enhanced patterns with universal character support."""
        for name, pattern in self.patterns.items():
            if not name.startswith('_'):  # Skip metadata
                # Convert existing word boundary patterns to universal patterns
                enhanced_pattern = self._convert_to_universal_pattern(pattern)
                self.enhanced_patterns[name] = enhanced_pattern
                logger.debug(f"Enhanced pattern '{name}': {pattern} -> {enhanced_pattern}")
    
    def _convert_to_universal_pattern(self, pattern: str) -> str:
        """
        Convert word boundary patterns to universal patterns that work with any character.
        
        Args:
            pattern: Original regex pattern
            
        Returns:
            Enhanced pattern with universal character support
        """
        # Replace negative lookbehind word boundary with non-digit boundary
        # Use fixed-width lookbehind that works with Python regex
        enhanced = pattern.replace(r'(?<!\w)', r'(?<![0-9])')
        
        # Replace negative lookahead word boundary with non-digit boundary
        # This ensures "77" is not part of a larger number
        enhanced = enhanced.replace(r'(?!\w)', r'(?![0-9])')
        
        # For patterns that don't have lookbehind but start with 77, add a boundary condition
        # that prevents matching when 77 is part of a larger number sequence
        if '(?<!' not in enhanced and enhanced.startswith('77'):
            # Only prevent matching if 77 is immediately preceded by a digit AND not followed by a hyphen
            # This allows "77-" patterns even after digits, but prevents "177" from matching as "77"
            enhanced = r'(?<!(?<!\d)\d)' + enhanced
        
        return enhanced
    
    def _compile_patterns(self):
        """Compile all patterns for efficient matching."""
        for name, pattern in self.patterns.items():
            if not name.startswith('_'):  # Skip metadata
                try:
                    if self.enhanced_mode and name in self.enhanced_patterns:
                        # Use enhanced pattern
                        enhanced_pattern = self.enhanced_patterns[name]
                        compiled_pattern = re.compile(enhanced_pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[name] = compiled_pattern
                        logger.debug(f"Compiled enhanced pattern '{name}': {enhanced_pattern}")
                    else:
                        # Use original pattern
                        compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[name] = compiled_pattern
                        logger.debug(f"Compiled original pattern '{name}': {pattern}")
                except re.error as e:
                    logger.error(f"Invalid pattern '{name}': {e}")
                    # Try to compile the original pattern as fallback
                    try:
                        compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[name] = compiled_pattern
                        logger.error(f"Original pattern also invalid for '{name}': {e}")
                    except re.error as e2:
                        logger.error(f"Original pattern also invalid for '{name}': {e2}")
        
        logger.info(f"DEBUG: Compiled {len(self.compiled_patterns)} patterns: {list(self.compiled_patterns.keys())}")
    
    def find_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find all pattern matches in text, deduplicating overlapping matches.
        Only returns matches that have mappings.
        
        Args:
            text: Text to search
            
        Returns:
            List of tuples (pattern_name, matched_text, start_pos, end_pos)
        """
        all_matches = []
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            for match in compiled_pattern.finditer(text):
                matched_text = match.group()
                
                # Check if we have a mapping for this text
                replacement = self.get_replacement(matched_text)
                if replacement:
                    all_matches.append((pattern_name, matched_text, match.start(), match.end()))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x[2])
        
        # Deduplicate overlapping matches - keep only the first match for each position
        deduplicated_matches = []
        used_positions = set()
        
        for pattern_name, matched_text, start_pos, end_pos in all_matches:
            # Check if this position range overlaps with any already used position
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                deduplicated_matches.append((pattern_name, matched_text, start_pos, end_pos))
                used_positions.update(position_range)
        
        return deduplicated_matches
    
    def find_all_pattern_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find ALL pattern matches in text, regardless of whether they have mappings.
        This is used for comprehensive detection reporting.
        
        Args:
            text: Text to search
            
        Returns:
            List of tuples (pattern_name, matched_text, start_pos, end_pos)
        """
        all_matches = []
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            for match in compiled_pattern.finditer(text):
                matched_text = match.group()
                all_matches.append((pattern_name, matched_text, match.start(), match.end()))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x[2])
        
        # Deduplicate overlapping matches - keep only the first match for each position
        deduplicated_matches = []
        used_positions = set()
        
        for pattern_name, matched_text, start_pos, end_pos in all_matches:
            # Check if this position range overlaps with any already used position
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                deduplicated_matches.append((pattern_name, matched_text, start_pos, end_pos))
                used_positions.update(position_range)
        
        return deduplicated_matches
    
    def find_matches_universal(self, text: str, context_chars: int = 5, include_unmapped: bool = False) -> List[UniversalMatch]:
        """
        Find matches with universal character support and context information.
        
        Args:
            text: Text to search
            context_chars: Number of context characters to include
            include_unmapped: When True, include detections even if there is no mapping. When False, only include mapped.
            
        Returns:
            List of UniversalMatch objects
        """
        all_matches = []
        
        logger.debug(f"PatternMatcher processing text: '{text}' (length: {len(text)})")
        logger.debug(f"Searching text: '{text}' with {len(self.compiled_patterns)} patterns")
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            logger.debug(f"Testing pattern '{pattern_name}' against text '{text}'")
            try:
                for match in compiled_pattern.finditer(text):
                    matched_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    logger.debug(f"Raw match found: '{matched_text}' at {start_pos}-{end_pos}")
                    
                    # Extract only the pattern part from the matched text
                    extracted_pattern = self._extract_pattern_part(matched_text, pattern_name)
                    logger.debug(f"Extracted pattern: '{extracted_pattern}' from '{matched_text}'")
                    if extracted_pattern:
                        matched_text = extracted_pattern
                        # Adjust positions to reflect the extracted pattern
                        pattern_start = matched_text.find(extracted_pattern)
                        if pattern_start != -1:
                            start_pos = match.start() + pattern_start
                            end_pos = start_pos + len(extracted_pattern)
                    else:
                        logger.debug(f"No pattern extracted, using original: '{matched_text}'")
                    
                    # Determine if we have a mapping for this text
                    replacement = self.get_replacement(matched_text)
                    logger.debug(f"Replacement for '{matched_text}': '{replacement}'")

                    # Decide inclusion
                    if include_unmapped or replacement:
                        # Get context
                        preceding_context = self.get_match_context(text, start_pos, context_chars, before=True)
                        following_context = self.get_match_context(text, end_pos, context_chars, before=False)
                        
                        universal_match = UniversalMatch(
                            pattern_name=pattern_name,
                            matched_text=matched_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            preceding_context=preceding_context,
                            following_context=following_context,
                            confidence=1.0
                        )
                        
                        all_matches.append(universal_match)
                        logger.debug(f"Found match: '{matched_text}' at {start_pos}-{end_pos} "
                                   f"with context: '{preceding_context}|{matched_text}|{following_context}'")
            
            except Exception as e:
                logger.error(f"Error matching pattern '{pattern_name}': {e}")
        
        # Sort matches by position and deduplicate
        all_matches.sort(key=lambda x: x.start_pos)
        deduplicated_matches = self._deduplicate_universal_matches(all_matches)
        
        logger.debug(f"Found {len(deduplicated_matches)} unique matches after deduplication")
        return deduplicated_matches
    
    def _deduplicate_universal_matches(self, matches: List[UniversalMatch]) -> List[UniversalMatch]:
        """
        Remove overlapping matches, keeping the first match for each position.
        
        Args:
            matches: List of matches to deduplicate
            
        Returns:
            Deduplicated list of matches
        """
        if not matches:
            return matches
        
        deduplicated = []
        used_positions = set()
        
        for match in matches:
            position_range = set(range(match.start_pos, match.end_pos))
            if not position_range.intersection(used_positions):
                deduplicated.append(match)
                used_positions.update(position_range)
                logger.debug(f"Kept match: '{match.matched_text}' at {match.start_pos}-{match.end_pos}")
            else:
                logger.debug(f"Skipped overlapping match: '{match.matched_text}' at {match.start_pos}-{match.end_pos}")
        
        return deduplicated
    
    def get_match_context(self, text: str, position: int, context_chars: int = 5, before: bool = True) -> str:
        """
        Get surrounding context for a match position.
        
        Args:
            text: Full text
            position: Position in text
            context_chars: Number of context characters
            before: Whether to get context before (True) or after (False) position
            
        Returns:
            Context string
        """
        if before:
            start = max(0, position - context_chars)
            return text[start:position]
        else:
            end = min(len(text), position + context_chars)
            return text[position:end]
    
    def get_replacement(self, original_text: str) -> Optional[str]:
        """
        Get replacement text for original text with enhanced matching.
        
        Args:
            original_text: Original text to replace
            
        Returns:
            Replacement text or None if no mapping exists
        """
        logger.debug(f"DEBUG: Looking for replacement for: '{original_text}'")
        logger.debug(f"DEBUG: Available mappings: {list(self.mappings.keys())[:10]}...")  # Show first 10 mappings
        
        # Try exact match first
        if original_text in self.mappings:
            logger.debug(f"Exact mapping found: '{original_text}' -> '{self.mappings[original_text]}'")
            return self.mappings[original_text]
        
        # Try case-insensitive match
        for key, value in self.mappings.items():
            if original_text.lower() == key.lower():
                logger.debug(f"Case-insensitive mapping found: '{original_text}' -> '{value}'")
                return value
        
        # Try normalized match (remove spaces and hyphens)
        normalized_original = re.sub(r'[\s-]', '', original_text)
        logger.debug(f"DEBUG: Normalized original: '{normalized_original}'")
        for key, value in self.mappings.items():
            normalized_key = re.sub(r'[\s-]', '', key)
            if normalized_original.lower() == normalized_key.lower():
                logger.debug(f"Normalized mapping found: '{original_text}' -> '{value}'")
                return value
        
        logger.debug(f"No mapping found for: '{original_text}'")
        return None
    
    def _extract_pattern_part(self, matched_text: str, pattern_name: str) -> Optional[str]:
        """
        Extract only the pattern part from the matched text.
        
        Args:
            matched_text: The full text that was matched by the regex
            pattern_name: Name of the pattern that was matched
            
        Returns:
            The extracted pattern part or None if extraction failed
        """
        # For 77- patterns, extract the part that starts with "77-"
        if pattern_name.startswith('pattern_77'):
            # Find the "77-" part in the matched text
            import re
            pattern_match = re.search(r'77-[0-9]{3}-[A-Za-z0-9]+(?:-[0-9]{2,3})?', matched_text)
            if pattern_match:
                return pattern_match.group()
        
        # For other patterns, return the original matched text
        return matched_text
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about the pattern matcher."""
        return {
            'enhanced_mode': self.enhanced_mode,
            'original_patterns_count': len(self.patterns),
            'enhanced_patterns_count': len(self.enhanced_patterns) if self.enhanced_mode else 0,
            'compiled_patterns_count': len(self.compiled_patterns),
            'mappings_count': len(self.mappings),
            'original_patterns': {k: v for k, v in self.patterns.items() if not k.startswith('_')},
            'enhanced_patterns': self.enhanced_patterns if self.enhanced_mode else {},
            'sample_mappings': dict(list(self.mappings.items())[:5])  # First 5 mappings
        }

def create_pattern_matcher(patterns: Dict[str, str], mappings: Dict[str, str], enhanced_mode: bool = True) -> PatternMatcher:
    """
    Factory function to create a PatternMatcher instance.
    
    Args:
        patterns: Dictionary of pattern names to regex patterns
        mappings: Dictionary of original text to replacement text
        enhanced_mode: Whether to use enhanced universal pattern matching
        
    Returns:
        PatternMatcher instance
    """
    return PatternMatcher(patterns, mappings, enhanced_mode)