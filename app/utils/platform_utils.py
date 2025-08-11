"""
Cross-platform utilities for document processing pipeline.
Handles platform detection, tool discovery, and path management.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .shared_constants import (
    PLATFORM_COMMANDS, CONVERSION_TOOLS, SUPPORTED_PLATFORMS,
    get_current_platform, is_macos, is_windows, is_linux
)

logger = logging.getLogger(__name__)

class PlatformUtils:
    """Utilities for cross-platform operations."""
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """Get comprehensive platform information."""
        return {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'current_platform': get_current_platform()
        }
    
    @staticmethod
    def get_platform_paths() -> Dict[str, str]:
        """Get platform-appropriate paths."""
        current_platform = get_current_platform()
        
        if current_platform == "darwin":
            return {
                'home': str(Path.home()),
                'applications': '/Applications',
                'temp': '/tmp',
                'path_separator': ':',
                'line_ending': '\n'
            }
        elif current_platform == "windows":
            return {
                'home': str(Path.home()),
                'program_files': os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
                'program_files_x86': os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'),
                'temp': os.environ.get('TEMP', 'C:\\Temp'),
                'path_separator': ';',
                'line_ending': '\r\n'
            }
        else:  # Linux
            return {
                'home': str(Path.home()),
                'usr_bin': '/usr/bin',
                'usr_local_bin': '/usr/local/bin',
                'temp': '/tmp',
                'path_separator': ':',
                'line_ending': '\n'
            }
    
    @staticmethod
    def normalize_path(path: str) -> Path:
        """Normalize path for current platform."""
        return Path(path).resolve()
    
    @staticmethod
    def get_executable_extension() -> str:
        """Get executable file extension for current platform."""
        return '.exe' if is_windows() else ''

class ConversionToolDetector:
    """Detects and manages document conversion tools."""
    
    _instance = None
    _detection_done = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._detection_done:
            self.platform = get_current_platform()
            self.detected_tools = {}
            self._detect_tools()
            ConversionToolDetector._detection_done = True
    
    def _detect_tools(self):
        """Detect available conversion tools on the current platform."""
        logger.info(f"Detecting conversion tools on {self.platform}")
        
        # Log which tools we're checking for
        logger.info(f"Checking for conversion tools: {CONVERSION_TOOLS}")
        
        # Detect LibreOffice
        logger.info("Checking for LibreOffice...")
        libreoffice_path = self._detect_libreoffice()
        if libreoffice_path:
            self.detected_tools['libreoffice'] = libreoffice_path
            logger.info(f"✓ LibreOffice detected at: {libreoffice_path}")
        else:
            logger.warning("✗ LibreOffice not found")
        
        # Detect Word COM (Windows only)
        if is_windows():
            logger.info("Checking for Microsoft Word COM interface...")
            word_com_available = self._detect_word_com()
            if word_com_available:
                self.detected_tools['word_com'] = True
                logger.info("✓ Microsoft Word COM interface detected")
            else:
                logger.warning("✗ Microsoft Word COM interface not available")
        else:
            logger.info("Skipping Word COM detection (not Windows platform)")
        
        # Log final detection summary
        if self.detected_tools:
            logger.info(f"Detection complete. Available tools: {list(self.detected_tools.keys())}")
        else:
            logger.warning("Detection complete. No conversion tools found!")
    
    def _detect_libreoffice(self) -> Optional[str]:
        """Detect LibreOffice installation."""
        commands = PLATFORM_COMMANDS.get(self.platform, {})
        
        # Try primary LibreOffice command
        primary_cmd = commands.get('libreoffice')
        if primary_cmd:
            logger.debug(f"Testing primary LibreOffice command: {primary_cmd}")
            if self._test_command(primary_cmd):
                logger.debug(f"✓ Primary command works: {primary_cmd}")
                return primary_cmd
            else:
                logger.debug(f"✗ Primary command failed: {primary_cmd}")
        
        # Try alternative command
        alt_cmd = commands.get('libreoffice_alt')
        if alt_cmd:
            logger.debug(f"Testing alternative LibreOffice command: {alt_cmd}")
            if self._test_command(alt_cmd):
                logger.debug(f"✓ Alternative command works: {alt_cmd}")
                return alt_cmd
            else:
                logger.debug(f"✗ Alternative command failed: {alt_cmd}")
        
        # Try finding in PATH
        libreoffice_exe = 'soffice' + PlatformUtils.get_executable_extension()
        logger.debug(f"Searching for '{libreoffice_exe}' in PATH...")
        path_cmd = shutil.which(libreoffice_exe)
        if path_cmd:
            logger.debug(f"Found in PATH: {path_cmd}")
            if self._test_command(path_cmd):
                logger.debug(f"✓ PATH command works: {path_cmd}")
                return path_cmd
            else:
                logger.debug(f"✗ PATH command failed: {path_cmd}")
        else:
            logger.debug(f"'{libreoffice_exe}' not found in PATH")
        
        logger.debug("LibreOffice detection failed - no working installation found")
        return None
    
    def _detect_word_com(self) -> bool:
        """Detect Microsoft Word COM interface (Windows only)."""
        if not is_windows():
            return False
        
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Quit()
            return True
        except Exception as e:
            logger.debug(f"Word COM detection failed: {e}")
            return False
    
    def _test_command(self, command: str) -> bool:
        """Test if a command is available and working."""
        try:
            # Test with version flag
            result = subprocess.run(
                [command, '--version'],
                capture_output=True,
                timeout=10,
                check=False
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    def get_available_tools(self) -> List[str]:
        """Get list of available conversion tools."""
        return list(self.detected_tools.keys())
    
    def get_tool_path(self, tool: str) -> Optional[str]:
        """Get path for a specific tool."""
        return self.detected_tools.get(tool)
    
    def get_preferred_tool(self) -> Optional[str]:
        """Get the preferred conversion tool for the current platform."""
        available_tools = self.get_available_tools()
        
        if not available_tools:
            return None
        
        # Preference order: LibreOffice first, then Word COM
        if 'libreoffice' in available_tools:
            return 'libreoffice'
        elif 'word_com' in available_tools:
            return 'word_com'
        
        return available_tools[0] if available_tools else None
    
    def is_tool_available(self, tool: str) -> bool:
        """Check if a specific tool is available."""
        return tool in self.detected_tools

class PathManager:
    """Manages file paths across different platforms."""
    
    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if necessary."""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_temp_directory() -> Path:
        """Get platform-appropriate temporary directory."""
        platform_paths = PlatformUtils.get_platform_paths()
        temp_dir = Path(platform_paths['temp'])
        return PathManager.ensure_directory(temp_dir)
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Get a safe filename by removing problematic characters."""
        import re
        # Remove or replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(safe_name) > 200:
            name_part, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name_part[:190] + ('.' + ext if ext else '')
        return safe_name
    
    @staticmethod
    def get_unique_filename(directory: Path, base_name: str, extension: str) -> Path:
        """Get a unique filename in the specified directory."""
        counter = 1
        filename = f"{base_name}{extension}"
        file_path = directory / filename
        
        while file_path.exists():
            filename = f"{base_name}_{counter}{extension}"
            file_path = directory / filename
            counter += 1
        
        return file_path
    
    @staticmethod
    def copy_with_metadata(source: Path, destination: Path) -> bool:
        """Copy file with metadata preservation."""
        try:
            import shutil
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source} to {destination}: {e}")
            return False

def get_system_info() -> Dict[str, any]:
    """Get comprehensive system information for debugging."""
    detector = ConversionToolDetector()
    
    return {
        'platform_info': PlatformUtils.get_platform_info(),
        'platform_paths': PlatformUtils.get_platform_paths(),
        'available_tools': detector.get_available_tools(),
        'preferred_tool': detector.get_preferred_tool(),
        'tool_paths': {tool: detector.get_tool_path(tool) for tool in detector.get_available_tools()}
    }

def validate_platform_support() -> bool:
    """Validate that the current platform is supported."""
    current_platform = get_current_platform()
    supported = current_platform in SUPPORTED_PLATFORMS
    
    if not supported:
        logger.warning(f"Platform {current_platform} is not officially supported. Supported platforms: {SUPPORTED_PLATFORMS}")
    
    return supported