"""
Platform utilities for cross-platform compatibility.

This module provides platform-specific utilities and system information
for the document processing pipeline.
"""

import platform
import sys
from typing import Dict, Any
from pathlib import Path


def validate_platform_support() -> bool:
    """
    Validate that the current platform is supported.
    
    Returns:
        True if platform is supported, False otherwise
    """
    current_platform = platform.system().lower()
    supported_platforms = ['darwin', 'windows', 'linux']
    
    # Map platform.system() output to our supported platforms
    platform_mapping = {
        'darwin': 'darwin',
        'windows': 'windows', 
        'win32': 'windows',
        'linux': 'linux'
    }
    
    mapped_platform = platform_mapping.get(current_platform, current_platform)
    return mapped_platform in supported_platforms


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for the current platform.
    
    Returns:
        Dictionary containing system information
    """
    current_platform = platform.system().lower()
    
    # Determine available conversion tools
    available_tools = []
    preferred_tool = None
    
    if current_platform == 'darwin':
        # Check for LibreOffice on macOS
        import shutil
        if shutil.which('soffice') or Path('/Applications/LibreOffice.app/Contents/MacOS/soffice').exists():
            available_tools.append('libreoffice')
            preferred_tool = 'libreoffice'
    elif current_platform in ['windows', 'win32']:
        # Check for tools on Windows
        import shutil
        if shutil.which('soffice.exe'):
            available_tools.append('libreoffice')
        # Note: Word COM is harder to detect, would require trying to create COM object
        available_tools.append('word_com')  # Assume available on Windows
        preferred_tool = 'word_com' if 'word_com' in available_tools else 'libreoffice'
    elif current_platform == 'linux':
        # Check for LibreOffice on Linux
        import shutil
        if shutil.which('soffice') or shutil.which('libreoffice'):
            available_tools.append('libreoffice')
            preferred_tool = 'libreoffice'
    
    if not preferred_tool and available_tools:
        preferred_tool = available_tools[0]
    
    return {
        'platform_info': {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'architecture': platform.machine()
        },
        'available_tools': available_tools,
        'preferred_tool': preferred_tool,
        'platform_supported': validate_platform_support()
    }


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
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
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
            print(f"Failed to copy {source} to {destination}: {e}")
            return False