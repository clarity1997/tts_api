"""
File processing utilities for VibeVoice API
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import mimetypes

logger = logging.getLogger(__name__)


def is_text_file(file_path: str) -> bool:
    """Check if file is a text file based on content type"""
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type is not None and mime_type.startswith('text/')
    except:
        return False


def is_audio_file(file_path: str) -> bool:
    """Check if file is an audio file based on extension"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions


def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    path = Path(file_path)
    
    info = {
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix.lower(),
        "size_bytes": get_file_size(file_path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_text": is_text_file(file_path),
        "is_audio": is_audio_file(file_path)
    }
    
    # Add size in human readable format
    size_bytes = info["size_bytes"]
    if size_bytes < 1024:
        info["size_human"] = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        info["size_human"] = f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        info["size_human"] = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        info["size_human"] = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    return info


def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read text file with error handling"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try alternative encodings
        for alt_encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=alt_encoding) as f:
                    logger.warning(f"Used {alt_encoding} encoding for {file_path}")
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file {file_path} with any supported encoding")
    except IOError as e:
        raise ValueError(f"Could not read file {file_path}: {e}")


def write_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """Write text file with error handling"""
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
            
    except IOError as e:
        raise ValueError(f"Could not write file {file_path}: {e}")


def validate_text_content(content: str, max_length: int = 50000) -> Dict[str, Any]:
    """Validate text content for TTS processing"""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Basic stats
    char_count = len(content)
    line_count = len(content.splitlines())
    word_count = len(content.split())
    
    validation["stats"] = {
        "characters": char_count,
        "lines": line_count,
        "words": word_count
    }
    
    # Validation checks
    if not content.strip():
        validation["valid"] = False
        validation["errors"].append("Content is empty or contains only whitespace")
    
    if char_count > max_length:
        validation["valid"] = False
        validation["errors"].append(f"Content too long: {char_count} characters (max {max_length})")
    
    # Count potential speaker assignments
    speaker_lines = [line for line in content.splitlines() 
                    if line.strip() and ('speaker' in line.lower() and ':' in line)]
    
    if speaker_lines:
        validation["stats"]["speaker_lines"] = len(speaker_lines)
        validation["warnings"].append(f"Found {len(speaker_lines)} lines with speaker format")
    
    # Check for very long lines
    long_lines = [i for i, line in enumerate(content.splitlines(), 1) 
                 if len(line) > 1000]
    
    if long_lines:
        validation["warnings"].append(f"Found {len(long_lines)} very long lines (>1000 chars)")
        validation["stats"]["long_lines"] = long_lines[:5]  # Show first 5
    
    return validation


def scan_directory_for_files(directory: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """Scan directory for files with specific extensions"""
    if extensions is None:
        extensions = ['.txt', '.wav', '.mp3']
    
    extensions = [ext.lower() for ext in extensions]
    files = []
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return files
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                file_info = get_file_info(str(file_path))
                files.append(file_info)
        
        # Sort by name
        files.sort(key=lambda x: x['name'])
        
    except OSError as e:
        logger.error(f"Error scanning directory {directory}: {e}")
    
    return files


def clean_filename(filename: str) -> str:
    """Clean filename for safe use"""
    # Remove or replace problematic characters
    cleaned = filename.replace(' ', '_')
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c in '._-')
    
    # Ensure it doesn't start with a dot or dash
    cleaned = cleaned.lstrip('.-')
    
    # Limit length
    if len(cleaned) > 100:
        name, ext = os.path.splitext(cleaned)
        cleaned = name[:100-len(ext)] + ext
    
    return cleaned or 'unnamed'


def ensure_directory_exists(directory: str) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Could not create directory {directory}: {e}")
        return False


def get_temp_filename(prefix: str = "vibevoice", suffix: str = ".wav") -> str:
    """Generate temporary filename"""
    import tempfile
    import time
    
    timestamp = int(time.time())
    random_part = os.urandom(4).hex()
    
    return f"{prefix}_{timestamp}_{random_part}{suffix}"


def copy_file_safe(source: str, destination: str) -> bool:
    """Safely copy file with error handling"""
    try:
        import shutil
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination)
        if dest_dir:
            ensure_directory_exists(dest_dir)
        
        shutil.copy2(source, destination)
        return True
        
    except (IOError, OSError) as e:
        logger.error(f"Error copying {source} to {destination}: {e}")
        return False