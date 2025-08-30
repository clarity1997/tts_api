"""
Audio processing utilities for VibeVoice API
"""
import base64
import io
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from scipy.io.wavfile import write as write_wav

logger = logging.getLogger(__name__)


def convert_to_16_bit_wav(data: np.ndarray) -> np.ndarray:
    """Convert audio data to 16-bit PCM format"""
    # Handle tensor input
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure numpy array
    data = np.array(data, dtype=np.float32)
    
    # Normalize to [-1, 1] range
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Convert to 16-bit integer
    data_16bit = (data * 32767).astype(np.int16)
    return data_16bit


def audio_to_base64(audio: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio array to base64 encoded string"""
    try:
        # Convert to 16-bit PCM
        audio_16bit = convert_to_16_bit_wav(audio)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        write_wav(buffer, sample_rate, audio_16bit)
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return audio_base64
        
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        raise ValueError(f"Audio conversion failed: {e}")


def base64_to_audio(audio_base64: str, sample_rate: int = 24000) -> Tuple[np.ndarray, int]:
    """Convert base64 encoded audio back to numpy array"""
    try:
        # Decode from base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Read WAV from bytes
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer)
        
        # Ensure float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Handle stereo to mono conversion
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, sr
        
    except Exception as e:
        logger.error(f"Error converting base64 to audio: {e}")
        raise ValueError(f"Base64 audio decoding failed: {e}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32)
        
    except ImportError:
        logger.warning("Librosa not available, using simple resampling")
        # Simple resampling (not ideal but works)
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level"""
    if len(audio) == 0:
        return audio
    
    current_peak = np.max(np.abs(audio))
    if current_peak > 0:
        scaling_factor = target_peak / current_peak
        return audio * scaling_factor
    
    return audio


def apply_fade_in_out(audio: np.ndarray, fade_samples: int = 1000) -> np.ndarray:
    """Apply fade-in and fade-out to audio"""
    if len(audio) <= 2 * fade_samples:
        fade_samples = len(audio) // 4
    
    if fade_samples <= 0:
        return audio
    
    audio_faded = audio.copy()
    
    # Fade in
    fade_curve = np.linspace(0, 1, fade_samples)
    audio_faded[:fade_samples] *= fade_curve
    
    # Fade out
    fade_curve = np.linspace(1, 0, fade_samples)
    audio_faded[-fade_samples:] *= fade_curve
    
    return audio_faded


def detect_silence(audio: np.ndarray, threshold: float = 0.01, min_silence_ms: int = 500, sample_rate: int = 24000) -> list:
    """Detect silence regions in audio"""
    min_silence_samples = int(min_silence_ms * sample_rate / 1000)
    
    # Find samples below threshold
    is_silent = np.abs(audio) < threshold
    
    # Find consecutive silent regions
    silent_regions = []
    start = None
    
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_silence_samples:
                silent_regions.append((start, i))
            start = None
    
    # Handle case where audio ends in silence
    if start is not None and len(audio) - start >= min_silence_samples:
        silent_regions.append((start, len(audio)))
    
    return silent_regions


def trim_silence(audio: np.ndarray, threshold: float = 0.01, sample_rate: int = 24000) -> np.ndarray:
    """Trim silence from beginning and end of audio"""
    if len(audio) == 0:
        return audio
    
    # Find first non-silent sample
    start = 0
    for i, sample in enumerate(audio):
        if abs(sample) > threshold:
            start = i
            break
    
    # Find last non-silent sample
    end = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold:
            end = i + 1
            break
    
    # Return trimmed audio
    return audio[start:end]


def get_audio_stats(audio: np.ndarray, sample_rate: int = 24000) -> dict:
    """Get basic statistics about audio"""
    if len(audio) == 0:
        return {
            "duration": 0.0,
            "samples": 0,
            "peak": 0.0,
            "rms": 0.0,
            "dynamic_range": 0.0
        }
    
    duration = len(audio) / sample_rate
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    dynamic_range = peak / (rms + 1e-8)  # Avoid division by zero
    
    return {
        "duration": duration,
        "samples": len(audio),
        "peak": float(peak),
        "rms": float(rms),
        "dynamic_range": float(dynamic_range),
        "sample_rate": sample_rate
    }