"""
Pydantic models for VibeVoice FastAPI
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"


class Language(str, Enum):
    """Supported languages"""
    EN = "en"
    ZH = "zh"


class Gender(str, Enum):
    """Speaker genders"""
    MALE = "male"
    FEMALE = "female"


# Request Models
class GenerateRequest(BaseModel):
    """Request model for basic audio generation"""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to convert to speech")
    speakers: List[str] = Field(..., min_items=1, max_items=4, description="List of speaker names")
    cfg_scale: float = Field(1.3, ge=1.0, le=2.0, description="CFG guidance scale")
    max_length: Optional[int] = Field(90, ge=1, le=90, description="Maximum audio length in minutes")
    format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
    
    @validator('speakers')
    def validate_speakers(cls, v):
        if not all(isinstance(speaker, str) and speaker.strip() for speaker in v):
            raise ValueError("All speakers must be non-empty strings")
        return [speaker.strip() for speaker in v]
    
    @validator('text')
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        return v


class GenerateStreamingRequest(GenerateRequest):
    """Request model for streaming audio generation"""
    chunk_size: Optional[int] = Field(24000, ge=1000, le=100000, description="Audio chunk size in samples")


class GenerateFromFileRequest(BaseModel):
    """Request model for file-based generation"""
    speakers: List[str] = Field(..., min_items=1, max_items=4, description="List of speaker names")
    cfg_scale: float = Field(1.3, ge=1.0, le=2.0, description="CFG guidance scale")
    max_length: Optional[int] = Field(90, ge=1, le=90, description="Maximum audio length in minutes")
    format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")


# Response Models
class GenerateResponse(BaseModel):
    """Response model for audio generation"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    duration: float = Field(..., ge=0, description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate")
    speakers_used: List[str] = Field(..., description="List of speakers used")
    generation_time: float = Field(..., ge=0, description="Generation time in seconds")
    format: AudioFormat = Field(..., description="Audio format")


class StreamingChunk(BaseModel):
    """Streaming audio chunk"""
    chunk: str = Field(..., description="Base64 encoded audio chunk")
    chunk_id: int = Field(..., ge=0, description="Chunk sequence ID")
    is_final: bool = Field(False, description="Whether this is the final chunk")


class StreamingComplete(BaseModel):
    """Streaming completion message"""
    status: str = Field("complete", description="Completion status")
    total_duration: float = Field(..., ge=0, description="Total audio duration in seconds")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks sent")


# Voice Models
class VoiceInfo(BaseModel):
    """Information about a voice preset"""
    name: str = Field(..., description="Voice internal name")
    display_name: str = Field(..., description="Voice display name")
    language: Language = Field(..., description="Voice language")
    gender: Gender = Field(..., description="Voice gender")
    sample_url: Optional[str] = Field(None, description="URL to voice sample")


class VoicesResponse(BaseModel):
    """Response model for available voices"""
    voices: List[VoiceInfo] = Field(..., description="List of available voices")
    total: int = Field(..., ge=0, description="Total number of voices")


# Model Information Models
class ModelInfo(BaseModel):
    """Information about the AI model"""
    name: str = Field(..., description="Model name")
    max_speakers: int = Field(..., ge=1, description="Maximum number of speakers")
    max_duration_minutes: int = Field(..., ge=1, description="Maximum audio duration in minutes")
    sample_rate: int = Field(..., description="Audio sample rate")
    languages: List[Language] = Field(..., description="Supported languages")


class HardwareInfo(BaseModel):
    """Hardware information"""
    device: str = Field(..., description="Compute device (cuda/cpu)")
    gpu_memory_used: Optional[str] = Field(None, description="GPU memory used")
    gpu_memory_total: Optional[str] = Field(None, description="Total GPU memory")
    gpu_name: Optional[str] = Field(None, description="GPU model name")


class ModelsResponse(BaseModel):
    """Response model for model information"""
    current_model: str = Field(..., description="Currently loaded model")
    model_info: ModelInfo = Field(..., description="Model information")
    hardware: HardwareInfo = Field(..., description="Hardware information")
    
    model_config = {"protected_namespaces": ()}


# System Status Models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    timestamp: str = Field(..., description="Timestamp of health check")
    
    model_config = {"protected_namespaces": ()}


class SystemStatusResponse(BaseModel):
    """Detailed system status response"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_status: str = Field(..., description="Model loading status")
    gpu_info: HardwareInfo = Field(..., description="GPU information")
    uptime_seconds: int = Field(..., ge=0, description="Service uptime in seconds")
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    active_requests: int = Field(..., ge=0, description="Currently active requests")
    
    model_config = {"protected_namespaces": ()}


# Error Models
class ErrorDetail(BaseModel):
    """Error detail information"""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")