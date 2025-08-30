"""
Configuration settings for VibeVoice FastAPI
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service Information
    service_name: str = "vibevoice-api"
    service_version: str = "1.0.0"
    
    # API Configuration
    api_title: str = "VibeVoice API"
    api_description: str = "High-quality multi-speaker conversational text-to-speech API"
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # Model Configuration
    model_name: str = "VibeVoice-7B"  # Upgraded for A800 deployment
    model_path: str = "./models/vibevoice-7b"
    voices_path: str = "./voices"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Model Download Configuration
    auto_download_model: bool = True
    huggingface_repo_id: str = "WestZhang/VibeVoice-Large-pt"  # 7B parameter model for A800
    download_timeout: int = 7200  # 2 hours for larger model download
    retry_attempts: int = 3
    verify_model_integrity: bool = True
    
    # Generation Parameters (A800 optimized)
    default_cfg_scale: float = 1.5  # Higher quality for A800
    max_speakers: int = 4
    max_duration_minutes: int = 180  # Extended duration support
    sample_rate: int = 24000
    inference_steps: int = 20  # More steps for better quality
    
    # Audio Settings
    supported_formats: List[str] = ["wav", "mp3"]
    max_text_length: int = 50000
    chunk_size: int = 24000
    
    # Performance Settings (A800 optimized)
    enable_flash_attention: bool = True
    enable_streaming: bool = True
    max_concurrent_requests: int = 8  # A800 can handle more concurrent requests
    gpu_memory_fraction: float = 0.9  # Use 90% of A800's 80GB memory
    batch_size: int = 4  # Larger batch size for A800
    
    # Paths (computed)
    @property
    def absolute_model_path(self) -> Path:
        """Get absolute path to model directory"""
        base_path = Path(__file__).parent
        return (base_path / self.model_path).resolve()
    
    @property
    def absolute_voices_path(self) -> Path:
        """Get absolute path to voices directory"""
        base_path = Path(__file__).parent
        return (base_path / self.voices_path).resolve()
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Configuration
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["GET", "POST", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    
    class Config:
        """Pydantic config"""
        env_prefix = "VIBEVOICE_"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings