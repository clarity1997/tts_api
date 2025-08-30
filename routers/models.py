"""
Model information routes for VibeVoice API
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from schemas.models import VoicesResponse, ModelsResponse, ModelInfo, HardwareInfo, Language
from services.vibevoice_service import get_vibevoice_service, VibeVoiceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Information"])


@router.get(
    "/voices",
    response_model=VoicesResponse,
    summary="Get available voices",
    description="List all available voice presets with their information"
)
async def get_voices(
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Get list of available voice presets"""
    try:
        voices = service.get_voice_info_list()
        
        return VoicesResponse(
            voices=voices,
            total=len(voices)
        )
        
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voices")


@router.get(
    "/voices/preview/{voice_name}",
    summary="Get voice preview sample",
    description="Download a sample audio file for the specified voice"
)
async def get_voice_preview(
    voice_name: str,
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Get voice preview sample file"""
    try:
        sample_path = service.get_voice_sample_path(voice_name)
        
        if not sample_path:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_name}' not found"
            )
        
        # Determine media type based on file extension
        media_type = "audio/wav"  # Default
        if sample_path.lower().endswith('.mp3'):
            media_type = "audio/mpeg"
        elif sample_path.lower().endswith('.ogg'):
            media_type = "audio/ogg"
        elif sample_path.lower().endswith('.flac'):
            media_type = "audio/flac"
        
        return FileResponse(
            path=sample_path,
            media_type=media_type,
            filename=f"{voice_name}_sample.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice preview {voice_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice preview")


@router.get(
    "/info",
    response_model=ModelsResponse,
    summary="Get model information",
    description="Get detailed information about the current model and hardware"
)
async def get_model_info(
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Get current model and hardware information"""
    try:
        settings = service.settings
        stats = service.get_system_stats()
        
        # Model information
        model_info = ModelInfo(
            name=settings.model_name,
            max_speakers=settings.max_speakers,
            max_duration_minutes=settings.max_duration_minutes,
            sample_rate=settings.sample_rate,
            languages=[Language.EN, Language.ZH]
        )
        
        # Hardware information
        gpu_info = stats.get("gpu_info", {})
        hardware_info = HardwareInfo(
            device=stats["device"],
            gpu_memory_used=gpu_info.get("memory_used"),
            gpu_memory_total=gpu_info.get("memory_total"),
            gpu_name=gpu_info.get("device_name")
        )
        
        return ModelsResponse(
            current_model=settings.model_name,
            model_info=model_info,
            hardware=hardware_info
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.get(
    "/capabilities",
    response_model=Dict[str, Any],
    summary="Get model capabilities",
    description="Get detailed capabilities and features of the current model"
)
async def get_model_capabilities(
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Get model capabilities and features"""
    try:
        settings = service.settings
        voices = service.get_voice_info_list()
        
        # Count voices by language and gender
        language_counts = {}
        gender_counts = {}
        
        for voice in voices:
            lang = voice.language.value
            gender = voice.gender.value
            
            language_counts[lang] = language_counts.get(lang, 0) + 1
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        return {
            "model_name": settings.model_name,
            "version": "1.0.0",
            "capabilities": {
                "max_speakers": settings.max_speakers,
                "max_duration_minutes": settings.max_duration_minutes,
                "streaming_support": settings.enable_streaming,
                "flash_attention": settings.enable_flash_attention,
                "multi_language": True,
                "voice_cloning": False,
                "real_time_factor": "0.1-0.3"  # Approximate
            },
            "audio_specs": {
                "sample_rate": settings.sample_rate,
                "bit_depth": 16,
                "channels": 1,
                "supported_formats": settings.supported_formats
            },
            "voices": {
                "total": len(voices),
                "by_language": language_counts,
                "by_gender": gender_counts
            },
            "generation_parameters": {
                "cfg_scale": {
                    "default": settings.default_cfg_scale,
                    "range": [1.0, 2.0],
                    "description": "Classifier-free guidance scale"
                },
                "inference_steps": {
                    "default": settings.inference_steps,
                    "description": "Number of denoising steps"
                }
            },
            "limitations": {
                "max_text_length": settings.max_text_length,
                "concurrent_requests": settings.max_concurrent_requests,
                "memory_requirements": "3-6GB GPU memory"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model capabilities")


@router.get(
    "/voices/search",
    response_model=VoicesResponse,
    summary="Search voices",
    description="Search voices by language, gender, or name"
)
async def search_voices(
    language: str = None,
    gender: str = None,
    name_contains: str = None,
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Search voices with filters"""
    try:
        all_voices = service.get_voice_info_list()
        filtered_voices = []
        
        for voice in all_voices:
            # Filter by language
            if language and voice.language.value != language.lower():
                continue
                
            # Filter by gender
            if gender and voice.gender.value != gender.lower():
                continue
                
            # Filter by name
            if name_contains and name_contains.lower() not in voice.name.lower():
                continue
            
            filtered_voices.append(voice)
        
        return VoicesResponse(
            voices=filtered_voices,
            total=len(filtered_voices)
        )
        
    except Exception as e:
        logger.error(f"Error searching voices: {e}")
        raise HTTPException(status_code=500, detail="Voice search failed")