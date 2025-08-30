"""
System monitoring routes for VibeVoice API
"""
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from schemas.models import HealthResponse, SystemStatusResponse, HardwareInfo
from services.vibevoice_service import get_vibevoice_service, VibeVoiceService
from config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Basic health check endpoint for load balancers and monitoring"
)
async def health_check(
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Basic health check"""
    try:
        stats = service.get_system_stats()
        
        return HealthResponse(
            status="healthy" if stats["model_loaded"] else "unhealthy",
            model_loaded=stats["model_loaded"],
            gpu_available=stats["device"] == "cuda",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@router.get(
    "/api/v1/status",
    response_model=SystemStatusResponse,
    summary="Detailed system status",
    description="Comprehensive system status including performance metrics"
)
async def get_system_status(
    service: VibeVoiceService = Depends(get_vibevoice_service),
    settings: Settings = Depends(get_settings)
):
    """Get detailed system status"""
    try:
        stats = service.get_system_stats()
        gpu_info = stats.get("gpu_info", {})
        
        hardware_info = HardwareInfo(
            device=stats["device"],
            gpu_memory_used=gpu_info.get("memory_used"),
            gpu_memory_total=gpu_info.get("memory_total"),
            gpu_name=gpu_info.get("device_name")
        )
        
        return SystemStatusResponse(
            service=settings.service_name,
            version=settings.service_version,
            model_status="loaded" if stats["model_loaded"] else "not_loaded",
            gpu_info=hardware_info,
            uptime_seconds=stats["uptime_seconds"],
            total_requests=stats["total_requests"],
            active_requests=stats["active_requests"]
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get(
    "/api/v1/metrics",
    response_model=Dict[str, Any],
    summary="Performance metrics",
    description="Get detailed performance and usage metrics"
)
async def get_metrics(
    service: VibeVoiceService = Depends(get_vibevoice_service),
    settings: Settings = Depends(get_settings)
):
    """Get performance metrics"""
    try:
        stats = service.get_system_stats()
        
        # Calculate request rate (requests per hour)
        requests_per_hour = 0
        if stats["uptime_seconds"] > 0:
            requests_per_hour = (stats["total_requests"] * 3600) / stats["uptime_seconds"]
        
        # Memory usage metrics
        gpu_info = stats.get("gpu_info", {})
        gpu_memory_percent = 0
        if gpu_info.get("memory_used") and gpu_info.get("memory_total"):
            try:
                used = float(gpu_info["memory_used"].replace("GB", ""))
                total = float(gpu_info["memory_total"].replace("GB", ""))
                gpu_memory_percent = (used / total) * 100
            except (ValueError, AttributeError):
                pass
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": {
                "name": settings.service_name,
                "version": settings.service_version,
                "uptime_seconds": stats["uptime_seconds"],
                "status": "running"
            },
            "requests": {
                "total": stats["total_requests"],
                "active": stats["active_requests"],
                "rate_per_hour": round(requests_per_hour, 2)
            },
            "model": {
                "name": settings.model_name,
                "status": "loaded" if stats["model_loaded"] else "not_loaded",
                "device": stats["device"]
            },
            "hardware": {
                "gpu": {
                    "available": stats["device"] == "cuda",
                    "name": gpu_info.get("device_name", "N/A"),
                    "memory": {
                        "used": gpu_info.get("memory_used", "N/A"),
                        "total": gpu_info.get("memory_total", "N/A"),
                        "percent": round(gpu_memory_percent, 1)
                    }
                }
            },
            "configuration": {
                "max_speakers": settings.max_speakers,
                "max_duration_minutes": settings.max_duration_minutes,
                "sample_rate": settings.sample_rate,
                "inference_steps": settings.inference_steps,
                "streaming_enabled": settings.enable_streaming
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")


@router.get(
    "/api/v1/config",
    response_model=Dict[str, Any],
    summary="Service configuration",
    description="Get current service configuration (non-sensitive values only)"
)
async def get_configuration(
    settings: Settings = Depends(get_settings)
):
    """Get service configuration"""
    return {
        "service": {
            "name": settings.service_name,
            "version": settings.service_version,
            "debug": settings.debug
        },
        "api": {
            "title": settings.api_title,
            "version": settings.api_version,
            "prefix": settings.api_prefix
        },
        "model": {
            "name": settings.model_name,
            "device": settings.device,
            "torch_dtype": settings.torch_dtype
        },
        "audio": {
            "sample_rate": settings.sample_rate,
            "supported_formats": settings.supported_formats,
            "max_text_length": settings.max_text_length
        },
        "generation": {
            "default_cfg_scale": settings.default_cfg_scale,
            "max_speakers": settings.max_speakers,
            "max_duration_minutes": settings.max_duration_minutes,
            "inference_steps": settings.inference_steps
        },
        "features": {
            "flash_attention": settings.enable_flash_attention,
            "streaming": settings.enable_streaming,
            "max_concurrent_requests": settings.max_concurrent_requests
        }
    }


@router.post(
    "/api/v1/gc",
    response_model=Dict[str, Any],
    summary="Trigger garbage collection",
    description="Manually trigger garbage collection and GPU memory cleanup"
)
async def trigger_garbage_collection():
    """Manually trigger garbage collection"""
    try:
        import gc
        import torch
        
        # Python garbage collection
        collected = gc.collect()
        
        # GPU memory cleanup
        gpu_memory_before = 0
        gpu_memory_after = 0
        
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            torch.cuda.empty_cache()
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "python_objects_collected": collected,
            "gpu_memory": {
                "before_mb": round(gpu_memory_before, 2),
                "after_mb": round(gpu_memory_after, 2),
                "freed_mb": round(gpu_memory_before - gpu_memory_after, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Garbage collection failed: {e}")
        raise HTTPException(status_code=500, detail="Garbage collection failed")


@router.get(
    "/api/v1/version",
    response_model=Dict[str, str],
    summary="Get version information",
    description="Get version information for the service and dependencies"
)
async def get_version_info(
    settings: Settings = Depends(get_settings)
):
    """Get version information"""
    try:
        import torch
        import transformers
        import fastapi
        
        return {
            "service_version": settings.service_version,
            "api_version": settings.api_version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "fastapi_version": fastapi.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        
    except Exception as e:
        logger.error(f"Version info collection failed: {e}")
        return {
            "service_version": settings.service_version,
            "api_version": settings.api_version,
            "error": "Failed to collect dependency versions"
        }