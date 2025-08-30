"""
VibeVoice FastAPI Application
High-quality multi-speaker conversational text-to-speech API
"""
import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for VibeVoice imports
sys.path.append(str(Path(__file__).parent))

from config import get_settings
from routers import audio, models, system
from services.vibevoice_service import get_vibevoice_service
from schemas.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting VibeVoice API service...")
    
    try:
        # Initialize the service
        service = await get_vibevoice_service()
        logger.info("VibeVoice service initialized successfully")
        
        # Log service information
        logger.info(f"Model: {settings.model_name}")
        logger.info(f"Device: {settings.device}")
        logger.info(f"Voices loaded: {len(service.voice_presets)}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize VibeVoice service: {e}")
        raise
    finally:
        logger.info("Shutting down VibeVoice API service...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "type": "not_found",
                "message": "The requested resource was not found",
                "code": "RESOURCE_NOT_FOUND"
            },
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": None
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An internal server error occurred",
                "code": "INTERNAL_ERROR"
            },
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "code": f"HTTP_{exc.status_code}"
            },
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": None
        }
    )


# Include routers
app.include_router(audio.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
app.include_router(system.router)  # System routes don't need prefix


# Static files for voice previews (if needed)
if settings.absolute_voices_path.exists():
    app.mount("/static/voices", StaticFiles(directory=str(settings.absolute_voices_path)), name="voices")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with service information"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "title": settings.api_title,
        "description": settings.api_description,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "audio_generation": f"{settings.api_prefix}/audio",
            "model_information": f"{settings.api_prefix}/models",
            "system_monitoring": "/health, /api/v1/status"
        }
    }


# API information endpoint
@app.get("/api", tags=["API Information"])
async def api_info():
    """Get API information and available endpoints"""
    return {
        "api_version": settings.api_version,
        "service_version": settings.service_version,
        "endpoints": {
            "audio": {
                "generate": f"{settings.api_prefix}/audio/generate",
                "generate_streaming": f"{settings.api_prefix}/audio/generate-streaming",
                "generate_from_file": f"{settings.api_prefix}/audio/generate-from-file",
                "formats": f"{settings.api_prefix}/audio/formats",
                "limits": f"{settings.api_prefix}/audio/limits"
            },
            "models": {
                "voices": f"{settings.api_prefix}/models/voices",
                "voice_preview": f"{settings.api_prefix}/models/voices/preview/{{voice_name}}",
                "info": f"{settings.api_prefix}/models/info",
                "capabilities": f"{settings.api_prefix}/models/capabilities",
                "search_voices": f"{settings.api_prefix}/models/voices/search"
            },
            "system": {
                "health": "/health",
                "status": "/api/v1/status",
                "metrics": "/api/v1/metrics",
                "config": "/api/v1/config",
                "version": "/api/v1/version"
            }
        },
        "authentication": "none",
        "rate_limiting": "none",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting VibeVoice API on {settings.host}:{settings.port}")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Debug mode: {settings.debug}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )