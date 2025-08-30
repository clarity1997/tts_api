"""
Audio generation routes for VibeVoice API
"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from schemas.models import (
    GenerateRequest, GenerateResponse, GenerateStreamingRequest,
    GenerateFromFileRequest, StreamingChunk, StreamingComplete,
    ErrorResponse, AudioFormat
)
from services.vibevoice_service import get_vibevoice_service, VibeVoiceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio Generation"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate audio from text",
    description="Convert text to speech using specified speakers"
)
async def generate_audio(
    request: GenerateRequest,
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Generate audio from text with specified speakers"""
    try:
        logger.info(f"Received generation request: {len(request.text)} chars, {len(request.speakers)} speakers")
        
        # Generate audio
        audio_array, metadata = await service.generate_audio(
            text=request.text,
            speakers=request.speakers,
            cfg_scale=request.cfg_scale,
            max_length=request.max_length
        )
        
        # Convert audio to base64
        audio_base64 = service._audio_to_base64(audio_array)
        
        # Prepare response
        response = GenerateResponse(
            audio_data=audio_base64,
            duration=metadata["duration"],
            sample_rate=metadata["sample_rate"],
            speakers_used=metadata["speakers_used"],
            generation_time=metadata["generation_time"],
            format=request.format
        )
        
        logger.info(f"Generation completed: {metadata['duration']:.2f}s audio in {metadata['generation_time']:.2f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/generate-streaming",
    summary="Generate audio with streaming",
    description="Convert text to speech with real-time streaming output"
)
async def generate_audio_streaming(
    request: GenerateStreamingRequest,
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Generate audio with streaming response"""
    try:
        logger.info(f"Received streaming request: {len(request.text)} chars, {len(request.speakers)} speakers")
        
        async def generate_events():
            """Generate Server-Sent Events for streaming"""
            try:
                chunk_count = 0
                async for chunk_data in service.generate_audio_streaming(
                    text=request.text,
                    speakers=request.speakers,
                    cfg_scale=request.cfg_scale,
                    max_length=request.max_length,
                    chunk_size=request.chunk_size
                ):
                    if "error" in chunk_data:
                        yield {
                            "event": "error",
                            "data": {"error": chunk_data["error"]}
                        }
                        break
                    elif "status" in chunk_data and chunk_data["status"] == "complete":
                        yield {
                            "event": "complete",
                            "data": {
                                "status": "complete",
                                "total_chunks": chunk_data["total_chunks"],
                                "generation_time": chunk_data["generation_time"]
                            }
                        }
                        break
                    else:
                        yield {
                            "event": "chunk",
                            "data": {
                                "chunk": chunk_data["chunk"],
                                "chunk_id": chunk_data["chunk_id"],
                                "is_final": chunk_data["is_final"]
                            }
                        }
                        chunk_count += 1
                        
                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.01)
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield {
                    "event": "error",
                    "data": {"error": str(e)}
                }
        
        return EventSourceResponse(generate_events())
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/generate-from-file",
    response_model=GenerateResponse,
    summary="Generate audio from uploaded text file",
    description="Convert uploaded text file to speech"
)
async def generate_from_file(
    file: UploadFile = File(..., description="Text file to convert"),
    speakers: str = Form(..., description="Comma-separated list of speaker names"),
    cfg_scale: float = Form(1.3, description="CFG guidance scale"),
    max_length: int = Form(90, description="Maximum audio length in minutes"),
    format: AudioFormat = Form(AudioFormat.WAV, description="Output audio format"),
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Generate audio from uploaded text file"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('text/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a text file"
            )
        
        # Read file content
        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded"
            )
        
        # Parse speakers list
        speakers_list = [s.strip() for s in speakers.split(',') if s.strip()]
        if not speakers_list:
            raise HTTPException(
                status_code=400,
                detail="At least one speaker must be specified"
            )
        
        # Create request object for validation
        request = GenerateRequest(
            text=text,
            speakers=speakers_list,
            cfg_scale=cfg_scale,
            max_length=max_length,
            format=format
        )
        
        logger.info(f"File upload request: {file.filename}, {len(text)} chars, {len(speakers_list)} speakers")
        
        # Generate audio
        audio_array, metadata = await service.generate_audio(
            text=request.text,
            speakers=request.speakers,
            cfg_scale=request.cfg_scale,
            max_length=request.max_length
        )
        
        # Convert to base64
        audio_base64 = service._audio_to_base64(audio_array)
        
        response = GenerateResponse(
            audio_data=audio_base64,
            duration=metadata["duration"],
            sample_rate=metadata["sample_rate"],
            speakers_used=metadata["speakers_used"],
            generation_time=metadata["generation_time"],
            format=request.format
        )
        
        logger.info(f"File generation completed: {metadata['duration']:.2f}s audio")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")


@router.get(
    "/formats",
    response_model=Dict[str, Any],
    summary="Get supported audio formats",
    description="List all supported audio output formats"
)
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "formats": [
            {
                "format": "wav",
                "description": "Waveform Audio File Format",
                "mime_type": "audio/wav",
                "extension": ".wav"
            },
            {
                "format": "mp3",
                "description": "MPEG Audio Layer III",
                "mime_type": "audio/mpeg",
                "extension": ".mp3"
            }
        ]
    }


@router.get(
    "/limits",
    response_model=Dict[str, Any],
    summary="Get generation limits",
    description="Get current limits for text length, speakers, etc."
)
async def get_generation_limits(
    service: VibeVoiceService = Depends(get_vibevoice_service)
):
    """Get current generation limits"""
    settings = service.settings
    
    return {
        "max_text_length": settings.max_text_length,
        "max_speakers": settings.max_speakers,
        "max_duration_minutes": settings.max_duration_minutes,
        "sample_rate": settings.sample_rate,
        "supported_formats": settings.supported_formats,
        "default_cfg_scale": settings.default_cfg_scale,
        "cfg_scale_range": {
            "min": 1.0,
            "max": 2.0
        }
    }