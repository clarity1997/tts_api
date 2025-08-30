# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Server
- **Primary startup**: `python start.py` (handles model download, requirements check, and server start)
- **Development mode**: `python start.py --reload --log-level DEBUG`
- **Check requirements only**: `python start.py --check-only`
- **Force model re-download**: `python start.py --force-download`
- **Custom host/port**: `python start.py --host 0.0.0.0 --port 8080`
- **Direct FastAPI run**: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

### Docker Commands
- **Build**: `docker build -t vibevoice-api .`
- **Run with GPU**: `docker run --gpus all -p 8000:8000 -v ./models:/app/models vibevoice-api`

### Testing and Validation
No automated test suite is present. Use these endpoints for manual testing:
- Health check: `GET /health`
- System status: `GET /api/v1/status`
- API documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a FastAPI-based text-to-speech service using the VibeVoice-1.5B model for multi-speaker conversational audio generation.

### Core Architecture
- **FastAPI application** (`main.py`) - Main application entry point with lifespan management
- **Enhanced startup script** (`start.py`) - Handles model download, validation, and server startup
- **Configuration management** (`config.py`) - Centralized settings using Pydantic
- **Service layer** (`services/vibevoice_service.py`) - Core VibeVoice model wrapper and audio generation
- **API routes** (`routers/`) - Modularized endpoints for audio, models, and system monitoring
- **Data models** (`schemas/`) - Pydantic request/response models
- **Utilities** (`utils/`) - Audio and file processing helpers

### Model Management
- **Auto-download capability**: First startup automatically downloads models from HuggingFace
- **Default model**: VibeVoice-7B (~15GB) optimized for A800 deployment
- **Alternative models**: VibeVoice-1.5B (~5GB) for smaller GPUs
- **Model path**: `./models/vibevoice-7b/` (excluded from git)  
- **Voice presets**: `./voices/` directory contains 9 built-in voice samples (English & Chinese)
- **GPU-optimized**: Designed for high-end CUDA GPUs (A800/A100), supports RTX series

### Configuration
All settings use `VIBEVOICE_` environment variable prefix:
- `VIBEVOICE_HOST`, `VIBEVOICE_PORT` - Server binding
- `VIBEVOICE_AUTO_DOWNLOAD_MODEL` - Enable/disable model auto-download
- `VIBEVOICE_HUGGINGFACE_REPO_ID` - Model repository (default: WestZhang/VibeVoice-Large-pt)
- `VIBEVOICE_DEVICE` - torch device (cuda/cpu)
- `VIBEVOICE_MAX_SPEAKERS` - Maximum speakers per request (default: 4)
- `VIBEVOICE_MAX_CONCURRENT_REQUESTS` - Concurrent request limit (default: 8 for A800)
- `VIBEVOICE_GPU_MEMORY_FRACTION` - GPU memory usage fraction (default: 0.9)

### Key API Endpoints
- **Audio generation**: `POST /api/v1/audio/generate`
- **Streaming audio**: `POST /api/v1/audio/generate-streaming`
- **Voice presets**: `GET /api/v1/models/voices`
- **System monitoring**: `GET /api/v1/status`, `GET /api/v1/metrics`

### Development Notes
- **No test framework** - Manual testing via `/docs` interface and health endpoints
- **A800 optimized** - Default configuration supports 8 concurrent requests with 80GB VRAM
- **Model loading time** - Initial request takes 15-30 seconds for large model loading
- **Voice preset discovery** - Automatically scans `./voices/` directory for `.wav` files

## A800 Deployment Guide

### Hardware Requirements (A800)
- **GPU**: NVIDIA A800 (80GB VRAM) 
- **RAM**: 32GB+ system memory recommended
- **Storage**: 20GB+ available space for model files
- **CUDA**: Version 12.1+ with cuDNN

### A800 Deployment Steps
1. **Load A800 configuration**: `source .env.a800` or use environment variables
2. **Start with optimized settings**: `python start.py`
3. **First startup**: Downloads VibeVoice-7B model (~15GB, may take 30-60 minutes)
4. **Docker deployment**: `docker run --gpus all --env-file .env.a800 -p 8000:8000 vibevoice-api`

### Performance Expectations (A800)
- **Concurrent requests**: 8-12 simultaneous generations
- **Generation speed**: 0.05-0.15x real-time (much faster than smaller GPUs)
- **Model loading**: ~20-30 seconds for initial startup
- **Memory usage**: ~70GB VRAM for full capacity

### Configuration Files for A800
- **Main config**: `config.py` (pre-configured for A800)
- **Environment**: `.env.a800` (A800-specific settings)
- **Docker**: `Dockerfile` (optimized CUDA devel image)