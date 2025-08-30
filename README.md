# VibeVoice FastAPI

🎙️ **High-quality multi-speaker conversational text-to-speech API**

A production-ready FastAPI service for VibeVoice, providing RESTful endpoints for generating expressive, long-form, multi-speaker conversational audio from text.

## ✨ Features

- **Multi-speaker Support**: Generate conversations with up to 4 different speakers
- **High Quality**: VibeVoice-1.5B model for natural, expressive speech synthesis
- **Streaming Audio**: Real-time audio generation with Server-Sent Events
- **Multiple Formats**: Support for WAV and MP3 output
- **Voice Presets**: 9 built-in voice presets (English & Chinese)
- **Self-contained**: Includes all model files, no external dependencies
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Auto Documentation**: Interactive API docs with Swagger UI

## 🏗️ Architecture

```
VibeVoice API/
├── models/                    # Auto-downloaded model files (excluded from git)
├── voices/                    # Voice preset audio samples
├── main.py                    # FastAPI application
├── start.py                   # Enhanced startup script with auto-download
├── config.py                  # Configuration management
├── services/                  # Core business logic
│   └── vibevoice_service.py   # VibeVoice model service
├── routers/                   # API route handlers
│   ├── audio.py              # Audio generation endpoints
│   ├── models.py             # Model information endpoints
│   └── system.py             # System monitoring endpoints
├── schemas/                   # Pydantic data models
└── utils/                    # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 6GB+ available disk space
- **Internet connection** (for first-time model download)

### 🎯 Automated Deployment (Recommended)

The API now supports **automatic model download** on first startup!

1. **Clone or extract the API code**:
```bash
git clone https://your-repo/vibevoice-api.git
cd vibevoice-api
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start the API server** (models download automatically):
```bash
python start.py
```

4. **First startup process**:
   - 🔄 API detects missing model files
   - 📥 Automatically downloads VibeVoice-1.5B (~5GB)
   - ✅ Verifies model integrity
   - 🚀 Starts the service

5. **Access the API**:
   - API Base URL: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### ⚡ Subsequent Startups

After the first startup, models are cached locally:
```bash
python start.py  # Starts immediately using cached models
```

### 🔧 Advanced Startup Options

```bash
# Check requirements without starting server
python start.py --check-only

# Force model re-download
python start.py --force-download

# Skip model check (use existing models)
python start.py --skip-model-check

# Custom host and port
python start.py --host 0.0.0.0 --port 8080

# Enable debug mode with auto-reload
python start.py --reload --log-level DEBUG
```

### 🐳 Docker Deployment

```bash
# Build image
docker build -t vibevoice-api .

# Run with GPU support (model downloads on first run)
docker run --gpus all -p 8000:8000 -v ./models:/app/models vibevoice-api
```

## 📡 API Endpoints

### 🎵 Audio Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/audio/generate` | POST | Generate audio from text |
| `/api/v1/audio/generate-streaming` | POST | Stream audio generation in real-time |
| `/api/v1/audio/generate-from-file` | POST | Generate audio from uploaded text file |
| `/api/v1/audio/formats` | GET | Get supported audio formats |
| `/api/v1/audio/limits` | GET | Get generation limits |

### 🎭 Model Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models/voices` | GET | List available voice presets |
| `/api/v1/models/voices/preview/{voice_name}` | GET | Download voice preview sample |
| `/api/v1/models/info` | GET | Get model and hardware information |
| `/api/v1/models/capabilities` | GET | Get detailed model capabilities |
| `/api/v1/models/voices/search` | GET | Search voices by filters |

### 🔧 System Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/api/v1/status` | GET | Detailed system status |
| `/api/v1/metrics` | GET | Performance metrics |
| `/api/v1/config` | GET | Service configuration |
| `/api/v1/version` | GET | Version information |

## 💡 Usage Examples

### Basic Audio Generation

```bash
curl -X POST "http://localhost:8000/api/v1/audio/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Speaker 1: Hello! Welcome to our podcast.\nSpeaker 2: Thanks for having me. This is exciting!",
       "speakers": ["en-Alice_woman", "en-Carter_man"],
       "cfg_scale": 1.3
     }'
```

### Streaming Generation

```bash
curl -X POST "http://localhost:8000/api/v1/audio/generate-streaming" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{
       "text": "Speaker 1: This is a streaming test.",
       "speakers": ["en-Maya_woman"]
     }'
```

### Get Available Voices

```bash
curl "http://localhost:8000/api/v1/models/voices"
```

## 🎭 Available Voices

The API includes 9 built-in voice presets:

**English Voices:**
- `en-Alice_woman` - Female voice
- `en-Carter_man` - Male voice  
- `en-Frank_man` - Male voice
- `en-Mary_woman_bgm` - Female voice with background music
- `en-Maya_woman` - Female voice
- `in-Samuel_man` - Male voice (Indian English)

**Chinese Voices:**
- `zh-Anchen_man_bgm` - Male voice with background music
- `zh-Bowen_man` - Male voice
- `zh-Xinran_woman` - Female voice

## ⚙️ Configuration

Configure the service via environment variables:

```bash
# Model settings
VIBEVOICE_MODEL_NAME="VibeVoice-1.5B"
VIBEVOICE_DEVICE="cuda"
VIBEVOICE_TORCH_DTYPE="bfloat16"

# Model download settings (NEW!)
VIBEVOICE_AUTO_DOWNLOAD_MODEL="true"           # Enable auto-download
VIBEVOICE_HUGGINGFACE_REPO_ID="microsoft/VibeVoice-1.5B"
VIBEVOICE_DOWNLOAD_TIMEOUT="3600"              # 1 hour timeout
VIBEVOICE_RETRY_ATTEMPTS="3"                   # Download retry attempts

# Server settings
VIBEVOICE_HOST="0.0.0.0"
VIBEVOICE_PORT="8000"
VIBEVOICE_DEBUG="false"

# Generation settings
VIBEVOICE_DEFAULT_CFG_SCALE="1.3"
VIBEVOICE_MAX_SPEAKERS="4"
VIBEVOICE_MAX_DURATION_MINUTES="90"
```

### 🎯 Auto-Download Configuration

The API supports flexible model management:

```bash
# Disable auto-download (manual model management)
VIBEVOICE_AUTO_DOWNLOAD_MODEL=false

# Use different model version
VIBEVOICE_HUGGINGFACE_REPO_ID="WestZhang/VibeVoice-Large-pt"  # 7B model

# Custom download settings
VIBEVOICE_DOWNLOAD_TIMEOUT=7200    # 2 hours for slower connections
VIBEVOICE_RETRY_ATTEMPTS=5         # More retry attempts
```

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu20.04

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t vibevoice-api .
docker run --gpus all -p 8000:8000 vibevoice-api
```

## 📊 Performance

### Hardware Requirements

| Configuration | Concurrent Requests | Memory Usage |
|---------------|-------------------|--------------|
| RTX 4060 (8GB) | 1-2 requests | ~4-6GB VRAM |
| RTX 3090 (24GB) | 3-6 requests | ~8-15GB VRAM |
| A100 (80GB) | 10+ requests | ~20-40GB VRAM |

### Benchmarks

- **Generation Speed**: 0.1-0.3x real-time
- **Model Loading**: ~10-15 seconds
- **Memory Footprint**: 3-4GB base + 1-2GB per request

## 🔧 Development

### Project Structure

```bash
api/
├── main.py              # Application entry point
├── config.py            # Settings and configuration
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Model files (not in git)
│   └── vibevoice-1.5b/ # Model weights and config
├── voices/             # Voice preset files
├── services/           # Business logic
├── routers/           # API route handlers  
├── schemas/           # Data models
└── utils/             # Utility functions
```

### Running in Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Voices

1. Add voice audio files to the `voices/` directory
2. Use naming convention: `{language}-{name}_{gender}.wav`
3. Restart the service to reload voice presets

## 🚨 Error Handling

The API provides detailed error responses:

```json
{
  "error": {
    "type": "validation_error",
    "message": "Speaker 'unknown_voice' not found in available voices",
    "code": "INVALID_SPEAKER"
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123"
}
```

## 🔍 Monitoring

Monitor your deployment using the built-in endpoints:

- **Health**: `/health` - Load balancer health checks
- **Metrics**: `/api/v1/metrics` - Detailed performance metrics  
- **Status**: `/api/v1/status` - System status and statistics

## 📚 API Documentation

Visit `/docs` for interactive Swagger UI documentation with:

- Complete endpoint documentation
- Request/response schemas  
- Interactive testing interface
- Example requests and responses

## ⚠️ Limitations

- **GPU Memory**: Limited by available VRAM
- **Concurrent Requests**: Depends on hardware configuration
- **Audio Length**: Maximum 90 minutes per generation
- **Text Length**: Maximum 50,000 characters per request
- **Languages**: Currently supports English and Chinese only

## 🤝 Support

For issues and questions:

1. Check the interactive documentation at `/docs`
2. Monitor system health at `/api/v1/status`  
3. Review logs for detailed error information
4. Refer to the original VibeVoice repository

## 📄 License

This API wrapper follows the same license as the original VibeVoice project.

---

**Made with ❤️ for the VibeVoice community**