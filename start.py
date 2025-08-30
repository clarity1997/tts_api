#!/usr/bin/env python3
"""
VibeVoice API Startup Script
Simple script to start the VibeVoice API server with proper configuration
"""
import argparse
import logging
import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vibevoice-api.log')
        ]
    )


def check_model_integrity(model_path: Path) -> bool:
    """Check if all required model files exist and are valid"""
    required_files = [
        "config.json",
        "model.safetensors.index.json", 
        "preprocessor_config.json"
    ]
    
    # Check if all required files exist
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            return False
    
    # Check if at least one safetensors file exists
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        return False
    
    # Check total model size (should be > 1GB for VibeVoice)
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    if total_size < 1024 * 1024 * 1024:  # Less than 1GB
        return False
        
    return True


def download_model(settings):
    """Download model from HuggingFace if not present"""
    logger = logging.getLogger(__name__)
    model_path = settings.absolute_model_path
    
    logger.info(f"🔄 Model not found at {model_path}")
    logger.info(f"📥 Starting automatic download of {settings.huggingface_repo_id}")
    logger.info("   This may take several minutes depending on your internet connection...")
    
    try:
        # Import HuggingFace hub here to avoid startup dependency
        from huggingface_hub import snapshot_download
        
        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        start_time = time.time()
        
        snapshot_download(
            repo_id=settings.huggingface_repo_id,
            local_dir=str(model_path),
            resume_download=True,
            ignore_patterns=["*.git*", "README.md", "*.md"]
        )
        
        download_time = time.time() - start_time
        logger.info(f"✅ Model download completed in {download_time:.1f} seconds")
        
        # Verify integrity
        if not check_model_integrity(model_path):
            raise RuntimeError("Downloaded model failed integrity check")
        
        # Log model info
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        logger.info(f"📊 Model size: {total_size / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        # Clean up partial download
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)
        return False


def ensure_model_available():
    """Ensure model is available, download if necessary"""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    model_path = settings.absolute_model_path
    
    # Check if model exists and is complete
    if check_model_integrity(model_path):
        logger.info(f"✅ Model found and verified: {model_path}")
        return True
    
    # Download model if auto_download is enabled
    if settings.auto_download_model:
        logger.info("🚀 Auto-download enabled, downloading model...")
        
        for attempt in range(settings.retry_attempts):
            if attempt > 0:
                logger.info(f"🔄 Retry attempt {attempt + 1}/{settings.retry_attempts}")
                time.sleep(5)  # Wait before retry
            
            if download_model(settings):
                return True
        
        logger.error(f"❌ Failed to download model after {settings.retry_attempts} attempts")
        return False
    else:
        logger.error("❌ Model not found and auto-download is disabled")
        logger.error(f"   Please download model manually to: {model_path}")
        logger.error(f"   Or enable auto-download with: VIBEVOICE_AUTO_DOWNLOAD_MODEL=true")
        return False


def check_other_requirements():
    """Check other requirements besides model"""
    errors = []
    settings = get_settings()
    
    # Check voices directory
    voices_path = settings.absolute_voices_path
    if not voices_path.exists():
        errors.append(f"Voices directory not found: {voices_path}")
    
    # Check for voice files
    if voices_path.exists():
        voice_files = list(voices_path.glob("*.wav"))
        if not voice_files:
            errors.append("No voice files found in voices directory")
    
    return errors


def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="VibeVoice API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--check-only", action="store_true", help="Check requirements and exit")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip model availability check")
    parser.add_argument("--force-download", action="store_true", help="Force model re-download")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 VibeVoice API Server Starting...")
    
    # Check model availability (most important)
    if not args.skip_model_check:
        logger.info("📋 Checking model availability...")
        
        # Force re-download if requested
        if args.force_download:
            settings = get_settings()
            model_path = settings.absolute_model_path
            if model_path.exists():
                logger.info("🗑️  Removing existing model for fresh download...")
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)
        
        if not ensure_model_available():
            logger.error("❌ Cannot start server: Model not available")
            if args.check_only:
                sys.exit(1)
            else:
                sys.exit(1)
    
    # Check other requirements
    logger.info("📋 Checking other requirements...")
    errors = check_other_requirements()
    
    if errors:
        logger.error("System requirements check failed:")
        for error in errors:
            logger.error(f"  - {error}")
        
        if args.check_only:
            sys.exit(1)
        else:
            logger.error("Cannot start server due to missing requirements")
            sys.exit(1)
    
    logger.info("✅ All requirements check passed")
    
    if args.check_only:
        logger.info("✅ Check completed successfully")
        return
    
    # Import and check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("CUDA not available, will use CPU (much slower)")
    except ImportError:
        logger.warning("PyTorch not available")
    
    # Set environment variables
    os.environ["VIBEVOICE_HOST"] = args.host
    os.environ["VIBEVOICE_PORT"] = str(args.port)
    os.environ["VIBEVOICE_LOG_LEVEL"] = args.log_level
    
    # Start server
    logger.info("Starting VibeVoice API server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Documentation: http://{args.host}:{args.port}/docs")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()