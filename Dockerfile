# VibeVoice API Dockerfile - A800 Optimized
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables (A800 optimized)
ENV PYTHONUNBUFFERED=1
ENV VIBEVOICE_HOST=0.0.0.0
ENV VIBEVOICE_PORT=9883
ENV VIBEVOICE_MAX_CONCURRENT_REQUESTS=8
ENV VIBEVOICE_GPU_MEMORY_FRACTION=0.9

# Install additional system dependencies for VibeVoice
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and install VibeVoice from Microsoft repository with comprehensive error handling
# NOTE: We keep the source code as editable install requires it
RUN set -e && \
    echo "Cloning VibeVoice repository to /opt/VibeVoice..." && \
    git clone https://github.com/microsoft/VibeVoice.git /opt/VibeVoice && \
    cd /opt/VibeVoice && \
    echo "Installing VibeVoice with editable mode..." && \
    pip install -e . && \
    echo "Checking pip list for vibevoice..." && \
    pip list | grep -i vibevoice && \
    echo "Checking Python site-packages..." && \
    python -c "import site; print('Site packages:', site.getsitepackages())" && \
    echo "Verifying VibeVoice installation..." && \
    python -c "import sys; print('Python path:', sys.path)" && \
    python -c "import vibevoice; print('✓ vibevoice base module imported from:', vibevoice.__file__)" && \
    python -c "from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig; print('✓ VibeVoiceConfig imported')" && \
    python -c "from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference; print('✓ VibeVoiceForConditionalGenerationInference imported')" && \
    python -c "from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor; print('✓ VibeVoiceProcessor imported')" && \
    python -c "from vibevoice.modular.streamer import AudioStreamer; print('✓ AudioStreamer imported')" && \
    echo "All VibeVoice modules imported successfully!"

# Set PYTHONPATH to include VibeVoice
ENV PYTHONPATH="/opt/VibeVoice:${PYTHONPATH}"

# Install flash-attn for optimal A800 performance (if not already included)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention already available or failed to install"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 vibevoice && \
    chown -R vibevoice:vibevoice /app

# Switch to non-root user
USER vibevoice

# Expose port
EXPOSE 9883

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9883/health || exit 1

# Start the application with A800 optimizations
CMD ["python3", "start.py", "--host", "0.0.0.0", "--port", "9883", "--workers", "1"]