# VibeVoice API Dockerfile - A800 Optimized
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables (A800 optimized)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIBEVOICE_HOST=0.0.0.0
ENV VIBEVOICE_PORT=9883
ENV VIBEVOICE_MAX_CONCURRENT_REQUESTS=8
ENV VIBEVOICE_GPU_MEMORY_FRACTION=0.9

# Install system dependencies (including additional tools for VibeVoice)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone and install VibeVoice from Microsoft repository
RUN git clone https://github.com/microsoft/VibeVoice.git /tmp/VibeVoice \
    && cd /tmp/VibeVoice \
    && pip3 install . \
    && cd / \
    && rm -rf /tmp/VibeVoice

# Install flash-attn for optimal A800 performance
RUN pip3 install flash-attn --no-build-isolation

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