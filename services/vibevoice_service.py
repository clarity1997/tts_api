"""
Core VibeVoice service for audio generation
"""
import os
import sys
import time
import base64
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import set_seed

# Add parent directory to path to import vibevoice modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.streamer import AudioStreamer
except ImportError as e:
    print(f"Failed to import VibeVoice modules: {e}")
    print("Make sure you're running from the VibeVoice root directory")
    raise

from config import get_settings
from schemas.models import VoiceInfo, Language, Gender

logger = logging.getLogger(__name__)


class VibeVoiceService:
    """Core service for VibeVoice audio generation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.processor = None
        self.device = self.settings.device
        self.voice_presets = {}
        self.is_initialized = False
        self.stats = {
            "total_requests": 0,
            "active_requests": 0,
            "start_time": time.time()
        }
        self._init_lock = threading.Lock()
        
    def _verify_model_files(self) -> bool:
        """Verify that all required model files exist"""
        model_path = self.settings.absolute_model_path
        
        if not model_path.exists():
            logger.error(f"Model directory does not exist: {model_path}")
            return False
        
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Missing required model files: {missing_files}")
            return False
        
        # Check for safetensors files
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            logger.error("No .safetensors model files found")
            return False
        
        logger.info(f"Model verification passed: {len(safetensor_files)} safetensors files found")
        return True

    async def initialize(self) -> None:
        """Initialize the VibeVoice model and processor"""
        if self.is_initialized:
            return
            
        with self._init_lock:
            if self.is_initialized:
                return
                
            logger.info("Initializing VibeVoice service...")
            
            try:
                # Verify model files first
                if not self._verify_model_files():
                    if self.settings.auto_download_model:
                        logger.info("Model files missing, but auto-download is enabled")
                        raise RuntimeError("Model files not found. Please ensure model is downloaded first.")
                    else:
                        raise RuntimeError("Model files not found and auto-download is disabled")
                
                # Set random seed for reproducibility
                set_seed(42)
                
                # Load processor
                logger.info(f"Loading processor from {self.settings.absolute_model_path}")
                self.processor = VibeVoiceProcessor.from_pretrained(
                    str(self.settings.absolute_model_path)
                )
                
                # Load model
                logger.info(f"Loading model from {self.settings.absolute_model_path}")
                torch_dtype = getattr(torch, self.settings.torch_dtype)
                
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(self.settings.absolute_model_path),
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    attn_implementation="flash_attention_2" if self.settings.enable_flash_attention else None,
                )
                self.model.eval()
                
                # Configure scheduler
                self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                    self.model.model.noise_scheduler.config, 
                    algorithm_type='sde-dpmsolver++',
                    beta_schedule='squaredcos_cap_v2'
                )
                self.model.set_ddpm_inference_steps(num_steps=self.settings.inference_steps)
                
                # Load voice presets
                self._load_voice_presets()
                
                self.is_initialized = True
                logger.info("✅ VibeVoice service initialized successfully")
                
                # Log model information
                total_size = sum(f.stat().st_size for f in self.settings.absolute_model_path.rglob("*") if f.is_file())
                logger.info(f"📊 Model size: {total_size / 1024**3:.2f} GB")
                
                if hasattr(self.model.model, 'language_model'):
                    logger.info(f"🧠 Language model attention: {self.model.model.language_model.config._attn_implementation}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize VibeVoice service: {e}")
                raise RuntimeError(f"Service initialization failed: {e}")
    
    def _load_voice_presets(self) -> None:
        """Load voice presets from the voices directory"""
        voices_dir = self.settings.absolute_voices_path
        
        if not voices_dir.exists():
            logger.warning(f"Voices directory not found: {voices_dir}")
            self.voice_presets = {}
            return
        
        # Supported audio extensions
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        # Scan for audio files
        self.voice_presets = {}
        for file_path in voices_dir.iterdir():
            if file_path.suffix.lower() in audio_extensions and file_path.is_file():
                name = file_path.stem
                self.voice_presets[name] = str(file_path)
        
        # Sort alphabetically
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        logger.info(f"Loaded {len(self.voice_presets)} voice presets: {list(self.voice_presets.keys())}")
    
    def get_voice_info_list(self) -> List[VoiceInfo]:
        """Get list of available voice information"""
        voices = []
        
        for name, path in self.voice_presets.items():
            # Parse voice information from filename
            display_name = name.replace('_', ' ').replace('-', ' ')
            
            # Determine language
            if name.startswith('en-'):
                language = Language.EN
            elif name.startswith('zh-'):
                language = Language.ZH
            else:
                language = Language.EN  # Default
            
            # Determine gender
            if any(keyword in name.lower() for keyword in ['woman', 'female']):
                gender = Gender.FEMALE
            elif any(keyword in name.lower() for keyword in ['man', 'male']):
                gender = Gender.MALE
            else:
                gender = Gender.MALE  # Default
            
            voices.append(VoiceInfo(
                name=name,
                display_name=display_name.title(),
                language=language,
                gender=gender,
                sample_url=f"/voices/preview/{name}"
            ))
        
        return voices
    
    def get_voice_sample_path(self, voice_name: str) -> Optional[str]:
        """Get path to voice sample file"""
        return self.voice_presets.get(voice_name)
    
    def _read_audio(self, audio_path: str, target_sr: int = None) -> np.ndarray:
        """Read and preprocess audio file"""
        if target_sr is None:
            target_sr = self.settings.sample_rate
            
        try:
            wav, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            
            # Resample if needed
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                
            return wav.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error reading audio {audio_path}: {e}")
            return np.array([], dtype=np.float32)
    
    def _format_script(self, text: str, num_speakers: int) -> str:
        """Format text script with speaker assignments"""
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line already has speaker format
            if line.startswith('Speaker ') and ':' in line:
                formatted_lines.append(line)
            else:
                # Auto-assign speakers in rotation
                speaker_id = len(formatted_lines) % num_speakers
                formatted_lines.append(f"Speaker {speaker_id}: {line}")
        
        return '\n'.join(formatted_lines)
    
    async def generate_audio(
        self,
        text: str,
        speakers: List[str],
        cfg_scale: float = 1.3,
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate audio from text and speakers"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        self.stats["active_requests"] += 1
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Validate inputs
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            if not speakers or len(speakers) > self.settings.max_speakers:
                raise ValueError(f"Number of speakers must be between 1 and {self.settings.max_speakers}")
            
            # Validate speaker names
            for speaker in speakers:
                if speaker not in self.voice_presets:
                    raise ValueError(f"Speaker '{speaker}' not found in available voices")
            
            # Format script with speaker assignments
            formatted_script = self._format_script(text, len(speakers))
            
            # Load voice samples
            voice_samples = []
            for speaker_name in speakers:
                audio_path = self.voice_presets[speaker_name]
                audio_data = self._read_audio(audio_path)
                
                if len(audio_data) == 0:
                    raise ValueError(f"Failed to load audio for speaker '{speaker_name}'")
                
                voice_samples.append(audio_data)
            
            # Prepare inputs
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Generate audio
            logger.info(f"Generating audio with {len(speakers)} speakers, CFG scale: {cfg_scale}")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                refresh_negative=True,
            )
            
            # Extract audio from outputs
            logger.info(f"Model outputs type: {type(outputs)}")
            logger.info(f"Model outputs attributes: {dir(outputs) if hasattr(outputs, '__dict__') else 'No attributes'}")
            
            if hasattr(outputs, 'audio_codes') and outputs.audio_codes is not None:
                logger.info(f"Found audio_codes with shape: {outputs.audio_codes.shape}")
                # Convert audio codes to waveform
                audio_array = self._decode_audio(outputs.audio_codes)
            elif hasattr(outputs, 'audio') and outputs.audio is not None:
                logger.info(f"Found audio attribute with shape: {outputs.audio.shape}")
                audio_array = self._decode_audio(outputs.audio)
            elif hasattr(outputs, 'waveform') and outputs.waveform is not None:
                logger.info(f"Found waveform attribute with shape: {outputs.waveform.shape}")
                audio_array = self._decode_audio(outputs.waveform)
            elif torch.is_tensor(outputs):
                logger.info(f"Outputs is tensor with shape: {outputs.shape}")
                audio_array = self._decode_audio(outputs)
            else:
                logger.error(f"Model output structure: {outputs}")
                raise RuntimeError("No audio generated from model - check output structure")
            
            generation_time = time.time() - start_time
            duration = len(audio_array) / self.settings.sample_rate
            
            # Prepare metadata
            metadata = {
                "duration": duration,
                "sample_rate": self.settings.sample_rate,
                "speakers_used": speakers,
                "generation_time": generation_time,
                "num_speakers": len(speakers),
                "cfg_scale": cfg_scale
            }
            
            logger.info(f"Audio generated successfully: {duration:.2f}s in {generation_time:.2f}s")
            
            return audio_array, metadata
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise
        finally:
            self.stats["active_requests"] -= 1
    
    async def generate_audio_streaming(
        self,
        text: str,
        speakers: List[str],
        cfg_scale: float = 1.3,
        max_length: Optional[int] = None,
        chunk_size: int = 24000
    ) -> Iterator[Dict[str, Any]]:
        """Generate audio with streaming support"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        self.stats["active_requests"] += 1
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Similar validation as generate_audio
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            if not speakers or len(speakers) > self.settings.max_speakers:
                raise ValueError(f"Number of speakers must be between 1 and {self.settings.max_speakers}")
            
            for speaker in speakers:
                if speaker not in self.voice_presets:
                    raise ValueError(f"Speaker '{speaker}' not found in available voices")
            
            # Format and prepare inputs (same as non-streaming)
            formatted_script = self._format_script(text, len(speakers))
            voice_samples = []
            
            for speaker_name in speakers:
                audio_path = self.voice_presets[speaker_name]
                audio_data = self._read_audio(audio_path)
                if len(audio_data) == 0:
                    raise ValueError(f"Failed to load audio for speaker '{speaker_name}'")
                voice_samples.append(audio_data)
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Start generation in background thread
            generation_thread = threading.Thread(
                target=self._generate_streaming_worker,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for generation to start
            time.sleep(1.0)
            
            # Stream audio chunks
            chunk_count = 0
            audio_stream = audio_streamer.get_stream(0)
            
            for audio_chunk in audio_stream:
                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure 1D
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to base64 for transmission
                audio_base64 = self._audio_to_base64(audio_np)
                
                yield {
                    "chunk": audio_base64,
                    "chunk_id": chunk_count,
                    "is_final": False
                }
                
                chunk_count += 1
            
            # Wait for generation thread to complete
            generation_thread.join(timeout=10.0)
            
            # Send completion message
            generation_time = time.time() - start_time
            yield {
                "status": "complete",
                "total_chunks": chunk_count,
                "generation_time": generation_time
            }
            
            logger.info(f"Streaming generation completed: {chunk_count} chunks in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {
                "error": str(e),
                "status": "error"
            }
        finally:
            self.stats["active_requests"] -= 1
    
    def _generate_streaming_worker(self, inputs, cfg_scale, audio_streamer):
        """Background worker for streaming generation"""
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                audio_streamer=audio_streamer,
                verbose=False,
                refresh_negative=True,
            )
        except Exception as e:
            logger.error(f"Streaming generation worker error: {e}")
            audio_streamer.end()
    
    def _decode_audio(self, audio_codes) -> np.ndarray:
        """Decode audio codes to waveform"""
        logger.info(f"Decoding audio: type={type(audio_codes)}, tensor={torch.is_tensor(audio_codes)}")
        
        if torch.is_tensor(audio_codes):
            logger.info(f"Audio tensor shape: {audio_codes.shape}, dtype: {audio_codes.dtype}")
            if audio_codes.dtype == torch.bfloat16:
                audio_codes = audio_codes.float()
            audio_array = audio_codes.cpu().numpy()
        else:
            logger.info(f"Converting non-tensor audio: {type(audio_codes)}")
            audio_array = np.array(audio_codes)
        
        logger.info(f"Audio array shape before processing: {audio_array.shape}")
        
        # Ensure proper shape and type
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
            
        logger.info(f"Audio array final shape: {audio_array.shape}, dtype: {audio_array.dtype}")
        
        # Check if we have valid audio data
        if audio_array.size == 0:
            raise RuntimeError("Decoded audio array is empty")
        
        return audio_array.astype(np.float32)
    
    def _audio_to_base64(self, audio: np.ndarray) -> str:
        """Convert audio array to base64 string"""
        # Normalize and convert to 16-bit PCM
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        audio_16bit = (audio * 32767).astype(np.int16)
        return base64.b64encode(audio_16bit.tobytes()).decode('utf-8')
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        uptime = time.time() - self.stats["start_time"]
        
        gpu_info = {}
        if torch.cuda.is_available() and self.device == "cuda":
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB",
                "utilization": f"{torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'N/A'}%"
            }
        
        return {
            "uptime_seconds": int(uptime),
            "total_requests": self.stats["total_requests"],
            "active_requests": self.stats["active_requests"],
            "gpu_info": gpu_info,
            "model_loaded": self.is_initialized,
            "device": self.device
        }


# Global service instance
_vibevoice_service = None


async def get_vibevoice_service() -> VibeVoiceService:
    """Get or create the global VibeVoice service instance"""
    global _vibevoice_service
    
    if _vibevoice_service is None:
        _vibevoice_service = VibeVoiceService()
        await _vibevoice_service.initialize()
    
    return _vibevoice_service