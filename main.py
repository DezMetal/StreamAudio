import sys
import time
import sounddevice as sd
from contextlib import contextmanager, nullcontext
from piper import PiperVoice, SynthesisConfig
import numpy as np
import re
import wave
import io
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
from scipy.signal import resample
from random import choice, uniform, randint
import json
import os
from perlin_noise import PerlinNoise
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
# --- NEW IMPORTS FOR DAEMON ---
from multiprocessing.connection import Listener, Client
import subprocess
import atexit
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- HELPER FUNCTIONS ---
def _update_settings(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            original[key] = _update_settings(original.get(key, {}), value)
        else:
            original[key] = value
    return original

# --- SETTINGS ---
DEFAULT_SETTINGS = {
    "model_path": os.path.join(script_dir, "models/femV1/en_US-hfc_female-medium.onnx"),
    "config_path": os.path.join(script_dir, "models/femV1/en_US-hfc_female-medium.onnx.json"),
    "start_phrases": ["initializing", "starting program", "hello", "ready", "initialize"],
    "stop_phrases": ["terminating", "ending program", "goodbye", "terminated", "shutting down"],
    "piper": {
        "speed": 1.0,
        "noise_scale": 0.67,
        "noise_w": 0.8
    },
    "dynamics": {
        "pause_comma_range": [50, 80],
        "pause_period_range": [110, 350],
        "noise_step": 0.2,
        "pitch_range": [-.3, 1.25],
        "pitch_deviation": 0.075,
        "scale_range": [0.85, 1.2],
        "scale_deviation": 0.075
    },
    "rvc": {
        "enabled": True,            # RVC enabled by default
        "model_path": "models/miku/miku_default_rvc.onnx",  # Default to included model
        "index_path": None,         # Optional path to .index file for better quality
        "pitch_shift": 1,           # Slight pitch adjustment
        "pitch_method": "rmvpe",    # Pitch extraction: rmvpe (best quality, slow), pm (fastest), harvest (balanced)
        "index_rate": 0.5,          # Feature search ratio (0.0-1.0)
        "protect": 0.5,             # Protection for voiceless consonants (0.0-0.5)
        "device": "cpu",            # Computation device: "cpu" or "cuda:0" (GPU recommended for speed)
        "fast_mode": True           # Skip dynamics processing for lower latency (recommended when RVC enabled)
    }
}

# Load settings_default.json if it exists (provides better defaults than hardcoded)
_default_settings_path = os.path.join(script_dir, "settings_default.json")
if os.path.exists(_default_settings_path):
    try:
        with open(_default_settings_path, 'r') as f:
            DEFAULT_SETTINGS = _update_settings(DEFAULT_SETTINGS, json.load(f))
    except Exception:
        pass  # Fall back to hardcoded defaults

# Resolve relative paths to be relative to script_dir (not CWD)
def _resolve_path(path):
    """Resolve path relative to StreamAudio directory if not absolute."""
    if path and not os.path.isabs(path):
        return os.path.join(script_dir, path)
    return path

# Fix model paths
DEFAULT_SETTINGS["model_path"] = _resolve_path(DEFAULT_SETTINGS.get("model_path"))
DEFAULT_SETTINGS["config_path"] = _resolve_path(DEFAULT_SETTINGS.get("config_path"))
if "rvc" in DEFAULT_SETTINGS and DEFAULT_SETTINGS["rvc"].get("model_path"):
    DEFAULT_SETTINGS["rvc"]["model_path"] = _resolve_path(DEFAULT_SETTINGS["rvc"]["model_path"])
if "rvc" in DEFAULT_SETTINGS and DEFAULT_SETTINGS["rvc"].get("index_path"):
    DEFAULT_SETTINGS["rvc"]["index_path"] = _resolve_path(DEFAULT_SETTINGS["rvc"]["index_path"])



@contextmanager
def _open_stream(sample_rate, channels, check_interrupted_callback=None):
    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype='int16')
        stream.start()
        yield stream
    except Exception as e:
        print(e)
        yield None
    finally:
        if 'stream' in locals() and stream:
            stream.stop()
            stream.close()

# --- CORE CLASSES ---
class DynamicsGenerator:
    def __init__(self, settings, seed=None):
        self.settings = settings
        if seed is None:
            seed = randint(1, 1000)
        self.seed = seed
        self.reset()

    def reset(self):
        self.noise_pos = 0
        self.pitch_noise = PerlinNoise(octaves=4, seed=self.seed)
        self.scale_noise = PerlinNoise(octaves=4, seed=self.seed + 1)

    def _map_noise_to_range(self, noise_val, value_range, deviation):
        center = (value_range[0] + value_range[1]) / 2
        span = (value_range[1] - value_range[0]) / 2
        return center + (noise_val * span * deviation)

    def process(self, text):
        parts = re.split(r'(\[dynamics=(?:on|off)\])', text, flags=re.IGNORECASE)
        processed_text = ""
        dynamics_on = True

        for part in parts:
            if not part: continue
            if re.match(r'\[dynamics=on\]', part, re.IGNORECASE):
                dynamics_on = True; continue
            elif re.match(r'\[dynamics=off\]', part, re.IGNORECASE):
                dynamics_on = False; continue

            if dynamics_on:
                sentences = re.split(r'(?<=[.?!])\s+', part)
                for sentence in sentences:
                    if not sentence.strip(): continue
                    self.noise_pos += self.settings['noise_step']
                    pitch_val = self._map_noise_to_range(self.pitch_noise(self.noise_pos), self.settings['pitch_range'], self.settings['pitch_deviation'])
                    scale_val = self._map_noise_to_range(self.scale_noise(self.noise_pos), self.settings['scale_range'], self.settings['scale_deviation'])
                    temp_sentence = f"[scale={scale_val:.2f}][pitch={pitch_val:.2f}]{sentence}"
                    temp_sentence = temp_sentence.replace(',', f', [pause={randint(*self.settings["pause_comma_range"])}]')
                    if temp_sentence.endswith('.') or temp_sentence.endswith(':'): temp_sentence = temp_sentence[:-1] + f'. [pause={randint(*self.settings["pause_period_range"])}]'
                    elif temp_sentence.endswith('?'): temp_sentence = temp_sentence[:-1] + f'? [pause={randint(*self.settings["pause_period_range"])}]'
                    elif temp_sentence.endswith('!'): temp_sentence = temp_sentence[:-1] + f'! [pause={randint(*self.settings["pause_period_range"])}]'
                    processed_text += temp_sentence + " "
            else:
                processed_text += part
        return f"{processed_text.strip()} [scale=1][pitch=0]"

class Voice:
    def __init__(self, settings_path="settings.json", headless=False):
        self.console = Console()
        self.headless = headless
        self.settings = DEFAULT_SETTINGS.copy()
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    self.settings = _update_settings(self.settings, json.load(f))
                if not self.headless: self.console.print(f"[green]✔ Loaded settings from {settings_path}[/green]")
            except Exception as e:
                if not self.headless: self.console.print(f"[bold red]Error loading {settings_path}: {e}[/bold red]")
        
        self.dynamics_generator = DynamicsGenerator(self.settings['dynamics'], seed=1) # Use a fixed seed for deterministic tests
        self._interrupted = False
        
        if not self.headless: self.console.print("[cyan]Initializing TTS model...[/cyan]")
        start_time = time.time()
        self.voice = PiperVoice.load(self.settings['model_path'], self.settings['config_path'])
        if not self.headless: self.console.print(f"[green]✔ Model loaded in {time.time() - start_time:.2f} seconds.[/green]")
        
        # RVC voice converter - initialized in background to avoid blocking startup
        self.rvc_converter = None
        self._rvc_settings = self.settings.get('rvc', {})
        self._rvc_ready = False
        self._rvc_init_thread = None
        self._rvc_ready_callback = None
        
        # Start background RVC initialization if enabled
        if self._rvc_settings.get('enabled') and self._rvc_settings.get('model_path'):
            if not self.headless: self.console.print("[cyan]RVC voice conversion enabled - loading in background...[/cyan]")
            self._rvc_init_thread = threading.Thread(target=self._init_rvc_background, daemon=True)
            self._rvc_init_thread.start()
    
    def set_rvc_ready_callback(self, callback):
        """Set a callback to be called when RVC is ready. Callback receives (success: bool, message: str)."""
        self._rvc_ready_callback = callback
        # If already ready, call immediately
        if self._rvc_ready:
            callback(self.rvc_converter is not None, "RVC already initialized")
    
    def _init_rvc_background(self):
        """Initialize RVC converter in background thread."""
        try:
            from rvc_inference import create_rvc_converter
            rvc_model = self._rvc_settings['model_path']
            # Resolve relative paths
            if not os.path.isabs(rvc_model):
                rvc_model = os.path.join(script_dir, rvc_model)
            
            # Determine if ONNX or PyTorch model
            is_onnx = rvc_model.lower().endswith('.onnx')
            
            rvc_start = time.time()
            
            if is_onnx:
                # ONNX model - use simpler params
                self.rvc_converter = create_rvc_converter(
                    model_path=rvc_model,
                    device=self._rvc_settings.get('device', 'cpu'),
                    pitch_shift=self._rvc_settings.get('pitch_shift', 0)
                )
            else:
                # PyTorch model - use full params
                self.rvc_converter = create_rvc_converter(
                    model_path=rvc_model,
                    index_path=self._rvc_settings.get('index_path'),
                    device=self._rvc_settings.get('device', 'cpu'),
                    pitch_method=self._rvc_settings.get('pitch_method', 'rmvpe'),
                    pitch_shift=self._rvc_settings.get('pitch_shift', 0),
                    index_rate=self._rvc_settings.get('index_rate', 0.5),
                    protect=self._rvc_settings.get('protect', 0.5)
                )
            
            # Pre-warm the pipeline multiple times with actual audio
            # Silent audio doesn't properly warm the ONNX model
            if not self.headless: self.console.print("[cyan]Warming up RVC pipeline...[/cyan]")
            try:
                # Generate actual audio from Piper for warmup (silence doesn't work)
                warmup_text = "test"
                warmup_chunks = list(self.voice.synthesize(warmup_text))
                if warmup_chunks:
                    warmup_audio = np.concatenate([
                        np.frombuffer(c.audio_int16_bytes, dtype=np.int16) 
                        for c in warmup_chunks
                    ])
                    # Run warmup multiple times to fully initialize model
                    for i in range(3):
                        self.rvc_converter.convert(warmup_audio, sample_rate=self.voice.config.sample_rate)
            except Exception as warmup_err:
                if not self.headless: self.console.print(f"[yellow]Warmup issue: {warmup_err}[/yellow]")
            
            load_time = time.time() - rvc_start
            self._rvc_ready = True
            if not self.headless: self.console.print(f"[green]✔ RVC converter ready in {load_time:.2f} seconds.[/green]")
            
            # Notify via callback
            if self._rvc_ready_callback:
                self._rvc_ready_callback(True, f"RVC voice conversion ready ({load_time:.1f}s)")
        except Exception as e:
            self._rvc_ready = True  # Mark as "done" even on failure to prevent waiting forever
            if not self.headless: self.console.print(f"[bold yellow]⚠ RVC initialization failed: {e}[/bold yellow]")
            self.rvc_converter = None
            
            # Notify via callback
            if self._rvc_ready_callback:
                self._rvc_ready_callback(False, f"RVC init failed: {e}")

    def reset_dynamics(self):
        self.dynamics_generator.reset()
    
    def stop(self):
        """Signals the current speech to stop."""
        self._interrupted = True
        # Logic to clear buffer if needed, though the loop break handles most
    
    def say_fast(self, text: str) -> None:
        """
        Fast-path TTS with RVC. Bypasses dynamics processing for minimal latency.
        Streams audio in chunks to reduce initial wait time.
        """
        self._interrupted = False
        
        if not text.strip():
            return
        
        sample_rate = self.voice.config.sample_rate
        channels = 1
        
        if not self.headless: self.console.print(f"[cyan]DEBUG: say_fast called for: '{text[:20]}...'[/cyan]")
        
        # Buffer for chunked processing
        chunk_buffer = []
        buffer_len = 0
        max_accumulate_len = int(sample_rate * 1.5)  # Max buffer size before forced flush
        overlap_len = int(sample_rate * 0.3)         # 0.3 seconds overlap
        last_context = np.array([], dtype=np.int16)
        
        # Open stream once
        try:
            stream = _open_stream(sample_rate, channels)
            if hasattr(stream, '__enter__'):
                st = stream.__enter__()
            else:
                st = stream
            
            if st is None:
                if not self.headless: self.console.print("[bold red]CRITICAL: Stream init FAILED (st is None)[/bold red]")
                return
            else:
                if not self.headless: self.console.print("[green]DEBUG: Stream opened successfully[/green]")
        except Exception as e:
            if not self.headless: self.console.print(f"[bold red]CRITICAL: Stream init Exception: {e}[/bold red]")
            return

        try:
            synth_config = SynthesisConfig()
            
            # Apply Piper settings from config
            piper_settings = self.settings.get('stream_audio', {}).get('piper', {})
            speed = piper_settings.get('speed', 1.0)
            if speed > 0:
                synth_config.length_scale = 1.0 / speed
            synth_config.noise_scale = piper_settings.get('noise_scale', 0.667)
            synth_config.noise_w = piper_settings.get('noise_w', 0.8)
            
            if not self.headless: self.console.print("[cyan]DEBUG: Starting Piper synthesis loop...[/cyan]")
            chunk_count = 0
            
            for chunk in self.voice.synthesize(text, synth_config):
                if self._interrupted:
                    if not self.headless: self.console.print("[yellow]DEBUG: Interrupted in synth loop[/yellow]")
                    break
                
                chunk_count += 1
                data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                chunk_buffer.append(data)
                buffer_len += len(data)
                
                # Check if we should process
                if buffer_len >= max_accumulate_len:
                    if not self.headless: self.console.print(f"[cyan]DEBUG: Flushing buffer (len={buffer_len})[/cyan]")
                    
                    # Flatten current buffer - PROCESS AS ONE PIECE
                    full_buffer = np.concatenate(chunk_buffer)
                    chunk_buffer = [] # Clear buffer list
                    buffer_len = 0
                    
                    # Prepare Chunk
                    raw_chunk = full_buffer
                    audio_out = raw_chunk # Default Fallback
                    
                    if self.rvc_converter is not None and self._rvc_ready:
                        try:
                            # Prepare input with context
                            if len(last_context) > 0:
                                rvc_input = np.concatenate([last_context, raw_chunk])
                            else:
                                rvc_input = raw_chunk
                            
                            # Convert
                            rvc_output = self.rvc_converter.convert(rvc_input, sample_rate)
                            
                            # Log but DO NOT FALLBACK (User Request)
                            in_max = np.abs(rvc_input).max()
                            out_max = np.abs(rvc_output).max()
                            ratio = out_max / (in_max + 1e-6)
                            
                            if not self.headless: 
                                self.console.print(f"[cyan]DEBUG: Chunk Amp In={in_max} Out={out_max} Ratio={ratio:.2f}[/cyan]")
                            
                            if out_max < 500 or (in_max > 1000 and ratio < 0.1):
                                 if not self.headless:
                                     self.console.print(f"[red]WARNING: RVC Low Output (Max={out_max}). Playing anyway.[/red]")
                            
                            # Trim Context from Output
                            if len(last_context) > 0:
                                trim_len = len(last_context)
                                if trim_len < len(rvc_output):
                                    audio_out = rvc_output[trim_len:]
                                else:
                                    audio_out = rvc_output
                            else:
                                audio_out = rvc_output

                        except Exception as e:
                            if not self.headless:
                                self.console.print(f"[yellow]RVC chunk failed: {e}[/yellow]")
                            audio_out = raw_chunk
                    
                    # Save context (raw) for next
                    last_context = raw_chunk[-overlap_len:] if len(raw_chunk) > overlap_len else raw_chunk
                    
                    # Play
                    if st and len(audio_out) > 0:
                        st.write(audio_out)
            
            if not self.headless: self.console.print(f"[cyan]DEBUG: Piper loop finished. Chunks: {chunk_count}. Buffer remnant: {buffer_len}[/cyan]")
            
            # Process remaining buffer
            if chunk_buffer and not self._interrupted:
                raw_chunk = np.concatenate(chunk_buffer)
                audio_out = raw_chunk
                
                if self.rvc_converter is not None and self._rvc_ready:
                    try:
                         # Prepare input with context
                        if len(last_context) > 0:
                            rvc_input = np.concatenate([last_context, raw_chunk])
                        else:
                            rvc_input = raw_chunk
                        
                        rvc_output = self.rvc_converter.convert(rvc_input, sample_rate)
                        
                        # Log but DO NOT FALLBACK (User Request)
                        in_max = np.abs(rvc_input).max()
                        out_max = np.abs(rvc_output).max()
                        ratio = out_max / (in_max + 1e-6)
                        
                        if not self.headless: 
                            self.console.print(f"[cyan]DEBUG: Final Chunk Amp In={in_max} Out={out_max} Ratio={ratio:.2f}[/cyan]")
                        
                        if out_max < 500 or (in_max > 1000 and ratio < 0.1):
                             if not self.headless: self.console.print(f"[red]WARNING: RVC Low Output (Final) (Max={out_max}). Playing anyway.[/red]")
                             
                        # Trim Context from Output
                        if len(last_context) > 0:
                            trim_len = len(last_context)
                            if trim_len < len(rvc_output):
                                audio_out = rvc_output[trim_len:]
                            else:
                                audio_out = rvc_output
                        else:
                            audio_out = rvc_output
                    except Exception as e:
                        print(f"RVC remaining failed: {e}")
                        audio_out = raw_chunk
                
                if st and len(audio_out) > 0:
                    st.write(audio_out)
            
            if not self.headless: self.console.print("[green]DEBUG: say_fast FINISHED completely[/green]")
                    
        finally:
            if hasattr(stream, '__exit__'):
                stream.__exit__(None, None, None)
            if not self.headless: self.console.print("[cyan]DEBUG: Stream Closed[/cyan]")
                    


    def _write_wav_to_buffer(self, audio_data, sample_rate, channels=1):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        buffer.seek(0)
        return buffer.getvalue()

    def say(self, text, speak_override=None):
        self._interrupted = False
        speak_flag = True if speak_override is None else speak_override
        save_path_flag = None
        return_bytes_flag = False

        # If already interrupted before starting
        if self._interrupted: return None
        
        # Fast mode: bypass dynamics when RVC is ready and fast_mode is enabled
        # Only use fast path for simple speech (no special tags)
        use_fast = (
            self._rvc_settings.get('fast_mode', False) 
            and self.rvc_converter is not None 
            and self._rvc_ready
            and speak_flag
            and not re.search(r'\[(save|bytes|speak|file)[=\]]', text, re.IGNORECASE)
        )
        if use_fast:
            self.say_fast(text)
            return None

        save_pattern = r"\[save=(?:'([^']*)'|\"([^\"]*)\"|([^\]\s]*))\]"
        file_pattern = r"\[file=(?:'([^']*)'|\"([^\"]*)\"|([^\]\s]*))\]"

        match = re.search(save_pattern, text, re.IGNORECASE)
        if match:
            save_path_flag = next((g for g in match.groups() if g is not None), None)
            if save_path_flag:
                text = text.replace(match.group(0), '', 1).strip()
        if re.search(r'\[bytes\]', text, re.IGNORECASE):
            return_bytes_flag, speak_flag = True, False
            text = re.sub(r'\[bytes\]', '', text, flags=re.IGNORECASE).strip()
        if re.search(r'\[speak=(false|off)\]', text, re.IGNORECASE):
            speak_flag = False
            text = re.sub(r'\[speak=(false|off)\]', '', text, flags=re.IGNORECASE).strip()
        
        def _replace_file_tag(match):
            filepath = next((g for g in match.groups() if g is not None), None)
            if filepath:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    self.console.print(f"[bold red]Error reading file {filepath}: {e}[/bold red]")
                    return ""
            return ""
        text = re.sub(file_pattern, _replace_file_tag, text, flags=re.IGNORECASE)

        processed_text = self.dynamics_generator.process(text)
        if not self.headless:
            self.console.print("\n---")
            self.console.print(Syntax(processed_text, "ssml", theme="monokai", line_numbers=False))
        
        tag_regex = r'(\[pause=\d+\]|\[scale=[\d.]+\]|\[pitch=[\d.-]+\]|\[noise_scale=[\d.]+\]|\[noise_w=[\d.]+\])'
        parts = [p for p in re.split(tag_regex, processed_text) if p]
        
        sample_rate, channels = self.voice.config.sample_rate, 1
        synth_config = SynthesisConfig()
        
        # Apply Piper settings from config
        piper_settings = self.settings.get('stream_audio', {}).get('piper', {})
        speed = piper_settings.get('speed', 1.0)
        if speed > 0:
            synth_config.length_scale = 1.0 / speed
        synth_config.noise_scale = piper_settings.get('noise_scale', 0.667)
        synth_config.noise_w = piper_settings.get('noise_w', 0.8)

        current_pitch_semitones = 0
        all_audio_data = []

        # Pass a lambda to check interruption status
        check_interrupt = lambda: self._interrupted
        
        # When RVC is enabled, we need to collect all audio first, convert it, then play
        # So we disable streaming during synthesis and play after conversion
        rvc_enabled = self.rvc_converter is not None
        stream_during_synthesis = speak_flag and not rvc_enabled

        with (_open_stream(sample_rate, channels, check_interrupt) if stream_during_synthesis else nullcontext()) as st:
            progress_context = Progress(console=self.console) if not self.headless else nullcontext()
            with progress_context as progress:
                task = progress.add_task("[cyan]Synthesizing...", total=len(parts)) if not self.headless else None
                
                for part in parts:
                    if self._interrupted:
                        if not self.headless: self.console.print("[yellow]Speech interrupted.[/yellow]")
                        break
                        
                    if not self.headless: progress.advance(task)
                    if part.startswith('[pause='):
                        duration_ms = int(part.strip('[]').split('=')[1])
                        silence = np.zeros(int(sample_rate * (duration_ms / 1000.0)), dtype=np.int16)
                        if stream_during_synthesis and st: st.write(silence)
                        all_audio_data.append(silence)
                    elif part.startswith('[scale='):
                        scale_val = float(part.strip('[]').split('=')[1]); synth_config.length_scale = 1.0 / scale_val if scale_val != 0 else 1.0
                    elif part.startswith('[pitch='):
                        pitch_val = float(part.strip('[]').split('=')[1]); current_pitch_semitones = -1 * pitch_val
                    elif part.startswith('[noise_scale='):
                        synth_config.noise_scale = float(part.strip('[]').split('=')[1])
                    elif part.startswith('[noise_w='):
                        synth_config.noise_w = float(part.strip('[]').split('=')[1])
                    else:
                        if not part.strip(): continue
                        raw_chunks = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in self.voice.synthesize(part, synth_config)]
                        if not raw_chunks: continue
                        audio_int16 = np.concatenate(raw_chunks)
                        final_part_audio = audio_int16

                        if current_pitch_semitones != 0:
                            speed = 2.0 ** (current_pitch_semitones / 12.0)
                            audio_float32 = (audio_int16 / 32768.0).astype(np.float32)
                            reader, writer = ArrayReader(audio_float32.reshape(channels, -1)), ArrayWriter(channels=channels)
                            wsola(channels=channels, speed=speed).run(reader, writer)
                            stretched_audio = writer.data
                            resampled_audio = resample(stretched_audio, len(audio_int16), axis=1)
                            final_part_audio = (resampled_audio.flatten() * 32768.0).astype(np.int16)

                        if stream_during_synthesis and st: st.write(final_part_audio)
                        all_audio_data.append(final_part_audio)
        
        if not self.headless: self.console.print("---")
        final_audio = np.concatenate(all_audio_data) if all_audio_data else np.array([], dtype=np.int16)
        
        # Apply RVC voice conversion if enabled
        if self.rvc_converter is not None and len(final_audio) > 0 and not self._interrupted:
            if not self.headless: self.console.print("[cyan]Applying RVC voice conversion...[/cyan]")
            try:
                final_audio = self.rvc_converter.convert(final_audio, sample_rate)
                if not self.headless: self.console.print("[OK] RVC conversion complete.")
            except Exception as e:
                if not self.headless: self.console.print(f"[bold yellow]RVC conversion failed: {e}[/bold yellow]")
        
        # Play audio after RVC conversion (if RVC was enabled and speak is on)
        if rvc_enabled and speak_flag and len(final_audio) > 0 and not self._interrupted:
            if not self.headless: self.console.print("[cyan]Playing converted audio...[/cyan]")
            with _open_stream(sample_rate, channels, check_interrupt) as st:
                if st:
                    st.write(final_audio)

        if save_path_flag and not self._interrupted:
                with wave.open(save_path_flag, 'wb') as wf:
                    wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(sample_rate)
                    wf.writeframes(final_audio.tobytes())
                self.console.print(f"[bold green]Audio saved to {save_path_flag}[/bold green]")

        if return_bytes_flag and not self._interrupted:
            return self._write_wav_to_buffer(final_audio, sample_rate, channels)
        return None

# #############################################################################
# #################### NEW DAEMON AND CLI IMPLEMENTATION ####################
# #############################################################################

# --- DAEMON SETTINGS ---
ADDRESS = ('localhost', 6000)
AUTH_KEY = b'piper-tts-daemon-key'
PID_FILE = os.path.join(script_dir, "tts_daemon.pid")

def cleanup_pid_file():
    """Ensures the PID file is removed on exit."""
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def run_server():
    """The server process that holds the 'hot' Voice instance."""
    # Write PID to file and register cleanup
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(cleanup_pid_file)

    # Initialize the voice model in headless mode
    print("Background TTS service starting...")
    tts_voice = Voice(settings_path="settings.json", headless=True)
    print("Background TTS service ready.")
    
    with Listener(ADDRESS, authkey=AUTH_KEY) as listener:
        while True:
            try:
                conn = listener.accept()
                with conn:
                    msg = conn.recv()
                    if msg == '--SHUTDOWN--':
                        conn.send('--ACK-SHUTDOWN--')
                        break # Exit the loop to shut down
                    elif msg == '--PING--':
                        conn.send('--PONG--')
                    else:
                        # Any other message is treated as text to speak
                        tts_voice.say(msg)
            except Exception as e:
                print(f"Server error: {e}")

def is_server_running():
    """Checks if the server is running by pinging it."""
    if not os.path.exists(PID_FILE):
        return False
    try:
        with Client(ADDRESS, authkey=AUTH_KEY) as conn:
            conn.send('--PING--')
            # poll(1) waits for 1 second. It returns True if data is available.
            if conn.poll(1):
                return conn.recv() == '--PONG--'
            # If poll returns False, we timed out.
            return False
    except (FileNotFoundError, ConnectionRefusedError):
        # If connection fails, server is dead. Clean up stale PID file.
        cleanup_pid_file()
        return False

if __name__ == '__main__':
    # Mode 1: Run as the background daemon
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        run_server()
        sys.exit(0)

    console = Console()
    args = sys.argv[1:]

    # Mode 2: Act as a client to control or use the daemon
    if not args:
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  python main.py 'Text to speak'")
        console.print("  python main.py --status")
        console.print("  python main.py --clear")
        sys.exit(1)

    command = " ".join(args)

    if command == '--status':
        if is_server_running():
            console.print("[green]✔ TTS background service is running.[/green]")
        else:
            console.print("[yellow] TTS background service is not running.[/yellow]")
        sys.exit(0)

    if command == '--clear':
        if not is_server_running():
            console.print("[yellow]TTS background service was not running.[/yellow]")
            sys.exit(0)
        try:
            with Client(ADDRESS, authkey=AUTH_KEY) as conn:
                conn.send('--SHUTDOWN--')
                # Wait up to 5 seconds for an acknowledgement from the server
                if conn.poll(5):
                    conn.recv() # Read the acknowledgement to clear the buffer
                else:
                    console.print("[yellow]Warning: Did not receive shutdown confirmation.[/yellow]")

            console.print("[bold green]TTS background service stopped.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error stopping service: {e}[/bold red]")
            console.print("[yellow]It may have already stopped. Forcing PID file cleanup.[/yellow]")
            cleanup_pid_file()
        sys.exit(0)

    # Default action: Speak an utterance
    if not is_server_running():
        console.print("[cyan]Starting TTS background service...[/cyan]")
        # Start the script itself as a detached background process
        # Using sys.executable ensures we use the same python interpreter
        subprocess.Popen(
            [sys.executable, __file__, '--daemon'],
            start_new_session=True # This detaches it from the current terminal
        )
        # Give it a moment to initialize the model
        time.sleep(5) # Adjust if model loading takes longer
        if not is_server_running():
            console.print("[bold red]Error: Failed to start the background service.[/bold red]")
            sys.exit(1)

    # Send the utterance to the server
    try:
        with Client(ADDRESS, authkey=AUTH_KEY) as conn:
            conn.send(command)
    except Exception as e:
        console.print(f"[bold red]Error communicating with TTS service: {e}[/bold red]")