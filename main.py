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

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- SETTINGS ---
DEFAULT_SETTINGS = {
    "model_path": os.path.join(script_dir, "models/cori/en_GB-cori-high.onnx"),
    "config_path": os.path.join(script_dir, "models/cori/en_GB-cori-high.onnx.json"),
    "start_phrases": ["initializing", "starting program", "hello", "ready", "initialize"],
    "stop_phrases": ["terminating", "ending program", "goodbye", "terminated", "shutting down"],
    "dynamics": {
        "pause_comma_range": [50, 80],
        "pause_period_range": [110, 350],
        "noise_step": 0.2,
        "pitch_range": [-.3, 1.25],
        "pitch_deviation": 0.075,
        "scale_range": [0.85, 1.2],
        "scale_deviation": 0.075
    }
}

# --- HELPER FUNCTIONS ---
def _update_settings(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            original[key] = _update_settings(original.get(key, {}), value)
        else:
            original[key] = value
    return original

@contextmanager
def _open_stream(sample_rate, channels):
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
        
        if not self.headless: self.console.print("[cyan]Initializing TTS model...[/cyan]")
        start_time = time.time()
        self.voice = PiperVoice.load(self.settings['model_path'], self.settings['config_path'])
        if not self.headless: self.console.print(f"[green]✔ Model loaded in {time.time() - start_time:.2f} seconds.[/green]")

    def reset_dynamics(self):
        self.dynamics_generator.reset()

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
        speak_flag = True if speak_override is None else speak_override
        save_path_flag = None
        return_bytes_flag = False

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
        current_pitch_semitones = 0
        all_audio_data = []

        with (_open_stream(sample_rate, channels) if speak_flag else nullcontext()) as st:
            progress_context = Progress(console=self.console) if not self.headless else nullcontext()
            with progress_context as progress:
                task = progress.add_task("[cyan]Synthesizing...", total=len(parts)) if not self.headless else None
                
                for part in parts:
                    if not self.headless: progress.advance(task)
                    if part.startswith('[pause='):
                        duration_ms = int(part.strip('[]').split('=')[1])
                        silence = np.zeros(int(sample_rate * (duration_ms / 1000.0)), dtype=np.int16)
                        if speak_flag and st: st.write(silence)
                        if save_path_flag or return_bytes_flag: all_audio_data.append(silence)
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

                        if speak_flag and st: st.write(final_part_audio)
                        if save_path_flag or return_bytes_flag: all_audio_data.append(final_part_audio)
        
        if not self.headless: self.console.print("---")
        final_audio = np.concatenate(all_audio_data) if all_audio_data else np.array([], dtype=np.int16)

        if save_path_flag:
                with wave.open(save_path_flag, 'wb') as wf:
                    wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(sample_rate)
                    wf.writeframes(final_audio.tobytes())
                self.console.print(f"[bold green]Audio saved to {save_path_flag}[/bold green]")

        if return_bytes_flag:
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