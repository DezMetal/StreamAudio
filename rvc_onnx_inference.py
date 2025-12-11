"""
ONNX RVC Voice Conversion Module for StreamAudio
=================================================

Provides ONNX-based RVC inference for faster voice conversion.
Supports chunked processing for lower latency.

Requires:
- onnxruntime (pip install onnxruntime)
- vec-768-layer-12.onnx embedder model
"""

import os
import numpy as np
from typing import Optional
import onnxruntime as ort


class RVCOnnxConverter:
    """
    ONNX-based RVC Voice Converter with chunked processing support.
    
    Usage:
        converter = RVCOnnxConverter("models/miku/miku_default_rvc.onnx")
        converted_audio = converter.convert(audio_int16, sample_rate=22050)
    """
    
    def __init__(
        self,
        model_path: str,
        embedder_path: Optional[str] = None,
        device: str = "cpu",
        pitch_shift: int = 0,
        hop_size: int = 512,
        sample_rate: int = 40000
    ):
        """
        Initialize the ONNX RVC converter.
        
        Args:
            model_path: Path to the .onnx RVC model
            embedder_path: Path to vec-768-layer-12.onnx (auto-detected if None)
            device: "cpu", "cuda", or "dml"
            pitch_shift: Pitch adjustment in semitones
            hop_size: Processing hop size (affects latency/quality tradeoff)
            sample_rate: Model's expected sample rate (usually 40000 for RVC ONNX)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX RVC model not found: {model_path}")
        
        self.model_path = model_path
        self.pitch_shift = pitch_shift
        self.hop_size = hop_size
        self.target_sr = sample_rate
        
        # Find embedder model
        if embedder_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            embedder_path = os.path.join(script_dir, "models", "vec-768-layer-12.onnx")
        
        if not os.path.exists(embedder_path):
            raise FileNotFoundError(
                f"Embedder model not found: {embedder_path}\n"
                "Download from: https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main"
            )
        
        self.embedder_path = embedder_path
        
        # Set up ONNX execution providers
        providers = self._get_providers(device)
        
        # Load models
        print(f"Loading ONNX RVC model: {os.path.basename(model_path)}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        print(f"Loading embedder: {os.path.basename(embedder_path)}")
        self.embedder = ort.InferenceSession(embedder_path, providers=providers)
        
        # Get model info
        self._analyze_model()
        
        print(f"ONNX RVC ready (device: {device})")
    
    def _get_providers(self, device: str) -> list:
        """Get ONNX execution providers based on device."""
        available = ort.get_available_providers()
        
        if device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml" and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    def _analyze_model(self):
        """Analyze ONNX model inputs/outputs."""
        self.model_inputs = {inp.name: inp for inp in self.session.get_inputs()}
        self.model_outputs = {out.name: out for out in self.session.get_outputs()}
        
        self.embedder_inputs = {inp.name: inp for inp in self.embedder.get_inputs()}
        self.embedder_outputs = {out.name: out for out in self.embedder.get_outputs()}
        
        # Warmup the model with dummy audio to initialize internal states
        self._warmup()
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Hubert features using the embedder model."""
        # Resample to 16kHz for embedder
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples).astype(np.float32)
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        
        # Embedder expects [1, 1, sample_length] - batch, channels, samples
        audio_input = audio.reshape(1, 1, -1)
        
        # Verify inputs
        if np.abs(audio_input).max() == 0:
            print("[ERROR] Embedder input is SILENT!")
        
        # Run embedder
        input_name = list(self.embedder_inputs.keys())[0]
        output_name = list(self.embedder_outputs.keys())[0]
        
        features = self.embedder.run([output_name], {input_name: audio_input})[0]
        
        # Verify features
        if np.abs(features).max() == 0:
            print("[ERROR] Embedder output is SILENT (Features are all zero)!")
        
        return features
    
    def _estimate_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple pitch estimation using autocorrelation."""
        # Simple F0 estimation - could be improved with proper algorithm
        frame_size = 1024
        hop = self.hop_size
        
        num_frames = max(1, (len(audio) - frame_size) // hop + 1)
        f0 = np.zeros(num_frames, dtype=np.float32)
        
        for i in range(num_frames):
            start = i * hop
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            
            # Autocorrelation-based pitch estimation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after initial decay
            d = np.diff(autocorr)
            start_idx = max(20, sr // 500)  # Min ~500Hz
            end_idx = min(len(d), sr // 50)  # Max ~50Hz
            
            if start_idx < end_idx and len(d) > end_idx:
                peak_idx = start_idx + np.argmax(autocorr[start_idx:end_idx])
                # Lower threshold to catch breathy/quiet parts (was 0.1, then 0.05, now 0.02)
                if autocorr[peak_idx] > 0.02 * autocorr[0]:
                    f0[i] = sr / peak_idx
        
        # Interpolate 0 values (unvoiced) to prevent silence in RVC
        # Find indices of non-zero pitch
        idx = np.flatnonzero(f0)
        if len(idx) > 0:
            # Interpolate zeros
            f0 = np.interp(np.arange(len(f0)), idx, f0[idx])
        else:
            # If completely unvoiced, use default mid-range pitch
            f0[:] = 200.0
        
        # Apply pitch shift
        if self.pitch_shift != 0:
            f0 = f0 * (2 ** (self.pitch_shift / 12))
        
        return f0.reshape(1, -1)
    
    def _warmup(self):
        """Run a warmup inference to initialize model state."""
        # Create 0.5s of low-level noise as warmup input
        warmup_len = int(self.target_sr * 0.5)
        warmup_audio = (np.random.randn(warmup_len) * 0.01).astype(np.float32)
        
        try:
            # Run through embedder and model once to warm up
            features = self._extract_features(warmup_audio, self.target_sr)
            f0 = self._estimate_pitch(warmup_audio, self.target_sr)
            
            # Just run once to initialize - discard output
            import scipy.ndimage
            features = scipy.ndimage.zoom(features, (1, 2, 1), order=1)
            feat_len = features.shape[1]
            
            pitch_hz = np.interp(
                np.linspace(0, 1, feat_len),
                np.linspace(0, 1, f0.shape[1]),
                f0[0]
            ).astype(np.float32)
            pitch_hz[:] = 200.0  # Default pitch for warmup
            
            f0_bin = 256
            f0_min = 50.0
            f0_max = 1100.0
            pitch_coarse = np.clip(
                np.round((pitch_hz - f0_min) / (f0_max - f0_min) * f0_bin).astype(np.int64),
                1, f0_bin - 1
            )
            
            inputs = {
                "feats": features.astype(np.float32),
                "p_len": np.array([feat_len], dtype=np.int64),
                "pitch": pitch_coarse.reshape(1, -1),
                "pitchf": pitch_hz.reshape(1, -1).astype(np.float32),
                "sid": np.array([0], dtype=np.int64)
            }
            
            output_name = list(self.model_outputs.keys())[0]
            _ = self.session.run([output_name], inputs)[0]
            # print("ONNX RVC warmup complete")
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")
    
    def convert(self, audio_int16: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """
        Convert audio using ONNX RVC.
        
        Args:
            audio_int16: Input audio as int16 numpy array
            sample_rate: Sample rate of input audio
            
        Returns:
            Converted audio as int16 numpy array
        """
        if len(audio_int16) == 0:
            return audio_int16
        
        # NO CHUNKING - pass full audio through to preserve natural speech flow
        # (breaths, pauses, transitions are essential for natural RVC output)
        
        # Convert to float
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Resample to model's expected rate
        if sample_rate != self.target_sr:
            from scipy.signal import resample
            num_samples = int(len(audio_float) * self.target_sr / sample_rate)
            audio_float = resample(audio_float, num_samples).astype(np.float32)
        
        # Extract features
        features = self._extract_features(audio_float, self.target_sr)
        
        # Estimate pitch
        f0 = self._estimate_pitch(audio_float, self.target_sr)
        
        # Prepare inputs based on model requirements
        # Model expects: feats, p_len, pitch, pitchf, sid
        try:
            # RVC model typically produces ~10ms audio per frame, but Hubert features are 20ms.
            # This causes 2x speedup. We need to upsample features by 2x to match.
            # Upsample features 2x (linear interpolation)
            import scipy.ndimage
            features = scipy.ndimage.zoom(features, (1, 2, 1), order=1)
            
            # Feature length for p_len
            feat_len = features.shape[1]
            
            # Resize pitch (F0 in Hz) to match NEW feature length
            pitch_hz = np.interp(
                np.linspace(0, 1, feat_len),
                np.linspace(0, 1, f0.shape[1]),
                f0[0]
            ).astype(np.float32)
            
            # For TTS input, if no pitch detected, use default female voice pitch
            if pitch_hz.max() < 50:
                # TTS audio typically has consistent pitch - use 200Hz default (female)
                pitch_hz = np.full(feat_len, 200.0, dtype=np.float32)
            
            # Convert F0 Hz to coarse pitch index
            # RVC expects coarse pitch as integer representing pitch bins
            f0_bin = 256  # Number of pitch bins
            f0_min = 50.0
            f0_max = 1100.0
            
            # Linear mapping to bin indices
            pitch_coarse = np.zeros(feat_len, dtype=np.int64)
            voiced_mask = pitch_hz > f0_min
            if voiced_mask.any():
                pitch_coarse[voiced_mask] = np.clip(
                    np.round((pitch_hz[voiced_mask] - f0_min) / (f0_max - f0_min) * f0_bin).astype(np.int64),
                    1, f0_bin - 1
                )
            
            # pitchf is the ACTUAL F0 in Hz (not normalized!)
            pitch_f = pitch_hz.copy()
            
            # Prepare all inputs
            inputs = {
                "feats": features.astype(np.float32),                    # [1, feat_len, 768]
                "p_len": np.array([feat_len], dtype=np.int64),           # [1]
                "pitch": pitch_coarse.reshape(1, -1),                     # [1, feat_len] - coarse bin index
                "pitchf": pitch_f.reshape(1, -1).astype(np.float32),      # [1, feat_len] - actual F0 Hz
                "sid": np.array([0], dtype=np.int64)                      # [1]
            }
            
            # Debug print
            pitch_min, pitch_max = pitch_f.min(), pitch_f.max()
            pitch_mean = pitch_f.mean()
            print(f"ONNX Pitch: min={pitch_min:.1f} max={pitch_max:.1f} mean={pitch_mean:.1f} voiced={(pitch_coarse>0).sum()}/{feat_len}")
            print(f"ONNX inputs: feats={inputs['feats'].shape}, p_len={inputs['p_len']}, pitch={inputs['pitch'].shape}")
            
            # Run inference
            output_name = list(self.model_outputs.keys())[0]
            result = self.session.run([output_name], inputs)[0]
            
            # Process output
            audio_out = result.flatten()
            print(f"ONNX output: {audio_out.shape}, range: [{audio_out.min():.3f}, {audio_out.max():.3f}]")
            
        except Exception as e:
            print(f"ONNX inference error: {e}")
            print(f"Model inputs: {list(self.model_inputs.keys())}")
            print(f"Model outputs: {list(self.model_outputs.keys())}")
            raise
        
        # Resample back to original rate
        if sample_rate != self.target_sr:
            from scipy.signal import resample
            num_samples = int(len(audio_out) * sample_rate / self.target_sr)
            audio_out = resample(audio_out, num_samples)
        
        # Output normalization - preserve full spectrum including quiet parts
        out_max = np.abs(audio_out).max()
        
        if out_max > 0 and out_max < 0.3:
            # Normalize quiet output to reasonable level (preserves natural dynamics)
            gain = min(0.3 / out_max, 5.0)  # Conservative boost
            audio_out = audio_out * gain
        
        # Convert back to int16
        audio_out = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)
        
        return audio_out
    
    def __repr__(self):
        return f"RVCOnnxConverter(model={os.path.basename(self.model_path)})"
