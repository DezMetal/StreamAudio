# SPDX-License-Identifier: LicenseRef-DNetLab-SourceAvailable-1.0
# SPDX-FileCopyrightText: Copyright (c) 2026 D-Net Lab <https://lab.dnet.live>
#
# Part of StreamAudio, created and maintained by D-Net Lab.
# Free for non-commercial use; commercial use requires a licence.
# Attribution to D-Net Lab is required and may not be removed. See LICENSE.
"""
RVC Voice Conversion Module for StreamAudio
============================================

Provides a togglable RVC (Retrieval-based Voice Conversion) post-processing
layer that converts Piper TTS output to a personalized voice.

Requires: rvc-python (pip install rvc-python)
"""

import os
import tempfile
import wave
import numpy as np
from typing import Optional

# Lazy import to avoid loading RVC dependencies when not needed
_rvc_inference = None

def _get_rvc_inference():
    """Lazy load RVCInference to avoid import overhead when RVC is disabled."""
    global _rvc_inference
    if _rvc_inference is None:
        try:
            from rvc_python.infer import RVCInference
            _rvc_inference = RVCInference
        except ImportError as e:
            raise ImportError(
                "rvc-python is required for RVC voice conversion. "
                "Install with: pip install rvc-python"
            ) from e
    return _rvc_inference


def create_rvc_converter(model_path: str, **kwargs):
    """
    Factory function to create the appropriate RVC converter based on model type.
    
    Args:
        model_path: Path to RVC model (.pth for PyTorch, .onnx for ONNX)
        **kwargs: Additional arguments passed to converter constructor
        
    Returns:
        RVCConverter or RVCOnnxConverter instance
    """
    if model_path.lower().endswith('.onnx'):
        from rvc_onnx_inference import RVCOnnxConverter
        return RVCOnnxConverter(model_path, **kwargs)
    else:
        return RVCConverter(model_path, **kwargs)


class RVCConverter:
    """
    RVC Voice Converter that transforms audio from one voice to another.
    
    Usage:
        converter = RVCConverter("models/miku/weights/miku_default_rvc.pth")
        converted_audio = converter.convert(audio_int16, sample_rate=22050)
    """
    
    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "cpu",
        pitch_method: str = "rmvpe",
        pitch_shift: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5
    ):
        """
        Initialize the RVC converter.
        
        Args:
            model_path: Path to the .pth RVC model file
            index_path: Optional path to .index file for better quality
            device: Computation device ("cpu" or "cuda:0")
            pitch_method: Pitch extraction method (rmvpe, harvest, crepe, pm)
            pitch_shift: Pitch adjustment in semitones
            index_rate: Feature search ratio (0.0-1.0)
            protect: Protection for voiceless consonants (0.0-0.5)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RVC model not found: {model_path}")
        
        self.model_path = model_path
        self.index_path = index_path
        self.device = device
        self.pitch_method = pitch_method
        self.pitch_shift = pitch_shift
        self.index_rate = index_rate
        self.protect = protect
        
        # Initialize RVC inference
        RVCInference = _get_rvc_inference()
        self.rvc = RVCInference(device=device)
        self.rvc.load_model(model_path, version="v2")
        
        # Set parameters
        self.rvc.set_params(
            f0method=pitch_method,
            f0up_key=pitch_shift,
            index_rate=index_rate,
            protect=protect
        )

    def convert(self, audio_int16: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """
        Convert audio using RVC with direct numpy processing (no temp files).
        
        Args:
            audio_int16: Input audio as int16 numpy array
            sample_rate: Sample rate of input audio
            
        Returns:
            Converted audio as int16 numpy array at original sample rate
        """
        if len(audio_int16) == 0:
            return audio_int16
        
        # TODO: Direct numpy conversion is producing static - use file fallback for now
        # The _convert_direct method needs debugging
        return self._convert_via_file(audio_int16, sample_rate)
    
    def _convert_direct(self, audio_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Fast path: Convert using direct numpy arrays, bypassing file I/O.
        
        rvc-python's load_audio() accepts (sample_rate, audio_data) tuples,
        which triggers the AttributeError path for direct array processing.
        """
        # Access the underlying VC module for direct pipeline access
        vc = self.rvc.vc
        
        if vc.hubert_model is None:
            from rvc_python.modules.vc.utils import load_hubert
            vc.hubert_model = load_hubert(vc.config, vc.lib_dir)
        
        # Prepare audio as float32 normalized (what the pipeline expects)
        audio_float = (audio_int16.astype(np.float32) / 32768.0)
        
        # Resample to 16kHz if needed (pipeline expects 16kHz input)
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio_float) * 16000 / sample_rate)
            audio_float = resample(audio_float, num_samples).astype(np.float32)
        
        # Normalize
        audio_max = np.abs(audio_float).max() / 0.95
        if audio_max > 1:
            audio_float /= audio_max
        
        # Get model info
        model_info = self.rvc.models.get(self.rvc.current_model, {})
        file_index = model_info.get("index", "") or ""
        
        # Run the pipeline directly
        times = [0, 0, 0]
        audio_opt = vc.pipeline.pipeline(
            vc.hubert_model,
            vc.net_g,
            0,  # sid
            audio_float,
            None,  # input_audio_path (not used when audio provided)
            times,
            self.pitch_shift,
            self.pitch_method,
            file_index,
            self.index_rate,
            vc.if_f0,
            3,  # filter_radius
            vc.tgt_sr,
            0,  # resample_sr (0 = no resample)
            1,  # rms_mix_rate
            vc.version,
            self.protect,
            None  # f0_file
        )
        
        # Resample back to original rate if needed
        if vc.tgt_sr != sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio_opt) * sample_rate / vc.tgt_sr)
            audio_opt = resample(audio_opt, num_samples)
        
        # Convert to int16
        return (audio_opt * 32768.0).astype(np.int16)
    
    def _convert_via_file(self, audio_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        """Fallback: Convert via temporary files (slower but more robust)."""
        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_file:
            in_path = in_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name
        
        try:
            # Write input audio to temp file
            with wave.open(in_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Run RVC conversion
            self.rvc.infer_file(in_path, out_path)
            
            # Read converted audio
            with wave.open(out_path, 'rb') as wf:
                converted_rate = wf.getframerate()
                converted_data = wf.readframes(wf.getnframes())
                converted_audio = np.frombuffer(converted_data, dtype=np.int16)
            
            # Resample back to original rate if needed
            if converted_rate != sample_rate:
                from scipy.signal import resample
                num_samples = int(len(converted_audio) * sample_rate / converted_rate)
                converted_audio = resample(converted_audio, num_samples).astype(np.int16)
            
            return converted_audio
            
        finally:
            # Clean up temp files
            if os.path.exists(in_path):
                os.unlink(in_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def __repr__(self):
        return f"RVCConverter(model={os.path.basename(self.model_path)}, device={self.device})"
