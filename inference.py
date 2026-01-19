"""
Inference Pipeline для Supertonic v2 TTS

Повний inference pipeline:
1. Text tokenization
2. Duration prediction
3. Latent generation (Flow-matching з ODE solver)
4. Waveform decoding

Параметри за замовчуванням:
- NFE: 32 (optimal quality/speed)
- CFG scale: 3.0
- Sample rate: 44.1kHz

Usage:
    python inference.py --text "Hello world" --reference audio.wav --output output.wav
    
    # або з Python:
    tts = SupertonicTTS.from_pretrained("checkpoints/")
    audio = tts.synthesize("Hello world", reference_audio)
    tts.save_audio(audio, "output.wav")

Референс: Supertonic v2 paper
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union, Tuple
import time

import torch
import torch.nn as nn
import torchaudio
import numpy as np

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
from supertonic.models.text_to_latent import TextToLatent
from supertonic.models.duration_predictor import DurationPredictor
from supertonic.losses.flow_matching_loss import ODESolver, compress_latents, decompress_latents
from supertonic.data.preprocessing import AudioProcessor, load_audio
from supertonic.data.tokenizer import CharacterTokenizer, detect_language_simple, LANGUAGE_CODES


class SupertonicTTS(nn.Module):
    """
    Supertonic v2 TTS - Повний inference pipeline.
    
    Генерує 44.1kHz high-fidelity speech з тексту.
    
    Args:
        latent_encoder: Pretrained latent encoder
        latent_decoder: Pretrained latent decoder  
        text_to_latent: Text-to-latent flow-matching model
        duration_predictor: Duration predictor
        tokenizer: Character tokenizer
        audio_processor: Audio processor
        config: Model configuration
    """
    
    def __init__(
        self,
        latent_encoder: LatentEncoder,
        latent_decoder: LatentDecoder,
        text_to_latent: TextToLatent,
        duration_predictor: DurationPredictor,
        tokenizer: CharacterTokenizer,
        audio_processor: AudioProcessor,
        config: dict
    ):
        super().__init__()
        
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.text_to_latent = text_to_latent
        self.duration_predictor = duration_predictor
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.config = config
        
        # ODE solver для flow-matching
        self.ode_solver = ODESolver(
            nfe=config.get("nfe", 32),
            cfg_scale=config.get("cfg_scale", 3.0)
        )
        
        # Default parameters
        self.sample_rate = config.get("sample_rate", 44100)
        self.compression_factor = config.get("compression_factor", 6)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ) -> "SupertonicTTS":
        """
        Завантажує pretrained моделі.
        
        Args:
            checkpoint_dir: Directory з checkpoints
            config_path: Path до config file
            device: Target device
            
        Returns:
            SupertonicTTS instance
        """
        checkpoint_dir = Path(checkpoint_dir)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        if config_path is None:
            config_path = checkpoint_dir / "config.yaml"
        
        config = OmegaConf.load(config_path)
        
        # Create tokenizer
        tokenizer = CharacterTokenizer(languages=config.languages.supported)
        
        # Create audio processor
        audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            n_mels=config.audio.n_mels
        )
        
        # Create models
        latent_encoder = LatentEncoder(
            input_dim=config.autoencoder.encoder.input_dim,
            hidden_dim=config.autoencoder.encoder.hidden_dim,
            output_dim=config.autoencoder.encoder.output_dim,
            num_blocks=config.autoencoder.encoder.num_blocks
        )
        
        latent_decoder = LatentDecoder(
            input_dim=config.autoencoder.decoder.input_dim,
            hidden_dim=config.autoencoder.decoder.hidden_dim,
            num_blocks=config.autoencoder.decoder.num_blocks,
            dilations=config.autoencoder.decoder.dilations,
            causal=config.autoencoder.decoder.causal
        )
        
        text_to_latent = TextToLatent(
            vocab_size=tokenizer.vocab_size,
            text_embed_dim=config.text_to_latent.text_encoder.embed_dim,
            text_hidden_dim=config.text_to_latent.text_encoder.hidden_dim,
            ref_input_dim=config.text_to_latent.reference_encoder.input_dim,
            ref_hidden_dim=config.text_to_latent.reference_encoder.hidden_dim,
            vf_hidden_dim=config.text_to_latent.vector_field.hidden_dim,
            gamma=config.larope.gamma
        )
        
        duration_predictor = DurationPredictor(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config.duration_predictor.hidden_dim,
            num_convnext_blocks=config.duration_predictor.num_convnext_blocks
        )
        
        # Load checkpoints
        ae_ckpt_path = checkpoint_dir / "autoencoder" / "checkpoint_final.pt"
        tts_ckpt_path = checkpoint_dir / "tts" / "checkpoint_final.pt"
        dur_ckpt_path = checkpoint_dir / "duration" / "checkpoint_final.pt"
        
        if ae_ckpt_path.exists():
            ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
            latent_encoder.load_state_dict(ae_ckpt["encoder"])
            latent_decoder.load_state_dict(ae_ckpt["decoder"])
            print(f"Loaded autoencoder from {ae_ckpt_path}")
        
        if tts_ckpt_path.exists():
            tts_ckpt = torch.load(tts_ckpt_path, map_location=device)
            text_to_latent.load_state_dict(tts_ckpt["model"])
            print(f"Loaded TTS model from {tts_ckpt_path}")
        
        if dur_ckpt_path.exists():
            dur_ckpt = torch.load(dur_ckpt_path, map_location=device)
            duration_predictor.load_state_dict(dur_ckpt["model"])
            print(f"Loaded duration predictor from {dur_ckpt_path}")
        
        # Move to device
        latent_encoder = latent_encoder.to(device).eval()
        latent_decoder = latent_decoder.to(device).eval()
        text_to_latent = text_to_latent.to(device).eval()
        duration_predictor = duration_predictor.to(device).eval()
        
        # Config dict for inference
        inference_config = {
            "sample_rate": config.audio.sample_rate,
            "hop_length": config.audio.hop_length,
            "compression_factor": config.latent.temporal_compression,
            "nfe": config.flow_matching.nfe,
            "cfg_scale": config.flow_matching.cfg_scale
        }
        
        return cls(
            latent_encoder=latent_encoder,
            latent_decoder=latent_decoder,
            text_to_latent=text_to_latent,
            duration_predictor=duration_predictor,
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            config=inference_config
        )
    
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        reference_audio: Union[str, Path, torch.Tensor],
        language: Optional[str] = None,
        duration_scale: float = 1.0,
        nfe: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Синтезує speech з тексту.
        
        Args:
            text: Input text
            reference_audio: Reference audio для speaker/style conditioning
            language: Language code (auto-detected if None)
            duration_scale: Scale factor for duration (>1 = slower)
            nfe: Number of function evaluations (override default)
            cfg_scale: CFG scale (override default)
            return_latent: Also return generated latents
            
        Returns:
            audio: Generated waveform [T]
            latent: (optional) Generated latent [24, T_latent]
        """
        device = next(self.parameters()).device
        
        # 1. Process reference audio
        if isinstance(reference_audio, (str, Path)):
            reference_audio = self.audio_processor.load(reference_audio)
        
        if reference_audio.dim() == 1:
            reference_audio = reference_audio.unsqueeze(0)  # [1, T]
        
        reference_audio = reference_audio.to(device)
        reference_mel = self.audio_processor.compute_mel(reference_audio)
        
        if reference_mel.dim() == 2:
            reference_mel = reference_mel.unsqueeze(0)  # [1, C, T]
        
        # 2. Tokenize text
        if language is None:
            language = detect_language_simple(text)
        
        lang_id = LANGUAGE_CODES.get(language, 0)
        text_ids = self.tokenizer.encode(text).unsqueeze(0).to(device)  # [1, L]
        text_mask = torch.ones(1, text_ids.size(1), dtype=torch.bool, device=device)
        lang_ids = torch.tensor([lang_id], device=device)
        
        # 3. Encode reference to latent
        ref_latent = self.latent_encoder(reference_mel)  # [1, 24, T]
        ref_compressed = compress_latents(ref_latent, self.compression_factor)  # [1, 144, T/6]
        ref_mask = torch.ones(1, ref_compressed.size(-1), dtype=torch.bool, device=device)
        
        # 4. Predict duration
        predicted_duration = self.duration_predictor(
            text_ids=text_ids,
            text_mask=text_mask,
            reference_latent=ref_compressed,
            reference_mask=ref_mask
        )  # [1]
        
        # Apply duration scale
        predicted_duration = predicted_duration * duration_scale
        num_frames = int(predicted_duration.item())
        num_frames = max(num_frames, 1)
        
        # 5. Encode text and reference
        text_encoding = self.text_to_latent.encode_text(
            text_ids, text_mask, lang_id=lang_ids
        )
        reference_encoding = self.text_to_latent.encode_reference(
            ref_compressed, ref_mask
        )
        
        # 6. Generate latents via ODE solver
        if nfe is not None:
            self.ode_solver.nfe = nfe
        if cfg_scale is not None:
            self.ode_solver.cfg_scale = cfg_scale
        
        z_shape = (1, 144, num_frames)
        
        generated_compressed = self.ode_solver.solve(
            model=self.text_to_latent.vector_field,
            z_shape=z_shape,
            text_encoding=text_encoding,
            reference_encoding=reference_encoding,
            z_ref=ref_compressed[:, :, :num_frames] if ref_compressed.size(-1) >= num_frames else None,
            text_mask=text_mask,
            device=device
        )
        
        # 7. Decompress latents
        generated_latent = decompress_latents(generated_compressed, self.compression_factor)
        
        # 8. Decode to waveform
        audio = self.latent_decoder(generated_latent)  # [1, T_audio]
        audio = audio.squeeze(0)  # [T_audio]
        
        if return_latent:
            return audio, generated_latent.squeeze(0)
        return audio
    
    def save_audio(
        self,
        audio: torch.Tensor,
        path: Union[str, Path],
        normalize: bool = True
    ):
        """
        Зберігає аудіо в файл.
        
        Args:
            audio: Audio tensor [T]
            path: Output path (.wav)
            normalize: Normalize audio before saving
        """
        if normalize:
            audio = audio / audio.abs().max().clamp(min=1e-8)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(
            str(path),
            audio.cpu(),
            self.sample_rate
        )
    
    @torch.no_grad()
    def synthesize_batch(
        self,
        texts: list[str],
        reference_audios: list[Union[str, Path, torch.Tensor]],
        **kwargs
    ) -> list[torch.Tensor]:
        """
        Batch synthesis (sequential, для простоти).
        
        Args:
            texts: List of texts
            reference_audios: List of reference audios
            **kwargs: Additional arguments for synthesize()
            
        Returns:
            List of generated audio tensors
        """
        results = []
        
        for text, ref_audio in zip(texts, reference_audios):
            audio = self.synthesize(text, ref_audio, **kwargs)
            results.append(audio)
        
        return results


def benchmark(
    tts: SupertonicTTS,
    text: str,
    reference_audio: torch.Tensor,
    num_runs: int = 10,
    warmup: int = 3
) -> dict:
    """
    Benchmark inference speed.
    
    Args:
        tts: TTS model
        text: Test text
        reference_audio: Reference audio
        num_runs: Number of benchmark runs
        warmup: Warmup runs
        
    Returns:
        Dict with timing statistics
    """
    device = next(tts.parameters()).device
    
    # Warmup
    for _ in range(warmup):
        _ = tts.synthesize(text, reference_audio)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    
    # Benchmark
    times = []
    audio_lengths = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        audio = tts.synthesize(text, reference_audio)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        
        times.append(end - start)
        audio_lengths.append(len(audio) / tts.sample_rate)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_audio_len = np.mean(audio_lengths)
    rtf = avg_time / avg_audio_len  # Real-Time Factor
    
    return {
        "avg_inference_time": avg_time,
        "std_inference_time": std_time,
        "avg_audio_length": avg_audio_len,
        "rtf": rtf,
        "speedup": 1.0 / rtf,
        "num_runs": num_runs
    }


def main():
    """CLI для inference."""
    parser = argparse.ArgumentParser(description="Supertonic v2 TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--reference", type=str, required=True, help="Reference audio path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    parser.add_argument("--language", type=str, default=None, help="Language code (auto-detected)")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Duration scale factor")
    parser.add_argument("--nfe", type=int, default=32, help="Number of function evaluations")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {args.checkpoint_dir}...")
    tts = SupertonicTTS.from_pretrained(
        args.checkpoint_dir,
        config_path=args.config,
        device=device
    )
    
    print(f"Synthesizing: '{args.text}'")
    print(f"Reference: {args.reference}")
    
    if args.benchmark:
        reference_audio, _ = load_audio(args.reference)
        results = benchmark(tts, args.text, reference_audio)
        
        print(f"\n=== Benchmark Results ===")
        print(f"Average inference time: {results['avg_inference_time']*1000:.2f} ms")
        print(f"Average audio length: {results['avg_audio_length']:.2f} s")
        print(f"Real-Time Factor (RTF): {results['rtf']:.4f}")
        print(f"Speed: {results['speedup']:.1f}× real-time")
    
    start = time.perf_counter()
    
    audio = tts.synthesize(
        text=args.text,
        reference_audio=args.reference,
        language=args.language,
        duration_scale=args.duration_scale,
        nfe=args.nfe,
        cfg_scale=args.cfg_scale
    )
    
    inference_time = time.perf_counter() - start
    audio_duration = len(audio) / tts.sample_rate
    
    tts.save_audio(audio, args.output)
    
    print(f"\nGenerated {audio_duration:.2f}s audio in {inference_time*1000:.0f}ms")
    print(f"Speed: {audio_duration/inference_time:.1f}× real-time")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
