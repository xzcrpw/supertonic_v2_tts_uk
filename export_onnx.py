"""
ONNX Export для Supertonic v2 TTS

Експортує всі модулі в ONNX формат для production inference:
1. Text Encoder
2. Reference Encoder
3. Vector Field Estimator
4. Latent Decoder (Vocoder)
5. Duration Predictor

ONNX конфігурація:
- Opset version: 17
- Dynamic axes для batch/sequence
- FP16 optimization опційно

Референс: Офіційні ONNX моделі (~260MB)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
from supertonic.models.text_to_latent import TextToLatent, ReferenceEncoder, TextEncoder
from supertonic.models.duration_predictor import DurationPredictor
from supertonic.data.tokenizer import CharacterTokenizer


class TextEncoderWrapper(nn.Module):
    """Wrapper для Text Encoder з фіксованими inputs."""
    
    def __init__(self, text_to_latent: TextToLatent):
        super().__init__()
        self.text_encoder = text_to_latent.text_encoder
    
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        lang_id: torch.Tensor
    ) -> torch.Tensor:
        return self.text_encoder(text_ids, text_mask, lang_id=lang_id)


class ReferenceEncoderWrapper(nn.Module):
    """Wrapper для Reference Encoder."""
    
    def __init__(self, text_to_latent: TextToLatent):
        super().__init__()
        self.reference_encoder = text_to_latent.reference_encoder
    
    def forward(
        self,
        reference_latent: torch.Tensor,
        reference_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.reference_encoder(reference_latent, reference_mask)


class VectorFieldWrapper(nn.Module):
    """Wrapper для Vector Field Estimator."""
    
    def __init__(self, text_to_latent: TextToLatent):
        super().__init__()
        self.vector_field = text_to_latent.vector_field
    
    def forward(
        self,
        z_t: torch.Tensor,
        z_ref: torch.Tensor,
        text_encoding: torch.Tensor,
        reference_encoding: torch.Tensor,
        timestep: torch.Tensor,
        text_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.vector_field(
            z_t=z_t,
            z_ref=z_ref,
            text_encoding=text_encoding,
            reference_encoding=reference_encoding,
            timestep=timestep,
            text_mask=text_mask
        )


class DurationPredictorWrapper(nn.Module):
    """Wrapper для Duration Predictor."""
    
    def __init__(self, duration_predictor: DurationPredictor):
        super().__init__()
        self.predictor = duration_predictor
    
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        reference_latent: torch.Tensor,
        reference_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(
            text_ids=text_ids,
            text_mask=text_mask,
            reference_latent=reference_latent,
            reference_mask=reference_mask
        )


def export_text_encoder(
    text_to_latent: TextToLatent,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Text Encoder в ONNX."""
    print("Exporting Text Encoder...")
    
    wrapper = TextEncoderWrapper(text_to_latent).eval()
    
    # Dummy inputs
    batch_size = 1
    text_len = 100
    
    text_ids = torch.randint(0, 100, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    lang_id = torch.tensor([0])
    
    # Export
    torch.onnx.export(
        wrapper,
        (text_ids, text_mask, lang_id),
        str(output_path),
        input_names=["text_ids", "text_mask", "lang_id"],
        output_names=["text_encoding"],
        dynamic_axes={
            "text_ids": {0: "batch", 1: "text_len"},
            "text_mask": {0: "batch", 1: "text_len"},
            "lang_id": {0: "batch"},
            "text_encoding": {0: "batch", 1: "text_len"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def export_reference_encoder(
    text_to_latent: TextToLatent,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Reference Encoder в ONNX."""
    print("Exporting Reference Encoder...")
    
    wrapper = ReferenceEncoderWrapper(text_to_latent).eval()
    
    # Dummy inputs
    batch_size = 1
    ref_len = 50
    latent_dim = 144
    
    reference_latent = torch.randn(batch_size, latent_dim, ref_len)
    reference_mask = torch.ones(batch_size, ref_len, dtype=torch.bool)
    
    # Export
    torch.onnx.export(
        wrapper,
        (reference_latent, reference_mask),
        str(output_path),
        input_names=["reference_latent", "reference_mask"],
        output_names=["reference_encoding"],
        dynamic_axes={
            "reference_latent": {0: "batch", 2: "ref_len"},
            "reference_mask": {0: "batch", 1: "ref_len"},
            "reference_encoding": {0: "batch"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def export_vector_field(
    text_to_latent: TextToLatent,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Vector Field Estimator в ONNX."""
    print("Exporting Vector Field Estimator...")
    
    wrapper = VectorFieldWrapper(text_to_latent).eval()
    
    # Dummy inputs
    batch_size = 1
    latent_dim = 144
    latent_len = 50
    text_len = 100
    hidden_dim = 128
    num_ref_vectors = 50
    
    z_t = torch.randn(batch_size, latent_dim, latent_len)
    z_ref = torch.randn(batch_size, latent_dim, latent_len)
    text_encoding = torch.randn(batch_size, text_len, hidden_dim)
    reference_encoding = torch.randn(batch_size, num_ref_vectors, hidden_dim)
    timestep = torch.rand(batch_size)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    
    # Export
    torch.onnx.export(
        wrapper,
        (z_t, z_ref, text_encoding, reference_encoding, timestep, text_mask),
        str(output_path),
        input_names=["z_t", "z_ref", "text_encoding", "reference_encoding", "timestep", "text_mask"],
        output_names=["velocity"],
        dynamic_axes={
            "z_t": {0: "batch", 2: "latent_len"},
            "z_ref": {0: "batch", 2: "latent_len"},
            "text_encoding": {0: "batch", 1: "text_len"},
            "reference_encoding": {0: "batch"},
            "timestep": {0: "batch"},
            "text_mask": {0: "batch", 1: "text_len"},
            "velocity": {0: "batch", 2: "latent_len"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def export_latent_decoder(
    latent_decoder: LatentDecoder,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Latent Decoder (Vocoder) в ONNX."""
    print("Exporting Latent Decoder...")
    
    latent_decoder.eval()
    
    # Dummy input
    batch_size = 1
    latent_dim = 24
    latent_len = 300
    
    latent = torch.randn(batch_size, latent_dim, latent_len)
    
    # Export
    torch.onnx.export(
        latent_decoder,
        (latent,),
        str(output_path),
        input_names=["latent"],
        output_names=["audio"],
        dynamic_axes={
            "latent": {0: "batch", 2: "latent_len"},
            "audio": {0: "batch", 1: "audio_len"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def export_latent_encoder(
    latent_encoder: LatentEncoder,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Latent Encoder в ONNX."""
    print("Exporting Latent Encoder...")
    
    latent_encoder.eval()
    
    # Dummy input
    batch_size = 1
    n_mels = 228
    mel_len = 200
    
    mel = torch.randn(batch_size, n_mels, mel_len)
    
    # Export
    torch.onnx.export(
        latent_encoder,
        (mel,),
        str(output_path),
        input_names=["mel"],
        output_names=["latent"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_len"},
            "latent": {0: "batch", 2: "latent_len"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def export_duration_predictor(
    duration_predictor: DurationPredictor,
    output_path: Path,
    opset_version: int = 17
):
    """Експортує Duration Predictor в ONNX."""
    print("Exporting Duration Predictor...")
    
    wrapper = DurationPredictorWrapper(duration_predictor).eval()
    
    # Dummy inputs
    batch_size = 1
    text_len = 100
    ref_len = 50
    latent_dim = 144
    
    text_ids = torch.randint(0, 100, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    reference_latent = torch.randn(batch_size, latent_dim, ref_len)
    reference_mask = torch.ones(batch_size, ref_len, dtype=torch.bool)
    
    # Export
    torch.onnx.export(
        wrapper,
        (text_ids, text_mask, reference_latent, reference_mask),
        str(output_path),
        input_names=["text_ids", "text_mask", "reference_latent", "reference_mask"],
        output_names=["duration"],
        dynamic_axes={
            "text_ids": {0: "batch", 1: "text_len"},
            "text_mask": {0: "batch", 1: "text_len"},
            "reference_latent": {0: "batch", 2: "ref_len"},
            "reference_mask": {0: "batch", 1: "ref_len"},
            "duration": {0: "batch"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def verify_onnx_model(onnx_path: Path, test_inputs: Dict[str, torch.Tensor]) -> bool:
    """Верифікує ONNX модель."""
    print(f"Verifying {onnx_path.name}...")
    
    try:
        # Load and check model
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        # Run inference
        session = ort.InferenceSession(str(onnx_path))
        
        # Convert inputs to numpy
        inputs = {k: v.numpy() for k, v in test_inputs.items()}
        
        # Run
        outputs = session.run(None, inputs)
        
        print(f"  ✓ Verification passed")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def export_all(
    checkpoint_dir: Path,
    output_dir: Path,
    config_path: Path,
    opset_version: int = 17,
    verify: bool = True
):
    """Експортує всі модулі."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Create tokenizer for vocab size
    tokenizer = CharacterTokenizer(languages=config.languages.supported)
    
    device = torch.device("cpu")  # Export on CPU
    
    # Load models
    print("Loading models...")
    
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
    ae_ckpt = checkpoint_dir / "autoencoder" / "checkpoint_final.pt"
    tts_ckpt = checkpoint_dir / "tts" / "checkpoint_final.pt"
    dur_ckpt = checkpoint_dir / "duration" / "checkpoint_final.pt"
    
    if ae_ckpt.exists():
        ckpt = torch.load(ae_ckpt, map_location=device)
        latent_encoder.load_state_dict(ckpt["encoder"])
        latent_decoder.load_state_dict(ckpt["decoder"])
    
    if tts_ckpt.exists():
        ckpt = torch.load(tts_ckpt, map_location=device)
        text_to_latent.load_state_dict(ckpt["model"])
    
    if dur_ckpt.exists():
        ckpt = torch.load(dur_ckpt, map_location=device)
        duration_predictor.load_state_dict(ckpt["model"])
    
    # Export all
    print("\n=== Exporting ONNX Models ===\n")
    
    exported = []
    
    # Latent Encoder
    path = export_latent_encoder(latent_encoder, output_dir / "latent_encoder.onnx", opset_version)
    exported.append(path)
    
    # Latent Decoder
    path = export_latent_decoder(latent_decoder, output_dir / "latent_decoder.onnx", opset_version)
    exported.append(path)
    
    # Text Encoder
    path = export_text_encoder(text_to_latent, output_dir / "text_encoder.onnx", opset_version)
    exported.append(path)
    
    # Reference Encoder
    path = export_reference_encoder(text_to_latent, output_dir / "reference_encoder.onnx", opset_version)
    exported.append(path)
    
    # Vector Field
    path = export_vector_field(text_to_latent, output_dir / "vector_field.onnx", opset_version)
    exported.append(path)
    
    # Duration Predictor
    path = export_duration_predictor(duration_predictor, output_dir / "duration_predictor.onnx", opset_version)
    exported.append(path)
    
    print(f"\n=== Export Complete ===")
    print(f"Exported {len(exported)} models to {output_dir}")
    
    # Calculate total size
    total_size = sum(p.stat().st_size for p in exported)
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    
    return exported


def main():
    """CLI для ONNX export."""
    parser = argparse.ArgumentParser(description="Export Supertonic v2 to ONNX")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="onnx_models")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    
    args = parser.parse_args()
    
    export_all(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        opset_version=args.opset,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
