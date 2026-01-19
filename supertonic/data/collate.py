"""
Collate functions для DataLoader

Забезпечують правильний batching з padding для:
- Audio (variable length)
- Mel spectrograms
- Text sequences
- Reference audio/mel
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


def pad_sequence_1d(
    sequences: List[torch.Tensor],
    padding_value: float = 0.0,
    max_len: Optional[int] = None
) -> torch.Tensor:
    """
    Pad 1D sequences to same length.
    
    Args:
        sequences: List of [T] tensors
        padding_value: Value for padding
        max_len: Optional maximum length
        
    Returns:
        Padded tensor [B, T_max]
    """
    lengths = [len(s) for s in sequences]
    max_length = max_len or max(lengths)
    
    padded = torch.full(
        (len(sequences), max_length),
        padding_value,
        dtype=sequences[0].dtype
    )
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    
    return padded


def pad_sequence_2d(
    sequences: List[torch.Tensor],
    padding_value: float = 0.0,
    max_len: Optional[int] = None
) -> torch.Tensor:
    """
    Pad 2D sequences [C, T] to same length.
    
    Args:
        sequences: List of [C, T] tensors
        padding_value: Value for padding
        max_len: Optional maximum length
        
    Returns:
        Padded tensor [B, C, T_max]
    """
    lengths = [s.shape[-1] for s in sequences]
    max_length = max_len or max(lengths)
    channels = sequences[0].shape[0]
    
    padded = torch.full(
        (len(sequences), channels, max_length),
        padding_value,
        dtype=sequences[0].dtype
    )
    
    for i, seq in enumerate(sequences):
        length = min(seq.shape[-1], max_length)
        padded[i, :, :length] = seq[:, :length]
    
    return padded


def create_mask(lengths: List[int], max_len: int) -> torch.Tensor:
    """
    Створює attention mask з lengths.
    
    Args:
        lengths: List of sequence lengths
        max_len: Maximum length
        
    Returns:
        mask: [B, max_len] boolean tensor (True = valid)
    """
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def autoencoder_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function для AutoencoderDataset.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Extract audio
    audios = [sample["audio"] for sample in batch]
    audio_lengths = [len(a) for a in audios]
    
    # Pad audio
    audio_padded = pad_sequence_1d(audios, padding_value=0.0)
    
    # Create mask
    max_len = audio_padded.shape[-1]
    audio_mask = create_mask(audio_lengths, max_len)
    
    result = {
        "audio": audio_padded,
        "audio_lengths": torch.tensor(audio_lengths),
        "audio_mask": audio_mask
    }
    
    # Optional mel
    if "mel" in batch[0]:
        mels = [sample["mel"] for sample in batch]
        mel_padded = pad_sequence_2d(mels, padding_value=0.0)
        mel_lengths = [m.shape[-1] for m in mels]
        mel_mask = create_mask(mel_lengths, mel_padded.shape[-1])
        
        result["mel"] = mel_padded
        result["mel_lengths"] = torch.tensor(mel_lengths)
        result["mel_mask"] = mel_mask
    
    return result


def tts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function для TTSDataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Audio
    audios = [sample["audio"] for sample in batch]
    audio_lengths = [len(a) for a in audios]
    audio_padded = pad_sequence_1d(audios, padding_value=0.0)
    audio_mask = create_mask(audio_lengths, audio_padded.shape[-1])
    
    # Mel
    mels = [sample["mel"] for sample in batch]
    mel_lengths = [m.shape[-1] for m in mels]
    mel_padded = pad_sequence_2d(mels, padding_value=0.0)
    mel_mask = create_mask(mel_lengths, mel_padded.shape[-1])
    
    # Text
    text_ids = [sample["text_ids"] for sample in batch]
    text_lengths = [len(t) for t in text_ids]
    text_padded = pad_sequence_1d(text_ids, padding_value=0)
    text_mask = create_mask(text_lengths, text_padded.shape[-1])
    
    # Reference mel
    ref_mels = [sample["reference_mel"] for sample in batch]
    ref_lengths = [r.shape[-1] for r in ref_mels]
    ref_padded = pad_sequence_2d(ref_mels, padding_value=0.0)
    ref_mask = create_mask(ref_lengths, ref_padded.shape[-1])
    
    # Language IDs
    lang_ids = torch.tensor([sample.get("lang_id", 0) for sample in batch])
    
    # Durations
    durations = torch.tensor([sample["duration"] for sample in batch])
    
    return {
        "audio": audio_padded,
        "audio_lengths": torch.tensor(audio_lengths),
        "audio_mask": audio_mask,
        "mel": mel_padded,
        "mel_lengths": torch.tensor(mel_lengths),
        "mel_mask": mel_mask,
        "text_ids": text_padded,
        "text_lengths": torch.tensor(text_lengths),
        "text_mask": text_mask,
        "reference_mel": ref_padded,
        "reference_lengths": torch.tensor(ref_lengths),
        "reference_mask": ref_mask,
        "lang_ids": lang_ids,
        "durations": durations
    }


def duration_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function для DurationDataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Text
    text_ids = [sample["text_ids"] for sample in batch]
    text_lengths = [len(t) for t in text_ids]
    text_padded = pad_sequence_1d(text_ids, padding_value=0)
    text_mask = create_mask(text_lengths, text_padded.shape[-1])
    
    # Reference mel
    ref_mels = [sample["reference_mel"] for sample in batch]
    ref_lengths = [r.shape[-1] for r in ref_mels]
    ref_padded = pad_sequence_2d(ref_mels, padding_value=0.0)
    ref_mask = create_mask(ref_lengths, ref_padded.shape[-1])
    
    # Durations
    durations = torch.stack([sample["duration"] for sample in batch])
    
    return {
        "text_ids": text_padded,
        "text_lengths": torch.tensor(text_lengths),
        "text_mask": text_mask,
        "reference_mel": ref_padded,
        "reference_lengths": torch.tensor(ref_lengths),
        "reference_mask": ref_mask,
        "durations": durations
    }


@dataclass
class TTSBatch:
    """
    Structured batch для TTS training.
    
    Provides named access to batch components.
    """
    audio: torch.Tensor
    audio_lengths: torch.Tensor
    audio_mask: torch.Tensor
    mel: torch.Tensor
    mel_lengths: torch.Tensor
    mel_mask: torch.Tensor
    text_ids: torch.Tensor
    text_lengths: torch.Tensor
    text_mask: torch.Tensor
    reference_mel: torch.Tensor
    reference_lengths: torch.Tensor
    reference_mask: torch.Tensor
    lang_ids: torch.Tensor
    durations: torch.Tensor
    
    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "TTSBatch":
        return cls(**d)
    
    def to(self, device: torch.device) -> "TTSBatch":
        """Move all tensors to device."""
        return TTSBatch(
            audio=self.audio.to(device),
            audio_lengths=self.audio_lengths.to(device),
            audio_mask=self.audio_mask.to(device),
            mel=self.mel.to(device),
            mel_lengths=self.mel_lengths.to(device),
            mel_mask=self.mel_mask.to(device),
            text_ids=self.text_ids.to(device),
            text_lengths=self.text_lengths.to(device),
            text_mask=self.text_mask.to(device),
            reference_mel=self.reference_mel.to(device),
            reference_lengths=self.reference_lengths.to(device),
            reference_mask=self.reference_mask.to(device),
            lang_ids=self.lang_ids.to(device),
            durations=self.durations.to(device)
        )


# ============================================================================
# Unit tests
# ============================================================================

def _test_collate():
    """Тест collate functions."""
    print("Testing Collate Functions...")
    
    # Test pad_sequence_1d
    seqs = [torch.randn(10), torch.randn(15), torch.randn(8)]
    padded = pad_sequence_1d(seqs)
    assert padded.shape == (3, 15)
    print(f"  pad_sequence_1d: {[len(s) for s in seqs]} → {padded.shape} ✓")
    
    # Test pad_sequence_2d
    seqs_2d = [torch.randn(24, 50), torch.randn(24, 80), torch.randn(24, 30)]
    padded_2d = pad_sequence_2d(seqs_2d)
    assert padded_2d.shape == (3, 24, 80)
    print(f"  pad_sequence_2d: {[s.shape for s in seqs_2d]} → {padded_2d.shape} ✓")
    
    # Test create_mask
    lengths = [10, 15, 8]
    mask = create_mask(lengths, 15)
    assert mask.shape == (3, 15)
    assert mask[0, 9] == True
    assert mask[0, 10] == False
    print(f"  create_mask: {lengths} → {mask.shape} ✓")
    
    # Test full collate
    batch = [
        {
            "audio": torch.randn(44100),
            "mel": torch.randn(228, 100),
            "text_ids": torch.randint(0, 100, (20,)),
            "reference_mel": torch.randn(228, 50),
            "lang_id": 0,
            "duration": 1.0
        },
        {
            "audio": torch.randn(88200),
            "mel": torch.randn(228, 200),
            "text_ids": torch.randint(0, 100, (30,)),
            "reference_mel": torch.randn(228, 80),
            "lang_id": 5,
            "duration": 2.0
        }
    ]
    
    collated = tts_collate_fn(batch)
    
    print(f"  TTS collate:")
    print(f"    audio: {collated['audio'].shape}")
    print(f"    mel: {collated['mel'].shape}")
    print(f"    text_ids: {collated['text_ids'].shape}")
    print(f"    reference_mel: {collated['reference_mel'].shape}")
    print(f"    lang_ids: {collated['lang_ids']}")
    
    # Test TTSBatch
    tts_batch = TTSBatch.from_dict(collated)
    print(f"  TTSBatch created ✓")
    
    print("All Collate tests passed! ✓\n")


if __name__ == "__main__":
    _test_collate()
