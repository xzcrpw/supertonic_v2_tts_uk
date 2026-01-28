"""
Dataset classes для Supertonic v2 TTS

Datasets:
1. AutoencoderDataset - для тренування Speech Autoencoder
2. TTSDataset - для тренування Text-to-Latent (з context-sharing)
3. DurationDataset - для тренування Duration Predictor
4. MultilingualTTSDataset - об'єднує всі датасети

Підтримувані мовні датасети:
- en: LJSpeech, VCTK, Hi-Fi TTS, LibriTTS
- uk: M-AILABS Ukrainian, Common Voice Ukrainian
- ko: (Korean datasets)
- es, pt, fr: (соответствующие датасети)
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np

from supertonic.data.preprocessing import AudioProcessor, load_audio, normalize_audio
from supertonic.data.tokenizer import CharacterTokenizer, detect_language_simple, LANGUAGE_CODES


class AutoencoderDataset(Dataset):
    """
    Dataset для тренування Speech Autoencoder.
    
    Повертає тільки аудіо — без тексту.
    
    Args:
        manifest_path: Path to manifest JSON file
        audio_processor: AudioProcessor instance
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        return_mel: Return mel spectrogram instead of raw audio
        segment_length: Fixed segment length in samples (for memory efficiency!)
        data_dir: Base directory for audio paths (if paths in manifest are relative)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_processor: Optional[AudioProcessor] = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        return_mel: bool = False,
        cache_audio: bool = False,
        segment_length: Optional[int] = None,  # e.g., 176400 for 4 seconds at 44.1kHz
        data_dir: Optional[Union[str, Path]] = None  # Base directory for relative paths
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_processor = audio_processor or AudioProcessor()
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.return_mel = return_mel
        self.cache_audio = cache_audio
        self.segment_length = segment_length  # Critical for OOM prevention!
        
        # Data directory - use manifest parent if not specified
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.manifest_path.parent.parent  # manifests/ -> data/
        
        # Load manifest
        self.samples = self._load_manifest()
        
        # Audio cache
        self._cache = {} if cache_audio else None
    
    def _load_manifest(self) -> List[Dict]:
        """Завантажує manifest file (JSON array or JSON Lines)."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try JSON array first, then JSON Lines
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            # JSON Lines format
            data = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        # Filter by duration
        samples = []
        for item in data:
            duration = item.get("duration", float("inf"))
            if self.min_duration <= duration <= self.max_duration:
                samples.append(item)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        
        # Resolve full path if relative
        full_path = Path(audio_path)
        if not full_path.is_absolute():
            full_path = self.data_dir / audio_path
        
        # Check cache
        if self._cache is not None and audio_path in self._cache:
            audio = self._cache[audio_path]
        else:
            audio = self.audio_processor.load(str(full_path))
            if self._cache is not None:
                self._cache[audio_path] = audio
        
        # CRITICAL: Random crop to segment_length to prevent OOM!
        if self.segment_length is not None and len(audio) > self.segment_length:
            max_start = len(audio) - self.segment_length
            start = random.randint(0, max_start)
            audio = audio[start:start + self.segment_length]
        
        result = {"audio": audio}
        
        if self.return_mel:
            result["mel"] = self.audio_processor.compute_mel(audio)
        
        # Add metadata
        result["duration"] = len(audio) / self.audio_processor.sample_rate
        result["audio_path"] = audio_path
        
        return result


class TTSDataset(Dataset):
    """
    Dataset для тренування Text-to-Latent module.
    
    Повертає:
    - audio: Raw waveform
    - mel: Mel spectrogram
    - text: Tokenized text
    - reference_audio: Reference audio для speaker conditioning
    - duration: Audio duration
    
    Args:
        manifest_path: Path to manifest JSON
        audio_processor: AudioProcessor instance
        tokenizer: CharacterTokenizer instance
        max_duration: Maximum audio duration
        min_duration: Minimum audio duration
        max_text_length: Maximum text length
        reference_mode: "self" (use same audio) or "random" (random from same speaker)
        data_dir: Base directory for audio paths (if paths in manifest are relative)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[CharacterTokenizer] = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        max_text_length: int = 500,
        reference_mode: str = "self",
        reference_min_duration: float = 1.0,
        reference_max_duration: float = 10.0,
        data_dir: Optional[Union[str, Path]] = None
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_text_length = max_text_length
        self.reference_mode = reference_mode
        self.reference_min_duration = reference_min_duration
        self.reference_max_duration = reference_max_duration
        
        # Data directory
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.manifest_path.parent.parent  # manifests/ -> data/
        
        # Load manifest
        self.samples = self._load_manifest()
        
        # Group by speaker for reference selection
        self.speaker_to_samples = self._group_by_speaker()
    
    def _load_manifest(self) -> List[Dict]:
        """Завантажує manifest (JSON array or JSON Lines)."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try JSON array first, then JSON Lines
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            # JSON Lines format
            data = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        samples = []
        for item in data:
            duration = item.get("duration", float("inf"))
            text_len = len(item.get("text", ""))
            
            if (self.min_duration <= duration <= self.max_duration and
                text_len <= self.max_text_length and text_len > 0):
                samples.append(item)
        
        return samples
    
    def _group_by_speaker(self) -> Dict[str, List[int]]:
        """Групує samples по speaker ID."""
        speaker_to_samples = {}
        for idx, sample in enumerate(self.samples):
            speaker = sample.get("speaker_id", "unknown")
            if speaker not in speaker_to_samples:
                speaker_to_samples[speaker] = []
            speaker_to_samples[speaker].append(idx)
        return speaker_to_samples
    
    def _resolve_path(self, audio_path: str) -> Path:
        """Resolve audio path (relative or absolute)."""
        full_path = Path(audio_path)
        if not full_path.is_absolute():
            full_path = self.data_dir / audio_path
        return full_path
    
    def _get_reference_audio(self, idx: int) -> torch.Tensor:
        """Отримує reference audio."""
        sample = self.samples[idx]
        
        if self.reference_mode == "self":
            # Use same audio as reference
            audio_path = self._resolve_path(sample["audio_path"])
            return self.audio_processor.load(str(audio_path))
        
        elif self.reference_mode == "random":
            # Random sample from same speaker
            speaker = sample.get("speaker_id", "unknown")
            speaker_samples = self.speaker_to_samples.get(speaker, [idx])
            
            # Filter by duration
            valid_samples = []
            for sidx in speaker_samples:
                if sidx != idx:
                    dur = self.samples[sidx].get("duration", 0)
                    if self.reference_min_duration <= dur <= self.reference_max_duration:
                        valid_samples.append(sidx)
            
            if len(valid_samples) == 0:
                valid_samples = [idx]
            
            ref_idx = random.choice(valid_samples)
            audio_path = self._resolve_path(self.samples[ref_idx]["audio_path"])
            return self.audio_processor.load(str(audio_path))
        
        else:
            raise ValueError(f"Unknown reference_mode: {self.reference_mode}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self._resolve_path(sample["audio_path"])
        audio = self.audio_processor.load(str(audio_path))
        mel = self.audio_processor.compute_mel(audio)
        
        # Tokenize text
        text = sample["text"]
        text_ids = self.tokenizer.encode(text)
        
        # Get reference audio
        reference_audio = self._get_reference_audio(idx)
        reference_mel = self.audio_processor.compute_mel(reference_audio)
        
        # Language
        language = sample.get("language", detect_language_simple(text))
        lang_id = LANGUAGE_CODES.get(language, 0)
        
        return {
            "audio": audio,
            "mel": mel,
            "text": text,
            "text_ids": text_ids,
            "reference_audio": reference_audio,
            "reference_mel": reference_mel,
            "duration": len(audio) / self.audio_processor.sample_rate,
            "language": language,
            "lang_id": lang_id,
            "speaker_id": sample.get("speaker_id", "unknown"),
            "audio_path": sample["audio_path"]
        }


class DurationDataset(Dataset):
    """
    Dataset для тренування Duration Predictor.
    
    Повертає:
    - text_ids: Tokenized text
    - reference_mel: Reference mel spectrogram
    - duration: Ground-truth duration (frames або seconds)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[CharacterTokenizer] = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        duration_unit: str = "frames"  # "frames" або "seconds"
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.duration_unit = duration_unit
        
        self.samples = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Завантажує manifest (JSON array or JSON Lines)."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try JSON array first, then JSON Lines
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            # JSON Lines format
            data = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        samples = []
        for item in data:
            duration = item.get("duration", float("inf"))
            if self.min_duration <= duration <= self.max_duration:
                samples.append(item)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Tokenize text
        text = sample["text"]
        text_ids = self.tokenizer.encode(text)
        
        # Load reference audio
        audio = self.audio_processor.load(sample["audio_path"])
        reference_mel = self.audio_processor.compute_mel(audio)
        
        # Duration
        duration_seconds = len(audio) / self.audio_processor.sample_rate
        
        if self.duration_unit == "frames":
            # Compressed frames (Kc = 6)
            num_mel_frames = reference_mel.shape[-1]
            duration = num_mel_frames / 6  # Temporal compression
        else:
            duration = duration_seconds
        
        return {
            "text_ids": text_ids,
            "reference_mel": reference_mel,
            "duration": torch.tensor(duration, dtype=torch.float),
            "text": text
        }


class MultilingualTTSDataset(Dataset):
    """
    Об'єднаний multilingual dataset.
    
    Підтримує:
    - Weighted sampling між мовами
    - Automatic language detection
    - Language embeddings
    
    Args:
        manifest_paths: Dict of {language: manifest_path}
        weights: Optional dict of {language: weight} for sampling
    """
    
    def __init__(
        self,
        manifest_paths: Dict[str, Union[str, Path]],
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[CharacterTokenizer] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer or CharacterTokenizer()
        
        # Create datasets for each language
        self.datasets = {}
        self.all_samples = []
        
        for lang, manifest_path in manifest_paths.items():
            ds = TTSDataset(
                manifest_path=manifest_path,
                audio_processor=self.audio_processor,
                tokenizer=self.tokenizer,
                **kwargs
            )
            self.datasets[lang] = ds
            
            # Add samples with language info
            for i in range(len(ds)):
                self.all_samples.append((lang, i))
        
        # Weights for sampling
        if weights is not None:
            self.weights = []
            for lang, idx in self.all_samples:
                self.weights.append(weights.get(lang, 1.0))
            self.weights = np.array(self.weights)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lang, sample_idx = self.all_samples[idx]
        sample = self.datasets[lang][sample_idx]
        sample["language"] = lang
        sample["lang_id"] = LANGUAGE_CODES.get(lang, 0)
        return sample
    
    def get_weighted_sampler(self):
        """Повертає WeightedRandomSampler для balanced sampling."""
        if self.weights is None:
            return None
        
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self),
            replacement=True
        )


# ============================================================================
# Context-Sharing Batch Expansion
# ============================================================================

class ContextSharingDataset(Dataset):
    """
    Dataset wrapper для context-sharing batch expansion.
    
    Ключова інновація Supertonic v2:
    - Для кожного sample: один раз кодуємо text/reference
    - Потім застосовуємо Ke різних noise samples з різними timesteps
    
    Переваги:
    - Ефективність пам'яті: B=16, Ke=4 використовує 6.92 GiB vs B=64, Ke=1 = 8.61 GiB
    - Стабілізує alignment learning
    - Прискорює convergence
    
    Args:
        base_dataset: Base TTSDataset
        expansion_factor: Ke - number of noise samples per text/reference
    """
    
    def __init__(
        self,
        base_dataset: TTSDataset,
        expansion_factor: int = 4
    ):
        self.base_dataset = base_dataset
        self.expansion_factor = expansion_factor
    
    def __len__(self) -> int:
        return len(self.base_dataset) * self.expansion_factor
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map expanded index to base index
        base_idx = idx // self.expansion_factor
        expansion_idx = idx % self.expansion_factor
        
        sample = self.base_dataset[base_idx]
        
        # Add expansion info (for different noise/timestep in training)
        sample["expansion_idx"] = expansion_idx
        sample["base_idx"] = base_idx
        
        return sample


# ============================================================================
# Manifest creation utilities
# ============================================================================

def create_manifest_from_ljspeech(
    ljspeech_path: Union[str, Path],
    output_path: Union[str, Path]
):
    """
    Створює manifest з LJSpeech dataset.
    
    LJSpeech structure:
    - wavs/LJ001-0001.wav
    - metadata.csv: filename|text|normalized_text
    """
    ljspeech_path = Path(ljspeech_path)
    metadata_path = ljspeech_path / "metadata.csv"
    
    samples = []
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[-1]  # Use normalized text
                
                audio_path = ljspeech_path / "wavs" / f"{filename}.wav"
                
                if audio_path.exists():
                    # Get duration
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                    
                    samples.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "speaker_id": "LJ",
                        "language": "en",
                        "duration": duration
                    })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created manifest with {len(samples)} samples")


def create_manifest_from_common_voice(
    cv_path: Union[str, Path],
    output_path: Union[str, Path],
    language: str = "uk",
    split: str = "train"
):
    """
    Створює manifest з Common Voice dataset.
    
    Common Voice structure:
    - clips/common_voice_uk_12345.mp3
    - train.tsv, test.tsv, validated.tsv
    """
    cv_path = Path(cv_path)
    tsv_path = cv_path / f"{split}.tsv"
    
    samples = []
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        path_idx = header.index("path")
        sentence_idx = header.index("sentence")
        
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(path_idx, sentence_idx):
                audio_file = parts[path_idx]
                text = parts[sentence_idx]
                
                audio_path = cv_path / "clips" / audio_file
                
                if audio_path.exists():
                    try:
                        info = torchaudio.info(str(audio_path))
                        duration = info.num_frames / info.sample_rate
                        
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text,
                            "speaker_id": f"cv_{hash(audio_file) % 10000}",
                            "language": language,
                            "duration": duration
                        })
                    except:
                        pass
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created manifest with {len(samples)} samples")


# ============================================================================
# Unit tests
# ============================================================================

def _test_datasets():
    """Тест datasets."""
    print("Testing Datasets...")
    
    # Create dummy manifest
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy audio
        audio_path = Path(tmpdir) / "test.wav"
        audio = torch.randn(44100 * 2)  # 2 seconds
        torchaudio.save(str(audio_path), audio.unsqueeze(0), 44100)
        
        # Create manifest
        manifest = [
            {
                "audio_path": str(audio_path),
                "text": "Hello world",
                "speaker_id": "test",
                "language": "en",
                "duration": 2.0
            },
            {
                "audio_path": str(audio_path),
                "text": "Привіт світ",
                "speaker_id": "test",
                "language": "uk",
                "duration": 2.0
            }
        ]
        
        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Test AutoencoderDataset
        ae_dataset = AutoencoderDataset(manifest_path)
        sample = ae_dataset[0]
        print(f"  AutoencoderDataset: audio shape {sample['audio'].shape} ✓")
        
        # Test TTSDataset
        tts_dataset = TTSDataset(manifest_path)
        sample = tts_dataset[0]
        print(f"  TTSDataset: mel {sample['mel'].shape}, text_ids {sample['text_ids'].shape} ✓")
        
        # Test DurationDataset
        dur_dataset = DurationDataset(manifest_path)
        sample = dur_dataset[0]
        print(f"  DurationDataset: duration {sample['duration'].item():.2f} frames ✓")
        
        # Test ContextSharingDataset
        cs_dataset = ContextSharingDataset(tts_dataset, expansion_factor=4)
        print(f"  ContextSharingDataset: {len(tts_dataset)} → {len(cs_dataset)} samples ✓")
    
    print("All Dataset tests passed! ✓\n")


if __name__ == "__main__":
    _test_datasets()
