# 🇺🇦 Supertonic v2 TTS - Ukrainian

Повна реімплементація **Supertonic v2 TTS** для тренування з української мови.

> **Paper**: [Supertonic: Lightweight Text-to-Speech for Super-Diverse Settings](https://arxiv.org/abs/2509.11084)

## 🎯 Features

- **66M параметрів** (компактна архітектура)
- **44.1kHz** високоякісний аудіо вихід
- **Українська мова** з нуля
- **Character-level** токенізація (без G2P)
- **Flow-matching** для якісної генерації
- **ONNX export** для production

## 📊 Архітектура

| Module | Parameters | Description |
|--------|-----------|-------------|
| Speech Autoencoder | ~47M | Vocos-based encoder/decoder з ISTFT |
| Text-to-Latent | ~19M | Flow-matching з LARoPE (γ=10) |
| Duration Predictor | ~0.5M | Швидке L1 тренування |

## 🚀 Quick Start

### Vast.ai Setup

```bash
# 1. Клонуйте репо
git clone https://github.com/your-username/supertonic_v2_tts_uk.git
cd supertonic_v2_tts_uk

# 2. Запустіть setup скрипт
chmod +x scripts/setup_vast.sh
./scripts/setup_vast.sh --minimal
```

### Тренування

```bash
# 1. Autoencoder (спочатку) - ~7-8 днів на 1×5090
python train_autoencoder.py --config config/default.yaml

# 2. Text-to-Latent - ~4-5 днів на 1×5090
python train_text_to_latent.py --config config/default.yaml \
    --autoencoder-checkpoint checkpoints/autoencoder/checkpoint_final.pt

# 3. Duration Predictor - ~20 хвилин
python train_duration_predictor.py --config config/default.yaml
```

### Inference

```bash
python inference.py \
    --text "Привіт, як справи?" \
    --reference samples/reference.wav \
    --output output.wav
```

## 📚 Датасети

### Українська мова

| Датасет | Годин | Спікерів | Посилання |
|---------|-------|----------|-----------|
| **M-AILABS Ukrainian** | ~20 | 2 | [Download](http://www.caito.de/data/Training/stt_tts/uk_UK.tgz) |
| **OpenTTS-UK** | ~10 | 5 | [HuggingFace](https://huggingface.co/datasets/Yehor/opentts-uk) |
| **Common Voice UK** | ~80 | 1000+ | [Mozilla](https://commonvoice.mozilla.org/uk/datasets) |
| **Voice of America** | ~390 | Many | [HuggingFace](https://huggingface.co/datasets/speech-uk/voice-of-america) |
| **Broadcast Speech** | ~300 | Many | [HuggingFace](https://huggingface.co/datasets/Yehor/broadcast-speech-uk) |

### Англійська (для pretrain)

| Датасет | Годин | Посилання |
|---------|-------|-----------|
| LJSpeech | 24 | [Link](https://keithito.com/LJ-Speech-Dataset/) |
| LibriTTS-R | 585 | [Link](https://www.openslr.org/141/) |

### Завантаження

```bash
# Мінімальний набір (~50GB)
python scripts/download_datasets.py --minimal

# Повний набір (~500GB)
python scripts/download_datasets.py --full
```

## 🖥️ Vast.ai Configuration

### Рекомендований Template

- **PyTorch (Vast)** з Jupyter
- CUDA 12.x

### Рекомендований Instance

| GPU | Ціна/год | Storage | Час тренування |
|-----|---------|---------|----------------|
| 1× RTX 5090 | $0.19-0.22 | 200 GB | ~12-14 днів |
| 2× RTX 5090 | $0.35-0.45 | 200 GB | ~6-7 днів |

**Загальна вартість**: ~$55-70

### Найкращі варіанти (станом на 2026):

- `host:96199` (Washington) - **$0.188/hr** - найдешевший
- `host:155385` (CN) - $0.213/hr - verified 5 months

## 📁 Структура проекту

```
supertonic_v2_tts_uk/
├── config/
│   └── default.yaml           # Конфігурація
├── supertonic/
│   ├── models/
│   │   ├── attention.py       # Multi-head attention з RoPE
│   │   ├── convnext.py        # ConvNeXt blocks
│   │   ├── larope.py          # Length-Aware RoPE
│   │   ├── speech_autoencoder.py  # Encoder/Decoder/Discriminators
│   │   ├── text_to_latent.py  # Text→Latent flow-matching
│   │   └── duration_predictor.py
│   ├── losses/
│   │   ├── autoencoder_loss.py    # GAN + Mel + FM loss
│   │   ├── flow_matching_loss.py  # CFM loss + ODE solver
│   │   └── duration_loss.py
│   └── data/
│       ├── preprocessing.py   # Audio processing
│       ├── tokenizer.py       # Multilingual tokenizer
│       ├── dataset.py         # Dataset classes
│       └── collate.py         # Batch collation
├── scripts/
│   ├── setup_vast.sh          # Vast.ai setup
│   ├── download_datasets.py   # Dataset downloader
│   └── prepare_manifest.py    # Manifest preparation
├── train_autoencoder.py       # Autoencoder training
├── train_text_to_latent.py    # TTS training
├── train_duration_predictor.py
├── inference.py               # Synthesis pipeline
├── export_onnx.py             # ONNX export
└── requirements.txt
```

## ⚙️ Конфігурація

Основні параметри в `config/default.yaml`:

```yaml
# Audio
audio:
  sample_rate: 44100
  n_fft: 2048
  hop_length: 512
  n_mels: 228

# Latent space
latent:
  dim: 24
  temporal_compression: 6  # Kc

# Flow matching
flow_matching:
  sigma_min: 1.0e-8
  p_uncond: 0.05           # CFG probability
  cfg_scale: 3.0           # Inference CFG scale
  nfe: 32                  # ODE steps

# LARoPE
larope:
  gamma: 10                # Critical for alignment!
```

## 📈 Тренування

### Етап 1: Autoencoder

```bash
python train_autoencoder.py \
    --config config/default.yaml \
    --data-dir data/raw \
    --batch-size 16 \
    --epochs 50
```

**Loss weights**: λ_recon=45, λ_adv=1, λ_fm=0.1

### Етап 2: Text-to-Latent

```bash
python train_text_to_latent.py \
    --config config/default.yaml \
    --autoencoder-checkpoint checkpoints/autoencoder/checkpoint_final.pt \
    --batch-size 64 \
    --expansion-factor 4 \
    --iterations 700000
```

**Context-Sharing**: B=64, Ke=4 → effective batch = 256

### Етап 3: Duration Predictor

```bash
python train_duration_predictor.py \
    --config config/default.yaml \
    --iterations 3000
```

## 🔧 ONNX Export

```bash
python export_onnx.py \
    --checkpoint-dir checkpoints \
    --output-dir onnx_models \
    --opset 17
```

Outputs:
- `latent_encoder.onnx`
- `latent_decoder.onnx`
- `text_encoder.onnx`
- `reference_encoder.onnx`
- `vector_field.onnx`
- `duration_predictor.onnx`

Total: ~260MB

## 📊 Benchmarks

Target metrics (based on paper):

| Metric | Target |
|--------|--------|
| Word Error Rate (WER) | <3% |
| Speaker Similarity | >0.85 |
| MOS | >4.0 |
| RTF (1×5090) | <0.1 |

## 🔗 Resources

- [Supertonic v2 Paper](https://arxiv.org/abs/2509.11084)
- [Ukrainian TTS Resources](https://github.com/egorsmkv/speech-recognition-uk)
- [HuggingFace speech-uk](https://huggingface.co/speech-uk)
- [Discord: Ukrainian Data Science](https://bit.ly/discord-uds)

## 📝 Citation

```bibtex
@article{supertonic2025,
  title={Supertonic: Lightweight Text-to-Speech for Super-Diverse Settings},
  author={...},
  journal={arXiv preprint arXiv:2509.11084},
  year={2025}
}
```

## 📜 License

MIT License

## 🙏 Acknowledgements

- [egorsmkv/speech-recognition-uk](https://github.com/egorsmkv/speech-recognition-uk) - Ukrainian speech resources
- [Yehor/opentts-uk](https://huggingface.co/datasets/Yehor/opentts-uk) - OpenTTS voices
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Ukrainian dataset
- [speech-uk](https://huggingface.co/speech-uk) - HuggingFace organization
