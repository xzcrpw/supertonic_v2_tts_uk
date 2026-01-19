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
| **OpenTTS LADA** | ~5 | 1 (female) | [HuggingFace](https://huggingface.co/datasets/speech-uk/opentts-lada) ✅ |
| **OpenTTS TETIANA** | ~5 | 1 (female) | [HuggingFace](https://huggingface.co/datasets/speech-uk/opentts-tetiana) ✅ |
| **OpenTTS KATERYNA** | ~5 | 1 (female) | [HuggingFace](https://huggingface.co/datasets/speech-uk/opentts-kateryna) ✅ |
| **OpenTTS MYKYTA** | ~5 | 1 (male) | [HuggingFace](https://huggingface.co/datasets/speech-uk/opentts-mykyta) ✅ |
| **OpenTTS OLEKSA** | ~5 | 1 (male) | [HuggingFace](https://huggingface.co/datasets/speech-uk/opentts-oleksa) ✅ |
| **Ukrainian Podcasts** | ~100+ | Many | [HuggingFace](https://huggingface.co/datasets/taras-sereda/uk-pods) ✅ |
| **Common Voice UK** | ~80 | 1000+ | [Mozilla](https://commonvoice.mozilla.org/uk/datasets) |
| **Voice of America** | ~390 | Many | [HuggingFace](https://huggingface.co/datasets/speech-uk/voice-of-america) ✅ |
| **Broadcast Speech** | ~300 | Many | [HuggingFace](https://huggingface.co/datasets/Yehor/broadcast-speech-uk) ✅ |
| **Compiled Dataset** | ~1200 | Many | [NextCloud](https://nx16725.your-storageshare.de/s/cAbcBeXtdz7znDN) / [Torrent](https://academictorrents.com/details/fcf8bb60c59e9eb583df003d54ed61776650beb8) |

> ⚠️ **M-AILABS** (caito.de) наразі **недоступний**. Використовуйте OpenTTS voices замість нього.

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

### 🏆 Рекомендовані GPU (ціна/швидкість)

| GPU | Ціна/год | Час | Вартість | Рекомендація |
|-----|----------|-----|----------|--------------|
| **A100 40GB (Italy)** | $0.152 | ~7 днів | **~$26** | 💰 Найдешевше |
| 2× A100 40GB (Italy) | $0.299 | ~4 дні | ~$29 | Швидко + дешево |
| **H100 SXM (India)** | $0.746 | ~2-3 дні | **~$35-45** | 🚀 Найшвидше |
| RTX 4090 (Portugal) | $0.155 | ~14 днів | ~$52 | Backup |
| RTX PRO 6000 Blackwell | $0.413 | ~5-6 днів | ~$55 | 96GB VRAM |

### 🚀 H100 SXM - Найшвидший варіант

```bash
# Конфігурація для H100 80GB
./scripts/train_h100.sh
# або
python train_autoencoder.py --config config/h100_optimized.yaml
```

**Переваги H100:**
- Transformer Engine (FP8) — 2× speedup
- 3,350 GB/s memory bandwidth
- 80GB VRAM → batch_size=48-128

### Рекомендований Template

- **PyTorch (Vast)** з Jupyter
- CUDA 12.x

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
