# Vocos Integration для Supertonic v2

Використання pretrained Vocos decoder замість власного недотренованого.

## Переваги

✅ **Швидше:** ~50k ітерацій замість 200k  
✅ **Дешевше:** ~2 дні на RTX 6000 замість 7 днів  
✅ **Якість:** Pretrained Vocos decoder (без металевого звуку)  
✅ **Мова-агностик:** Працює на українській без проблем  

## Встановлення

```bash
# Install Vocos
pip install vocos

# Verify installation
python -c "from vocos import Vocos; print('✓ Vocos installed')"
```

## Тренування

### 1. Підготовка

Переконайся що дані готові:
```bash
ls data/processed/audio/*.wav | head -5
```

### 2. Запуск тренування

**На локальній машині:**
```bash
python train_autoencoder_vocos.py --config config/vocos_adapter.yaml
```

**На vast.ai сервері:**
```bash
cd /workspace/supertonic_v2_tts_uk

# Stop existing training if needed
pkill -9 python

# Start Vocos adapter training
nohup python train_autoencoder_vocos.py \
  --config config/vocos_adapter.yaml \
  > training_vocos.log 2>&1 &

# Monitor
tail -f training_vocos.log
```

### 3. Моніторинг

```bash
# Check progress
tail -20 training_vocos.log

# GPU usage
nvidia-smi
```

## Після тренування

### Тест якості

```bash
python scripts/test_vocos_reconstruction.py \
  --checkpoint checkpoints/vocos_adapter/checkpoint_50000.pt \
  --audio data/processed/audio/0001.wav \
  --output test_vocos_recon.wav
```

### Використання в TTS

Замість оригінального автоенкодера використай Vocos adapter в Stage 2:

```bash
python train_text_to_latent.py \
  --config config/rtx6000_optimal.yaml \
  --autoencoder-checkpoint checkpoints/vocos_adapter/checkpoint_50000.pt \
  --use-vocos \
  --batch-size 128 \
  --no-wandb
```

## Конфігурація

**`config/vocos_adapter.yaml`** - головний конфіг

Ключові параметри:
- `audio.sample_rate: 24000` - Vocos працює на 24kHz
- `audio.n_mels: 100` - Vocos використовує 100 mel bands
- `encoder.output_dim: 100` - Encoder виводить 100-dim features
- `vocos.freeze_decoder: true` - Vocos decoder frozen
- `train.total_iterations: 50000` - Швидше ніж звичайний autoencoder

## Очікувані результати

| Ітерації | Loss | Якість |
|----------|------|--------|
| 10k | ~0.5 | Базова реконструкція |
| 30k | ~0.2 | Добра якість |
| 50k | ~0.1 | Відмінна якість |

## Troubleshooting

### ImportError: vocos not installed
```bash
pip install vocos
```

### CUDA OOM
Зменш batch_size в конфігу:
```yaml
train:
  batch_size: 16  # замість 32
```

### Метал все ще є
Можливо encoder ще не навчився. Почекай до 50k ітерацій або збільш `total_iterations`.

## Вартість

| GPU | Час (50k) | Вартість |
|-----|-----------|----------|
| RTX 6000 | ~2 дні | ~$20 |
| H100 | ~4 год | ~$16 |
| A100 | ~8 год | ~$20 |

## Порівняння з оригінальним

| Підхід | Ітерації | Час | Якість decoder |
|--------|----------|-----|----------------|
| Оригінальний | 200k | 7 днів | Тренується з нуля ⚠️ |
| **Vocos** | **50k** | **2 дні** | **Pretrained ✅** |

## Файли

- `supertonic/models/vocos_wrapper.py` - Wrapper для Vocos
- `config/vocos_adapter.yaml` - Конфіг
- `train_autoencoder_vocos.py` - Скрипт тренування
- Оригінальні файли **НЕ ЗМІНЕНО** ✅
