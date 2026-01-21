import librosa
import numpy as np
import os
import sys
from pathlib import Path

def calculate_mcd(target_wav, synth_wav):
    try:
        y_target, sr = librosa.load(target_wav, sr=44100)
        y_synth, _ = librosa.load(synth_wav, sr=44100)
        
        # Обрізаємо тишу на початку і в кінці
        y_target, _ = librosa.effects.trim(y_target)
        y_synth, _ = librosa.effects.trim(y_synth)
        
        # MFCC
        mfcc_target = librosa.feature.mfcc(y=y_target, sr=sr, n_mfcc=13)
        mfcc_synth = librosa.feature.mfcc(y=y_synth, sr=sr, n_mfcc=13)
        
        # DTW
        _, wp = librosa.sequence.dtw(X=mfcc_target, Y=mfcc_synth, metric='euclidean')
        target_aligned = mfcc_target[:, wp[:, 0]]
        synth_aligned = mfcc_synth[:, wp[:, 1]]
        
        # MCD calculation
        diff = target_aligned[1:, :] - synth_aligned[1:, :]
        mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
        return mcd * (10.0 / np.log(10.0)) * np.sqrt(2.0) / 10.0
    except Exception as e:
        return None

def main():
    # Шлях до оригіналу (має бути ТИМ САМИМ, що ти вибрав у тест ері)
    original = "data/raw/opentts/lada/extracted_wavs/lada_0.wav"
    results_dir = Path("test_results")
    
    print("\n" + "="*60)
    print("📊 ЗВІТ ПРО ПРОГРЕС НАВЧАННЯ (MCD Score)")
    print("="*60)
    
    if not results_dir.exists():
        print("❌ Папка test_results не знайдена. Спочатку запусти interactive_test.py")
        return

    # Шукаємо всі згенеровані файли для цього оригіналу
    synth_files = list(results_dir.glob(f"{Path(original).stem}__checkpoint_*.wav"))
    # Сортуємо по номеру кроку
    synth_files.sort(key=lambda x: int(x.stem.split('_')[-1]))

    if not synth_files:
        print(f"❌ Не знайдено файлів для {Path(original).stem} у test_results")
        return

    scores = []
    for f in synth_files:
        step = f.stem.split('_')[-1]
        score = calculate_mcd(original, f)
        if score:
            scores.append((int(step), score))
            status = "🚀" if len(scores) == 1 or score < scores[-2][1] else "⚠️ "
            print(f"{status} Step {step:6}: MCD = {score:.4f}")

    if len(scores) >= 2:
        print("-" * 60)
        start_mcd = scores[0][1]
        end_mcd = scores[-1][1]
        total_imp = ((start_mcd - end_mcd) / start_mcd) * 100
        print(f"📈 Загальне покращення: {total_imp:.2f}%")
        
    print("="*60)
    print("💡 Пояснення: Чим менше MCD, тим ближче голос до оригіналу.")
    print("   MCD 13-15 — це ще дуже шумно. Очікуй 7.0-8.0 на 50к-100к.")

if __name__ == "__main__":
    main()