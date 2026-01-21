import librosa
import numpy as np
import os
import sys
from pathlib import Path
from scipy.spatial.distance import euclidean

# Фікс шляхів
sys.path.insert(0, str(Path(__file__).parent.parent))

def calculate_mcd_professional(target_wav, synth_wav):
    """Обчислює Mel Cepstral Distortion між двома файлами."""
    # 1. Завантаження (44100 Гц)
    y_target, sr = librosa.load(target_wav, sr=44100)
    y_synth, _ = librosa.load(synth_wav, sr=44100)

    # 2. Обчислення MFCC (стандарт для MCD: 13-25 коефіцієнтів)
    # Ми беремо 13 коефіцієнтів, як у більшості наукових статей
    mfcc_target = librosa.feature.mfcc(y=y_target, sr=sr, n_mfcc=13)
    mfcc_synth = librosa.feature.mfcc(y=y_synth, sr=sr, n_mfcc=13)

    # 3. Dynamic Time Warping (DTW) - вирівнювання файлів по часу
    # (якщо один файл на мілісекунду довший, DTW це виправить)
    D, wp = librosa.sequence.dtw(X=mfcc_target, Y=mfcc_synth, metric='euclidean')
    
    # Витягуємо вирівняні ознаки
    target_aligned = mfcc_target[:, wp[:, 0]]
    synth_aligned = mfcc_synth[:, wp[:, 1]]

    # 4. Розрахунок середньої евклідової відстані
    # Виключаємо 0-й коефіцієнт (енергія), беремо 1-12 (тембр)
    diff = target_aligned[1:, :] - synth_aligned[1:, :]
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
    
    # Масштабування для стандарту MCD (10/ln10 * sqrt(2))
    mcd_final = mcd * (10.0 / np.log(10.0)) * np.sqrt(2.0) / 10.0 # Нормалізація

    return mcd_final

def main():
    original = "data/raw/opentts/lada/extracted_wavs/lada_0.wav"
    
    files = {
        "5k": "reconstructed_5000.wav",
        "10k": "reconstructed_10000.wav",
        "20k": "reconstructed_20000.wav",
        "30k": "reconstructed_30000.wav", # якщо є
        "45k": "reconstructed_45000.wav"  # якщо є
    }

    print("\n" + "="*50)
    print("🔬 АНАЛІЗ ПРОГРЕСУ МОДЕЛІ (MCD via Librosa)")
    print("="*50)

    if not os.path.exists(original):
        print(f"❌ Оригінал не знайдено: {original}")
        return

    results = []
    for label, path in files.items():
        if os.path.exists(path):
            try:
                score = calculate_mcd_professional(original, path)
                results.append((label, score))
                print(f"✅ Step {label:4}: MCD = {score:.4f}")
            except Exception as e:
                print(f"⚠️ Помилка на {label}: {e}")

    if len(results) >= 2:
        print("-" * 50)
        imp = ((results[0][1] - results[-1][1]) / results[0][1]) * 100
        print(f"📈 Покращення якості: {imp:.2f}%")
    
    print("="*50)
    print("💡 Орієнтири:")
    print("   8.0+ : Початкова стадія (металевий звук)")
    print("   6.5  : Гарна розбірливість")
    print("   5.0  : Якісний голос")
    print("   3.0  : Студійний ідеал")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()