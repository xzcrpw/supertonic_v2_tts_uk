import os
from pymcd.mcd import Calculate_MCD

def main():
    # 1. Шляхи до файлів (ЗМІНИ НА СВОЇ)
    # Дуже важливо порівнювати РЕКОНСТРУКЦІЮ з тим самим ОРИГІНАЛОМ
    original_file = "data/raw/opentts/lada/extracted_wavs/lada_0.wav" 
    
    checkpoints = {
        "5k": "reconstructed_5000.wav",
        "10k": "reconstructed_10000.wav",
        "20k": "reconstructed_20000.wav",
        "45k": "reconstructed_45000.wav"
    }

    # 2. Ініціалізація калькулятора
    # mode="dtw" — це обов'язково, воно вирівнює файли по часу, 
    # якщо вони трохи зсунуті
    mcd_toolbox = Calculate_MCD(mcd_mode="dtw")

    print("\n" + "="*50)
    print("🔬 ПРОФЕСІЙНИЙ АНАЛІЗ ЯКОСТІ (MCD)")
    print("="*50)
    print(f"Оригінал: {original_file}")
    print("-"*50)

    if not os.path.exists(original_file):
        print(f"❌ Помилка: Оригінальний файл не знайдено за шляхом {original_file}")
        return

    results = []

    for label, path in checkpoints.items():
        if os.path.exists(path):
            try:
                # Обчислюємо MCD
                mcd_value = mcd_toolbox.calculate_mcd(original_file, path)
                results.append((label, mcd_value))
                print(f"✅ Checkpoint {label:4}: MCD = {mcd_value:.4f}")
            except Exception as e:
                print(f"⚠️ Помилка при обробці {label}: {e}")
        else:
            print(f"⏭️  Checkpoint {label:4}: Файл не знайдено (пропускаю)")

    # 3. Аналіз прогресу
    if len(results) >= 2:
        print("-"*50)
        first_val = results[0][1]
        last_val = results[-1][1]
        improvement = ((first_val - last_val) / first_val) * 100
        print(f"📈 Загальний прогрес: {improvement:.2f}% покращення")
    
    print("="*50)
    print("💡 ГАЙД ПО ЦИФРАХ:")
    print("   > 8.0  : Жахливо (робот у бочці)")
    print("   6.0-8.0: Початковий рівень (слова розбірливі, але метал)")
    print("   4.0-6.0: Хороший рівень (схоже на людину)")
    print("   < 3.0  : Професійна якість (майже як оригінал)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()