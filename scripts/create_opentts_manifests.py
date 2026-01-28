import json
from pathlib import Path

def create_subset(source_name, target_name):
    src_path = Path(f"data/manifests_stage2/{source_name}")
    tgt_path = Path(f"data/manifests_stage2/{target_name}")
    
    if not src_path.exists():
        print(f"❌ Файл не знайдено: {src_path}")
        return

    print(f"📖 Читаю {src_path}...")
    with open(src_path, "r", encoding="utf-8") as f:
        # Підтримка і списку, і JSONL
        try:
            data = json.load(f)
        except:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # Фільтруємо тільки OpenTTS
    filtered = [
        item for item in data 
        if "opentts" in item.get("audio_path", "").lower() 
        or item.get("source") == "opentts"
    ]
    
    print(f"   Всього: {len(data)}")
    print(f"   OpenTTS: {len(filtered)}")

    if not filtered:
        print("❌ Не знайдено записів OpenTTS! Перевір шляхи.")
        return

    print(f"💾 Зберігаю в {tgt_path}...")
    with open(tgt_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print("✅ Готово.\n")

if __name__ == "__main__":
    create_subset("train.json", "train_opentts.json")
    create_subset("val.json", "val_opentts.json")