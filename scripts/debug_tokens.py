import sys
import os
from pathlib import Path

# Додаємо корінь проекту в шлях, щоб Python побачив папку 'supertonic'
sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.data.tokenizer import CharacterTokenizer

# Імітуємо твій конфіг
print("Initializing tokenizer with ['uk', 'en']...")
tokenizer = CharacterTokenizer(languages=["uk", "en"])

uk_text = "Привіт, як справи?"
en_text = "Hello, how are you?"

uk_ids = tokenizer.encode(uk_text)
en_ids = tokenizer.encode(en_text)

print(f"\nVocab size: {tokenizer.vocab_size}")

print(f"\n🇺🇦 UK: '{uk_text}'")
print(f"IDs: {uk_ids}")

print(f"\n🇬🇧 EN: '{en_text}'")
print(f"IDs: {en_ids}")

# Перевірка на UNKNOWN
# Зазвичай ID=0 - це padding, ID=1/2 - це unknown (залежить від реалізації)
unique_en = set(en_ids.tolist())
unique_uk = set(uk_ids.tolist())

print(f"\nUnique EN tokens: {len(unique_en)}")
print(f"Unique UK tokens: {len(unique_uk)}")

if len(unique_en) <= 2:
    print("\n❌ ПИЗДА! Англійська мова перетворюється на сміття (однакові токени)!")
    print("Твоя модель не може вчити англійську, бо не бачить літер.")
elif len(unique_uk) <= 2:
    print("\n❌ ПИЗДА! Українська мова перетворюється на сміття!")
else:
    print("\n✅ Токенізатор працює коректно. Літери кодуються різними цифрами.")