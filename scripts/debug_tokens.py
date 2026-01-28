# scripts/debug_tokens.py
from supertonic.data.tokenizer import CharacterTokenizer

# Імітуємо твій конфіг
tokenizer = CharacterTokenizer(languages=["uk", "en"])

uk_text = "Привіт, як справи?"
en_text = "Hello, how are you?"

uk_ids = tokenizer.encode(uk_text)
en_ids = tokenizer.encode(en_text)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"\n🇺🇦 UK: '{uk_text}'")
print(f"IDs: {uk_ids}")

print(f"\n🇬🇧 EN: '{en_text}'")
print(f"IDs: {en_ids}")

# ПЕРЕВІРКА НА UNKNOWN (0 або 1 зазвичай)
if all(x == 0 for x in en_ids) or len(set(en_ids)) <= 2:
    print("\n❌ ПИЗДА! Англійська мова перетворюється на сміття/нулі!")
else:
    print("\n✅ Токенізатор працює коректно.")