"""
Character Tokenizer для Supertonic v2 TTS

Ключова особливість: NO G2P REQUIRED!
SupertonicTTS працює на raw character-level input:
- Unicode character embeddings покривають Latin, Cyrillic, CJK, etc.
- Модель сама навчається pronunciation
- Numbers, dates, abbreviations обробляються without preprocessing

Підтримувані мови (v2):
- English (en)
- Korean (ko)  
- Spanish (es)
- Portuguese (pt)
- French (fr)
- Ukrainian (uk) - додаємо!
"""

import torch
from typing import Dict, List, Optional, Set, Union
import unicodedata
import json
from pathlib import Path


# ============================================================================
# Character sets для кожної мови
# ============================================================================

# Latin alphabet (базовий для en, es, pt, fr)
LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Extended Latin (для Spanish, Portuguese, French)
EXTENDED_LATIN = "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýÿœ"
EXTENDED_LATIN_UPPER = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝŸŒ"

# Ukrainian Cyrillic
UKRAINIAN_LOWER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINIAN_UPPER = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
UKRAINIAN_SPECIAL = "'"  # апостроф

# Russian Cyrillic (для можливого fine-tuning)
RUSSIAN_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RUSSIAN_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

# Korean Hangul Jamo (basic)
# Повний Hangul занадто великий, використовуємо compositional Jamo
KOREAN_JAMO = "".join([chr(c) for c in range(0x1100, 0x1200)])  # Initial/medial/final jamo
KOREAN_COMPAT_JAMO = "".join([chr(c) for c in range(0x3130, 0x3190)])  # Compatibility jamo

# Numbers та punctuation (universal)
DIGITS = "0123456789"
PUNCTUATION = ".,!?;:'-—–\"()[]{}…·•"
WHITESPACE = " \t\n"

# Special tokens
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<mask>": 4
}


def create_multilingual_vocab(
    include_languages: Optional[List[str]] = None,
    include_korean_full: bool = False
) -> Dict[str, int]:
    """
    Створює multilingual vocabulary.
    
    Args:
        include_languages: List of language codes to include
            Options: "en", "es", "pt", "fr", "uk", "ru", "ko"
            Default: all
        include_korean_full: Include full Hangul syllables (large!)
            
    Returns:
        vocab: Character to index mapping
    """
    if include_languages is None:
        include_languages = ["en", "es", "pt", "fr", "uk", "ko"]
    
    vocab = dict(SPECIAL_TOKENS)
    current_idx = len(SPECIAL_TOKENS)
    
    # Helper to add characters
    def add_chars(chars: str):
        nonlocal current_idx
        for c in chars:
            if c not in vocab:
                vocab[c] = current_idx
                current_idx += 1
    
    # Always include basic characters
    add_chars(WHITESPACE)
    add_chars(DIGITS)
    add_chars(PUNCTUATION)
    
    # Add language-specific characters
    if "en" in include_languages:
        add_chars(LATIN_LOWER)
        add_chars(LATIN_UPPER)
    
    if any(lang in include_languages for lang in ["es", "pt", "fr"]):
        add_chars(LATIN_LOWER)
        add_chars(LATIN_UPPER)
        add_chars(EXTENDED_LATIN)
        add_chars(EXTENDED_LATIN_UPPER)
    
    if "uk" in include_languages:
        add_chars(UKRAINIAN_LOWER)
        add_chars(UKRAINIAN_UPPER)
        add_chars(UKRAINIAN_SPECIAL)
    
    if "ru" in include_languages:
        add_chars(RUSSIAN_LOWER)
        add_chars(RUSSIAN_UPPER)
    
    if "ko" in include_languages:
        add_chars(KOREAN_COMPAT_JAMO)
        if include_korean_full:
            # Повний Hangul: U+AC00 - U+D7A3 (11,172 symbols!)
            # Це значно збільшує vocab, але потрібно для Korean
            for code in range(0xAC00, 0xD7A4):
                add_chars(chr(code))
    
    return vocab


class CharacterTokenizer:
    """
    Character-level tokenizer для Supertonic v2.
    
    NO G2P! Модель сама навчається pronunciation.
    
    Args:
        vocab: Character to index mapping
        unk_token: Token for unknown characters
        pad_token: Padding token
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
    """
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        languages: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        add_bos: bool = False,
        add_eos: bool = False,
        normalize_unicode: bool = True,
        lowercase: bool = False
    ):
        if vocab is None:
            vocab = create_multilingual_vocab(include_languages=languages)
        
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        self.unk_id = vocab[unk_token]
        self.pad_id = vocab[pad_token]
        self.bos_id = vocab[bos_token]
        self.eos_id = vocab[eos_token]
        
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
    
    @property
    def vocab_size(self) -> int:
        """Розмір словника."""
        return len(self.vocab)
    
    def _preprocess(self, text: str) -> str:
        """Попередня обробка тексту."""
        # Unicode normalization (NFC)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def encode(
        self,
        text: str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        return_tensor: bool = True
    ) -> Union[List[int], torch.Tensor]:
        """
        Кодує текст в токени.
        
        Args:
            text: Input text
            add_bos: Add BOS token (overrides default)
            add_eos: Add EOS token (overrides default)
            return_tensor: Return as torch.Tensor
            
        Returns:
            Token IDs
        """
        text = self._preprocess(text)
        
        # Character to ID
        ids = []
        
        if add_bos or (add_bos is None and self.add_bos):
            ids.append(self.bos_id)
        
        for char in text:
            ids.append(self.vocab.get(char, self.unk_id))
        
        if add_eos or (add_eos is None and self.add_eos):
            ids.append(self.eos_id)
        
        if return_tensor:
            return torch.tensor(ids, dtype=torch.long)
        return ids
    
    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special: bool = True
    ) -> str:
        """
        Декодує токени в текст.
        
        Args:
            ids: Token IDs
            skip_special: Skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        chars = []
        for idx in ids:
            if skip_special and idx in special_ids:
                continue
            chars.append(self.inverse_vocab.get(idx, self.unk_token))
        
        return "".join(chars)
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Кодує batch текстів з padding.
        
        Args:
            texts: List of texts
            max_length: Maximum length (truncate if longer)
            padding: Pad to same length
            return_mask: Return attention mask
            
        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        encoded = [self.encode(text, return_tensor=False) for text in texts]
        
        # Truncate
        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        
        # Get max length
        max_len = max(len(ids) for ids in encoded)
        
        # Pad
        if padding:
            padded = []
            masks = []
            for ids in encoded:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
            
            result = {"input_ids": torch.tensor(padded, dtype=torch.long)}
            
            if return_mask:
                result["attention_mask"] = torch.tensor(masks, dtype=torch.long)
            
            return result
        else:
            return {"input_ids": [torch.tensor(ids, dtype=torch.long) for ids in encoded]}
    
    def save(self, path: Union[str, Path]):
        """Зберігає tokenizer."""
        data = {
            "vocab": self.vocab,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "normalize_unicode": self.normalize_unicode,
            "lowercase": self.lowercase
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharacterTokenizer":
        """Завантажує tokenizer."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            vocab=data["vocab"],
            unk_token=data["unk_token"],
            pad_token=data["pad_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            add_bos=data["add_bos"],
            add_eos=data["add_eos"],
            normalize_unicode=data["normalize_unicode"],
            lowercase=data["lowercase"]
        )


# ============================================================================
# Language-specific utilities
# ============================================================================

# Language codes
LANGUAGE_CODES = {
    "en": 0,  # English
    "ko": 1,  # Korean
    "es": 2,  # Spanish
    "pt": 3,  # Portuguese
    "fr": 4,  # French
    "uk": 5,  # Ukrainian
    "ru": 6,  # Russian (optional)
}


def detect_language_simple(text: str) -> str:
    """
    Простий language detector на основі Unicode ranges.
    
    NOTE: Це спрощена версія. Для production використовуйте
    langdetect або fasttext.
    
    Args:
        text: Input text
        
    Returns:
        Language code
    """
    # Count characters in each script
    cyrillic_count = 0
    korean_count = 0
    latin_count = 0
    
    for char in text:
        code = ord(char)
        
        # Cyrillic
        if 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
            cyrillic_count += 1
        # Korean Hangul
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            korean_count += 1
        # Latin
        elif 0x0041 <= code <= 0x007A or 0x00C0 <= code <= 0x00FF:
            latin_count += 1
    
    if korean_count > cyrillic_count and korean_count > latin_count:
        return "ko"
    elif cyrillic_count > latin_count:
        # Distinguish Ukrainian from Russian by specific letters
        if "і" in text or "ї" in text or "є" in text or "ґ" in text:
            return "uk"
        return "ru"
    else:
        # Default to English for Latin
        return "en"


# ============================================================================
# Unit tests
# ============================================================================

def _test_tokenizer():
    """Тест tokenizer."""
    print("Testing CharacterTokenizer...")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(
        languages=["en", "uk", "ko"],
        add_bos=False,
        add_eos=False
    )
    
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Test English
    en_text = "Hello, world!"
    en_ids = tokenizer.encode(en_text)
    en_decoded = tokenizer.decode(en_ids)
    assert en_decoded == en_text
    print(f"  English: '{en_text}' → {en_ids.tolist()} → '{en_decoded}' ✓")
    
    # Test Ukrainian
    uk_text = "Привіт, світе!"
    uk_ids = tokenizer.encode(uk_text)
    uk_decoded = tokenizer.decode(uk_ids)
    assert uk_decoded == uk_text
    print(f"  Ukrainian: '{uk_text}' → {uk_ids.tolist()} → '{uk_decoded}' ✓")
    
    # Test batch encoding
    batch = ["Hello!", "Привіт!", "Test 123"]
    batch_encoded = tokenizer.batch_encode(batch)
    
    print(f"  Batch input_ids shape: {batch_encoded['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch_encoded['attention_mask'].shape}")
    
    # Test language detection
    assert detect_language_simple("Hello world") == "en"
    assert detect_language_simple("Привіт світ їжак") == "uk"
    print(f"  Language detection ✓")
    
    # Test vocab creation
    vocab = create_multilingual_vocab(include_languages=["en", "uk"])
    print(f"  Multilingual vocab (en+uk): {len(vocab)} characters")
    
    print("All Tokenizer tests passed! ✓\n")


if __name__ == "__main__":
    _test_tokenizer()
