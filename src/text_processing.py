"""
text_processing.py - Text processing and caption generation for FaceCraft.

Converts CelebA 40 binary attributes, CelebA-Dialog, and Face2Text into
unified natural language captions.
"""

import random
from typing import Dict, List, Optional
import pandas as pd

# The official 40 CelebA attributes
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]


def attributes_to_caption(attr_dict: Dict[str, int], seed: Optional[int] = None) -> str:
    """
    Synthesize a fluent, natural descriptive sentence from CelebA binary attributes.
    attr_dict maps attribute name -> 1 (present) or -1/0 (absent).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    is_male = attr_dict.get("Male", -1) == 1
    is_young = attr_dict.get("Young", -1) == 1

    # Base identity noun
    if is_young:
        base_noun = "young man" if is_male else "young woman"
    else:
        base_noun = "man" if is_male else "woman"

    # Hair characteristics
    hair_colors = []
    if attr_dict.get("Blond_Hair", -1) == 1:
        hair_colors.append("blond")
    if attr_dict.get("Brown_Hair", -1) == 1:
        hair_colors.append("brown")
    if attr_dict.get("Black_Hair", -1) == 1:
        hair_colors.append("black")
    if attr_dict.get("Gray_Hair", -1) == 1:
        hair_colors.append("gray")

    hair_style = []
    if attr_dict.get("Straight_Hair", -1) == 1:
        hair_style.append("straight")
    if attr_dict.get("Wavy_Hair", -1) == 1:
        hair_style.append("wavy")
    if attr_dict.get("Bangs", -1) == 1:
        hair_style.append("bangs")
    if attr_dict.get("Receding_Hairline", -1) == 1:
        hair_style.append("a receding hairline")

    is_bald = attr_dict.get("Bald", -1) == 1

    # Facial hair (for men)
    facial_hair = []
    if is_male:
        if attr_dict.get("Mustache", -1) == 1:
            facial_hair.append("a mustache")
        if attr_dict.get("Goatee", -1) == 1:
            facial_hair.append("a goatee")
        if attr_dict.get("Sideburns", -1) == 1:
            facial_hair.append("sideburns")
        if attr_dict.get("5_o_Clock_Shadow", -1) == 1 and not facial_hair:
            facial_hair.append("a 5 o'clock shadow")

    # Expressions & features
    features = []
    if attr_dict.get("Smiling", -1) == 1:
        features.append("smiling")
    elif attr_dict.get("Mouth_Slightly_Open", -1) == 1:
        features.append("with mouth slightly open")

    if attr_dict.get("Pale_Skin", -1) == 1:
        features.append("pale skin")
    if attr_dict.get("Rosy_Cheeks", -1) == 1:
        features.append("rosy cheeks")
    if attr_dict.get("High_Cheekbones", -1) == 1:
        features.append("high cheekbones")
    if attr_dict.get("Bushy_Eyebrows", -1) == 1:
        features.append("bushy eyebrows")
    elif attr_dict.get("Arched_Eyebrows", -1) == 1:
        features.append("arched eyebrows")
    if attr_dict.get("Narrow_Eyes", -1) == 1:
        features.append("narrow eyes")
    if attr_dict.get("Bags_Under_Eyes", -1) == 1:
        features.append("bags under eyes")
    if attr_dict.get("Pointy_Nose", -1) == 1:
        features.append("a pointy nose")
    elif attr_dict.get("Big_Nose", -1) == 1:
        features.append("a big nose")
    if attr_dict.get("Big_Lips", -1) == 1:
        features.append("big lips")
    if attr_dict.get("Chubby", -1) == 1 or attr_dict.get("Double_Chin", -1) == 1:
        features.append("a chubby face")

    # Accessories
    accessories = []
    if attr_dict.get("Eyeglasses", -1) == 1:
        accessories.append("eyeglasses")
    if attr_dict.get("Wearing_Hat", -1) == 1:
        accessories.append("a hat")
    if attr_dict.get("Wearing_Earrings", -1) == 1:
        accessories.append("earrings")
    if attr_dict.get("Wearing_Necklace", -1) == 1:
        accessories.append("a necklace")
    if attr_dict.get("Wearing_Necktie", -1) == 1:
        accessories.append("a necktie")
    if attr_dict.get("Wearing_Lipstick", -1) == 1:
        accessories.append("lipstick")
    if attr_dict.get("Heavy_Makeup", -1) == 1:
        accessories.append("makeup")

    # Sentence construction
    desc_clauses = []

    # Hair phrase
    if is_bald:
        desc_clauses.append("who is bald")
    elif hair_colors or hair_style:
        hair_desc_parts = []
        if hair_colors:
            hair_desc_parts.append(hair_colors[0])
        style_parts = [s for s in hair_style if s not in ("bangs", "a receding hairline")]
        if style_parts:
            hair_desc_parts.append(style_parts[0])

        hair_desc = " ".join(hair_desc_parts) + " hair" if hair_desc_parts else "hair"
        if "bangs" in hair_style:
            hair_desc += " with bangs"
        if "a receding hairline" in hair_style:
            hair_desc += " with a receding hairline"
        desc_clauses.append(f"with {hair_desc}")

    # Facial hair
    if facial_hair:
        desc_clauses.append("having " + " and ".join(facial_hair))

    # Facial features
    if features:
        top_features = features[:3]
        if "smiling" in top_features:
            top_features.remove("smiling")
            base_noun = f"smiling {base_noun}"
        if top_features:
            desc_clauses.append("with " + ", ".join(top_features))

    # Accessories
    if accessories:
        desc_clauses.append("wearing " + " and ".join(accessories[:2]))

    article = "An" if base_noun.startswith(("a", "e", "i", "o", "u")) else "A"
    if not desc_clauses:
        return f"{article} {base_noun}."

    if len(desc_clauses) == 1:
        return f"{article} {base_noun} {desc_clauses[0]}."
    else:
        return f"{article} {base_noun} {desc_clauses[0]}, {', '.join(desc_clauses[1:])}."


class SimpleTokenizer:
    """
    Lightweight character or word-level tokenizer for from-scratch Transformer encoder.
    Includes special tokens: <pad>, <sos>, <eos>, <unk>.
    """
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts: Dict[str, int] = {}

    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from list of text sentences."""
        for text in texts:
            tokens = self._tokenize_text(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words:
            if len(self.word2idx) >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def _tokenize_text(self, text: str) -> List[str]:
        # Lowercase and clean punctuation
        cleaned = text.lower().replace(".", " . ").replace(",", " , ")
        return [w.strip() for w in cleaned.split() if w.strip()]

    def encode(self, text: str, max_length: int = 32) -> List[int]:
        """Encode string to token indices with padding and truncation."""
        tokens = self._tokenize_text(text)
        token_ids = [self.word2idx[self.SOS_TOKEN]]
        for t in tokens[:max_length - 2]:
            token_ids.append(self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]))
        token_ids.append(self.word2idx[self.EOS_TOKEN])

        # Pad to max_length
        if len(token_ids) < max_length:
            token_ids += [self.word2idx[self.PAD_TOKEN]] * (max_length - len(token_ids))
        return token_ids[:max_length]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token indices back to readable string."""
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, self.UNK_TOKEN)
            if word in (self.SOS_TOKEN, self.PAD_TOKEN):
                continue
            if word == self.EOS_TOKEN:
                break
            words.append(word)
        return " ".join(words).replace(" .", ".").replace(" ,", ",")

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
