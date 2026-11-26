import re
import random

HOMOGLYPH_SWAPS = {
    "a": "\u0430",  # Cyrillic а
    "e": "\u0435",  # Cyrillic е
    "i": "\u0456",  # Cyrillic і
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "c": "\u0441",  # Cyrillic с
    "x": "\u0445",  # Cyrillic х
    "y": "\u0443",  # Cyrillic у
}

LEET_MAP = {
    "a": "4",
    "e": "3",
    "i": "1",
    "o": "0",
    "s": "5",
}

EMOJIS = ["📲", "🔥", "⚠️", "🎁", "💰", "✅", "🚨", "💸", "📦", "🎉"]
NOISE_STRINGS = ["wkrjfejejrn", "mnbvcx", "zzqwpo", "xkcdpl", "asdkjh", "qzxvbn", "lkjsdfoiu"]

BANGLISH_VARIANTS = [
    ("bhul", "vul"),
    ("korun", "koren"),
    ("bolen", "bolun"),
    ("pathan", "pathao"),
    ("taka", "tk"),
    ("apnar", "apnr"),
    ("ekhoni", "ekhn e"),
    ("taratari", "tara tari"),
    ("jitechen", "jitsen"),
    ("হয়েছে", "হয়েছে"),
    ("আয়", "আয়"),
    ("করুন", "করেন"),
    ("যোগ", "জোগ"),
]


def _corrupt_word(word: str, rng: random.Random, prob: float) -> str:
    if len(word) <= 2 or word in {"|||", "S:", "R:", "A:", "B:"} or word.startswith("http") or word.startswith("hxxp"):
        return word

    w = word
    # Bangla/Banglish variant
    if rng.random() < prob * 0.5:
        lw = w.lower()
        for src, dst in BANGLISH_VARIANTS:
            if lw == src:
                w = dst
                break
            elif lw == dst:
                w = src
                break

    # Zero-width char (\u200b or \u200c) inside word (works for both Latin & Bangla)
    if len(w) >= 4 and rng.random() < prob * 0.7:
        idx = rng.randint(1, len(w) - 1)
        zw = rng.choice(["\u200b", "\u200c"])
        w = w[:idx] + zw + w[idx:]

    # Latin-specific corruptions
    if re.search(r"[a-zA-Z]", w):
        chars = list(w)
        r = rng.random()
        if r < prob * 0.35 and len(chars) >= 4:
            # Swap two adjacent chars
            idx = rng.randint(1, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif r < prob * 0.6 and len(chars) >= 4:
            # Drop a char
            idx = rng.randint(1, len(chars) - 2)
            chars.pop(idx)
        elif r < prob * 0.85:
            # Repeat a char
            idx = rng.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx] * rng.randint(1, 2))
        w = "".join(chars)

        # Leet-speak or homoglyph swap
        if rng.random() < prob * 0.45:
            new_chars = []
            for ch in w:
                lch = ch.lower()
                if lch in HOMOGLYPH_SWAPS and rng.random() < 0.3:
                    new_chars.append(HOMOGLYPH_SWAPS[lch])
                elif lch in LEET_MAP and rng.random() < 0.25:
                    new_chars.append(LEET_MAP[lch])
                else:
                    new_chars.append(ch)
            w = "".join(new_chars)

        # Random capitalization
        if rng.random() < prob * 0.4:
            w = "".join(c.upper() if rng.random() < 0.4 else c.lower() for c in w)
    else:
        # Bangla script character drop / repeat / space split
        if len(w) >= 4 and rng.random() < prob * 0.35:
            idx = rng.randint(1, len(w) - 2)
            if rng.random() < 0.5:
                w = w[:idx] + " " + w[idx:]
            else:
                w = w[:idx] + w[idx + 1:]

    return w


def apply_noise(text: str, rng: random.Random, level: float = 0.3) -> str:
    """Apply realistic adversarial noise to English, Banglish, or Bangla text."""
    if not isinstance(text, str) or not text.strip():
        return text

    # Preserve ||| turns
    turns = text.split("|||")
    noisy_turns = []

    for turn in turns:
        tokens = turn.strip().split(" ")
        new_tokens = []
        for tok in tokens:
            # Defanged URL variation
            if "http" in tok.lower() or "[.]" in tok:
                if rng.random() < level:
                    tok = tok.replace("https://", "hxxps://").replace("http://", "hxxp://")
                    if "[.]" not in tok:
                        tok = tok.replace(".", "[.]")
                new_tokens.append(tok)
                continue

            if rng.random() < level:
                tok = _corrupt_word(tok, rng, prob=min(0.9, level + 0.2))
            new_tokens.append(tok)

            # Occasionally drop space between two tokens
            if len(new_tokens) >= 2 and rng.random() < level * 0.12:
                prev = new_tokens.pop()
                new_tokens[-1] = new_tokens[-1] + prev

        # Insert a random noise string
        if len(new_tokens) >= 3 and rng.random() < level * 0.4:
            pos = rng.randint(1, len(new_tokens) - 1)
            new_tokens.insert(pos, rng.choice(NOISE_STRINGS))

        # Insert an emoji
        if rng.random() < level * 0.5:
            pos = rng.choice([0, len(new_tokens)])
            new_tokens.insert(pos, rng.choice(EMOJIS))

        # Extra spaces occasionally
        joiner = "  " if rng.random() < level * 0.15 else " "
        noisy_turns.append(joiner.join(new_tokens))

    return " ||| ".join(noisy_turns)


if __name__ == "__main__":
    rng = random.Random(42)
    for sample in [
        "bKash Helpline: Apnar account verify korte PIN ar OTP pathan.",
        "আপনার পার্সেল আটকে আছে। ডেলিভারি চার্জ ৫০ টাকা পরিশোধ করুন।",
        "S: Hello, this is DHL ||| R: I didn't order anything ||| S: Pay Tk 2500 customs",
    ]:
        print("CLEAN:", sample)
        print("NOISY:", apply_noise(sample, rng, level=0.4))
        print("-" * 60)
