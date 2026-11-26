import re
import unicodedata

# Common Cyrillic / Greek homoglyphs mapped to Latin equivalents
HOMOGLYPH_MAP = {
    # Cyrillic lowercase
    "\u0430": "a",  # а -> a
    "\u0435": "e",  # е -> e
    "\u0451": "e",  # ё -> e
    "\u0456": "i",  # і -> i
    "\u0457": "i",  # ї -> i
    "\u0458": "j",  # ј -> j
    "\u043e": "o",  # о -> o
    "\u0440": "p",  # р -> p
    "\u0441": "c",  # с -> c
    "\u0443": "y",  # у -> y
    "\u0445": "x",  # х -> x
    "\u0455": "s",  # ѕ -> s
    "\u0501": "d",  # ԁ -> d
    "\u051b": "q",  # ԛ -> q
    "\u051d": "w",  # ԝ -> w
    # Cyrillic uppercase
    "\u0410": "A",  # А -> A
    "\u0412": "B",  # В -> B
    "\u0415": "E",  # Е -> E
    "\u041a": "K",  # К -> K
    "\u041c": "M",  # М -> M
    "\u041d": "H",  # Н -> H
    "\u041e": "O",  # О -> O
    "\u0420": "P",  # Р -> P
    "\u0421": "C",  # С -> C
    "\u0422": "T",  # Т -> T
    "\u0425": "X",  # Х -> X
    "\u04ae": "Y",  # Ү -> Y
    "\u0406": "I",  # І -> I
    # Greek lowercase / uppercase
    "\u03b1": "a",  # α -> a
    "\u03b5": "e",  # ε -> e
    "\u03b9": "i",  # ι -> i
    "\u03bf": "o",  # ο -> o
    "\u03c1": "p",  # ρ -> p
    "\u03c2": "c",  # ς -> c
    "\u0391": "A",  # Α -> A
    "\u0392": "B",  # Β -> B
    "\u0395": "E",  # Ε -> E
    "\u0397": "H",  # Η -> H
    "\u0399": "I",  # Ι -> I
    "\u039a": "K",  # Κ -> K
    "\u039c": "M",  # Μ -> M
    "\u039d": "N",  # Ν -> N
    "\u039f": "O",  # Ο -> O
    "\u03a1": "P",  # Ρ -> P
    "\u03a4": "T",  # Τ -> T
    "\u03a7": "X",  # Χ -> X
}
HOMOGLYPH_TRANS = str.maketrans(HOMOGLYPH_MAP)

# Defanged URL pattern: matches hxxp(s)://..., http(s)://..., or domain with [.]
URL_PATTERN = re.compile(
    r"(?:hxxps?|https?|ftp)://\S+|(?:[a-zA-Z0-9-]+\[\.\])+[a-zA-Z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)

# Digit runs (ASCII and Bengali digits)
DIGIT_RUN_PATTERN = re.compile(r"[0-9\u09e6-\u09ef]+(?:[.,][0-9\u09e6-\u09ef]+)*")

# Latin uppercase letters only (preserves Bangla Unicode untouched)
LATIN_UPPER_PATTERN = re.compile(r"[A-Z]+")


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Unicode NFKC normalization
    t = unicodedata.normalize("NFKC", text)

    # 2. Remove only \u200b, \u200c, \ufeff. Keep \u200d (ZWJ for Bangla).
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")

    # 3. Map common Cyrillic/Greek homoglyphs to Latin
    t = t.translate(HOMOGLYPH_TRANS)

    # 4. Replace defanged URLs with [URL] token
    t = URL_PATTERN.sub(" [URL] ", t)

    # 5. Lowercase Latin text only (leaves Bangla and [URL] handled cleanly)
    t = LATIN_UPPER_PATTERN.sub(lambda m: m.group(0).lower(), t)
    t = t.replace("[url]", "[URL]")

    # 6. Replace digit runs with [NUM] token
    t = DIGIT_RUN_PATTERN.sub(" [NUM] ", t)

    # 7. Preserve "|||" as a spaced separator token and collapse repeated whitespace
    t = t.replace("|||", " ||| ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def run_unit_tests():
    # Test 1: Homoglyphs (Cyrillic о, а, і, е, р, с)
    s1 = "Cоngratulаtіоns! Рay Tаkа 500 via bKаsh"
    out1 = preprocess_text(s1)
    assert "congratulations!" in out1, f"Failed homoglyph: {out1}"
    assert "pay taka [NUM] via bkash" in out1, f"Failed homoglyph+num: {out1}"

    # Test 2: Zero-width chars (\u200b, \u200c removed; \u200d kept!)
    s2 = "ব\u200cসে Trx\u200bID\ufeff র\u200d্যাংক"
    out2 = preprocess_text(s2)
    assert "\u200b" not in out2 and "\u200c" not in out2 and "\ufeff" not in out2, f"Failed ZW strip: {out2}"
    assert "\u200d" in out2, f"Failed to preserve ZWJ \\u200d: {out2}"
    assert "trxid" in out2, f"Failed Latin lowercase: {out2}"

    # Test 3: Defanged URLs and conversation separator |||
    s3 = "S: Click hxxps://gp-gift[.]xyz/claim ||| R: Okay 12345"
    out3 = preprocess_text(s3)
    assert "[URL]" in out3, f"Failed URL token: {out3}"
    assert "|||" in out3, f"Failed separator preservation: {out3}"
    assert "[NUM]" in out3, f"Failed NUM token: {out3}"

    # Test 4: Bangla text + Bengali digits
    s4 = "আপনার ৫০০ টাকা ক্যাশ আউট সফল হয়েছে।"
    out4 = preprocess_text(s4)
    assert "আপনার [NUM] টাকা ক্যাশ আউট সফল হয়েছে।" == out4, f"Failed Bangla: {out4}"

    print("All preprocess_text unit tests PASSED!")


if __name__ == "__main__":
    run_unit_tests()
