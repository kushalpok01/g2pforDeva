import regex as re

INPUT = "hindidata/hindi_dataset.tsv"
OUTPUT = "hindidata/hindi_preprocessed.tsv"

# ---- IPA NORMALIZATION RULES ----
IPA_MAP = {
    "ä": "a",
    "ɾ": "r",
    "ɦ": "ɦ",  
}

DEV_RE = r"^\p{Devanagari}+$"

def normalize_ipa(ipa):
    for k, v in IPA_MAP.items():
        ipa = ipa.replace(k, v)
    return ipa.strip()

clean = set()

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        word = parts[0]
        ipa = normalize_ipa(parts[1])

        if " " in word:
            continue
        if not re.match(DEV_RE, word):
            continue

        clean.add((word, ipa))

with open(OUTPUT, "w", encoding="utf-8") as f:
    for w, i in sorted(clean):
        f.write(f"{w}\t{i}\n")

print(f"Saved {len(clean)} clean entries")
