import mwxml, bz2, regex as re
from tqdm import tqdm

DUMP = r"C:\wiktionary\enwiktionary-latest-pages-articles.xml.bz2"
OUT = "hindi_pages.txt"

DEV = re.compile(r'^\p{Devanagari}+$')
pages = set()

with bz2.open(DUMP, "rb") as f:
    dump = mwxml.Dump.from_file(f)
    for page in tqdm(dump, desc="Scanning dump"):
        title = page.title.strip()
        if not DEV.match(title):
            continue

        for rev in page:
            if rev.text and "==Hindi==" in rev.text:
                pages.add(title)
            break

with open(OUT, "w", encoding="utf-8") as f:
    for p in sorted(pages):
        f.write(p + "\n")
