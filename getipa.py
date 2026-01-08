import requests
import time
import json
import logging
import re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# ================= CONFIG =================
INPUT_FILE = r"C:\Users\Kush al\Desktop\New folder\nepali_pages.txt"
OUTPUT_FILE = "nepali_word_ipa.tsv"
PROGRESS_FILE = "scrape_progress.json"
LOG_FILE = "scraper.log"

API_URL = "https://en.wiktionary.org/w/api.php"
SLEEP_SEC = 0.25
HEADERS = {"User-Agent": "NepaliIPA-Dataset/1.0 (research)"}

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= PROGRESS =================
def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_words):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_words), f, ensure_ascii=False)

# ================= API ====================
def get_wiktionary_html(page):
    try:
        r = requests.get(
            API_URL,
            params={
                "action": "parse",
                "page": page,
                "prop": "text",
                "format": "json",
                "formatversion": 2
            },
            headers=HEADERS,
            timeout=15
        )
        
        if r.status_code == 429:
            logger.warning(f"Rate limited, waiting 10s...")
            time.sleep(10)
            return get_wiktionary_html(page)
        
        if r.status_code != 200:
            return ""
        
        data = r.json()
        if "error" in data:
            return ""
            
        return data.get("parse", {}).get("text", "")
        
    except Exception:
        return ""

# ================= IPA EXTRACTION =================
def extract_nepali_ipa(html):
    """Extract IPA from Nepali section."""
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    ipas = []
    
    # Find the Nepali phonology Wikipedia link
    nepali_links = soup.find_all("a", class_="extiw", title="wikipedia:Nepali phonology")
    
    for link in nepali_links:
        # Go up to parent <li>
        li_parent = link.find_parent("li")
        if li_parent:
            # Find all IPA spans in this <li>
            for ipa_span in li_parent.find_all("span", class_="IPA"):
                ipa_text = ipa_span.get_text(strip=True)
                # Remove brackets and slashes
                ipa_text = ipa_text.strip("[]/ ")
                if ipa_text:
                    ipas.append(ipa_text)
    
    # Remove duplicates, keep order
    return list(dict.fromkeys(ipas))

# ================= MAIN =================
def main():
    if not Path(INPUT_FILE).exists():
        logger.error(f"Input file '{INPUT_FILE}' not found!")
        return
    
    with open(INPUT_FILE, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(words)} words")
    
    processed = load_progress()
    remaining = [w for w in words if w not in processed]
    
    if processed:
        logger.info(f"Resuming: {len(processed)} done, {len(remaining)} remaining")
    
    stats = {"found": 0, "not_found": 0, "errors": 0}
    
    mode = 'a' if processed else 'w'
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:
        if mode == 'w':
            out.write("word\tipa\tsource\n")
        
        for word in tqdm(remaining, desc="Fetching IPA"):
            html = get_wiktionary_html(word)
            
            if not html:
                out.write(f"{word}\t\tERROR\n")
                stats["errors"] += 1
            else:
                ipas = extract_nepali_ipa(html)
                
                if not ipas:
                    out.write(f"{word}\t\tNO_IPA\n")
                    stats["not_found"] += 1
                else:
                    for ipa in ipas:
                        out.write(f"{word}\t{ipa}\tWIKTIONARY\n")
                    stats["found"] += 1
                    logger.info(f"'{word}' → {ipas}")
            
            processed.add(word)
            if len(processed) % 50 == 0:
                save_progress(processed)
            
            time.sleep(SLEEP_SEC)
    
    save_progress(processed)
    
    logger.info(f"Done! Results in {OUTPUT_FILE}")
    logger.info(f"Found: {stats['found']}, No IPA: {stats['not_found']}, Errors: {stats['errors']}")
    
    if len(processed) == len(words):
        Path(PROGRESS_FILE).unlink(missing_ok=True)
        logger.info("Progress file cleaned up")

if __name__ == "__main__":
    main()