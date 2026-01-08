import requests
import time
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
INPUT_FILE = "hindi_pages.txt"
OUTPUT_FILE = "hindidataset.tsv"
PROGRESS_FILE = "scrape_progress.json"
LOG_FILE = "scraper.log"

API_URL = "https://en.wiktionary.org/w/api.php"
SLEEP_SEC = 0.25  # polite rate (4 req/sec)
HEADERS = {
    "User-Agent": "hindiIPA-Dataset/1.0 (research)"
}

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
    """Load already processed words."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_words):
    """Save progress to resume later."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_words), f, ensure_ascii=False)

# ================= API ====================
def get_ipa_from_template(word):
    """
    Get IPA by expanding the {{ne-IPA|word}} template.
    Returns list of IPA strings.
    """
    try:
        r = requests.get(
            API_URL,
            params={
                "action": "expandtemplates",
                "text": f"{{{{hi-IPA|{word}}}}}",
                "prop": "wikitext",
                "format": "json",
                "formatversion": 2
            },
            headers=HEADERS,
            timeout=15
        )
        
        if r.status_code == 429:
            logger.warning(f"Rate limited on '{word}', waiting 10s...")
            time.sleep(10)
            return get_ipa_from_template(word)  # retry
        
        if r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} for '{word}'")
            return []
        
        data = r.json()
        if "error" in data:
            logger.warning(f"API error for '{word}': {data['error'].get('info', '')}")
            return []
        
        # Extract wikitext
        wikitext = data.get("expandtemplates", {}).get("wikitext", "")
        
        # Extract IPA from the expanded template
        return extract_ipa_from_wikitext(wikitext)
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout for '{word}'")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for '{word}': {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for '{word}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error for '{word}': {e}")
        return []

def extract_ipa_from_wikitext(wikitext):
    """
    Extract IPA strings from expanded template wikitext.
    Example: <span class="IPA nowrap">[nepäl]</span>
    """
    if not wikitext:
        return []
    
    # Pattern to match IPA in brackets within span tags
    # Matches: [nepäl] or /nepäl/ within <span class="IPA...">
    pattern = r'<span[^>]*class="IPA[^"]*"[^>]*>[\s]*([/\[]([^\]/\[]+)[\]/])[\s]*</span>'
    
    matches = re.findall(pattern, wikitext)
    ipas = [match[1].strip() for match in matches if match[1].strip()]
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(ipas))

def get_page_wikitext(word):
    """
    Fallback: Get the raw wikitext of a page to check if {{ne-IPA}} exists.
    """
    try:
        r = requests.get(
            API_URL,
            params={
                "action": "query",
                "titles": word,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "format": "json",
                "formatversion": 2
            },
            headers=HEADERS,
            timeout=15
        )
        
        if r.status_code != 200:
            return ""
        
        data = r.json()
        pages = data.get("query", {}).get("pages", [])
        
        if not pages or "missing" in pages[0]:
            return ""
        
        return pages[0].get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
        
    except Exception as e:
        logger.error(f"Error getting wikitext for '{word}': {e}")
        return ""

def has_nepali_section(wikitext):
    """Check if page has a Nepali section."""
    return bool(re.search(r'==\s*Hindi\s*==', wikitext))

def extract_ipa_from_page(word):
    """
    Try multiple methods to get IPA:
    1. Expand {{hi-IPA|word}} template
    2. Check page wikitext for template usage
    """
    # Method 1: Try expanding template directly
    ipas = get_ipa_from_template(word)
    if ipas:
        return ipas
    
    # Method 2: Check if page exists and has Nepali section
    wikitext = get_page_wikitext(word)
    if not wikitext:
        logger.debug(f"Page '{word}' not found")
        return []
    
    if not has_nepali_section(wikitext):
        logger.debug(f"No Hindi section in '{word}'")
        return []
    
    # Look for {{ne-IPA}} template in the wikitext
    template_matches = re.findall(r'\{\{hi-IPA\|([^}]+)\}\}', wikitext)
    
    for template_word in template_matches:
        template_word = template_word.strip()
        ipas = get_ipa_from_template(template_word)
        if ipas:
            return ipas
    
    logger.debug(f"No IPA template found for '{word}'")
    return []

# ================= MAIN =================
def main():
    # Check input file exists
    if not Path(INPUT_FILE).exists():
        logger.error(f"Input file '{INPUT_FILE}' not found!")
        return
    
    # Load words
    with open(INPUT_FILE, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(words)} words from {INPUT_FILE}")
    
    # Load progress
    processed = load_progress()
    remaining = [w for w in words if w not in processed]
    
    if processed:
        logger.info(f"Resuming: {len(processed)} already done, {len(remaining)} remaining")
    
    # Open output file (append if resuming)
    mode = 'a' if processed else 'w'
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:
        # Write header if new file
        if mode == 'w':
            out.write("word\tipa\tsource\n")
        
        # Process words
        for word in tqdm(remaining, desc="Fetching IPA"):
            ipas = extract_ipa_from_page(word)
            
            if not ipas:
                out.write(f"{word}\t\tNO_IPA\n")
                logger.debug(f"No IPA for '{word}'")
            else:
                for ipa in ipas:
                    out.write(f"{word}\t{ipa}\tWIKTIONARY\n")
                logger.info(f"'{word}' → {ipas}")
            
            # Update progress
            processed.add(word)
            if len(processed) % 100 == 0:  # save every 100 words
                save_progress(processed)
            
            time.sleep(SLEEP_SEC)
    
    # Final progress save
    save_progress(processed)
    logger.info(f"✓ Done! Results in {OUTPUT_FILE}")
    
    # Clean up progress file if complete
    if len(processed) == len(words):
        Path(PROGRESS_FILE).unlink(missing_ok=True)
        logger.info("Progress file cleaned up")

# ================= RUN =================
if __name__ == "__main__":
    main()