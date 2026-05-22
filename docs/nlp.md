## `nlp/` — Multilingual Text Preprocessing & Analysis 🌍✂️

A production-ready NLP toolkit for English and Persian (Farsi) text: normalization, tokenization, stopword removal, Unicode sanitization, and frequency analysis. Built for fast prototyping on multilingual ML projects.

---

### 🎯 Quick Start

```python
from yumbox.nlp import Preprocessor

# Create a preprocessor with sensible defaults for English text
preproc = Preprocessor(
    remove_punctuation=True,
    remove_en_stopwords=True,
    do_en_normalizer=True,      # lowercasing
    do_en_lemmatizer=True,      # WordNet lemmatization
    do_hard_limit=True,         # filter words by length
    low_hard_limit=3,
    high_hard_limit=26,
)

# Process a single string
text = "The quick brown foxes jumped! 🦊"
clean = preproc(text)
# → "quick brown fox jump"

# Process a DataFrame column
import pandas as pd
df = pd.DataFrame({"review": ["Great product!", "Terrible 😠", "OK I guess"]})
df["clean"] = df["review"].apply(preproc)
```

#### Persian (Farsi) Support
```python
# Preprocessor with Persian-specific options
fa_preproc = Preprocessor(
    normalize_fa_chars=True,    # Normalize Arabic/Farsi character variants
    remove_fa_stopwords=True,   # Remove common Farsi stopwords via hazm
    do_fa_normalizer=True,      # hazm.Normalizer for Farsi text
    fa_tokenizer=True,          # Use hazm.word_tokenize (splits by script family)
)

fa_text = "این یک متن فارسی است. خرید و فروش کالاهای مختلف."
clean_fa = fa_preproc(fa_text)
# → "متن فارسی خرید فروش کالاهای مختلف"
```

---

### 🧰 Core Components Reference

#### `Preprocessor` — The All-in-One Text Cleaner

A configurable, flag-driven preprocessor that chains common NLP operations. All options are boolean flags — enable only what you need.

##### Initialization Flags (English)

| Flag | Purpose | Default |
|------|---------|---------|
| `remove_punctuation` | Strip `string.punctuation` chars | `True` |
| `remove_en_stopwords` | Remove NLTK English stopwords | `True` |
| `do_en_normalizer` | Lowercase text | `False` |
| `do_en_stemmer` | Porter stemming via NLTK | `False` |
| `do_en_lemmatizer` | WordNet lemmatization | `False` |
| `en_lemmatize_method` | `"all"`, `"verbs"`, or `"known"` (POS-aware) | `"all"` |
| `en_tokenizer` | Use `nltk.word_tokenize` instead of split | `False` |
| `en_sentence_tokenizer` | Use `nltk.sent_tokenize` | `False` |
| `remove_singular_digis` | Remove standalone digits (e.g., "5") | `False` |
| `do_hard_limit` | Filter words by character length | `False` |
| `low_hard_limit` / `high_hard_limit` | Min/max word length when filtering | `3` / `26` |

##### Initialization Flags (Persian/Farsi)

| Flag | Purpose | Default |
|------|---------|---------|
| `normalize_fa_chars` | Map Arabic variants to Farsi equivalents (ه→ه, ی→ی) | `True` |
| `remove_fa_stopwords` | Remove Farsi stopwords via `hazm` | `True` |
| `do_fa_normalizer` | Apply `hazm.Normalizer` (digit conversion, spacing) | `False` |
| `do_fa_stemmer` / `do_fa_lemmatizer` | hazm stemming/lemmatization | `False` |
| `fa_tokenizer` / `fa_sentence_tokenizer` | Use hazm tokenizers (script-aware) | `False` |

##### Unicode & Character Filtering

| Flag | Purpose |
|------|---------|
| `only_eng_unicode_range` | Keep only English letters (A-Z, a-z) |
| `only_fa_unicode_range` | Keep only Persian/Arabic script ranges |
| `only_num_unicode_range` | Keep only digits (0-9) |
| `remove_cjk` | Strip CJK Unified Ideographs (Chinese/Japanese/Korean) |
| `strip_accents` | Remove diacritics (é→e, ü→u) via Unicode NFD |
| `remove_ctrl_chars` | Strip control/format characters (Unicode categories C, M, P) |
| `remove_special_chars` | Remove chars defined in `Chars.get_burn_chars()` |

##### Web & HTML Preprocessing

| Flag | Purpose |
|------|---------|
| `parse_html` | Strip HTML tags via `yumbox.scraper.parse_html` |
| `remove_uri` / `remove_www` | Remove URLs and www.* domains |
| `split_group_annots` | Split on `/` and `,` (useful for product specs) |
| `remove_parentheticals` | Remove text in `()` or `[]` |

##### Calling the Preprocessor
```python
preproc = Preprocessor(...)
result = preproc(text, keep_lf=False)  # keep_lf preserves newlines between sentences
```

> 💡 **Pro tip**: Instantiate `Preprocessor` once and reuse it — heavy initializations (NLTK downloads, hazm models) happen in `__init__`, where you can `RUN` it in your Dockerfile to cache downloaders as a layer.

---

#### `chars.py` — Unicode Utilities

Low-level character handling for robust multilingual text processing.

##### `UnicodeRanges` — Regex from Codepoint Ranges
```python
from yumbox.nlp.chars import UnicodeRanges

# Build regex to match ONLY English letters
eng_regex = UnicodeRanges.regex_from_range(
    UnicodeRanges.english_letters_ranges, 
    negate=False  # False = match these ranges; True = exclude them
)
# → re.compile(r'[\u0041-\u005B\u0061-\u007B]')

# Build regex to REMOVE CJK characters
cjk_remove = UnicodeRanges.regex_cjk_chars()
text = re.sub(cjk_remove, " ", text)
```

##### `Unicode` — Category-Based Sanitization
```python
from yumbox.nlp.chars import Unicode

# Remove accents: é → e, ñ → n
clean = Unicode.strip_accents("café naïve")  # → "cafe naive"

# Remove control/format/punctuation chars (keep letters, numbers, symbols, separators)
clean = Unicode.sanitize_cmp("Hello\u200b\u00ad\u0009World!")  
# → "Hello World"  (zero-width chars and tab removed)

# Remove specific Unicode blocks (e.g., Hangul Jamo, Cyrillic)
clean = Unicode.sanitize_char_sets("Привет мир")  # → "  " (Cyrillic removed)
```

##### `Chars` — Character Mappings & Burn Lists
```python
from yumbox.nlp.chars import Chars

# Get all character normalization mappings (Arabic→Farsi, digits, etc.)
for mapping in Chars.get_mappings():
    for src, dst in mapping:
        text = text.replace(src, dst)
# Example: "۱۲۳" → "123" (Eastern Arabic digits to ASCII)

# Get regex pattern for "burn chars" (chars to always remove)
burn_pattern = Chars.get_burn_chars()  # → r"[\u200e\u200f\u200c®©™...]"
text = re.sub(burn_pattern, " ", text)

# Get CJK removal pattern
cjk = Chars.get_cjk()  # → re.compile(r'[\u4e00-\u9fff]')
```

---

#### `tools.py` — Frequency Analysis & Name Resolution

##### `MapRed` — Simple Map-Reduce for Word/Char Frequencies
```python
from yumbox.nlp.tools import MapRed

# Count word frequencies
mr = MapRed()
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    mr(word)

print(mr.dictionary)  # → {'apple': 3, 'banana': 2, 'cherry': 1}

# Filter by frequency
rare = mr.filter(2, op="lt")  # → {'cherry': 1}
common = mr.filter(2, op="gte")  # → {'apple': 3, 'banana': 2}

# Get pandas DataFrame for analysis
df = mr.df  # → DataFrame with columns: word, freq

# Set low-frequency threshold for fast lookup
mr.set_lowfreq(1)  # mr.lowfreq is now a set: {'cherry'}

# Sort by frequency
mr.sort("desc")  # Sort dictionary in-place: highest freq first
```

##### `defaultname` — Name Resolution / Alias Clustering
Resolve multiple aliases to a canonical name (e.g., product names, entity resolution).

```python
from yumbox.nlp.tools import defaultname

resolver = defaultname()

# Add alias pairs: (canonical, alias)
resolver.add("iPhone 15", "iphone15")
resolver.add("iPhone 15", "iphone-15")
resolver.add("iPhone 15", "Apple iPhone 15")

# Resolve any alias to canonical
print(resolver["iphone-15"])  # → "iPhone 15"
print(resolver["Apple iPhone 15"])  # → "iPhone 15"

# Get all aliases for a canonical name
cluster = resolver.get_cluster("iPhone 15")
# → {"iPhone 15", "iphone15", "iphone-15", "Apple iPhone 15"}

# Export as dataset (canonical → list of aliases)
dataset = resolver.get_dataset()
# → {"iPhone 15": ["iphone15", "iphone-15", "Apple iPhone 15"]}
```

> ⚡ **Performance note**: `defaultname` uses incremental backref updates — O(1) amortized per `add()` call, even for million-scale alias graphs.

---

#### `processors.py` — Legacy Functions & Analysis Helpers

> ⚠️ Most functions here are deprecated in favor of the `Preprocessor` class. Use for reference or one-off scripts.

| Function | Purpose | Replacement |
|----------|---------|------------|
| `sentence_preprocessor` | Basic text cleaning pipeline | `Preprocessor()` |
| `fa_sent_normalizer` / `en_sent_normalizer` | Language-specific tokenization + stopword removal | `Preprocessor(fa_*/en_* flags)` |
| `product_title_preprocessor` | E-commerce title cleanup | Custom `Preprocessor` config |
| `word_post_process` / `char_post_process` | Filter low-freq words/chars via `MapRed` | `MapRed.set_lowfreq()` + list comprehension |
| `run_analysis` / `run_post_process` | CLI-style corpus analysis pipelines | Build custom pipeline with `Preprocessor` + `MapRed` |

##### Quick Analysis Helpers
```python
from yumbox.nlp.processors import analyze_topten, analyze_head
from yumbox.nlp.tools import MapRed

# Top/bottom 10 by frequency
mr = MapRed()
# ... populate mr ...
analyze_topten(mr)  # Prints top 10 desc, then top 10 asc

# Analyze DataFrame sorted by length or frequency
df = mr.df
analyze_head(df, sort_key="freq")  # Shows top 2000 rows by frequency
```

---

### 🔁 Common Patterns

#### Multilingual Product Description Pipeline
```python
from yumbox.nlp import Preprocessor

# Configure for e-commerce text (mixed EN/FA, HTML, specs)
preproc = Preprocessor(
    # HTML & web cleanup
    parse_html=True,
    remove_uri=True,
    remove_www=True,
    split_group_annots=True,  # Split "Brand/Model/Color"
    
    # Character normalization
    normalize_fa_chars=True,
    strip_accents=True,
    remove_cjk=True,  # Remove Chinese spam
    
    # Language-agnostic cleaning
    remove_punctuation=True,
    remove_parentheticals=True,  # Remove (SKU: XYZ)
    do_hard_limit=True,
    low_hard_limit=2,
    high_hard_limit=30,
    
    # English processing
    do_en_normalizer=True,
    remove_en_stopwords=True,
    
    # Farsi processing (optional, enable if FA text detected)
    # remove_fa_stopwords=True,
    # do_fa_normalizer=True,
)

# Process product titles/descriptions
df["clean_text"] = df["description"].apply(preproc)
```

#### Incremental Corpus Cleaning with Frequency Filtering
```python
from yumbox.nlp import Preprocessor
from yumbox.nlp.tools import MapRed

# Phase 1: Preprocess and collect frequencies
preproc = Preprocessor(remove_en_stopwords=False)  # Keep stopwords for freq analysis
mr = MapRed()

for text in corpus:
    clean = preproc(text)
    for word in clean.split():
        mr(word)

# Phase 2: Filter low-frequency words
mr.set_lowfreq(5)  # Words appearing ≤5 times are "low freq"

# Phase 3: Re-process with stopwords + low-freq removal
preproc_final = Preprocessor(
    remove_en_stopwords=True,
    # Custom filter: skip words in mr.lowfreq
)

def filter_lowfreq(text, mr_lowfreq):
    return " ".join(w for w in text.split() if w not in mr_lowfreq)

df["final"] = df["raw"].apply(
    lambda t: filter_lowfreq(preproc_final(t), mr.lowfreq)
)
```

#### Entity Resolution for Product Catalogs
```python
from yumbox.nlp.tools import defaultname

# Resolve product name variants
resolver = defaultname()

# Seed with known canonical names
catalog = {
    "Sony WH-1000XM5": ["wh1000xm5", "sony-wh-1000xm5", "WH-1000XM5 headphones"],
    "Apple AirPods Pro": ["airpods-pro", "apple-airpods-pro-2nd-gen"],
}

for canonical, aliases in catalog.items():
    for alias in aliases:
        resolver.add(canonical, alias)

# Apply to dataset
def resolve_product(name):
    return resolver[name]  # Returns canonical or original if unknown

df["canonical_name"] = df["product_name"].apply(resolve_product)

# Group by canonical name for aggregation
grouped = df.groupby("canonical_name").agg({
    "price": "mean",
    "rating": "mean",
    "review_count": "sum"
})
```

#### Unicode-Aware Text Filtering
```python
from yumbox.nlp.chars import UnicodeRanges
import re

# Keep ONLY English letters + digits (remove everything else)
keep_regex = UnicodeRanges.regex_from_range(
    UnicodeRanges.english_letters_ranges + UnicodeRanges.numerical_digits_ranges,
    negate=True  # Negate = remove chars NOT in these ranges
)

def keep_eng_num(text):
    return re.sub(keep_regex, " ", text)

# Use in preprocessing pipeline
clean = keep_eng_num("Product: 苹果 🍎 (iPhone 15) - $999")
# → "Product  iPhone    "
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `hazm` requires Persian language models | Run `python -m hazm download` once, or handle `LookupError` gracefully |
| NLTK resources not downloaded | Call `init_nltk()` once at startup, or use `nltk.download(..., quiet=True)` |
| `do_fa_normalizer` converts English digits to Farsi | Only enable if you want Farsi digits; otherwise skip or post-convert back |
| `Preprocessor` is stateful (loads models in `__init__`) | Instantiate once per process; don't recreate in tight loops |
| `MapRed.set_lowfreq()` creates a `set` for O(1) lookup | Always call this before filtering in production — don't use `filter()` in loops |
| Unicode regex ranges are exclusive on end | `UnicodeRanges.english_letters_ranges` uses `(0x0041, 0x005B)` for A-Z (0x5B = '[' not included) |

#### Pro Tips
```python
# Tip 1: Detect language and switch preprocessor dynamically
def smart_preproc(text, en_preproc, fa_preproc):
    # Simple heuristic: count Farsi/Arabic codepoints
    fa_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if fa_chars / max(len(text), 1) > 0.3:
        return fa_preproc(text)
    return en_preproc(text)

# Tip 2: Cache preprocessing results to disk using Yumbox (survives restarts!)
from yumbox.cache import cache_kwargs_hash
from yumbox.config import BFG

BFG["cache_dir"] = "~/.cache/nlp_preproc"  # Auto-creates directory

@cache_kwargs_hash  # Hashes kwargs → saves to {func}_{md5}.pkl
def cached_preproc(text: str, mode: str = "auto") -> str:
    return smart_preproc(text, en_preproc, fa_preproc)

# Usage: First call computes & persists to disk. Subsequent runs load instantly.
result = cached_preproc("Long corpus text...", mode="mixed")

# Tip 3: Use MapRed for vocabulary pruning before tokenization
mr = MapRed()
for doc in corpus:
    for word in doc.split():
        mr(word)
mr.set_lowfreq(3)
# Now filter: [w for w in text.split() if w not in mr.lowfreq]

# Tip 4: Combine defaultname with Preprocessor for entity-aware cleaning
resolver = defaultname()
# ... populate resolver ...

def entity_aware_preproc(text, preproc, resolver):
    # 1. Extract potential entities (simple: capitalize words)
    # 2. Resolve to canonical names
    # 3. Preprocess the rest
    ...
```

---

### 📦 Module Organization

```
nlp/
├── __init__.py      # Preprocessor class, tokenizers, init helpers
├── chars.py         # Unicode ranges, character mappings, sanitization
├── processors.py    # Legacy functions, analysis helpers (deprecated)
├── tools.py         # MapRed (frequency), defaultname (alias resolution)
└── stop_words.txt   # Custom English stopwords (loaded by Preprocessor)
```

> 💡 **Why two normalization paths?** `Preprocessor` is the modern, configurable entry point. Legacy functions in `processors.py` remain for backward compatibility and quick scripts. For new code, always prefer `Preprocessor`.

---

### 🔐 Security & Best Practices

```python
# Never preprocess untrusted input without sanitization first
def safe_preproc(text, preproc):
    # 1. Truncate to avoid ReDoS on regex-heavy preprocessors
    text = text[:10000] if text else ""
    # 2. Remove null bytes and other injection vectors
    text = text.replace("\x00", "")
    # 3. Then preprocess
    return preproc(text)

# Log preprocessing stats for monitoring
from collections import Counter

def monitored_preproc(text, preproc, stats_counter):
    original_len = len(text)
    result = preproc(text)
    
    stats_counter["chars_removed"] += original_len - len(result)
    stats_counter["words_in_result"] += len(result.split())
    
    return result
```

---

> 🍱 **Yumbox Philosophy**: Text preprocessing should be *configurable, composable, and fast*. If you need a new normalization rule, add a flag to `Preprocessor`. If you need a new character mapping, extend `Chars`.

Happy preprocessing! 🚀 If you spot a missing language feature or a performance bottleneck, open an issue! If your use case doesn't fit, the code is intentionally modular — extend away.
