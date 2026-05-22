## `scraper/` — Web Scraping Utilities & Image Downloading 🕷️🖼️

Production-ready tools for HTML parsing, text extraction, URL manipulation, and robust image downloading with parallel processing and corruption detection.

---

### 🎯 Quick Start

```python
from yumbox.scraper import parse_html, html_to_text, ImageDownloader

# Extract clean text from HTML
html_content = "<div><h1>Product Title</h1><p>Price: $99.99</p></div>"
text = parse_html(html_content)
# → "Product Title Price: $99.99"

# Download images with parallel processing
from yumbox.scraper import parse_html, ImageDownloader

# Download images with sensible defaults
downloader = ImageDownloader(base_dir="./images", num_processes=4)
results = downloader.download_images(["https://example.com/img.jpg"])

# Check results
if results["https://example.com/img.jpg"]:
    print("✓ Downloaded!")
downloader.cleanup()  # Remove temp files when done
```

---

### 🧰 Function Reference

#### URL Utilities

##### `myurljoin()` — Smart URL Joining
```python
from yumbox.scraper import myurljoin

# Basic URL joining
base = "https://example.com/products/"
relative = "../category/electronics"
joined = myurljoin(base, relative)
# → "https://example.com/category/electronics"

# Add www subdomain automatically (avoid redirects)
joined = myurljoin(
    "https://lcsc.com/product",
    "/details/123",
    add_www=True
)
# → "https://www.lcsc.com/details/123"

# Replace localhost with real domain (for local testing)
joined = myurljoin(
    "http://localhost:8000/api/data",
    "",
    replace_domain_if_localhost="api.example.com",
    scheme="https"
)
# → "https://api.example.com/api/data"
```

**Parameters:**
| Param | Type | Default | Purpose |
|-------|------|---------|---------|
| `base` | `str` | — | Base URL for joining |
| `url` | `str` | — | Relative or absolute URL to join |
| `add_www` | `bool` | `False` | Add `www.` subdomain if missing (prevent 301 redirect → faster crawl) |
| `replace_domain_if_localhost` | `str \| None` | `None` | Replace localhost/127.0.0.1 with this domain (locally saved and served pages) |
| `scheme` | `str` | `"https"` | Scheme to use when replacing localhost |

---

#### HTML Parsing & Text Extraction

##### `parse_html()` — HTML to Clean Text
```python
from yumbox.scraper import parse_html

# Remove HTML tags and clean content
html = """
<div class="product">
    <h1>Wireless Headphones</h1>
    <p>High-quality sound with <a href="#">noise cancellation</a>.</p>
    <img src="headphones.jpg" alt="Headphones">
    <span class="price">$99.99</span>
</div>
"""

text = parse_html(html)
# → "Wireless Headphones High-quality sound with noise cancellation. $99.99"

# Filters applied:
# - Removes HTML tags
# - Removes URLs (http://, https://)
# - Removes www.* domains
# - Splits on "/" and "," for long tokens (>25 chars)
# - Removes tokens longer than 25 characters
```

---

#### Selector Extensions

##### `MySelector` — Enhanced Parsel Selector
```python
from yumbox.scraper import MySelector

# Auto-decode percent-encoded content
selector = MySelector("<div>Hello%20World%21</div>")
text = selector.css("div::text").get()
# → "Hello World!"  (unquoted + unescaped)

# Normalize Unicode (NFKD)
selector = MySelector("<p>Café</p>")
text = selector.css("p::text").get()
# → "Cafe"  (normalized)

# Handle non-HTML content gracefully
selector = MySelector("123")
text = selector.get()
# → "123"

selector = MySelector("true")
text = selector.get()
# → "1"
```

**Key Features:**
- Automatic URL decoding (`%20` → space)
- HTML entity unescaping (`&amp;` → `&`)
- Unicode normalization (NFKD form)
- Graceful fallback for non-HTML content

##### `MyResponse` — Response Wrapper
```python
from yumbox.scraper import MyResponse

# Wrap raw HTTP response for Parsel selection
response = MyResponse(
    url="https://example.com",
    body=b"<html><body><h1>Title</h1></body></html>",
    status=200
)

# CSS selection
titles = response.css("h1::text").getall()
# → ["Title"]

# XPath selection
titles = response.xpath("//h1/text()").getall()
# → ["Title"]
```

---

#### Image Downloader

##### `ImageDownloader` — Parallel Image Fetching with Validation

```python
from yumbox.scraper import ImageDownloader

# Configure with production-ready settings
downloader = ImageDownloader(
    base_dir="/tmp/images",      # Persistent storage location
    num_processes=8,             # Parallel threads for speed
    check_corruption=True,       # Auto-detect broken/truncated images via PIL
    min_width=100,               # Filter out tiny icons/thumbnails
    min_height=100,
)

# Mix of valid and invalid URLs to demonstrate error handling
urls = [
    "https://example.com/valid.jpg",
    "https://example.com/404.jpg",      # Will fail gracefully
    "https://example.com/corrupt.png",  # Will be caught by corruption check
]

results = downloader.download_images(urls)

# Process results with file metadata
for url, path in results.items():
    if path and path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"✓ {url} → {path.name} ({size_kb:.1f} KB)")
    else:
        print(f"✗ {url} failed (check logs for details)")

# Optional cleanup to avoid disk buildup
downloader.cleanup()
```

##### Constructor Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `base_dir` | `str \| None` | `None` | Base directory for temp folders (auto-creates `image_downloader` in system temp if `None`) |
| `num_processes` | `int \| None` | `None` | Number of parallel threads (defaults to CPU count) |
| `check_corruption` | `bool` | `True` | Validate images with PIL after download |
| `min_width` | `int` | `50` | Minimum image width in pixels |
| `min_height` | `int` | `50` | Minimum image height in pixels |
| `domain` | `str \| None` | `None` | Domain to resolve locally (e.g., `"mydomain.com"`) |
| `resolve_to` | `str` | `"127.0.0.1"` | IP address to resolve domain to |

##### Key Methods

**`download_images(urls: list[str]) → dict[str, Path | None]`**
```python
# Download with progress tracking
downloader = ImageDownloader(num_processes=4)

urls = ["https://example.com/img{}.jpg".format(i) for i in range(100)]
results = downloader.download_images(urls)

# Output:
# Starting download of 100 images using 4 processes (with quality checks: min 50x50px)
# ✓ Successfully downloaded abc123
# ✗ Failed to download https://example.com/broken.jpg
# 
# Download complete!
# Temp directory: /tmp/image_downloader/tmp_xyz
# Successfully downloaded: 95/100 images
# Failed downloads: 5 images

# Access downloaded files
successful = {url: path for url, path in results.items() if path}
print(f"Downloaded {len(successful)} images to {downloader.temp_dir}")
```

**`cleanup()` — Remove All Downloads**
```python
# Automatic cleanup with context manager pattern
downloader = ImageDownloader()
try:
    results = downloader.download_images(urls)
    # Process images...
finally:
    downloader.cleanup()  # Removes temp dir and all files
```

**`is_image_corrupted(filepath: Path) → bool`**
```python
# Manual corruption check
from pathlib import Path
from yumbox.scraper import ImageDownloader

downloader = ImageDownloader(check_corruption=True)
is_bad = downloader.is_image_corrupted(Path("/path/to/image.jpg"))

# Checks:
# - File size > 0
# - Can open with PIL
# - Dimensions >= min_width/min_height
# - Image data loads without errors
```

**`get_filename_from_url(url: str) → str`**
```python
# Generate unique filename from URL hash
downloader = ImageDownloader()
filename = downloader.get_filename_from_url("https://example.com/img.jpg")
# → "-3827409283740928374"  (hash-based, avoids collisions)
```

---

### 🔁 Common Patterns

#### E-commerce Product Image Scraping
```python
from yumbox.scraper import ImageDownloader, parse_html, myurljoin, MySelector
import requests

def scrape_product_images(product_url: str):
    # Fetch product page
    response = requests.get(product_url)
    selector = MySelector(response.text)
    
    # Extract image URLs
    img_urls = selector.css("img::attr(src)").getall()
    
    # Convert relative URLs to absolute
    base_url = product_url.rsplit("/", 1)[0] + "/"
    absolute_urls = [myurljoin(base_url, url, add_www=True) for url in img_urls]
    
    # Download images
    downloader = ImageDownloader(
        base_dir=f"/tmp/products/{product_url.split('/')[-1]}",
        num_processes=8,
        min_width=200,
        min_height=200,
    )
    
    results = downloader.download_images(absolute_urls)
    
    # Get successful downloads
    downloaded = [path for path in results.values() if path]
    print(f"Downloaded {len(downloaded)} product images")
    
    return downloaded

# Usage
images = scrape_product_images("https://example.com/product/123")
```

#### Local Development with Custom DNS
```python
# Test scraping against production domain, but resolve to local server
downloader = ImageDownloader(
    domain="api.myapp.com",
    resolve_to="192.168.8.18",  # Route production domain to LAN-IP
)

# Downloads from https://api.myapp.com/images/* will hit 192.168.8.18:443
# instead of going over the internet, which will accomplish the task faster
urls = ["https://api.myapp.com/images/product1.jpg"]
results = downloader.download_images(urls)
```

#### Batch Processing with Error Recovery
```python
from pathlib import Path
import json

def download_with_checkpoint(url_file: str, output_dir: str):
    # Load URLs
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Check for existing checkpoint
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = json.load(f)
        urls = [u for u in urls if u not in completed]
        print(f"Resuming: {len(urls)} remaining URLs")
    
    # Download
    downloader = ImageDownloader(base_dir=output_dir, num_processes=16)
    results = downloader.download_images(urls)
    
    # Save checkpoint
    completed.extend([url for url, path in results.items() if path])
    with open(checkpoint_file, "w") as f:
        json.dump(completed, f)
    
    print(f"Total downloaded: {len(completed)}")
    return results

# Usage
results = download_with_checkpoint("urls.txt", "/tmp/batch_download")
```

#### Image Quality Filtering Pipeline
```python
from yumbox.scraper import ImageDownloader
from PIL import Image

def download_high_quality(urls: list[str], min_size: tuple[int, int] = (500, 500)):
    downloader = ImageDownloader(
        check_corruption=True,
        min_width=min_size[0],
        min_height=min_size[1],
        num_processes=8,
    )
    
    results = downloader.download_images(urls)
    
    # Additional quality checks
    high_quality = []
    for url, path in results.items():
        if not path:
            continue
        
        try:
            with Image.open(path) as img:
                # Check aspect ratio (not too extreme)
                aspect = img.width / img.height
                if 0.25 <= aspect <= 4.0:
                    high_quality.append(path)
                else:
                    path.unlink()  # Remove extreme aspect ratios
        except:
            path.unlink()
    
    print(f"High-quality images: {len(high_quality)}/{len(urls)}")
    return high_quality

# Usage
good_images = download_high_quality(product_urls, min_size=(800, 600))
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `parse_html` removes ALL URLs | If you need to preserve URLs, use `html_to_text()` instead and filter manually |
| `MySelector.get()` normalizes Unicode | This may change characters (é → e); disable if you need exact preservation |
| Custom DNS resolution (`domain` + `resolve_to`) requires pycurl | Only works with pycurl's `RESOLVE` option; doesn't affect system DNS |
| `cleanup()` removes ALL files in temp dir | Don't store other files in `base_dir`; use separate directories |
| PIL `verify()` doesn't always catch corruption | Some corrupted images might pass `verify()`; consider additional validation if needed |

#### Pro Tips
```python
# Tip 1: Use session-based downloading for repeated requests
class SessionImageDownloader(ImageDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MyBot/1.0"})
    
    def download_image(self, url, filepath):
        # Use session for connection pooling
        try:
            response = self.session.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except:
            pass
        return False

# Tip 2: Add retry logic for flaky downloads using Yumbox's built-in @retry decorator
from yumbox.cache import retry

class RetryImageDownloader(ImageDownloader):
    @retry(
        max_tries=3,                  # 1 initial + 2 retries
        backoff_factor=2,             # Exponential backoff: 2s, 4s, 8s...
        max_wait=30,                  # Cap backoff at 30 seconds
        retry_on400=False,            # Don't retry client errors (4xx)
        return_exception=True,        # Return {"status":"error",...} instead of raising
    )
    def download_image(self, url, filepath):
        # Your original download logic — retry decorator wraps it automatically
        return super().download_image(url, filepath)

# Advanced: Add a validator to check response content, not just status
def validate_image(response):
    """Ensure downloaded file is a valid image before accepting."""
    if not response or not isinstance(response, dict):
        from yumbox.cache import ValidationError
        raise ValidationError("Invalid response format")
    # Add custom validation logic here

class ValidatingRetryDownloader(ImageDownloader):
    @retry(
        max_tries=3,
        backoff_factor=2,
        validator=validate_image,  # Custom validation hook
        return_exception=True,
    )
    def download_image(self, url, filepath):
        return super().download_image(url, filepath)

# Tip 3: Progress bar integration
from tqdm import tqdm

def download_with_progress(self, urls):
    results = {}
    for url in tqdm(urls, desc="Downloading"):
        _, filepath = self._download_single_image(url)
        results[url] = filepath
    return results

ImageDownloader.download_images = download_with_progress

# Tip 4: Automatic format conversion
def convert_to_webp(self, filepath: Path) -> Path:
    from PIL import Image
    img = Image.open(filepath)
    webp_path = filepath.with_suffix(".webp")
    img.save(webp_path, "WEBP", quality=85)
    filepath.unlink()  # Remove original
    return webp_path

# Monkey-patch or subclass to add conversion
```

---

### 🔐 Security & Best Practices

```python
# Validate URLs before downloading
import re
from urllib.parse import urlparse

def safe_download(downloader: ImageDownloader, urls: list[str]):
    allowed_schemes = {"http", "https"}
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    
    safe_urls = []
    for url in urls:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in allowed_schemes:
            logger.warning(f"Blocked non-HTTP URL: {url}")
            continue
        
        # Check for path traversal
        # Note: Current filename logic uses hash(url), so path doesn't affect local FS.
        # This check protects against future code changes or scheme extensions.
        if ".." in parsed.path:
            logger.warning(f"Blocked path traversal attempt: {url}")
            continue
        
        # Check extension
        ext = Path(parsed.path).suffix.lower()
        if ext not in allowed_extensions:
            logger.warning(f"Blocked non-image extension: {url}")
            continue
        
        safe_urls.append(url)
    
    return downloader.download_images(safe_urls)

# Rate limiting to avoid overwhelming servers
import time
from threading import Lock

class RateLimitedDownloader(ImageDownloader):
    def __init__(self, *args, requests_per_second=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.rps = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.last_request = 0
        self.lock = Lock()
    
    def download_image(self, url, filepath):
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request = time.time()
        
        return super().download_image(url, filepath)
```

---

> 💡 **Why pycurl over requests?** pycurl provides fine-grained control (custom DNS resolution, low-level socket options) and better performance for bulk downloads. For simple use cases, `requests` works fine, but pycurl shines in production scraping.

---

Happy scraping! 🚀 If you need a new parser, downloader feature, or Scrapy integration, the code is intentionally modular — extend away.
