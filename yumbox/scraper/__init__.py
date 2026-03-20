import html
import logging
import re
import tempfile
import unicodedata
import urllib.parse
from pathlib import Path

import pycurl
import six
from lxml import etree
from parsel import Selector
from PIL import Image

logger = logging.getLogger("YumBox")


def myurljoin(
    base,
    url,
    add_www=False,
    replace_domain_if_localhost: str | None = None,
    scheme="https",
):
    """
    Convert express local urls to actual website urls.
    Also ensures www. subdomain is present to avoid redirects.
    """

    joined_url = urllib.parse.urljoin(base, url)

    if replace_domain_if_localhost:
        parsed_url = urllib.parse.urlparse(joined_url)
        if parsed_url.netloc == "127.0.0.1" or parsed_url.netloc == "localhost":
            joined_url = parsed_url._replace(
                scheme=scheme, netloc=replace_domain_if_localhost
            ).geturl()

    # Add www. if it's lcsc.com without www (avoid redirect)
    if add_www:
        parsed_url = urllib.parse.urlparse(joined_url)
        if parsed_url.netloc and not parsed_url.netloc.startswith("www"):
            joined_url = parsed_url._replace(netloc="www" + parsed_url.netloc).geturl()

    return joined_url


class MySelector(Selector):
    def get(self):
        """
        My serialize and return the matched nodes in a single unicode string.
        Percent encoded content is unquoted.
        """
        try:
            t = etree.tostring(
                self.root,
                method=self._tostring_method,
                encoding="unicode",
                with_tail=False,
            )
            t = urllib.parse.unquote(t)
            t = html.unescape(t)
            t = unicodedata.normalize("NFKD", t)
            return t
        except (AttributeError, TypeError):
            if self.root is True:
                return "1"
            elif self.root is False:
                return "0"
            else:
                t = six.text_type(self.root)
                t = urllib.parse.unquote(t)
                t = html.unescape(t)
                t = unicodedata.normalize("NFKD", t)
                return t


class MyResponse:
    def __init__(self, url, body, status) -> None:
        self.url = url
        self.body = body
        self.css = lambda sel: MySelector(body.decode("utf-8")).css(sel)
        self.xpath = lambda sel: MySelector(body.decode("utf-8")).xpath(sel)
        self.status = status


def html_to_text(t: str) -> str:
    """Html or text to text.
    Parse does not fail if not html."""

    # if isinstance(t, int):
    #     return t
    # css method fails if input is all numbers with this weird error:
    # ValueError: Cannot use css on a Selector of type 'json'
    # if t.numeric():
    # return t
    if t == "null":
        return t
    try:
        float(t)
        return t
    except ValueError:
        pass

    selector = MySelector(t)
    parsed = selector.css("::text").getall()
    parsed = " ".join(parsed)
    return parsed


def parse_html(content) -> str:
    if content:
        content = html_to_text(content)
    if content:
        if any(a in content for a in ["http://", "https://"]):
            content = ""
    if content:
        www_pattern = re.compile("www\.[^\s]+\.[^\s]")
        content = [c for c in content.split() if not www_pattern.search(content)]
        content = " ".join(content)
    if content:
        content = [re.sub("/|,", " ", c) if len(c) > 25 else c for c in content.split()]
        content = " ".join(content)
    if content:
        content = [c for c in content.split() if len(c) < 26]
        content = " ".join(content)
    return content


def make_scrapy_log_colorful():
    import copy

    import scrapy.utils.log
    from colorlog import ColoredFormatter

    color_formatter = ColoredFormatter(
        (
            "%(log_color)s%(levelname)-5s%(reset)s "
            "%(yellow)s[%(asctime)s]%(reset)s"
            "%(white)s %(name)s %(funcName)s %(bold_purple)s:%(lineno)d%(reset)s "
            "%(log_color)s%(message)s%(reset)s"
        ),
        datefmt="%y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "blue",
            "INFO": "bold_cyan",
            "WARNING": "red",
            "ERROR": "bg_bold_red",
            "CRITICAL": "red,bg_white",
        },
    )

    _get_handler = copy.copy(scrapy.utils.log._get_handler)

    def _get_handler_custom(*args, **kwargs):
        handler = _get_handler(*args, **kwargs)
        handler.setFormatter(color_formatter)
        return handler

    scrapy.utils.log._get_handler = _get_handler_custom


class ImageDownloader:
    def __init__(
        self,
        base_dir: str | None = None,
        num_processes: int | None = None,
        check_corruption: bool = True,
        min_width: int = 50,
        min_height: int = 50,
        domain: str | None = None,
        resolve_to: str = "127.0.0.1",
    ) -> None:
        """
        Initialize the image downloader.

        Args:
            base_dir: Base directory where temp dirs will be created.
                     If None, creates 'image_downloader' in temp directory
            num_processes: Number of parallel processes/threads for downloading
            check_corruption: Whether to verify images aren't corrupted after download
            min_width: Minimum image width in pixels (default: 50)
            min_height: Minimum image height in pixels (default: 50)
            domain: Domain to resolve locally (e.g., 'mydomain.com'). If None, no custom resolution
            resolve_to: IP address to resolve the domain to (default: 127.0.0.1)
        """
        if base_dir is None:
            temp_base = Path(tempfile.gettempdir())
            self.base_dir = temp_base / "image_downloader"
        else:
            self.base_dir = Path(base_dir)

        if isinstance(num_processes, int) and num_processes > 0:
            self.num_processes = num_processes
        else:
            try:
                self.num_processes = int(num_processes)
                assert self.num_processes > 0
            except (TypeError, ValueError, AssertionError):
                import multiprocessing as mp

                self.num_processes = mp.cpu_count()
        self.check_corruption = check_corruption
        self.min_width = max(1, min_width)
        self.min_height = max(1, min_height)
        self.domain = domain
        self.resolve_to = resolve_to if resolve_to is not None else "127.0.0.1"
        self.temp_dir: Path | None = None
        self.downloaded_files: dict[str, Path | None] = {}

    def create_temp_dir(self) -> Path:
        """Create a temporary directory under the base directory."""
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(dir=self.base_dir))
        return self.temp_dir

    def get_filename_from_url(self, url: str) -> str:
        """Generate a filename from URL hash."""
        return str(hash(url))

    def is_image_corrupted(self, filepath: Path) -> bool:
        """
        Check if an image file is corrupted, invalid, or too small.

        Args:
            filepath: Path to the image file

        Returns:
            True if corrupted/invalid/too small, False if valid
        """
        try:
            # Check if file is empty
            if filepath.stat().st_size == 0:
                logger.debug(f"Image is empty: {filepath.name}")
                return True

            # Try to open and verify the image with PIL
            with Image.open(filepath) as img:
                # Check image dimensions
                width, height = img.size
                if width < self.min_width or height < self.min_height:
                    logger.debug(
                        f"Image too small ({width}x{height}): {filepath.name} (min: {self.min_width}x{self.min_height})"
                    )
                    return True

                # Verify the image by loading it completely
                img.verify()

            # Re-open to check if we can load the image data
            # (verify() closes the file, so we need to re-open)
            # with Image.open(filepath) as img:
            #     img.load()

            return False

        except (IOError, OSError, Image.DecompressionBombError) as e:
            logger.debug(f"Image corruption detected in {filepath.name}: {str(e)}")
            return True
        except Exception as e:
            logger.debug(f"Unexpected error checking image {filepath.name}: {str(e)}")
            return True

    def download_image(self, url: str, filepath: Path) -> bool:
        """
        Download a single image using pycurl.

        Args:
            url: URL of the image to download
            filepath: Local path where to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, "wb") as f:
                c = pycurl.Curl()
                c.setopt(c.URL, url)
                c.setopt(c.WRITEDATA, f)
                c.setopt(c.FOLLOWLOCATION, True)
                c.setopt(c.TIMEOUT, 30)
                c.setopt(c.USERAGENT, "Mozilla/5.0 (compatible; DataFusion/1.0)")
                c.setopt(c.SSL_VERIFYPEER, 0)
                c.setopt(c.SSL_VERIFYHOST, 0)

                # Add custom DNS resolution if domain is provided
                if self.domain:
                    # Format: "domain:port:ip"
                    c.setopt(c.RESOLVE, [f"{self.domain}:{self.resolve_to}"])

                c.perform()

                # Check HTTP response code
                response_code = c.getinfo(c.RESPONSE_CODE)
                c.close()

                if response_code == 200:
                    # Check for image corruption if enabled
                    if self.check_corruption and self.is_image_corrupted(filepath):
                        logger.error(
                            f"Downloaded image is corrupted or too small: {url}"
                        )
                        filepath.unlink()
                        return False
                    return True
                else:
                    # Remove failed download file
                    if filepath.exists():
                        filepath.unlink()
                    logger.error(f"HTTP {response_code} for {url}")
                    return False

        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            # Remove failed download file
            if filepath.exists():
                filepath.unlink()
            return False

    def _download_single_image(self, url: str) -> tuple[str, Path | None]:
        """
        Download a single image and return the result.

        Args:
            url: URL to download

        Returns:
            Tuple of (url, filepath_or_None)
        """
        filename = self.get_filename_from_url(url)
        filepath = self.temp_dir / filename

        logger.debug(f"Downloading {url} -> {filepath.name}")

        if self.download_image(url, filepath):
            logger.info(f"✓ Successfully downloaded {filepath.name}")
            return url, filepath
        else:
            logger.error(f"✗ Failed to download {url}")
            return url, None

    def download_images(self, image_urls: list[str]) -> dict[str, Path | None]:
        """
        Download multiple images with optional parallel processing.

        Args:
            image_urls: List of image URLs to download

        Returns:
            Dict mapping URLs to their local file paths (None if failed)
        """
        if not self.temp_dir:
            self.create_temp_dir()

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(image_urls))

        corruption_check_msg = (
            f" (with quality checks: min {self.min_width}x{self.min_height}px)"
            if self.check_corruption
            else ""
        )
        logger.info(
            f"Starting download of {len(unique_urls)} images using {self.num_processes} processes{corruption_check_msg}"
        )
        resolve_msg = (
            f"(resolving {self.domain} to {self.resolve_to})" if self.domain else ""
        )
        logger.debug(resolve_msg)

        if self.num_processes == 1:
            # Sequential download
            for url in unique_urls:
                _, filepath = self._download_single_image(url)
                self.downloaded_files[url] = filepath
        else:
            from concurrent.futures import ThreadPoolExecutor

            # Parallel download using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                results = executor.map(self._download_single_image, unique_urls)

                for url, filepath in results:
                    self.downloaded_files[url] = filepath

        successful_count = sum(
            1 for path in self.downloaded_files.values() if path is not None
        )
        failed_count = len(unique_urls) - successful_count

        logger.info(f"\nDownload complete!")
        logger.info(f"Temp directory: {self.temp_dir}")
        logger.info(
            f"Successfully downloaded: {successful_count}/{len(unique_urls)} images"
        )
        logger.info(f"Failed downloads: {failed_count} images")

        return self.downloaded_files

    def cleanup(self) -> None:
        """Remove all downloaded files and temp directory."""
        if self.downloaded_files:
            for url, filepath in self.downloaded_files.items():
                if filepath and filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception as e:
                        logger.error(f"Error removing {filepath}: {e}")

        if self.temp_dir and self.temp_dir.exists():
            try:
                # Remove any remaining files in temp dir
                for item in self.temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()

                self.temp_dir.rmdir()
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temp directory: {e}")
