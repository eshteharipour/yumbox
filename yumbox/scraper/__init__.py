import html
import re
import unicodedata
import urllib.parse

import six
from lxml import etree
from parsel import Selector


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
