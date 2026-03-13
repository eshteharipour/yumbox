import re
import unicodedata

import unicodeblock
from tqdm import tqdm


class UnicodeRanges:
    """
    References:
        https://en.wikipedia.org/wiki/Arabic_script_in_Unicode
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
    """

    english_letters = r"\u0041-\u005A\u0061-\u007A"
    persian_letters = r"\u0600-\u06FF\u0750-\u077F"
    numerical_digits = r"\u0030-\u0039"
    special_chars = "!@#$%^&*()_+{{}}\[\]:;\"'<>,.?/|~`×\u200c=.-]"

    # A-Z and a-z
    english_letters_ranges = [(0x0041, 0x005B), (0x0061, 0x007B)]
    # Arabic and Persian ranges
    persian_letters_ranges = [(0x0600, 0x0700), (0x0750, 0x0780)]
    # 0-9:;<=>?
    numerical_digits_ranges = [(0x0030, 0x0040)]
    specials = [(0x0032, 0x0047), (0x0058, 0x0064), (0x0091, 0x0096), (0x0123, 0x0126)]

    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        # (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        # (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        # (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        # (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        # (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
        # (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        # (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    ]

    @classmethod
    def regex_from_range(self, ranges: list[tuple[int, int]], negate=False):
        hex_to_unicode = lambda x: rf"\u{format(x, '04X')}"
        ranges = [(hex_to_unicode(s), hex_to_unicode(e)) for s, e in ranges]
        ranges = [f"{s}-{e}" for s, e in ranges]
        ranges = "".join(ranges)
        if negate == False:
            pattern = rf"[{ranges}]"
        else:
            pattern = rf"[^ {ranges}]"  # Added space
        return re.compile(pattern)

    @classmethod
    def unknown_chars(self):
        return rf"[^{self.english_letters}{self.persian_letters}{self.numerical_digits}{self.special_chars}"

    @classmethod
    def regex_cjk_chars(self):
        return self.regex_from_range(self.cjk_ranges)

    @classmethod
    def print_unknown_chars(self, words: list):
        regex = re.compile(self.unknown_chars())

        for word in tqdm(words):
            if bool(regex.search(word)):
                print(word)

    @classmethod
    def print_unicode_range(self, ranges: list[tuple[int, int]]):
        for start, end in ranges:
            for codepoint in range(start, end):
                print(chr(codepoint), end="")
            print()  # New line after each range

    @classmethod
    def print_known_ranges(self):
        print("English Letters:")
        self.print_unicode_range(self.english_letters_ranges)

        print("\nPersian Letters:")
        self.print_unicode_range(self.persian_letters_ranges)

        print("\nNumerical Digits:")
        self.print_unicode_range(self.numerical_digits_ranges)

        print("\nSpecial chars:")
        self.print_unicode_range(self.specials)


class Unicode:
    # https://www.fileformat.info/info/unicode/category/index.htm
    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string

    def strip_accents(text: str):
        """Strip accents from text (e.g., é -> e)."""

        return "".join(c for c in unicodedata.normalize("NFD", text))

    def sanitize_cmp(text: str):
        """Strip CMP categories from text:
        Other (Control, Format, Not Assigned, Private Use, Surrogate),
        Mark (Spacing Combining, Enclosing, Nonspacing),
        Punctuation (Connector, Dash, Close, Final quote, Initial quote, Other, Open).

        We keep LNSZ (Letter, Number, Symbol, Separator). Separator is handled with regex \s.
        """

        # Other (C) removes LF
        # Punctuation (P) removes dot

        return "".join(
            (
                c
                if unicodedata.category(c)[0] not in ["C", "M", "P"] or c in ["\n", "."]
                else " "
            )
            for c in text
        )

    # Remove trash character sets
    def sanitize_char_sets(text: str):
        return "".join(
            c if unicodeblock.blocks.of(c) not in ["HANGUL_JAMO", "CYRILLIC"] else " "
            for c in text
        )


class Chars:
    # Equivalent EXTENDED ARABIC-INDIC DIGIT
    mappings_extended_arin_en_digit = [
        ("۰", "0"),
        ("۱", "1"),
        ("۲", "2"),
        ("۳", "3"),
        ("۴", "4"),
        ("۵", "5"),
        ("۶", "6"),
        ("۷", "7"),
        ("۸", "8"),
        ("۹", "9"),
    ]

    # Equivalent ARABIC-INDIC DIGIT
    mappings_arin_en_digit = [
        ("٠", "0"),
        ("١", "1"),
        ("٢", "2"),
        ("٣", "3"),
        ("٤", "4"),
        ("٥", "5"),
        ("٦", "6"),
        ("٧", "7"),
        ("٨", "8"),
        ("٩", "9"),
    ]

    # Equivalent letters
    mappings_ar_fa_letters = [
        ("ە", "ه"),  # ARABIC LETTER AE, ARABIC LETTER HEH
        ("ہ", "ه"),  # ARABIC LETTER HEH GOAL, ARABIC LETTER HEH
        ("ٸ", "ی"),  # ARABIC LETTER HIGH HAMZA YEH, ARABIC LETTER FARSI YEH
        ("ھ", "ه"),  # ARABIC LETTER HEH DOACHASHMEE, ARABIC LETTER HEH
        ("ى", "ی"),  # ARABIC LETTER ALEF MAKSURA, ARABIC LETTER FARSI YEH
        ("ں", "ن"),  # ARABIC LETTER NOON GHUNNA, ARABIC LETTER NOON
        ("ے", "ی"),  # ARABIC LETTER YEH BARREE, ARABIC LETTER FARSI YEH
        ("ﯼ", "ی"),  # ARABIC LETTER FARSI YEH ISOLATED FORM # NEW!
        ("آ", "ا"),  # A kolah-dar
        ("ي", "ی"),  # ARABIC LETTER YEH
    ]

    # Equivalent special characters
    mappings_ar_ = [
        ("؛", ";"),
        ("٪", "%"),
        ("²", "2"),
        ("³", "3"),
    ]

    burn_chars_a = [
        "\u200e",  # RLM
        "\u200f",  # LRM
        "\u200c",  # ZWNJ
        "\u00ad",  # SOFT HYPHEN [SHY]
        "\u2026",  # HORIZONTAL ELLIPSIS (re.sub doesn't pick up … so used hex value)
        "®",
        "©",
        "™",
        "(",
        ")",
        "/",
        ",",
        ":",
        "[",
        "]",
        "«",
        "»",
        "<",
        ">",
        "'",
        '"',
        "#",
        "*",
        "،",  # ARABIC COMMA
        "-",
        "|",
    ]

    def get_mappings():
        classvars = list(filter(lambda a: a.startswith("mappings_"), vars(Chars)))
        mappings = []
        for cv in classvars:
            mappings.append(getattr(Chars, cv))
        return mappings

    def get_burn_chars():
        classvars = list(filter(lambda a: a.startswith("burn_"), vars(Chars)))
        burn_chars = []
        for cv in classvars:
            burn_chars += getattr(Chars, cv)
        burn_chars = "".join(re.escape(c) for c in burn_chars)
        return f"[{burn_chars}]"

    def get_cjk():
        return re.compile("[\u4e00-\u9fff]")
