import re
import string
from typing import Literal

import hazm
import nltk

from yumbox.scraper import parse_html

from .chars import Chars, Unicode, UnicodeRanges
from .tools import MapRed, defaultname, replace_fromstart


def init_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")  # For tokenization
    nltk.download("wordnet")  # For lemmatization
    nltk.download("omw-1.4")  # For wordnet lemmatizer support
    nltk.download("punkt_tab")


def get_wordnet_pos(self, word):
    from nltk.corpus import wordnet

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


class Preprocessor:
    def __init__(
        self,
        normalize_fa_chars=True,
        remove_special_chars=True,
        remove_cjk=True,
        strip_accents=True,
        remove_ctrl_chars=True,
        remove_fa_stopwords=True,
        remove_en_stopwords=True,
        remove_punctuation=True,
        only_eng_unicode_range=False,
        only_fa_unicode_range=False,
        only_num_unicode_range=False,
        do_fa_normalizer=False,
        do_fa_stemmer=False,
        do_fa_lemmatizer=False,
        do_en_normalizer=False,
        do_en_stemmer=False,
        do_en_lemmatizer=False,
        fa_tokenizer=False,
        en_tokenizer=False,
        fa_sentence_tokenizer=False,
        en_sentence_tokenizer=False,
        en_lemmatize_method: Literal["all", "verbs", "known"] = "all",
        remove_parentheticals=False,
        remove_singular_digis=False,
        do_hard_limit=False,
        high_hard_limit=26,
        low_hard_limit=3,
        parse_html=False,
        remove_uri=False,
        remove_www=False,
        split_group_annots=False,
    ):
        """Preprocessor from aggregated tools.
        Functions included:
            - sentence_preprocessor
            - fa_sent_normalizer
            - en_sent_normalizer
            - parse_html
            - product_title_preprocessor
        Functions to be included:
            - sentence_normalizer
            - custom_en_stop_words
            - calculated_stop_words
            - custom keep alphabet
            - input type json to text
        Functions half included:
            - word_post_process (we need to remove low freq words using tf-idf)
            - char_post_process (we need to remove low freq chars using mapred)
        """

        init_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        self.normalize_fa_chars = normalize_fa_chars
        self.remove_special_chars = remove_special_chars
        self.remove_cjk = remove_cjk
        self.strip_accents = strip_accents
        self.remove_ctrl_chars = remove_ctrl_chars
        self.remove_fa_stopwords = remove_fa_stopwords
        self.remove_en_stopwords = remove_en_stopwords
        self.remove_punctuation = remove_punctuation
        self.only_eng_unicode_range = only_eng_unicode_range
        self.only_fa_unicode_range = only_fa_unicode_range
        self.only_num_unicode_range = only_num_unicode_range
        self.do_fa_normalizer = do_fa_normalizer
        self.do_fa_stemmer = do_fa_stemmer
        self.do_fa_lemmatizer = do_fa_lemmatizer
        self.do_en_normalizer = do_en_normalizer
        self.do_en_stemmer = do_en_stemmer
        self.do_en_lemmatizer = do_en_lemmatizer
        self.fa_tokenizer = fa_tokenizer
        self.en_tokenizer = en_tokenizer
        self.fa_sentence_tokenizer = fa_sentence_tokenizer
        self.en_sentence_tokenizer = en_sentence_tokenizer
        self.en_lemmatize_method = en_lemmatize_method
        self.remove_parentheticals = remove_parentheticals
        self.remove_singular_digis = remove_singular_digis
        self.do_hard_limit = do_hard_limit
        self.high_hard_limit = high_hard_limit
        self.low_hard_limit = low_hard_limit
        self.parse_html = parse_html
        self.remove_uri = remove_uri
        self.remove_www = remove_www
        self.split_group_annots = split_group_annots

        if self.remove_fa_stopwords:
            self.fa_stopwords = set(hazm.stopwords_list())
            # TODO: calculated_stop_words
        if self.remove_en_stopwords:
            self.en_stopwords = set(stopwords.words("english"))
            # self.en_stopwords = [s for s in self.en_stopwords if len(s) > 2]
            # TODO: custom_en_stop_words

        # TODO: custom keep alphabet
        # Filter out characters not in ok_alphabet
        # text = "".join([t for t in text if t in ok_alphabet])

        if self.do_fa_normalizer:
            # !!! Changes en digit to fa digit !!!
            fa_normalizers = hazm.Normalizer()
            self.fa_normalizer = fa_normalizers.normalize
        if self.do_fa_stemmer:
            fa_stemmer = hazm.Stemmer()
            self.fa_stemmer = fa_stemmer.stem
        if self.do_fa_lemmatizer:
            fa_lemmatizer = hazm.Lemmatizer()
            self.fa_lemmatizer = fa_lemmatizer.lemmatize

        if self.do_en_stemmer:
            en_stemmer = PorterStemmer()
            self.en_stemmer = en_stemmer.stem
        if self.do_en_lemmatizer:
            en_lemmatizer = WordNetLemmatizer()
            self.en_lemmatizer = en_lemmatizer.lemmatize

        # Splits based on family (en, fa, numeric)
        self.fa_word_tokenizer = WordTokenizer(self.fa_tokenizer)
        self.en_word_tokenizer = WordTokenizer(self.en_tokenizer)
        self.fa_sent_tokenizer = SentTokenizer(self.fa_sentence_tokenizer)
        self.en_sent_tokenizer = SentTokenizer(self.en_sentence_tokenizer)

        self.ok_unicode_ranges = []
        if only_eng_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.english_letters_ranges
            )
        if only_fa_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.persian_letters_ranges
            )
        if only_num_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.numerical_digits_ranges
            )

    def en_normalizer(self, s: str):
        return s.lower()

    def parenthetical_remover(self, s: str):
        """Remove text inside parantheses or brackets"""
        return re.sub(r"\([^)]*\)|\[[^\]]*\]", "", s)

    def singular_digits_remover(self, tokens: list[str]):
        isnum = re.compile("^\d+$")
        return [t for t in tokens if not re.match(isnum, t)]

    def hard_limiter(self, tokens: list[str], low: int, high: int):
        """Keep words longer and shorter than [a, b] chars"""
        return [t for t in tokens if (len(t) >= low) and len(t) <= high]

    def uri_remover(self, s: str):
        return re.sub("( )?http(s)?://[^\s]+( )?", " ", s).strip()

    def www_remover(self, s: str):
        return re.sub("( )?www\.[^\s]+\.[^\s]( )?", " ", s).strip()

    def group_annots_splitter(self, s: str):
        return re.sub("/|,", " ", s).strip()

    def __call__(self, s: str, keep_lf=False) -> str:
        # TODO: Check with "from parsel import Selector" if type of input is json (double quotes escaped).
        # and `loads` it .

        if not s:
            return ""

        if self.parse_html:
            s = parse_html(s)
        if self.remove_uri:
            s = self.uri_remover(s)
        if self.remove_www:
            s = self.www_remover(s)
        if self.split_group_annots:
            s = self.group_annots_splitter(s)

        if self.do_en_normalizer == True:
            s = self.en_normalizer(s)

        # Replace with equivalent characters
        if self.normalize_fa_chars == True:
            for mapping in Chars.get_mappings():
                for x, y in mapping:
                    s = s.replace(x, y)

        # Remove trash chars
        if self.remove_special_chars:
            s = re.sub(Chars.get_burn_chars(), " ", s)
        if self.remove_cjk:
            s = re.sub(Chars.get_cjk(), " ", s)
            s = Unicode.sanitize_char_sets(s)
        if self.strip_accents:
            s = Unicode.strip_accents(s)
        if self.remove_ctrl_chars:
            s = Unicode.sanitize_cmp(s)
        if self.remove_punctuation:
            s = "".join([c for c in s if c not in string.punctuation])

        if self.ok_unicode_ranges:
            ur_regex = UnicodeRanges.regex_from_range(
                self.ok_unicode_ranges, negate=True
            )
            s = re.sub(ur_regex, "", s)

        if self.remove_parentheticals:
            s = self.parenthetical_remover(s)

        # Farsi
        sentenecs = self.fa_sent_tokenizer(s)
        procd_sentenecs = []
        for sent in sentenecs:
            words = self.fa_word_tokenizer(sent)

            if self.remove_fa_stopwords:
                words = [w for w in words if w not in self.fa_stopwords]
            if self.do_fa_normalizer:
                # !!! Changes en digit to fa digit !!!
                words = [self.fa_normalizer(w) for w in words]
            if self.do_fa_stemmer:
                words = [self.fa_stemmer(w) for w in words]
            if self.do_fa_lemmatizer:
                words = [self.fa_lemmatizer(w) for w in words]

            # words = [w for w in words if " " not in w]

            sent = " ".join(words)
            procd_sentenecs.append(sent)

        s = "\n".join(procd_sentenecs)

        # English + general
        sentences = self.en_sent_tokenizer(s)
        procd_sentences = []
        for sent in sentences:
            words = self.en_word_tokenizer(sent)

            if self.remove_singular_digis:
                words = self.singular_digits_remover(words)
            if self.do_hard_limit:
                words = self.hard_limiter(
                    words, self.low_hard_limit, self.high_hard_limit
                )

            # TODO: Per word English normalizer?
            if self.remove_en_stopwords:
                words = [w for w in words if w not in self.en_stopwords]
            if self.do_en_stemmer:
                words = [self.en_stemmer(w) for w in words]

            if self.do_en_lemmatizer:
                # Lemmatize
                if self.en_lemmatize_method == "all":
                    words = [self.en_lemmatizer(w) for w in words]

                # Lemmatize with part-of-speech (POS) tags
                # 'v' for verbs
                if self.en_lemmatize_method == "verbs":
                    words = [self.en_lemmatizer(word, pos="v") for word in words]

                # Lemmatize with Wordnet
                if self.en_lemmatize_method == "known":
                    words = [
                        self.en_lemmatizer(w, get_wordnet_pos(w))
                        for w in words
                        if self.en_lemmatizer(w, get_wordnet_pos(w)) != ""
                    ]

            # words = [w for w in words if " " not in w]

            sent = " ".join(words)
            procd_sentences.append(sent)

        s = "\n".join(procd_sentences)

        # Remove extra spaces
        if keep_lf:
            s = re.sub("[\t\r\n\v\f]", " ", s)
            s = re.sub("[\t\r\n\v\f][\t\r\n\v\f]+", " ", s)
        else:
            # s = re.sub("\s", " ", s)
            # s = re.sub("\s\s+", " ", s)
            s = " ".join(s.split())

        # Removed chars leave spaces
        s = s.strip()

        return s


class WordTokenizer:
    def __init__(self, fa_tokenizer=False, en_tokenizer=False):
        if fa_tokenizer ^ en_tokenizer:
            # TODO: Can also use both tokenizers
            self.word_tokenizer = lambda s: s.split(" ")
        elif fa_tokenizer:
            # Splits based on family (en, fa, numeric):
            self.word_tokenizer = hazm.word_tokenize
        else:
            self.word_tokenizer = nltk.word_tokenize

    def __call__(self, s: str) -> str:
        return self.word_tokenizer(s)


class SentTokenizer:
    def __init__(self, fa_sentence_tokenizer=False, en_sentence_tokenizer=False):
        if fa_sentence_tokenizer ^ en_sentence_tokenizer:
            # TODO: Can also use both tokenizers
            self.sent_tokenizer = lambda s: s.split("\n")
        elif fa_sentence_tokenizer:
            self.sent_tokenizer = hazm.sent_tokenize
        else:
            self.sent_tokenizer = nltk.sent_tokenize

    def __call__(self, s: str) -> str:
        return self.sent_tokenizer(s)
