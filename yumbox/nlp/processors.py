import os
import re
from typing import Literal

import hazm
import nltk
import pandas as pd
import unicodeblock.blocks
import unicodeblock.sequence
from nltk.corpus import stopwords
from tqdm import tqdm

from yumbox.data import fix_pandas_truncation

from .chars import Chars, Unicode, UnicodeRanges
from .tools import MapRed


def sentence_normalizer(s: str):
    """Sentence -> sentences normalized by unicode family.
    Deprecated in favor of Preprocessor class."""
    _o = "".join(s.split())

    # TODO: REMOVES FARSI CHARACTERS
    # import unicodedata
    # import unicodeblock.sequence
    s = " ".join([str(x) for x in unicodeblock.sequence.usplit(s)])

    # TODO: Replace specs
    # Replace v with volate
    # Replace a with ampere
    # Replace ... with ...

    # TODO: hazm.word_tokenize splits based on family (en, fa, numeric)

    _t = "".join(s.split())

    if _o != _t:
        print("WARNING: Split escaped some characters!")

    return s


def sentence_preprocessor(s: str, keep_lf=False) -> str:
    """Sentence -> sentence.
    Deprecated in favor of Preprocessor class."""

    # Normalize
    s = s.strip()
    s = s.lower()

    # Replace with equivalent characters
    for mapping in Chars.get_mappings():
        for x, y in mapping:
            s = s.replace(x, y)

    # Remove trash chars
    s = re.sub(Chars.get_burn_chars(), " ", s)
    s = re.sub(Chars.get_cjk(), " ", s)
    s = Unicode.strip_accents(s)
    s = Unicode.sanitize_cmp(s)
    s = Unicode.sanitize_char_sets(s)

    # Special preprocessor
    s = product_title_preprocessor(s)

    # Remove stop words
    s = " ".join([w for w in s.split(" ") if w not in online_shop_stop_words()])

    # Farsi normalizer
    s = fa_sent_normalizer(s)

    # English normalizer
    s = en_sent_normalizer(s)

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


def column_preprocessor(arr: list[str], join_by=" ", appendlf=True) -> str:
    arr = [a for a in arr if a]
    text = join_by.join(arr)

    text = sentence_preprocessor(text, keep_lf=True)

    if text and appendlf:
        text += "\n"

    return text


def debug_sentence_preprocessor(s: str) -> str:
    import inspect
    import re

    space_pattern = re.compile("^\s+")

    def inject_print(func):
        source = inspect.getsource(func)
        lines = source.splitlines()

        # Keep the function definition line
        modified_lines = [lines[0]]

        # Loop through the rest of the lines (the body of the function)
        for line in lines[1:]:
            modified_lines.append(line)
            if " = " in line:
                spaces = re.search(space_pattern, line)[0]
                modified_lines.append(spaces + f'print(""">>>{line.strip()}""")')
                modified_lines.append(spaces + "print(s)")

        # Join the modified lines back into a string
        modified_code = "\n".join(modified_lines)

        return modified_code

    injected_sentence_preprocessor = inject_print(sentence_preprocessor)
    injected_sentence_preprocessor = "\n".join(
        [injected_sentence_preprocessor, f's = r"""{s}"""', "sentence_preprocessor(s)"]
    )

    exec(injected_sentence_preprocessor)


# from nltk.corpus import wordnet
# from nltk.stem import PorterStemmer, WordNetLemmatizer

fa_stop_words = None
en_stop_words = None


def init_fa_stop_words():
    global fa_stop_words
    if fa_stop_words is None:
        fa_stop_words = set(hazm.stopwords_list())


def init_en_stop_words(use_custom=True):
    global en_stop_words
    if en_stop_words is None:
        if not use_custom:
            en_stop_words = list(stopwords.words("english"))
            # en_stop_words = [s for s in en_stop_words if len(s) > 2]
        else:
            en_stop_words = custom_en_stop_words()


def custom_en_stop_words():
    with open(os.path.join(os.path.dirname(__file__), "stop_words.txt")) as fd:
        esw = fd.readlines()
    return [s.strip() for s in esw if s.strip()]


def fa_sent_normalizer(text: str):
    """Sentence -> word -> sentence.
    Farsi tokenize, remove stop words."""
    if fa_stop_words is None:
        init_fa_stop_words()

    # normalizer = hazm.Normalizer()  # !!! Changes English digit to Farsi digit
    # stemmer = hazm.Stemmer()
    # lemmatizer = hazm.Lemmatizer()

    # Splits based on family (English, Farsi, numeric):
    # word_tokenizer = hazm.word_tokenize
    word_tokenizer = lambda s: s.split(" ")
    sent_tokenizer = hazm.sent_tokenize

    # Tokenize
    sentenecs = sent_tokenizer(text)
    procd_sentenecs = []
    for sent in sentenecs:
        words = word_tokenizer(sent)

        # Preprocess
        words = [w for w in words if w not in fa_stop_words]
        # words = [normalizer.normalize(w) for w in words]
        # words = [stemmer.stem(w) for w in words]
        # words = [lemmatizer.lemmatize(w) for w in words]

        # words = [w for w in words if " " not in w]

        sent = " ".join(words)
        procd_sentenecs.append(sent)

    return "\n".join(procd_sentenecs)


def en_sent_normalizer(text: str):
    """Sentence -> word -> sentence.
    English tokenize, remove stop words."""
    if en_stop_words is None:
        init_en_stop_words()

    # stemmer = PorterStemmer()  # en_stemmer.stem()
    # lemmatizer = WordNetLemmatizer()  # en_lemmatizer.lemmatize()

    word_tokenizer = nltk.word_tokenize
    sent_tokenizer = nltk.sent_tokenize

    # Tokenize
    sentences = sent_tokenizer(text)
    procd_sentences = []
    for sent in sentences:
        words = word_tokenizer(sent)

        # Preprocess
        words = [w for w in words if w not in en_stop_words]
        # words = [stemmer.stem(w) for w in words]

        # Lemmatize
        # words = [lemmatizer.lemmatize(w) for w in words]

        # Lemmatize with part-of-speech (POS) tags
        # words = [lemmatizer.lemmatize(word, pos="v") for word in words]  # 'v' for verbs

        # Lemmatize with Wordnet
        # words = [
        #     lemmatizer.lemmatize(w, get_wordnet_pos(w))
        #     for w in words
        #     if lemmatizer.lemmatize(w, get_wordnet_pos(w)) != ""
        # ]

        # words = [w for w in words if " " not in w]

        sent = " ".join(words)
        procd_sentences.append(sent)

    return "\n".join(procd_sentences)


def __debug(arr: list) -> str:
    RTL_CTRL_CHAR = "\u202b"
    arr = [a for a in arr if a]
    arr = [re.sub("\s", " ", a) for a in arr]
    sentence = "\n".join(arr)
    return sentence


def product_title_preprocessor(text: str):
    """Sentence -> comprehension -> sentence."""

    # Keep words longer than 2 chars
    # isnum = re.compile("^\d+$")
    # text = " ".join([t for t in text.split(" ") if (len(t) > 2) or (re.match(isnum, t))])

    # Remove text inside parantheses or brackets
    # text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)

    # Filter out characters not in ok_alphabet
    # text = "".join([t for t in text if t in ok_alphabet])

    return text


def online_shop_stop_words():
    """Returns one list of English and Farsi stopwords
    Used both in preprocess and postprocess.
    Preprocess: For inference on model.
    Postprocess: For cleaning corpus."""

    # Returns two lists of stopwords: English and Farsi
    # return ["rohs"], ["خرید"]
    return [
        "rohs",
        "خرید",
        "فروش",
        "عمده",
        "برای",
        "قیمت",
        "طی تماس",
        "تماس",
        "بگیرید",
        "جهت سفارش",
        "سفارش",
        "تلفنی",
        "تلفن",
        "های کپی",
        "کپی",
        "اصل",
        "org",
    ]


def word_post_process(sentence: str, mr: MapRed, verbose=False) -> str:
    """Sentence -> word -> sentence.
    Remove len 1 words, long words, low req words, stop words."""

    words = sentence.split(" ")

    # TODO: Convert this to comprehension
    pwords = []
    for w in words:
        # Remove len 1 words
        if len(w) < 2:
            verbose and print(f"skipping {w}")
            continue

        # 56 was the largest mfr_no word (split by space) from my inspection
        if len(w) > 56:
            verbose and print(f"skipping {w}")
            continue

        # Remove low freq words
        if w in mr.lowfreq:
            verbose and print(f"skipping {w}")
            continue

        # Remove stop words (ROHS)
        # Check separately than appending to mr.lowfreq made it twice as fast
        if w in online_shop_stop_words():
            verbose and print(f"skipping {w}")
            continue

        pwords.append(w)

    sentence = " ".join(pwords)

    return sentence


def char_post_process(text: str, mr: MapRed, verbose=False):
    """Sentence -> sentence.
    Remove low freq chars."""

    # Remove low freq chars
    verbose and print("Removing low frequency chars")
    for c in mr.lowfreq:
        verbose and print(f"Removing {c}")
        text = text.replace(c, " ")

    return text


def analyze_chars(mr_word: MapRed, mr_char: MapRed):
    """For every unique character, shows an example word.

    Args:
        mr_word (MapRed): MapRed instance for words.
        mr_char (MapRed): MapRed instance for characters.
    """

    def _word_contains_char(words, char):
        for word in words:
            if char in word:
                return word

    words = mr_word.dictionary.keys()
    mr_char.sort("desc")
    for char, freq in mr_char.dictionary.items():
        word = _word_contains_char(words, char)
        print(char, freq, word)


def analyze_topten(mr: MapRed):
    """Sorts desc and asc and prints top 10 word/char -> freq.

    Args:
        mr (MapRed): map reduce by word or character level
    """

    mr.sort("desc")
    keys = list(mr.dictionary.keys())[0:10]
    t = {k: mr.dictionary[k] for k in keys}
    print(t)

    mr.sort("asc")
    keys = list(mr.dictionary.keys())[0:10]
    t = {k: mr.dictionary[k] for k in keys}
    print(t)


def analyze_head(df: pd.DataFrame, sort_key: Literal["len", "freq"]):
    """Sorts MapRed dataframe by key specified in descending.

    Args:
        df (pd.DataFrame): MapRed dataframe
        sort_key (str): Sort based on len or freq of word or char.
    """

    df["len"] = df["word"].apply(lambda x: len(x))
    sdf = df.sort_values(by=[sort_key], ascending=False)

    fix_pandas_truncation()

    print(sdf.head(2000))


def run_analysis(input: str):
    regex = UnicodeRanges.regex_cjk_chars()

    mr_word = MapRed()
    mr_char = MapRed()
    with open(input, "r", encoding="utf-8") as fd:
        for l in tqdm(fd.readlines()):
            l = l.strip()
            for word in l.split(" "):
                mr_word(word)
                # bool(regex.search(word)) and print(l)
                for char in word:
                    mr_char(char)
    analyze_topten(mr_word)
    analyze_topten(mr_char)

    df_word = mr_word.df
    analyze_head(df_word, sort_key="len")

    df_char = mr_char.df
    df_char["block"] = df_char["word"].apply(lambda x: unicodeblock.blocks.of(x))
    analyze_head(df_char, sort_key="freq")

    analyze_chars(mr_word, mr_char)


def run_post_process(input: str, output: str):
    print("Reading dataset file...")
    corpus = []
    mr_word = MapRed()
    mr_char = MapRed()
    with open(input, "r", encoding="utf-8") as fd:
        for l in tqdm(fd.readlines()):
            l = l.strip()
            corpus.append(l)
            for word in l.split(" "):
                mr_word(word)
                for char in word:
                    mr_char(char)

    # Remove duplicate lines
    corpus = list(set(corpus))

    # Remove single word lines
    # Remove 1-2 character(s) lines
    corpus = [c for c in corpus if len(c) > 2 and len(c.split()) > 1]

    print("Running word process...")
    mr_word.set_lowfreq(10)  # 10 not good for isee dataset
    procd = []
    for l in tqdm(corpus):
        p = word_post_process(l, mr_word)
        procd.append(p)

    print("Running character process...")
    corpus = "\n".join(procd)
    mr_char.set_lowfreq(100)
    corpus = char_post_process(corpus, mr_char, verbose=True)

    # Trash consecutive sequence of numbers
    print("Removing consequitive sequence of digits...")
    corpus = re.sub("(\d+ )+\d+", " ", corpus)
    corpus = re.sub("\s\s+", " ", corpus)

    print("Saving processed corpus...")
    with open(output, "w", encoding="utf-8") as fd:
        fd.write(corpus)
    print("All done!")
