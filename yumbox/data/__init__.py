import logging
from collections.abc import Callable
from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset

from yumbox.config import BFG

from .trainer import *

logger = logging.getLogger("YumBox")

no_op = lambda x: x


class PairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        hash1_col: str,
        hash2_col: str,
        mode: Literal["text_pair", "image_pair", "text_image_pair"],
        features: dict[str, np.ndarray],
        image_storage: Literal["local", "web"] = "local",
        label_col: str | None = None,
        embed_dim: int = 0,
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
        transform: Callable | None = no_op,
    ):
        self.mode = mode
        self.image_storage = image_storage
        self.embed_dim = embed_dim

        self.preprocessor = no_op if preprocessor is None else preprocessor
        self.tokenizer = no_op if tokenizer is None else tokenizer
        self.transform = no_op if transform is None else transform

        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.has_label = label_col is not None

        # 1. Filter validity based on hashes (following Img/WebImgDataset EXACT logic)
        df_valid = df[
            df[hash1_col].astype(bool)
            & df[hash1_col].notna()
            & df[hash2_col].astype(bool)
            & df[hash2_col].notna()
        ].copy()

        # 2. Compute composite hash for pairs (e.g. "textHash_imageMD5")
        # Using astype(str) guarantees safe concatenation.
        # .tolist() + zip() is up to 100x faster than iterrows() for large datasets
        hashes = (
            df_valid[hash1_col].astype(str) + "_" + df_valid[hash2_col].astype(str)
        ).tolist()
        c1_vals = df_valid[col1].tolist()
        c2_vals = df_valid[col2].tolist()

        if self.has_label:
            labels = df_valid[label_col].tolist()
            id2data = {
                h: (v1, v2, l) for h, v1, v2, l in zip(hashes, c1_vals, c2_vals, labels)
            }
        else:
            id2data = {h: (v1, v2) for h, v1, v2 in zip(hashes, c1_vals, c2_vals)}

        # 3. EXACT missing_keys architecture, now correctly deduplicating at the *pair* level
        missing_keys = set(id2data.keys()).difference(set(features.keys()))
        self.data = [(k, *id2data[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def _process_text(self, text: str):
        """EXACT TextDataset logic"""
        if pd.isna(text):
            return None
        tok = self.preprocessor(text)
        tok = self.tokenizer(tok)
        if not isinstance(tok, str):
            tok = tok.squeeze()
        return tok

    def _process_image(self, path_or_url: str):
        """Combined ImgDataset & WebImgDataset logic toggled by image_storage"""
        if pd.isna(path_or_url):
            return None

        if self.image_storage == "local":
            try:
                img = Image.open(path_or_url).convert("RGB")
                return self.transform(img)
            except Exception as e:
                logger.error(f"Error while reading image: {path_or_url}")
                logger.error(e)
                raise
        else:  # web
            try:
                response = requests.get(
                    path_or_url, stream=False, timeout=10, headers=self.headers
                )
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                return self.transform(img)
            except Exception as e:
                print(f"WARNING: download/read failed with exception: {e}")
                return torch.empty(0, self.embed_dim, dtype=torch.float32)

    def __getitem__(self, index):
        item = self.data[index]
        key = item[0]
        val1 = item[1]
        val2 = item[2]

        # Route processing based on the 3 core modalities
        if self.mode == "text_pair":
            out1 = self._process_text(val1)
            out2 = self._process_text(val2)

        elif self.mode == "image_pair":
            out1 = self._process_image(val1)
            out2 = self._process_image(val2)

        elif self.mode == "text_image_pair":
            # Convention: col1 = text, col2 = image
            out1 = self._process_text(val1)
            out2 = self._process_image(val2)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.has_label:
            label = item[3]
            return key, out1, out2, label

        return key, out1, out2


class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        hash_col: str,
        features: dict[str, np.ndarray],
        transform: Callable | None = no_op,
    ):
        self.transform = no_op if transform is None else transform

        df_wimages = df[df[hash_col].astype(bool) & df[hash_col].notna()]
        hash2path = {r[hash_col]: r[path_col] for _, r in df_wimages.iterrows()}
        missing_keys = set(hash2path.keys()).difference(set(features.keys()))
        self.data = [(k, hash2path[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, path = self.data[index]
        try:
            img = Image.open(path).convert("RGB")
        # OSError: [Errno 12] Cannot allocate memory
        # except OSError as e:
        #     logger.error(f"Error reading corrupted image: {path}")
        #     logger.error(e)
        #     raise
        except Exception as e:
            logger.error(f"Error while reading image: {path}")
            logger.error(e)
            raise
        img = self.transform(img)
        return key, img


class TextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
    ):
        self.preprocessor = no_op if preprocessor is None else preprocessor
        self.tokenizer = no_op if tokenizer is None else tokenizer

        id2text = dict(zip(df[id_col], df[text_col]))
        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        self.data = [(k, id2text[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, text = self.data[index]
        tok = self.preprocessor(text)
        tok = self.tokenizer(tok)
        if not isinstance(tok, str):
            tok = tok.squeeze()
        return key, tok


def split_token_ids(ids, chunk_size, overlap):
    start = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        chunk = ids[start:end]
        yield chunk
        start = end - overlap if end != len(ids) else len(ids)


class TFDocumentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        max_seq_length: int,
        overlap: int,
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
    ):
        if preprocessor is None:
            self.preprocessor = no_op
        else:
            self.preprocessor = preprocessor

        if tokenizer is None:
            self.tokenizer = no_op
        else:
            self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length
        self.overlap = overlap

        assert hasattr(self.tokenizer, "encode"), "BertTokenizerFast expected"
        assert hasattr(self.tokenizer, "decode"), "BertTokenizerFast expected"

        id2text = dict(zip(df[id_col], df[text_col]))
        id2text = {k: v for k, v in id2text.items() if k and pd.notna(k)}

        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        self.data = [(k, id2text[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, text = self.data[index]
        prep = self.preprocessor(text)
        tok = self.tokenizer.encode(prep, truncation=False)
        if len(tok) > self.max_seq_length + self.overlap:
            token_chunks = split_token_ids(
                tok, chunk_size=self.max_seq_length, overlap=self.overlap
            )
            text_chunks = []
            for i, chunk in enumerate(token_chunks):
                chunk_text = self.tokenizer.decode(
                    chunk,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                text_chunks.append(chunk_text)
        else:
            text_chunks = [prep]

        return idx, text_chunks


class ZeroshotDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        templates: list[str],
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
    ):
        self.templates = templates
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        id2text = dict(zip(df[id_col], df[text_col]))
        id2text = {k: v for k, v in id2text.items() if k and pd.notna(k)}

        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        data = [(k, id2text[k]) for k in missing_keys]

        self.data = [d + (t,) for d in data for t in templates]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, cls, temp = self.data[index]
        prompt = self.tokenizer(temp.format(self.preprocessor(cls)))
        prompt = prompt.squeeze()
        return idx, prompt


def fix_pandas_truncation():
    # pd.options.display.x = None
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    # pd.set_option("display.max_seq_item", None)
    # pd.set_option("display.colheader_justify", "left")
