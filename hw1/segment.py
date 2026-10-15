"""jieba 斷詞、<s> </s> 標記、把文章編成本地 int32 ID 的 Chunk，支援多行程。"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Iterator

import jieba
import numpy as np

from hw1 import BOS, DICT_PATH, EOS
from hw1.corpus import ReadStats, iter_articles, split_sentences

_initialized = False


def init_jieba() -> None:
    """載入 dict.txt.big；每個行程只做一次。"""
    global _initialized
    if _initialized:
        return
    jieba.setLogLevel(logging.ERROR)
    jieba.set_dictionary(str(DICT_PATH))
    jieba.initialize()
    _initialized = True


def tokenize_sentence(sentence: str) -> list[str]:
    init_jieba()
    return [t for t in jieba.cut(sentence) if not t.isspace()]


def tokenize_article(text: str) -> list[list[str]]:
    out = []
    for sent in split_sentences(text):
        toks = tokenize_sentence(sent)
        if toks:
            out.append([BOS, *toks, EOS])
    return out


@dataclass
class Chunk:
    vocab: list[str]
    ids: np.ndarray            # int32，本地 ID
    article_starts: np.ndarray  # int64，每篇文章在 ids 中的起點
    n_sentences: int


def encode_articles(articles: list[str]) -> Chunk:
    index: dict[str, int] = {BOS: 0, EOS: 1}
    vocab = [BOS, EOS]
    ids: list[int] = []
    starts: list[int] = []
    n_sent = 0
    for art in articles:
        starts.append(len(ids))
        for sent in tokenize_article(art):
            n_sent += 1
            for tok in sent:
                gid = index.get(tok)
                if gid is None:
                    gid = len(vocab)
                    index[tok] = gid
                    vocab.append(tok)
                ids.append(gid)
    return Chunk(
        vocab=vocab,
        ids=np.asarray(ids, dtype=np.int32),
        article_starts=np.asarray(starts, dtype=np.int64),
        n_sentences=n_sent,
    )


def _batches(articles: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for a in articles:
        batch.append(a)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def segment_corpus(
    path: str | Path,
    limit: int | None = None,
    workers: int = 1,
    chunk_size: int = 2000,
    stats: ReadStats | None = None,
) -> Iterator[Chunk]:
    """依語料順序產出 Chunk。workers>1 時用 spawn Pool，每個行程各自載入 jieba。

    不用 Pool.imap：它會把整個輸入先讀進任務佇列。這裡手動維持最多 2*workers 個在途任務。
    """
    batches = _batches(iter_articles(path, limit=limit, stats=stats), chunk_size)
    if workers <= 1:
        for b in batches:
            yield encode_articles(b)
        return
    init_jieba()  # 先在主行程建好 jieba 的字典快取，避免子行程同時寫快取
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_jieba) as pool:
        pending: deque = deque()
        for b in batches:
            pending.append(pool.apply_async(encode_articles, (b,)))
            if len(pending) >= workers * 2:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()
