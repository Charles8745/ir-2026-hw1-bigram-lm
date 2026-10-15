"""Bigram / Trigram 語言模型：predict、prob、generate。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hw1.counting import Vocab, unpack
from hw1.segment import tokenize_article
from hw1.store import load_counts


@dataclass
class Step:
    w1: str
    w2: str
    count: int
    context_count: int
    prob: float


@dataclass
class ProbResult:
    logprob: float
    sentences: list[list[str]]
    steps: list[Step]
    first_unseen: tuple[str, str] | None


class BigramModel:
    def __init__(self, vocab: Vocab, unigram: np.ndarray, bigram_keys: np.ndarray, bigram_counts: np.ndarray, meta: dict):
        self.vocab = vocab
        self.unigram = unigram
        self.bigram_keys = bigram_keys
        self.bigram_counts = bigram_counts
        self.meta = meta
        # 每個 w1 的區間 [row_start[w1], row_start[w1+1])
        hi, _ = unpack(bigram_keys)
        self.row_start = np.searchsorted(hi, np.arange(len(vocab) + 1), side="left")
        # 每個 w1 作為前詞的總次數；float64 權重在 2^53 以下精確
        self.row_total = np.bincount(hi, weights=bigram_counts, minlength=len(vocab)).astype(np.int64)

    @classmethod
    def load(cls, counts_dir: str | Path) -> "BigramModel":
        b = load_counts(counts_dir, with_tokens=False)
        return cls(b.vocab, b.unigram, b.bigram_keys, b.bigram_counts, b.meta)

    # ---- 查表 ----
    def successors(self, w_id: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.row_start[w_id], self.row_start[w_id + 1]
        _, ids = unpack(self.bigram_keys[s:e])
        counts = self.bigram_counts[s:e]
        order = np.lexsort((ids, -counts))
        return ids[order], counts[order]

    def context_count(self, w_id: int) -> int:
        return int(self.row_total[w_id])

    def bigram_count(self, w1_id: int, w2_id: int) -> int:
        key = (int(w1_id) << 32) | int(w2_id)
        i = np.searchsorted(self.bigram_keys, key)
        if i < len(self.bigram_keys) and self.bigram_keys[i] == key:
            return int(self.bigram_counts[i])
        return 0

    # ---- Part 2 ----
    def predict(self, w: str, k: int) -> list[tuple[str, float]]:
        wid = self.vocab.id(w)
        if wid is None or self.context_count(wid) == 0:
            return []
        ids, counts = self.successors(wid)
        total = self.context_count(wid)
        return [(self.vocab.word(i), float(c) / total) for i, c in zip(ids[:k], counts[:k])]

    def _pair_prob(self, w1: str, w2: str) -> Step:
        a, b = self.vocab.id(w1), self.vocab.id(w2)
        if a is None or b is None:
            return Step(w1, w2, 0, 0 if a is None else self.context_count(a), 0.0)
        c = self.bigram_count(a, b)
        ctx = self.context_count(a)
        return Step(w1, w2, c, ctx, (c / ctx) if ctx else 0.0)

    def prob(self, sentence: str) -> ProbResult:
        sentences = tokenize_article(sentence)
        steps: list[Step] = []
        first_unseen = None
        logprob = 0.0
        for sent in sentences:
            for w1, w2 in zip(sent[:-1], sent[1:]):
                st = self._pair_prob(w1, w2)
                steps.append(st)
                if st.prob == 0.0:
                    logprob = -math.inf
                    if first_unseen is None:
                        first_unseen = (w1, w2)
                elif logprob != -math.inf:
                    logprob += math.log2(st.prob)
        return ProbResult(logprob, sentences, steps, first_unseen)
