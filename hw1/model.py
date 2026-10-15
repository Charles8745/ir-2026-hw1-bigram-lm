"""Bigram / Trigram 語言模型：predict、prob、generate。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hw1 import BOS_ID, EOS, EOS_ID
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
    w0: str | None = None  # trigram 步驟的前前詞；bigram 步驟為 None


@dataclass
class ProbResult:
    logprob: float
    sentences: list[list[str]]
    steps: list[Step]
    first_unseen: tuple[str, ...] | None


@dataclass
class GenerateResult:
    words: list[str]
    text: str
    mode: str
    cycle_start: int | None
    cycle_length: int | None
    n_sentences: int


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

    def _bigram_index(self, w1_id: int, w2_id: int) -> int | None:
        key = (int(w1_id) << 32) | int(w2_id)
        i = np.searchsorted(self.bigram_keys, key)
        if i < len(self.bigram_keys) and self.bigram_keys[i] == key:
            return int(i)
        return None

    def bigram_count(self, w1_id: int, w2_id: int) -> int:
        i = self._bigram_index(w1_id, w2_id)
        return 0 if i is None else int(self.bigram_counts[i])

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

    @staticmethod
    def _score(steps: list[Step]) -> tuple[float, tuple[str, ...] | None]:
        """累加 log2；遇到 prob 為 0 的步驟整體變 −inf，並記下第一個未見配對（trigram 步驟為三元組）。"""
        logprob = 0.0
        first_unseen = None
        for st in steps:
            if st.prob == 0.0:
                logprob = -math.inf
                if first_unseen is None:
                    first_unseen = (st.w0, st.w1, st.w2) if st.w0 is not None else (st.w1, st.w2)
            elif logprob != -math.inf:
                logprob += math.log2(st.prob)
        return logprob, first_unseen

    def prob(self, sentence: str) -> ProbResult:
        sentences = tokenize_article(sentence)
        steps = [self._pair_prob(w1, w2) for sent in sentences for w1, w2 in zip(sent[:-1], sent[1:])]
        logprob, first_unseen = self._score(steps)
        return ProbResult(logprob, sentences, steps, first_unseen)

    # ---- generate ----
    def _context_from_seed(self, seed: str) -> list[int]:
        """seed 斷詞後的 ID 序列（不含標記）；空或全 OOV 時回 [BOS_ID]。"""
        ids = [self.vocab.id(t) for sent in tokenize_article(seed) for t in sent[1:-1]]
        ids = [i for i in ids if i is not None]
        return ids or [BOS_ID]

    def _next_candidates(self, context: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (ids, counts)，已排除 <s>。子類別可覆寫以使用更長的前文。"""
        ids, counts = self.successors(context[-1])
        keep = ids != BOS_ID
        return ids[keep], counts[keep]

    def _state(self, context: list[int]) -> tuple[int, ...]:
        return (context[-1],)

    def generate(self, seed: str, n: int, mode: str = "greedy", rng_seed: int | None = None) -> GenerateResult:
        if mode not in ("greedy", "sample"):
            raise ValueError(f"unknown mode {mode!r}")
        rng = np.random.default_rng(rng_seed)
        context = self._context_from_seed(seed)
        words: list[str] = []
        seen: dict[tuple[int, ...], int] = {}
        cycle_start = cycle_length = None
        n_sent = 0
        step = 0
        while step < n:
            ids, counts = self._next_candidates(context)
            if len(ids) == 0:
                context = [BOS_ID]
                ids, counts = self._next_candidates(context)
            if mode == "greedy":
                state = self._state(context)
                if cycle_start is None and state in seen:
                    cycle_start = step
                    cycle_length = step - seen[state]
                seen.setdefault(state, step)
                nxt = int(ids[0])
            else:
                nxt = int(rng.choice(ids, p=counts / counts.sum()))
            words.append(self.vocab.word(nxt))
            step += 1
            if nxt == EOS_ID:
                n_sent += 1
                context = [BOS_ID]
            else:
                context = context + [nxt]
        text = "".join(w if w != EOS else " ｜ " for w in words).strip()
        return GenerateResult(words, text, mode, cycle_start, cycle_length, n_sent)


class TrigramModel(BigramModel):
    def __init__(self, vocab, unigram, bigram_keys, bigram_counts, trigram_keys, trigram_counts, meta):
        super().__init__(vocab, unigram, bigram_keys, bigram_counts, meta)
        if trigram_keys is None:
            raise ValueError("counts directory has no trigram table; run count with --trigram")
        self.trigram_keys = trigram_keys
        self.trigram_counts = trigram_counts

    @classmethod
    def load(cls, counts_dir: str | Path) -> "TrigramModel":
        b = load_counts(counts_dir, with_tokens=False)
        return cls(b.vocab, b.unigram, b.bigram_keys, b.bigram_counts, b.trigram_keys, b.trigram_counts, b.meta)

    def trigram_successors(self, w1_id: int, w2_id: int) -> tuple[np.ndarray, np.ndarray]:
        bi = self._bigram_index(w1_id, w2_id)
        if bi is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        # trigram key = (bigram_index << 32) | w3；同一 bigram 的三元組是連續區間
        s = np.searchsorted(self.trigram_keys, bi << 32)
        e = np.searchsorted(self.trigram_keys, (bi + 1) << 32)
        _, ids = unpack(self.trigram_keys[s:e])
        counts = self.trigram_counts[s:e]
        order = np.lexsort((ids, -counts))
        return ids[order], counts[order]

    def _next_candidates(self, context: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if len(context) >= 2:
            ids, counts = self.trigram_successors(context[-2], context[-1])
            if len(ids):
                keep = ids != BOS_ID
                return ids[keep], counts[keep]
        return super()._next_candidates(context)

    def _state(self, context: list[int]) -> tuple[int, ...]:
        return tuple(context[-2:])

    def _trigram_step(self, w0: str, w1: str, w2: str) -> Step:
        a, b, c = self.vocab.id(w0), self.vocab.id(w1), self.vocab.id(w2)
        bi = None if (a is None or b is None) else self._bigram_index(a, b)
        if bi is None:
            return Step(w1, w2, 0, 0, 0.0, w0)
        ctx = int(self.bigram_counts[bi])
        cnt = 0
        if c is not None:
            key = (bi << 32) | c
            i = int(np.searchsorted(self.trigram_keys, key))
            if i < len(self.trigram_keys) and self.trigram_keys[i] == key:
                cnt = int(self.trigram_counts[i])
        return Step(w1, w2, cnt, ctx, cnt / ctx if ctx else 0.0, w0)

    def prob(self, sentence: str) -> ProbResult:
        """第一對用 bigram，其後用 P(w3|w1,w2)。"""
        sentences = tokenize_article(sentence)
        steps: list[Step] = []
        for sent in sentences:
            for i in range(1, len(sent)):
                if i == 1:
                    steps.append(self._pair_prob(sent[0], sent[1]))
                else:
                    steps.append(self._trigram_step(sent[i - 2], sent[i - 1], sent[i]))
        logprob, first_unseen = self._score(steps)
        return ProbResult(logprob, sentences, steps, first_unseen)
