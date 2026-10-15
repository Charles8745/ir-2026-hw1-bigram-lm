"""token ID 串流建立與 unigram / bigram / trigram 計數（numpy）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from hw1 import BOS, BOS_ID, EOS, EOS_ID
from hw1.segment import Chunk


@dataclass
class Vocab:
    words: list[str]
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def id(self, word: str) -> int | None:
        return self.index.get(word)

    def word(self, i: int) -> str:
        return self.words[int(i)]


def build_token_stream(chunks: Iterable[Chunk]) -> tuple[np.ndarray, Vocab, np.ndarray, int]:
    """合併各 chunk 的本地詞表為全域詞表，重映射 ID，串成一條 int32 串流。"""
    index: dict[str, int] = {BOS: BOS_ID, EOS: EOS_ID}
    words = [BOS, EOS]
    parts: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    total = 0
    n_sent = 0
    for ch in chunks:
        remap = np.empty(len(ch.vocab), dtype=np.int32)
        for i, w in enumerate(ch.vocab):
            gid = index.get(w)
            if gid is None:
                gid = len(words)
                index[w] = gid
                words.append(w)
            remap[i] = gid
        parts.append(remap[ch.ids])
        offsets.append(ch.article_starts + total)
        total += len(ch.ids)
        n_sent += ch.n_sentences
    tokens = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    offsets.append(np.array([total], dtype=np.int64))
    return tokens, Vocab(words), np.concatenate(offsets), n_sent


def pack(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    return (hi.astype(np.int64) << 32) | lo.astype(np.int64)


def unpack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = keys.astype(np.int64)
    return keys >> 32, keys & 0xFFFFFFFF


def count_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """對 int64 key 陣列做排序計數。記憶體不足時只需改這個函式為分片合併。"""
    u, c = np.unique(keys, return_counts=True)
    return u, c.astype(np.int64)


def count_unigrams(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(tokens, minlength=vocab_size).astype(np.int64)


def count_bigrams(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w1 = tokens[:-1]
    w2 = tokens[1:]
    mask = w1 != EOS_ID
    return count_keys(pack(w1[mask], w2[mask]))


def count_trigrams(tokens: np.ndarray, bigram_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """key = (w1,w2) 在 bigram_keys 中的序號 << 32 | w3。排除 w1 或 w2 為 </s> 的位置。"""
    w1 = tokens[:-2]
    w2 = tokens[1:-1]
    w3 = tokens[2:]
    mask = (w1 != EOS_ID) & (w2 != EOS_ID)
    k2 = pack(w1[mask], w2[mask])
    del w1, w2
    idx = np.searchsorted(bigram_keys, k2)
    if not (np.all(idx < len(bigram_keys)) and np.array_equal(bigram_keys[idx], k2)):
        raise ValueError("bigram table does not cover all bigrams in the token stream")
    del k2
    k3 = pack(idx, w3[mask])
    del idx, w3, mask
    return count_keys(k3)


def run_count(
    corpus: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    workers: int = 10,
    chunk_size: int = 2000,
    trigram: bool = False,
    timings_path: str | Path = "results/timings.json",
) -> dict:
    """完整流程：斷詞 → token 串流 → unigram/bigram(/trigram) → 存檔。回傳 meta。"""
    from datetime import datetime, timezone

    import jieba

    from hw1.corpus import ReadStats
    from hw1.segment import segment_corpus
    from hw1.store import CountsBundle, save_counts
    from hw1.timing import Stopwatch, record_timing

    stats = ReadStats()
    with Stopwatch() as total:
        with Stopwatch() as t_seg:
            tokens, vocab, offsets, n_sent = build_token_stream(
                segment_corpus(corpus, limit=limit, workers=workers, chunk_size=chunk_size, stats=stats)
            )
        unigram = count_unigrams(tokens, len(vocab))
        with Stopwatch() as t_bi:
            bk, bc = count_bigrams(tokens)
        tk = tc = None
        t_tri_s = None
        if trigram:
            with Stopwatch() as t_tri:
                tk, tc = count_trigrams(tokens, bk)
            t_tri_s = t_tri.seconds
    n_markers = int(unigram[BOS_ID] + unigram[EOS_ID])
    meta = {
        "corpus": str(corpus),
        "limit": limit,
        "n_articles": int(len(offsets) - 1),
        "skipped_bad_lines": stats.skipped_bad,
        "skipped_empty_articles": stats.skipped_empty,
        "n_sentences": int(n_sent),
        "n_tokens": int(tokens.size),
        "n_tokens_no_markers": int(tokens.size - n_markers),
        "vocab_size": len(vocab),
        "vocab_size_no_markers": len(vocab) - 2,
        "n_bigrams": int(len(bk)),
        "bigram_tokens_total": int(bc.sum()),
        "n_trigrams": int(len(tk)) if tk is not None else None,
        "trigram_tokens_total": int(tc.sum()) if tc is not None else None,
        "time_segment_s": round(t_seg.seconds, 3),
        "time_count_bigram_s": round(t_bi.seconds, 3),
        "time_count_trigram_s": round(t_tri_s, 3) if t_tri_s is not None else None,
        "time_total_s": round(total.seconds, 3),
        "workers": workers,
        "chunk_size": chunk_size,
        "jieba_version": jieba.__version__,
        "dictionary": "dict.txt.big",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_counts(out_dir, CountsBundle(tokens, vocab, offsets, unigram, bk, bc, tk, tc, meta))
    record_timing("count", total.seconds, path=timings_path, out=str(out_dir),
                  n_articles=meta["n_articles"], n_tokens=meta["n_tokens"], trigram=trigram)
    return meta
