"""計數結果的存檔與載入。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hw1 import EOS_ID
from hw1.counting import Vocab

FILES = {
    "tokens": "tokens.npy",
    "article_offsets": "article_offsets.npy",
    "unigram": "unigram_counts.npy",
    "bigram_keys": "bigram_keys.npy",
    "bigram_counts": "bigram_counts.npy",
    "trigram_keys": "trigram_keys.npy",
    "trigram_counts": "trigram_counts.npy",
}


@dataclass
class CountsBundle:
    tokens: np.ndarray | None
    vocab: Vocab
    article_offsets: np.ndarray
    unigram: np.ndarray
    bigram_keys: np.ndarray
    bigram_counts: np.ndarray
    trigram_keys: np.ndarray | None
    trigram_counts: np.ndarray | None
    meta: dict


def save_counts(out_dir: str | Path, b: CountsBundle) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / FILES["tokens"], b.tokens)
    np.save(out / FILES["article_offsets"], b.article_offsets)
    np.save(out / FILES["unigram"], b.unigram)
    np.save(out / FILES["bigram_keys"], b.bigram_keys)
    np.save(out / FILES["bigram_counts"], b.bigram_counts)
    if b.trigram_keys is not None:
        np.save(out / FILES["trigram_keys"], b.trigram_keys)
        np.save(out / FILES["trigram_counts"], b.trigram_counts)
    else:
        # 同目錄先前的 trigram 檔其 key 是舊 bigram 表的序號，不可留下
        for name in ("trigram_keys", "trigram_counts"):
            (out / FILES[name]).unlink(missing_ok=True)
    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for w in b.vocab.words:
            f.write(w + "\n")
    (out / "meta.json").write_text(json.dumps(b.meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_counts(dir: str | Path, with_tokens: bool = True) -> CountsBundle:
    d = Path(dir)
    words = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if words and words[-1] == "":
        words.pop()
    tri_k = d / FILES["trigram_keys"]
    return CountsBundle(
        tokens=np.load(d / FILES["tokens"]) if with_tokens else None,
        vocab=Vocab(words),
        article_offsets=np.load(d / FILES["article_offsets"]),
        unigram=np.load(d / FILES["unigram"]),
        bigram_keys=np.load(d / FILES["bigram_keys"]),
        bigram_counts=np.load(d / FILES["bigram_counts"]),
        trigram_keys=np.load(tri_k) if tri_k.exists() else None,
        trigram_counts=np.load(d / FILES["trigram_counts"]) if tri_k.exists() else None,
        meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
    )


def dump_bigrams_text(tokens: np.ndarray, vocab: Vocab, path: str | Path) -> int:
    """每行 'w1<TAB>w2'，跳過 w1 == </s>。供 sort | uniq -c 對照。"""
    n = 0
    words = vocab.words
    with open(path, "w", encoding="utf-8") as f:
        w1 = tokens[:-1]
        w2 = tokens[1:]
        for a, b in zip(w1.tolist(), w2.tolist()):
            if a == EOS_ID:
                continue
            f.write(f"{words[a]}\t{words[b]}\n")
            n += 1
    return n
