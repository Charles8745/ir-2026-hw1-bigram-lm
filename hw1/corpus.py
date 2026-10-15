"""讀取維基語料（JSON 陣列、每行一篇）、分句、切子集。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_SENT_END = re.compile(r"(?<=[。！？])")
_CLOSERS = "」』）)〕】》"


@dataclass
class ReadStats:
    read: int = 0
    skipped_bad: int = 0
    skipped_empty: int = 0


def _iter_with_stats(path: str | Path, limit: int | None, stats: ReadStats) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats.read >= limit:
                return
            s = line.strip()
            if s in ("[", "]", ""):
                continue
            if s.endswith(","):
                s = s[:-1].rstrip()
            try:
                article = json.loads(s)
            except json.JSONDecodeError:
                stats.skipped_bad += 1
                continue
            if not isinstance(article, str) or not article.strip():
                stats.skipped_empty += 1
                continue
            stats.read += 1
            yield article


def iter_articles(path: str | Path, limit: int | None = None, stats: ReadStats | None = None) -> Iterator[str]:
    """逐篇產出文章字串。壞行與空文章跳過；傳入 stats 可取得筆數。"""
    yield from _iter_with_stats(path, limit, stats if stats is not None else ReadStats())


def read_articles(path: str | Path, limit: int | None = None) -> tuple[list[str], ReadStats]:
    stats = ReadStats()
    arts = list(_iter_with_stats(path, limit, stats))
    return arts, stats


def split_sentences(text: str) -> list[str]:
    """以 。！？ 切句，標點留在句尾；閉合引號併回前句；空句丟棄。"""
    out: list[str] = []
    for piece in _SENT_END.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if out and piece[0] in _CLOSERS:
            closers = ""
            while piece and piece[0] in _CLOSERS:
                closers += piece[0]
                piece = piece[1:]
            out[-1] += closers
            piece = piece.strip()
            if not piece:
                continue
        out.append(piece)
    return out


def write_subset(src: str | Path, dst: str | Path, n_articles: int) -> int:
    """把前 n 篇寫成與原語料同格式的 JSON 陣列檔。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, art in enumerate(iter_articles(src, limit=n_articles)):
            if i:
                f.write(",\n")
            f.write("  " + json.dumps(art, ensure_ascii=False))
            n += 1
        f.write("\n]\n")
    return n
