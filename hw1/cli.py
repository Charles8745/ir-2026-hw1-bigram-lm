"""python -m hw1 <command> ..."""
from __future__ import annotations

import argparse

import numpy as np


def cmd_subset(a: argparse.Namespace) -> None:
    from hw1.corpus import write_subset

    n = write_subset(a.corpus, a.out, a.n_articles)
    print(f"wrote {n} articles to {a.out}")


def cmd_count(a: argparse.Namespace) -> None:
    from hw1.counting import run_count

    meta = run_count(a.corpus, a.out, limit=a.limit, workers=a.workers,
                     chunk_size=a.chunk_size, trigram=a.trigram, timings_path=a.timings)
    print(f"N={meta['n_tokens']} (no markers {meta['n_tokens_no_markers']}) "
          f"V={meta['vocab_size']} (no markers {meta['vocab_size_no_markers']}) "
          f"B={meta['n_bigrams']} T={meta['n_trigrams']} "
          f"t_segment={meta['time_segment_s']}s t_bigram={meta['time_count_bigram_s']}s "
          f"t_total={meta['time_total_s']}s")


def cmd_dump_bigrams(a: argparse.Namespace) -> None:
    from hw1.store import dump_bigrams_text, load_counts

    b = load_counts(a.counts)
    n = dump_bigrams_text(b.tokens, b.vocab, a.out)
    print(f"wrote {n} bigram lines to {a.out}")


def cmd_top_bigrams(a: argparse.Namespace) -> None:
    from hw1.counting import unpack
    from hw1.store import load_counts

    b = load_counts(a.counts, with_tokens=False)
    order = np.argsort(-b.bigram_counts, kind="stable")[: a.k]
    w1, w2 = unpack(b.bigram_keys[order])
    print(f"B={len(b.bigram_keys)}")
    for c, x, y in zip(b.bigram_counts[order], w1, w2):
        print(f"{int(c)}\t{b.vocab.word(x)}\t{b.vocab.word(y)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hw1")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("subset", help="write the first N articles as a same-format JSON file")
    s.add_argument("--corpus", required=True)
    s.add_argument("--n-articles", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_subset)

    s = sub.add_parser("count", help="segment + count unigrams/bigrams(/trigrams)")
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--chunk-size", type=int, default=2000)
    s.add_argument("--trigram", action="store_true")
    s.add_argument("--timings", default="results/timings.json")
    s.set_defaults(func=cmd_count)

    s = sub.add_parser("dump-bigrams", help="dump 'w1<TAB>w2' lines for sort|uniq cross-check")
    s.add_argument("--counts", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_dump_bigrams)

    s = sub.add_parser("top-bigrams", help="print B and top-k bigrams")
    s.add_argument("--counts", required=True)
    s.add_argument("--k", type=int, default=20)
    s.set_defaults(func=cmd_top_bigrams)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
