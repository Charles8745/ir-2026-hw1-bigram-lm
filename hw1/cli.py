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


def _load_model(counts: str, trigram: bool = False):
    from hw1.model import BigramModel, TrigramModel

    return (TrigramModel if trigram else BigramModel).load(counts)


def cmd_predict(a: argparse.Namespace) -> None:
    m = _load_model(a.counts)
    rows = m.predict(a.word, a.k)
    if not rows:
        print(f"'{a.word}' not in vocabulary or has no successors")
        return
    for w, p in rows:
        print(f"{w}\t{p:.4f}")


def cmd_prob(a: argparse.Namespace) -> None:
    m = _load_model(a.counts, a.trigram)
    r = m.prob(a.sentence)
    print("tokens:", " ".join(" ".join(s) for s in r.sentences))
    for s in r.steps:
        print(f"  P({s.w2} | {s.w1}) = {s.count}/{s.context_count} = {s.prob:.6g}")
    print(f"log2 P = {r.logprob:.4f}")
    if r.first_unseen:
        print(f"first unseen pair: {r.first_unseen}")


def cmd_generate(a: argparse.Namespace) -> None:
    m = _load_model(a.counts, a.trigram)
    r = m.generate(a.seed, a.n, mode=a.mode, rng_seed=a.rng_seed)
    print(r.text)
    if r.mode == "greedy":
        print(f"cycle_start={r.cycle_start} cycle_length={r.cycle_length} sentences={r.n_sentences}")


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

    s = sub.add_parser("predict", help="top-k next words after W (bigram)")
    s.add_argument("word")
    s.add_argument("k", type=int)
    s.add_argument("--counts", default="data/counts/full")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("prob", help="log2 probability of a sentence")
    s.add_argument("sentence")
    s.add_argument("--counts", default="data/counts/full")
    s.add_argument("--trigram", action="store_true")
    s.set_defaults(func=cmd_prob)

    s = sub.add_parser("generate", help="generate n words from seed")
    s.add_argument("--seed", default="")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    s.add_argument("--rng-seed", type=int, default=None)
    s.add_argument("--counts", default="data/counts/full")
    s.add_argument("--trigram", action="store_true")
    s.set_defaults(func=cmd_generate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
