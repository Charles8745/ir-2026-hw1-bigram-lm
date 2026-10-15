import json
from pathlib import Path

import numpy as np

from hw1 import BOS, EOS
from hw1.counting import run_count
from hw1.store import dump_bigrams_text, load_counts
from hw1.timing import Stopwatch, record_timing

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def test_stopwatch_and_record_timing(tmp_path):
    p = tmp_path / "timings.json"
    with Stopwatch() as sw:
        pass
    assert sw.seconds >= 0
    record_timing("count", 1.5, path=p, n_articles=2)
    record_timing("count", 2.5, path=p, n_articles=3)
    data = json.loads(p.read_text())
    assert [d["seconds"] for d in data] == [1.5, 2.5]
    assert data[1]["n_articles"] == 3 and "timestamp" in data[1]


def test_run_count_toy_meta_and_roundtrip(tmp_path):
    out = tmp_path / "counts"
    meta = run_count(FIXTURE, out, workers=1, trigram=True, timings_path=tmp_path / "t.json")
    assert meta["n_articles"] == 2
    assert meta["skipped_bad_lines"] == 0 and meta["skipped_empty_articles"] == 0
    assert meta["n_sentences"] == 3
    assert meta["n_tokens"] == 18 and meta["n_tokens_no_markers"] == 12
    assert meta["vocab_size"] == 8 and meta["vocab_size_no_markers"] == 6
    assert meta["n_bigrams"] == 9 and meta["bigram_tokens_total"] == 15
    assert meta["n_trigrams"] == 10
    assert meta["dictionary"] == "dict.txt.big" and meta["jieba_version"]
    assert (out / "meta.json").exists()
    b = load_counts(out)
    assert b.vocab.words[:2] == [BOS, EOS]
    assert b.tokens.tolist()[:6] == [0, 2, 3, 4, 5, 1]
    assert b.article_offsets.tolist() == [0, 12, 18]
    assert b.unigram.tolist() == [3, 3, 3, 2, 2, 3, 1, 1]
    assert len(b.bigram_keys) == 9 and b.bigram_counts.sum() == 15
    assert len(b.trigram_keys) == 10
    assert b.meta["n_bigrams"] == 9
    timings = json.loads((tmp_path / "t.json").read_text())
    assert timings[-1]["command"] == "count"


def test_load_without_tokens(tmp_path):
    out = tmp_path / "counts"
    run_count(FIXTURE, out, workers=1, timings_path=tmp_path / "t.json")
    b = load_counts(out, with_tokens=False)
    assert b.tokens is None and b.trigram_keys is None
    assert len(b.bigram_keys) == 9


def test_dump_bigrams_text(tmp_path):
    out = tmp_path / "counts"
    run_count(FIXTURE, out, workers=1, timings_path=tmp_path / "t.json")
    b = load_counts(out)
    p = tmp_path / "bigrams.txt"
    n = dump_bigrams_text(b.tokens, b.vocab, p)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert n == 15 and len(lines) == 15
    assert lines[0] == f"{BOS}\t我"
    assert f"{EOS}\t{BOS}" not in lines


def test_rerun_without_trigram_clears_stale_trigram_files(tmp_path):
    out = tmp_path / "counts"
    run_count(FIXTURE, out, workers=1, trigram=True, timings_path=tmp_path / "t.json")
    assert (out / "trigram_keys.npy").exists()
    other = tmp_path / "other.json"
    other.write_text('[\n  "我討厭你。"\n]\n', encoding="utf-8")
    meta = run_count(other, out, workers=1, trigram=False, timings_path=tmp_path / "t.json")
    assert meta["n_trigrams"] is None
    b = load_counts(out)
    assert b.trigram_keys is None and b.trigram_counts is None
    assert not (out / "trigram_keys.npy").exists()
    assert not (out / "trigram_counts.npy").exists()
