import json
from pathlib import Path

from hw1.corpus import ReadStats, iter_articles, read_articles, split_sentences, write_subset

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def test_read_toy_articles():
    arts, stats = read_articles(FIXTURE)
    assert arts == ["我喜歡你。我喜歡他。", "我討厭你。"]
    assert stats == ReadStats(read=2, skipped_bad=0, skipped_empty=0)


def test_iter_articles_limit():
    assert list(iter_articles(FIXTURE, limit=1)) == ["我喜歡你。我喜歡他。"]


def test_iter_articles_fills_caller_stats():
    stats = ReadStats()
    list(iter_articles(FIXTURE, stats=stats))
    assert stats == ReadStats(read=2, skipped_bad=0, skipped_empty=0)


def test_reader_handles_bad_and_empty_lines(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('[\n  "甲。",\n  not json,\n  "",\n  "   ",\n  "乙。"\n]\n', encoding="utf-8")
    arts, stats = read_articles(p)
    assert arts == ["甲。", "乙。"]
    assert stats.skipped_bad == 1
    assert stats.skipped_empty == 2


def test_split_sentences_basic():
    assert split_sentences("我喜歡你。我喜歡他。") == ["我喜歡你。", "我喜歡他。"]


def test_split_sentences_keeps_tail_without_punct():
    assert split_sentences("今天很好。明天呢") == ["今天很好。", "明天呢"]


def test_split_sentences_attaches_closing_quote():
    assert split_sentences("他說：「你好！」然後走了？") == ["他說：「你好！」", "然後走了？"]


def test_split_sentences_drops_empty():
    assert split_sentences("。。 甲！") == ["。", "。", "甲！"]
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_write_subset_roundtrip(tmp_path):
    dst = tmp_path / "sub.json"
    n = write_subset(FIXTURE, dst, 1)
    assert n == 1
    arts, _ = read_articles(dst)
    assert arts == ["我喜歡你。我喜歡他。"]
    # 格式與原語料一致：第一行 [，最後一行 ]
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[" and lines[-1] == "]"
    assert json.loads(lines[1].rstrip(",")) == "我喜歡你。我喜歡他。"
