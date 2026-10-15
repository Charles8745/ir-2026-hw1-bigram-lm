import json
from pathlib import Path

import numpy as np

from hw1 import BOS, EOS
from hw1.corpus import ReadStats
from hw1.segment import Chunk, encode_articles, segment_corpus, tokenize_article, tokenize_sentence

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def test_tokenize_sentence_drops_whitespace_tokens():
    assert tokenize_sentence("我 喜歡\t你\n。") == ["我", "喜歡", "你", "。"]


def test_tokenize_article_wraps_markers():
    assert tokenize_article("我喜歡你。我喜歡他。") == [
        [BOS, "我", "喜歡", "你", "。", EOS],
        [BOS, "我", "喜歡", "他", "。", EOS],
    ]


def test_tokenize_article_skips_sentences_that_tokenize_to_nothing():
    assert tokenize_article("  \n  ") == []
    assert tokenize_article("我。\n\n") == [[BOS, "我", "。", EOS]]


def test_encode_articles_toy():
    ch = encode_articles(["我喜歡你。我喜歡他。", "我討厭你。"])
    assert ch.vocab[:2] == [BOS, EOS]
    assert ch.vocab == [BOS, EOS, "我", "喜歡", "你", "。", "他", "討厭"]
    assert ch.ids.dtype == np.int32
    assert ch.ids.tolist() == [0, 2, 3, 4, 5, 1, 0, 2, 3, 6, 5, 1, 0, 2, 7, 4, 5, 1]
    assert ch.article_starts.tolist() == [0, 12]
    assert ch.n_sentences == 3


def test_encode_empty_article_keeps_offsets_monotonic():
    ch = encode_articles(["我。", "   ", "你。"])
    assert ch.article_starts.tolist() == [0, 4, 4]
    assert len(ch.ids) == 8


def test_segment_corpus_matches_single_chunk():
    single = encode_articles(["我喜歡你。我喜歡他。", "我討厭你。"])
    chunks = list(segment_corpus(FIXTURE, workers=1, chunk_size=1))
    assert len(chunks) == 2
    assert isinstance(chunks[0], Chunk)
    # 兩個 chunk 各自有本地詞表，拼回字串後要和單一 chunk 的結果相同
    words = []
    for ch in chunks:
        words.extend(ch.vocab[i] for i in ch.ids)
    assert words == [single.vocab[i] for i in single.ids]


def test_segment_corpus_multiprocess_same_result():
    a = list(segment_corpus(FIXTURE, workers=1, chunk_size=1))
    b = list(segment_corpus(FIXTURE, workers=2, chunk_size=1))
    assert [c.ids.tolist() for c in a] == [c.ids.tolist() for c in b]
    assert [c.vocab for c in a] == [c.vocab for c in b]


def test_segment_corpus_bounds_in_flight_tasks(tmp_path):
    n, workers = 50, 2
    arts = [f"文章{i:03d}的內容很長。" for i in range(n)]
    corpus = tmp_path / "gen.json"
    corpus.write_text(
        "[\n" + ",\n".join("  " + json.dumps(a, ensure_ascii=False) for a in arts) + "\n]\n",
        encoding="utf-8",
    )
    st = ReadStats()
    chunks = []
    max_lead = 0
    for k, ch in enumerate(segment_corpus(corpus, workers=workers, chunk_size=1, stats=st)):
        # 已讀取但尚未產出的篇數即在途任務數；yield 當下上限為 2*workers-1
        lead = st.read - (k + 1)
        assert lead <= 2 * workers - 1
        max_lead = max(max_lead, lead)
        chunks.append(ch)
    assert max_lead == 2 * workers - 1  # 確認確實進入背壓分支
    assert len(chunks) == n
    assert st.read == n
    # 依序：第 k 個 chunk 必須對應第 k 篇（每篇唯一的三位數編號）
    numbers = [[w for w in (ch.vocab[i] for i in ch.ids) if w.isdigit()] for ch in chunks]
    assert numbers == [[f"{i:03d}"] for i in range(n)]
