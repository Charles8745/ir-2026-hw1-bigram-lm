import numpy as np
import pytest

from hw1 import BOS, EOS
from hw1.counting import (
    Vocab,
    build_token_stream,
    count_bigrams,
    count_keys,
    count_trigrams,
    count_unigrams,
    pack,
    unpack,
)
from hw1.segment import encode_articles

TOY = ["我喜歡你。我喜歡他。", "我討厭你。"]


def _toy_stream():
    chunks = [encode_articles(TOY[:1]), encode_articles(TOY[1:])]
    return build_token_stream(chunks)


def test_build_token_stream_merges_vocab_and_offsets():
    tokens, vocab, offsets, n_sent = _toy_stream()
    assert vocab.words == [BOS, EOS, "我", "喜歡", "你", "。", "他", "討厭"]
    assert vocab.id("討厭") == 7 and vocab.id("不存在") is None
    assert tokens.dtype == np.int32
    assert [vocab.word(i) for i in tokens] == [
        BOS, "我", "喜歡", "你", "。", EOS,
        BOS, "我", "喜歡", "他", "。", EOS,
        BOS, "我", "討厭", "你", "。", EOS,
    ]
    assert offsets.tolist() == [0, 12, 18]
    assert n_sent == 3


def test_pack_unpack_roundtrip():
    hi = np.array([0, 5, 2**31 - 1], dtype=np.int64)
    lo = np.array([1, 7, 2**31 - 1], dtype=np.int64)
    k = pack(hi, lo)
    h, l = unpack(k)
    assert h.tolist() == hi.tolist() and l.tolist() == lo.tolist()
    assert k.dtype == np.int64


def test_count_keys_sorted_with_counts():
    keys = np.array([5, 3, 5, 9, 3, 5], dtype=np.int64)
    u, c = count_keys(keys)
    assert u.tolist() == [3, 5, 9] and c.tolist() == [2, 3, 1]


def test_unigram_counts_toy():
    tokens, vocab, _, _ = _toy_stream()
    uni = count_unigrams(tokens, len(vocab))
    assert uni.tolist() == [3, 3, 3, 2, 2, 3, 1, 1]
    assert uni.sum() == 18


def test_bigram_counts_toy():
    tokens, vocab, _, _ = _toy_stream()
    keys, counts = count_bigrams(tokens)
    w1, w2 = unpack(keys)
    got = {(vocab.word(a), vocab.word(b)): int(c) for a, b, c in zip(w1, w2, counts)}
    assert got == {
        (BOS, "我"): 3, ("我", "喜歡"): 2, ("喜歡", "你"): 1, ("你", "。"): 2, ("。", EOS): 3,
        ("喜歡", "他"): 1, ("他", "。"): 1, ("我", "討厭"): 1, ("討厭", "你"): 1,
    }
    assert len(keys) == 9 and counts.sum() == 15   # 18 tokens − 3 句尾
    assert np.all(np.diff(keys) > 0)


def test_trigram_counts_toy():
    tokens, vocab, _, _ = _toy_stream()
    bk, bc = count_bigrams(tokens)
    tk, tc = count_trigrams(tokens, bk)
    bidx, w3 = unpack(tk)
    w1, w2 = unpack(bk[bidx])
    got = {(vocab.word(a), vocab.word(b), vocab.word(c)): int(n) for a, b, c, n in zip(w1, w2, w3, tc)}
    assert got == {
        (BOS, "我", "喜歡"): 2, ("我", "喜歡", "你"): 1, ("喜歡", "你", "。"): 1, ("你", "。", EOS): 2,
        ("我", "喜歡", "他"): 1, ("喜歡", "他", "。"): 1, ("他", "。", EOS): 1,
        (BOS, "我", "討厭"): 1, ("我", "討厭", "你"): 1, ("討厭", "你", "。"): 1,
    }
    assert len(tk) == 10 and tc.sum() == 12        # 18 − 3×2
    assert np.all(np.diff(tk) > 0)


def test_trigram_context_sum_equals_bigram_count():
    """每個 (w1,w2) 作為 trigram 前文的總次數 = 該 bigram 次數（w2 ≠ </s>）。"""
    tokens, vocab, _, _ = _toy_stream()
    bk, bc = count_bigrams(tokens)
    tk, tc = count_trigrams(tokens, bk)
    bidx, _ = unpack(tk)
    sums = np.bincount(bidx, weights=tc, minlength=len(bk)).astype(np.int64)
    _, w2 = unpack(bk)
    mask = w2 != 1
    assert sums[mask].tolist() == bc[mask].tolist()


def test_trigram_raises_when_bigram_table_incomplete():
    tokens, _, _, _ = _toy_stream()
    bk, _ = count_bigrams(tokens)
    with pytest.raises(ValueError, match="does not cover"):
        count_trigrams(tokens, bk[:-1])   # 漏掉最大的 bigram
