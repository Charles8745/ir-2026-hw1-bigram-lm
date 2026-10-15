import math

import pytest

from hw1 import BOS, EOS
from hw1.model import BigramModel


@pytest.fixture(scope="module")
def model(toy_counts):
    return BigramModel.load(toy_counts)


def test_predict_toy(model):
    assert model.predict("我", 2) == [("喜歡", pytest.approx(2 / 3)), ("討厭", pytest.approx(1 / 3))]
    assert model.predict("我", 10) == model.predict("我", 2)
    assert model.predict(BOS, 1) == [("我", pytest.approx(1.0))]


def test_predict_oov_returns_empty(model):
    assert model.predict("不存在的詞", 5) == []
    assert model.predict(EOS, 5) == []


def test_predict_probs_sum_to_one_and_context_equals_unigram(model):
    for w in ["我", "喜歡", "你", "。", BOS]:
        probs = [p for _, p in model.predict(w, 10_000)]
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
        wid = model.vocab.id(w)
        assert model.context_count(wid) == int(model.unigram[wid])


def test_prob_toy_sentence(model):
    r = model.prob("我喜歡你。")
    # P(我|<s>)=1 · P(喜歡|我)=2/3 · P(你|喜歡)=1/2 · P(。|你)=1 · P(</s>|。)=1 = 1/3
    assert r.logprob == pytest.approx(math.log2(1 / 3))
    assert r.first_unseen is None
    assert [s.w2 for s in r.steps] == ["我", "喜歡", "你", "。", EOS]
    assert r.steps[1].count == 2 and r.steps[1].context_count == 3


def test_prob_unseen_pair_is_neg_inf_with_detail(model):
    r = model.prob("我喜歡你。我討厭他。")
    assert r.logprob == -math.inf
    assert r.first_unseen == ("討厭", "他")
    assert len(r.sentences) == 2
    assert len(r.steps) == 10          # 兩句各 5 對，明細仍完整


def test_prob_oov_word(model):
    r = model.prob("我討厭蘋果。")      # jieba: 我 / 討厭 / 蘋果 / 。
    assert r.logprob == -math.inf
    assert r.first_unseen == ("討厭", "蘋果")
    assert r.steps[2].context_count == 1 and r.steps[2].count == 0


def test_prob_empty(model):
    r = model.prob("")
    assert r.logprob == 0.0 and r.steps == [] and r.sentences == []
