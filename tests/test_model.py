import math

import pytest

from hw1 import BOS, EOS
from hw1.model import BigramModel, GenerateResult, TrigramModel


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


def test_generate_greedy_toy_and_cycle(model):
    r = model.generate("我", 8, mode="greedy")
    assert isinstance(r, GenerateResult)
    # 我→喜歡(2)→你(1, id 小於 他)→。→</s>→(新句)我→喜歡→你→。
    assert r.words == ["喜歡", "你", "。", EOS, "我", "喜歡", "你", "。"]
    assert r.text == "喜歡你。 ｜ 我喜歡你。"
    assert r.n_sentences == 1
    # 狀態序列：我(0) 喜歡(1) 你(2) 。(3) <s>(4) 我(5) → 第 5 步回到第 0 步的狀態，循環長度 5
    assert r.cycle_start == 5 and r.cycle_length == 5


def test_generate_greedy_reproducible(model):
    assert model.generate("我", 20).words == model.generate("我", 20).words


def test_generate_sample_reproducible_with_seed(model):
    a = model.generate("我", 30, mode="sample", rng_seed=1).words
    b = model.generate("我", 30, mode="sample", rng_seed=1).words
    assert a == b
    assert set(a) <= {"我", "喜歡", "討厭", "你", "他", "。", EOS}
    assert BOS not in a


def test_generate_empty_or_oov_seed_starts_from_bos(model):
    assert model.generate("", 3).words == ["我", "喜歡", "你"]
    assert model.generate("不存在", 3).words == ["我", "喜歡", "你"]
    assert model.generate(EOS, 3).words == ["我", "喜歡", "你"]


def test_generate_n_zero(model):
    assert model.generate("我", 0).words == []


def test_trigram_model_predict_and_generate(toy_counts):
    tm = TrigramModel.load(toy_counts)
    assert tm.predict("我", 2) == [("喜歡", pytest.approx(2 / 3)), ("討厭", pytest.approx(1 / 3))]
    # trigram 前文 (我, 喜歡) → 你:1, 他:1；同分取 id 小者 你
    assert tm.generate("我喜歡", 3).words == ["你", "。", EOS]
    # trigram 前文 (<s>, 我) → 喜歡 2/3
    r = tm.generate("", 2)
    assert r.words == ["我", "喜歡"]


def test_trigram_prob_uses_trigram_counts(toy_counts):
    tm = TrigramModel.load(toy_counts)
    r = tm.prob("我喜歡你。")
    # P(我|<s>) bigram=1 ; P(喜歡|<s>,我)=2/3 ; P(你|我,喜歡)=1/2 ; P(。|喜歡,你)=1 ; P(</s>|你,。)=1
    assert r.logprob == pytest.approx(math.log2(1 / 3))
    assert r.steps[1].context_count == 3 and r.steps[2].context_count == 2
