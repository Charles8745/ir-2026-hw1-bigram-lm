import json
import re

from hw1.analysis import _md, _n_articles, q2_predictability, run_all
from hw1.model import BigramModel


def _bare_markers(text: str) -> list[str]:
    """code span 以外殘留的 <s> / </s>（會被 Markdown 當成 HTML 標籤）。"""
    return re.findall(r"</?s>", re.sub(r"`+[^`]*`+", "", text))


def test_run_all_produces_files(toy_counts, tmp_path):
    out = tmp_path / "results"
    run_all(toy_counts, out)
    for name in ["q1", "q2", "q3", "q4", "q5", "trigram"]:
        assert (out / f"{name}.json").exists(), name
        assert (out / f"{name}.md").read_text(encoding="utf-8").strip(), name
    for png in ["zipf_unigram.png", "zipf_bigram.png", "q4_growth.png"]:
        assert (out / png).stat().st_size > 1000
    q3 = json.loads((out / "q3.json").read_text())
    assert q3["B"] == 9 and q3["V_no_markers"] == 6
    assert abs(q3["fraction_seen"] - 9 / 36) < 1e-12
    q4 = json.loads((out / "q4.json").read_text())
    assert q4["rows"][-1]["B"] == 9
    q2 = json.loads((out / "q2.json").read_text())
    assert "的" in q2 and q2["的"]["in_vocab"] is False
    # R9: three spellings are analysed
    assert {"的", "台灣", "臺灣"} <= set(q2)
    # R6: the trigram table records how many own sentences get -inf under each model
    tri = json.loads((out / "trigram.json").read_text())
    assert tri["n_own_sentences"] == 8
    assert 0 <= tri["n_neg_inf_sentences"]["bigram"] <= 8
    assert 0 <= tri["n_neg_inf_sentences"]["trigram"] <= 8
    assert "−∞" in (out / "q5.md").read_text(encoding="utf-8")
    # q5 step detail carries w0 for trigram rows
    q5 = json.loads((out / "q5.json").read_text())
    tri_rows = [r for r in q5["rows"] if r["model"] == "trigram"]
    assert tri_rows and all(len(st) == 6 for r in tri_rows for st in r["steps"])
    # R16: six original sentences + two 臺 controls, for each model
    for model in ("bigram", "trigram"):
        assert sum(r["model"] == model for r in q5["rows"]) == 8
    assert sum(r["label"] == "百科體（臺）" for r in q5["rows"]) == 4
    # markdown: sentence markers must be inside code spans, never bare HTML-like tags
    for name in ["q1", "q2", "q5", "trigram"]:
        text = (out / f"{name}.md").read_text(encoding="utf-8")
        assert _bare_markers(text) == [], name
    q5_md = (out / "q5.md").read_text(encoding="utf-8")
    assert "`<s>" in q5_md and "| <s>" not in q5_md and "| <s> " not in q5_md
    # R15: per-seed side-by-side greedy table
    tri_md = (out / "trigram.md").read_text(encoding="utf-8")
    assert "| seed | bigram greedy | trigram greedy | bigram 循環（起點／回到／長度） | trigram 循環（起點／回到／長度） |" in tri_md
    assert tri_md.count("\n| `") >= 5  # one row per seed


def test_greedy_cycle_wording_reports_entry_repeat_and_length(toy_counts, tmp_path):
    # R22: cycle_start 是第一次回到已出現狀態的步數，循環起點是 cycle_start − cycle_length
    run_all(toy_counts, tmp_path)
    q1 = json.loads((tmp_path / "q1.json").read_text())
    q1_md = (tmp_path / "q1.md").read_text(encoding="utf-8")
    tri_md = (tmp_path / "trigram.md").read_text(encoding="utf-8")
    assert "進入長度" not in q1_md and "cycle (start/length)" not in tri_md
    greedy = [r for r in q1["rows"] if r["mode"] == "greedy" and r["cycle_start"] is not None]
    assert greedy
    for r in greedy:
        cs, L = r["cycle_start"], r["cycle_length"]
        assert f"（第 {cs} 步首次回到已出現的狀態；循環長度 {L}，循環從第 {cs - L} 步開始）" in q1_md
    # 玩具語料 seed「他」：bigram 6/5 → 起點 1，trigram 7/5 → 起點 2
    row = next(line for line in tri_md.splitlines() if line.startswith("| `他` |"))
    assert row.endswith("| 1 / 6 / 5 | 2 / 7 / 5 |")


def test_analyze_cli_records_timing(toy_counts, tmp_path):
    from hw1.cli import main

    out = tmp_path / "res"
    main(["analyze", "--counts", str(toy_counts), "--out", str(out)])
    timings = json.loads((out / "timings.json").read_text())
    assert timings[-1]["command"] == "analyze"
    assert (out / "summary.json").exists()


def test_md_helper_escapes_markers_and_pipes():
    assert _md("<s>") == "`<s>`"
    assert _md("</s>") == "`</s>`"
    assert _md("|", table=True) == "`\\|`"
    assert _md("|") == "`|`"
    assert _md("a`b") == "`` a`b ``"
    assert _md("a\nb") == "`a b`"


def test_q2_top10_words_are_escaped(toy_counts, tmp_path):
    bm = BigramModel.load(toy_counts)
    q2_predictability(bm, tmp_path, words=("。",))  # 。 的唯一後繼是 </s>
    text = (tmp_path / "q2.md").read_text(encoding="utf-8")
    assert "`</s>`" in text and _bare_markers(text) == []


def test_q4_prefix_sizes_use_floor():
    # 全語料 1,323,056 篇：1% = 13,230、10% = 132,305（不是 round 的 13,231 / 132,306）
    assert _n_articles(1_323_056, 0.01) == 13_230
    assert _n_articles(1_323_056, 0.1) == 132_305
    assert _n_articles(1_323_056, 1.0) == 1_323_056
    assert _n_articles(13_230, 0.03) == 396
    assert _n_articles(2, 0.001) == 1  # 至少 1 篇
