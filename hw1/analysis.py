"""Part 3 五題、Zipf 圖、trigram 比較。所有函式只讀計數檔，輸出到 results/。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from hw1 import BOS_ID, EOS_ID  # noqa: E402
from hw1.counting import count_bigrams, count_unigrams  # noqa: E402
from hw1.model import BigramModel, TrigramModel  # noqa: E402
from hw1.store import load_counts  # noqa: E402

SEEDS = ["台灣", "他", "的", "1990年", ""]
OWN_SENTENCES = [
    ("百科體", "台灣是一個位於東亞的島嶼。"),
    ("百科體", "他出生於1950年，畢業於國立臺灣大學。"),
    ("百科體", "該公司成立於2001年，總部位於台北市。"),
    ("口語", "我今天早上吃了一個很好吃的蛋餅。"),
    ("口語", "你等一下要不要一起去看電影？"),
    ("亂湊", "香蕉打電話給星期三的數學。"),
]

# 只保留 matplotlib 實際找得到的中文字型，避免每次畫圖都出現 findfont 警告。
_CJK_FONTS = ["PingFang TC", "Heiti TC", "PingFang HK", "Arial Unicode MS", "STHeiti", "Hiragino Sans GB"]
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in _CJK_FONTS if f in _available_fonts] + ["sans-serif"]


def _write(out: Path, name: str, data: dict, md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{name}.md").write_text(md, encoding="utf-8")


def _gen_row(m, seed: str, n: int, mode: str, rng_seed: int | None = None) -> dict:
    r = m.generate(seed, n, mode=mode, rng_seed=rng_seed)
    return {"seed": seed or "<s>", "mode": mode, "rng_seed": rng_seed, "text": r.text,
            "cycle_start": r.cycle_start, "cycle_length": r.cycle_length, "n_sentences": r.n_sentences}


def q1_greedy_vs_sample(bm: BigramModel, tm: TrigramModel | None, out: Path, n: int = 50) -> dict:
    rows = []
    for seed in SEEDS:
        rows.append({"model": "bigram", **_gen_row(bm, seed, n, "greedy")})
        for s in range(3):
            rows.append({"model": "bigram", **_gen_row(bm, seed, n, "sample", rng_seed=s)})
        if tm is not None:
            rows.append({"model": "trigram", **_gen_row(tm, seed, n, "greedy")})
            rows.append({"model": "trigram", **_gen_row(tm, seed, n, "sample", rng_seed=0)})
    data = {"n_words": n, "rows": rows}
    md = ["# Q1 貪婪生成 vs 抽樣", ""]
    for r in rows:
        tag = f"[{r['model']}/{r['mode']}" + (f"/seed{r['rng_seed']}" if r["rng_seed"] is not None else "") + "]"
        extra = f"（第 {r['cycle_start']} 步進入長度 {r['cycle_length']} 的循環）" if r["mode"] == "greedy" and r["cycle_start"] is not None else ""
        md.append(f"- **`{r['seed']}`** {tag} {extra}: {r['text']}")
    _write(out, "q1", data, "\n".join(md) + "\n")
    return data


def _predictability(bm: BigramModel, w: str) -> dict:
    wid = bm.vocab.id(w)
    if wid is None or bm.context_count(wid) == 0:
        return {"in_vocab": False}
    ids, counts = bm.successors(wid)
    p = counts / counts.sum()
    return {
        "in_vocab": True,
        "count": int(bm.unigram[wid]),
        "n_successors": int(len(ids)),
        "top1_prob": float(p[0]),
        "top10_mass": float(p[:10].sum()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "perplexity": float(2 ** (-(p * np.log2(p)).sum())),
        "top10": [(bm.vocab.word(i), float(x)) for i, x in zip(ids[:10], p[:10])],
    }


def q2_predictability(bm: BigramModel, out: Path, words=("的", "台灣", "臺灣")) -> dict:
    data = {w: _predictability(bm, w) for w in words}
    md = ["# Q2 哪個詞後面比較好預測", "", "| 詞 | 出現次數 | 後繼種類 | top1 機率 | top10 機率和 | 條件熵 (bits) | 困惑度 |", "|---|---|---|---|---|---|---|"]
    for w, d in data.items():
        if d["in_vocab"]:
            md.append(f"| {w} | {d['count']} | {d['n_successors']} | {d['top1_prob']:.4f} | {d['top10_mass']:.4f} | {d['entropy_bits']:.3f} | {d['perplexity']:.1f} |")
        else:
            md.append(f"| {w} | 不在詞表 | | | | | |")
    md.append("")
    for w, d in data.items():
        if d["in_vocab"]:
            md.append(f"- **{w}** 的前 10 名：" + "、".join(f"{t}({p:.3f})" for t, p in d["top10"]))
    _write(out, "q2", data, "\n".join(md) + "\n")
    return data


def q3_seen_fraction(bm: BigramModel, out: Path) -> dict:
    V = len(bm.vocab) - 2
    B = int(len(bm.bigram_keys))
    data = {"V_no_markers": V, "B": B, "V_squared": V * V, "fraction_seen": B / (V * V), "avg_successors_per_word": B / V}
    md = (f"# Q3 已見 bigram 佔 V×V 的比例\n\n- V（不含標記）= {V:,}\n- V² = {V*V:,}\n- B = {B:,}\n"
          f"- B / V² = {data['fraction_seen']:.3e}（{data['fraction_seen']*100:.5f}%）\n- 平均每個詞有 {data['avg_successors_per_word']:.2f} 種不同後繼\n")
    _write(out, "q3", data, md)
    return data


def q4_growth(counts_dir: Path, out: Path, fractions=(0.001, 0.01, 0.03, 0.1, 0.3, 1.0)) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = load_counts(counts_dir, with_tokens=True)
    n_art = len(b.article_offsets) - 1
    rows = []
    for f in fractions:
        k = max(1, int(round(n_art * f)))
        end = int(b.article_offsets[k])
        toks = b.tokens[:end]
        t0 = time.perf_counter()
        keys, counts = count_bigrams(toks)
        uni = count_unigrams(toks, len(b.vocab))
        rows.append({"fraction": f, "n_articles": k, "N": int(toks.size), "V": int((uni[2:] > 0).sum()),
                     "B": int(len(keys)), "seconds": round(time.perf_counter() - t0, 2)})
    xs = np.log([r["N"] for r in rows])
    ys = np.log([r["B"] for r in rows])
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(rows) > 1 and len(set(xs)) > 1 else float("nan")
    data = {"rows": rows, "log_log_slope_B_vs_N": slope}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([r["N"] for r in rows], [r["B"] for r in rows], "o-", label="distinct bigrams B")
    ax.loglog([r["N"] for r in rows], [r["V"] for r in rows], "s--", label="vocabulary V")
    ax.set_xlabel("N (tokens)"); ax.set_ylabel("count"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"growth of B and V vs N (slope of B ≈ {slope:.3f})")
    fig.tight_layout(); fig.savefig(out / "q4_growth.png", dpi=150); plt.close(fig)
    md = ["# Q4 語料量增加時相異 bigram 的成長", "", "V 不含 `<s>`、`</s>`，與 Q3 一致。", "",
          "| 比例 | 篇數 | N | V | B | B/N | 計數秒數 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['fraction']*100:g}% | {r['n_articles']:,} | {r['N']:,} | {r['V']:,} | {r['B']:,} | {r['B']/r['N']:.3f} | {r['seconds']} |")
    md += ["", f"log-log 擬合：B ∝ N^{slope:.3f}", "", "![](q4_growth.png)"]
    _write(out, "q4", data, "\n".join(md) + "\n")
    return data


def _prob_row(m, label: str, s: str) -> dict:
    r = m.prob(s)
    return {"label": label, "sentence": s, "logprob": None if math.isinf(r.logprob) else r.logprob,
            "n_tokens": sum(len(x) - 2 for x in r.sentences),
            "first_unseen": list(r.first_unseen) if r.first_unseen else None,
            "steps": [(st.w0, st.w1, st.w2, st.count, st.context_count, st.prob) for st in r.steps]}


def q5_own_sentences(bm: BigramModel, tm: TrigramModel | None, out: Path) -> dict:
    rows = [{"model": "bigram", **_prob_row(bm, l, s)} for l, s in OWN_SENTENCES]
    if tm is not None:
        rows += [{"model": "trigram", **_prob_row(tm, l, s)} for l, s in OWN_SENTENCES]
    md = ["# Q5 自己寫的句子機率", "", "| 模型 | 類型 | 句子 | 詞數 | log₂P | 第一個未見詞對 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lp = "−∞" if r["logprob"] is None else f"{r['logprob']:.2f}"
        unseen = " ".join(r["first_unseen"]) if r["first_unseen"] else ""
        md.append(f"| {r['model']} | {r['label']} | {r['sentence']} | {r['n_tokens']} | {lp} | {unseen} |")
    md.append("")
    for r in rows:
        md.append(f"**{r['sentence']}**（{r['model']}）")
        for w0, w1, w2, c, ctx, p in r["steps"]:
            cond = w1 if w0 is None else f"{w0} {w1}"
            md.append(f"- P({w2} | {cond}) = {c}/{ctx} = {p:.3g}")
        md.append("")
    _write(out, "q5", {"rows": rows}, "\n".join(md) + "\n")
    return {"rows": rows}


def _zipf(counts: np.ndarray, title: str, path: Path, fit_top: int = 1000, n_plot: int = 4000) -> dict:
    c = np.sort(counts[counts > 0])[::-1]
    n = len(c)
    k = min(fit_top, n)
    log_ranks = np.log(np.arange(1, k + 1))
    slope, intercept = np.polyfit(log_ranks, np.log(c[:k]), 1)
    # 全語料的 bigram 有數千萬點；畫圖只取 log 等距的 n_plot 個名次（擬合仍用精確的前 k 名）。
    idx = np.unique(np.geomspace(1, n, n_plot).astype(np.int64)) - 1
    fit_ranks = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(idx + 1, c[idx], ".", ms=3, label="observed")
    ax.loglog(fit_ranks, np.exp(intercept) * fit_ranks ** slope, "r-", label=f"fit top {k}: slope={slope:.2f}")
    ax.set_xlabel("rank"); ax.set_ylabel("frequency"); ax.set_title(title); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return {"n_types": int(n), "fit_top": int(k), "slope": float(slope)}


def zipf_plots(bm: BigramModel, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    uni = bm.unigram.copy(); uni[[BOS_ID, EOS_ID]] = 0
    data = {"unigram": _zipf(uni, "Zipf: unigram", out / "zipf_unigram.png"),
            "bigram": _zipf(bm.bigram_counts, "Zipf: bigram", out / "zipf_bigram.png")}
    (out / "zipf.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def _count_neg_inf(m) -> int:
    return sum(1 for _, s in OWN_SENTENCES if math.isinf(m.prob(s).logprob))


def trigram_cost(bm: BigramModel, tm: TrigramModel | None, out: Path) -> dict:
    meta = bm.meta
    data = {
        "n_bigrams": meta["n_bigrams"], "n_trigrams": meta.get("n_trigrams"),
        "time_count_bigram_s": meta["time_count_bigram_s"], "time_count_trigram_s": meta.get("time_count_trigram_s"),
        "bigram_table_bytes": int(bm.bigram_keys.nbytes + bm.bigram_counts.nbytes),
        "trigram_table_bytes": int(tm.trigram_keys.nbytes + tm.trigram_counts.nbytes) if tm else None,
        "n_own_sentences": len(OWN_SENTENCES),
        "n_neg_inf_sentences": {"bigram": _count_neg_inf(bm), "trigram": _count_neg_inf(tm) if tm else None},
    }

    def fmt(x, spec: str = ",") -> str:
        return "—" if x is None else format(x, spec)

    neg = data["n_neg_inf_sentences"]
    n_own = data["n_own_sentences"]
    md = ["# Trigram 與 bigram 的代價比較", "", "| 項目 | bigram | trigram |", "|---|---|---|",
          f"| 相異 n-gram 數 | {fmt(data['n_bigrams'])} | {fmt(data['n_trigrams'])} |",
          f"| 計數時間（秒） | {data['time_count_bigram_s']} | {fmt(data['time_count_trigram_s'], '')} |",
          f"| 表格大小（MB），作為載入記憶體的近似 | {data['bigram_table_bytes']/1e6:.1f} | "
          f"{'—' if data['trigram_table_bytes'] is None else format(data['trigram_table_bytes']/1e6, '.1f')} |",
          f"| 自寫 {n_own} 句中機率為 −∞ 的句數 | {neg['bigram']} | {fmt(neg['trigram'])} |",
          "", "生成品質比較見 q1.md；句子機率比較見 q5.md。"]
    _write(out, "trigram", data, "\n".join(md) + "\n")
    return data


def run_all(counts_dir: str | Path, out_dir: str | Path) -> None:
    counts_dir, out = Path(counts_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # q4 要載入整條 token 串流，先做完並釋放，避免與模型表格同時佔用記憶體。
    q4 = q4_growth(counts_dir, out)
    try:
        tm = TrigramModel.load(counts_dir)
        # bigram 模型與 trigram 模型共用同一份表格，不重複載入。
        bm = BigramModel(tm.vocab, tm.unigram, tm.bigram_keys, tm.bigram_counts, tm.meta)
    except ValueError:
        tm = None
        bm = BigramModel.load(counts_dir)
    summary = {
        "q1": q1_greedy_vs_sample(bm, tm, out),
        "q2": q2_predictability(bm, out),
        "q3": q3_seen_fraction(bm, out),
        "q4": q4,
        "q5": q5_own_sentences(bm, tm, out),
        "zipf": zipf_plots(bm, out),
        "trigram": trigram_cost(bm, tm, out),
        "meta": bm.meta,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
