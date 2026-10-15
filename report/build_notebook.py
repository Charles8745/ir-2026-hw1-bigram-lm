"""產生 report/showcase.ipynb。

必須在專案根目錄執行（腳本用相對路徑寫出 report/showcase.ipynb）：

    uv run python report/build_notebook.py

產生後再執行一次 notebook，把輸出存進檔案：

    uv run jupyter nbconvert --to notebook --execute --inplace report/showcase.ipynb --ExecutePreprocessor.timeout=600
"""
import nbformat as nbf

RES = "results/full"
COUNTS = "data/counts/full"

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# HW1 Word Bigram / Trigram 語言模型 — 結果展示\n\n本 notebook 只讀取 `results/full/` 與 `data/counts/full/` 的既有結果，不重跑計數。")
code(f"""import json, pathlib
from IPython.display import Markdown, Image, display
ROOT = pathlib.Path.cwd()
ROOT = ROOT.parent if ROOT.name == "report" else ROOT   # nbconvert 與 VS Code 以 report/ 為 cwd
RES = ROOT / "{RES}"; COUNTS = ROOT / "{COUNTS}"
meta = json.loads((COUNTS / "meta.json").read_text())
def show(name):
    # 結果檔裡的 ![](q4_growth.png) 會以 report/ 為基準而破圖；圖片另由下一格顯示
    lines = (RES / f"{{name}}.md").read_text().splitlines()
    display(Markdown("\\n".join(l for l in lines if not l.startswith("![]("))))""")
md("## Part 1 數：N、V、B、t")
code("""display(Markdown(f'''
| 指標 | 值 |
|---|---|
| N（含標記） | {meta["n_tokens"]:,} |
| N（不含標記） | {meta["n_tokens_no_markers"]:,} |
| V（含標記） | {meta["vocab_size"]:,} |
| V（不含標記） | {meta["vocab_size_no_markers"]:,} |
| B | {meta["n_bigrams"]:,} |
| T（相異 trigram） | {meta["n_trigrams"]:,} |
| 斷詞時間 | {meta["time_segment_s"]:.0f} s |
| bigram 計數時間 | {meta["time_count_bigram_s"]:.0f} s |
| trigram 計數時間 | {meta["time_count_trigram_s"]:.0f} s |
| 總時間 | {meta["time_total_s"]:.0f} s |
| 文章數 / 句數 | {meta["n_articles"]:,} / {meta["n_sentences"]:,} |
'''))""")
md("## Part 2 建：互動示範\n\n語料中多寫作「臺灣」，「台灣」出現次數極少，因此示範以「臺灣」為輸入；緊接著的一格以「台灣」對照拼寫差異。")
code("""from hw1.model import BigramModel
bm = BigramModel.load(COUNTS)
bm.predict("臺灣", 10)""")
code("""bm.predict("台灣", 10)""")
code("""r = bm.prob("臺灣是一個島嶼。")
print(r.logprob, r.first_unseen)
[(s.w1, s.w2, s.count, s.context_count, round(s.prob, 4)) for s in r.steps]""")
code("""print(bm.generate("臺灣", 30, mode="greedy").text)
print(bm.generate("臺灣", 30, mode="sample", rng_seed=0).text)""")
for q in ["q1", "q2", "q3", "q4", "q5"]:
    md(f"## Part 3：{q.upper()}")
    code(f'show("{q}")')
    if q == "q4":
        code('display(Image(filename=str(RES / "q4_growth.png")))')
md("## 加分：Zipf 圖")
code('display(Image(filename=str(RES / "zipf_unigram.png"))); display(Image(filename=str(RES / "zipf_bigram.png")))\nprint(json.loads((RES / "zipf.json").read_text()))')
md("## 加分：trigram 比較")
code('show("trigram")')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "report/showcase.ipynb")
print("wrote report/showcase.ipynb")
