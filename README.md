# 資訊檢索 HW1：中文維基 Word Bigram／Trigram 語言模型

用 2023 年繁體中文維基百科（`wiki_article_list_2023_tra.json`）建 word bigram 語言模型（加分題另做 trigram），純 MLE、不平滑；jieba 斷詞，numpy 計數。

- 報告：`report/report.md`（作業各題的答案、數字與出處都在這裡）
- 展示 notebook：`report/showcase.ipynb`（已執行，讀 `results/full/` 與 `data/counts/full/`）
- 分析結果：`results/full/`（全語料）、`results/p1/`（前 1%），計數紀錄在 `results/notes.md`

## 環境

Python 3.10，用 uv 管理（`.python-version` 固定 3.10）：`uv sync`。語料不在 repo 裡，請放在 `data/wiki_article_list_2023_tra.json`，例如 `mkdir -p data && ln -s /path/to/wiki_article_list_2023_tra.json data/`。

jieba 字典用 `hw1/resources/dict.txt.big`（已 commit），來源 https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big ，SHA-256 `b16011275c42955ccd81fc1adecc93a59dbb7926af69d93fc95d4943d40f6aad`。

## 重跑（在專案根目錄執行）

```bash
uv run pytest                                                   # 單元測試（玩具語料）
uv run python -m hw1 subset --corpus data/wiki_article_list_2023_tra.json --n-articles 13230 --out data/subsets/p1.json
uv run python -m hw1 count --corpus data/subsets/p1.json --out data/counts/p1 --workers 10 --trigram
tests/integration/check_sort_uniq.sh data/counts/p1             # 1% 子集：numpy 計數 vs sort | uniq -c
uv run python -m hw1 count --corpus data/wiki_article_list_2023_tra.json --out data/counts/full --workers 10 --trigram
uv run python -m hw1 analyze --counts data/counts/full --out results/full
uv run python -m hw1 analyze --counts data/counts/p1 --out results/p1
uv run python report/build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace report/showcase.ipynb --ExecutePreprocessor.timeout=600
```

`predict`、`prob`、`generate` 等其他指令見 `uv run python -m hw1 --help` 與報告第 0、2 節。

注意：全語料計數在 Apple M5（10 workers）約 29 分鐘，需要充足的可用記憶體（記憶體不足時會大量 swap，見報告 5.1 節）；`data/counts/full/` 在磁碟上約 4.9 GB。重跑 `analyze` 會覆寫 `results/`，其中 Q4 的計數秒數每次略有不同。

## 程式結構（`hw1/`）

- `corpus.py`：逐行讀 JSON 陣列、分句、輸出前 N 篇子集
- `segment.py`：jieba 斷詞、加 `<s>` `</s>`、多行程編成 int32 ID
- `counting.py`：合併詞表成 token 串流；unigram／bigram／trigram 計數（`run_count`）
- `store.py`：計數結果的存檔與載入（`.npy`、`vocab.txt`、`meta.json`）
- `timing.py`：計時並寫入 `timings.json`
- `model.py`：`BigramModel`、`TrigramModel` 的 `predict`、`prob`、`generate`
- `analysis.py`：Part 3 五題、Zipf 圖、trigram 比較（`analyze` 指令）
- `cli.py`：`python -m hw1` 命令列入口
