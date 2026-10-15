# HW1 設計文件：中文維基 Word Bigram / Trigram 語言模型

日期：2026-10-15
作業頁面：https://irclass.nlpnchu.org/assignment-02.html

## 1. 目標與範圍

用 2023 年繁體中文維基語料（`wiki_article_list_2023_tra.json`，1.65 GB，
1,323,056 行，每行一篇文章）建立 word bigram 語言模型，
P(w₂|w₁) = count(w₁,w₂) / count(w₁)，並完成作業頁面的三個 Part、
加分題中的 Zipf 圖與 trigram，產出程式碼與觀察報告。

範圍內：
- Part 1 數：unigram、bigram 計數，回報 N、V、B、t。
- Part 2 建：`predict(w, k)`、`prob(sentence)`、`generate(seed, n)`，
  generate 支援 greedy 與 sample 兩種模式。
- Part 3 看：五題觀察，每題附數據或實際輸出。
- 加分：unigram 與 bigram 的 Zipf 圖；trigram 模型與 bigram 的比較。
- 報告：Markdown 正式報告 + 展示用 notebook。

範圍外：平滑（smoothing）、UNK 處理、Web 介面、課堂投票句子的挑選自動化。

## 2. 固定決策

| 項目 | 決定 |
|---|---|
| 語言與工具 | Python 3.10，uv 管理虛擬環境，jieba 精確模式 + `dict.txt.big` |
| 分句 | 以 。！？ 為句尾，標點留在該句句尾；文章結尾算句尾；空句丟棄 |
| token 清理 | jieba 切出的純空白 token 丟棄，其他（含標點、數字）一律保留 |
| 句子標記 | 每句前後加 `<s>`、`</s>`，標記不經過 jieba，在斷詞後加入 |
| 跨句詞對 | 不計算 (`</s>`, `<s>`)；trigram 同樣排除任何跨句的三元組 |
| 機率 | 純 MLE，不平滑；未見詞對的句子機率為 −inf，並回報是哪一對 |
| 子集 | 取語料前綴：1% = 前 13,230 篇，10% = 前 132,305 篇 |
| 交付形式 | 核心套件 + CLI + Markdown 報告 + 只讀結果的展示 notebook |

N、V、B 的定義：
- N：含 `<s>` `</s>` 的總 token 數；另附不含標記的數字。
- V：相異 token 數，含兩個標記；另附不含標記的數字。
- B：相異 bigram 數，不含跨句詞對。
- 第三題的 V×V 使用不含標記的 V。

## 3. 專案結構

```
HW1/
├── pyproject.toml          # 依賴：jieba、numpy、matplotlib、jupyter、pytest
├── hw1/
│   ├── corpus.py           # 逐行讀 JSON、切子集、分句
│   ├── segment.py          # jieba 斷詞、<s> </s> 包裝、多行程分片
│   ├── counting.py         # ID 編碼、bigram/trigram 打包計數、存檔
│   ├── model.py            # BigramModel / TrigramModel：predict、prob、generate
│   ├── analysis.py         # Part 3 五題、Zipf 圖、trigram 比較
│   └── cli.py              # python -m hw1 <指令>
├── tests/
│   ├── fixtures/toy.json   # 作業頁面的三句玩具語料
│   ├── test_*.py           # pytest 單元測試
│   └── integration/        # 1% 子集與 sort|uniq 對照腳本，手動執行
├── data/
│   ├── wiki_article_list_2023_tra.json
│   ├── subsets/            # 1%、10% 子集
│   └── counts/<name>/      # tokens.npy、vocab.txt、article_offsets.npy、
│                           # unigram_counts.npy、bigram_keys.npy、bigram_counts.npy、
│                           # trigram_keys.npy、trigram_counts.npy、meta.json
├── results/                # q1..q5 的 md 與 json、Zipf PNG、timings.json、notes.md
├── report/
│   ├── report.md
│   └── showcase.ipynb
└── docs/superpowers/specs/
```

`data/` 與 `results/` 不進 git；原始語料不進 git。

## 4. 資料流與計數流程

### 4.1 讀取（corpus.py）
- 逐行讀檔。跳過只有 `[` 或 `]` 的行；其餘行去掉首尾空白與行尾逗號後
  `json.loads` 成一篇文章字串。
- 異常行（解析失敗、空字串）跳過並計數，寫入 `results/notes.md`。
- `--limit N` 只讀前 N 篇，用於子集。

### 4.2 分句與斷詞（segment.py）
- 分句規則見第 2 節。
- `multiprocessing.Pool(10)`，每個任務 2,000 篇文章，用 `imap` 保持順序。
- worker initializer 執行 `jieba.set_dictionary(dict.txt.big)` 與
  `jieba.initialize()`，字典每個行程只載入一次。
- worker 對每句 `jieba.lcut(sentence)`，丟棄純空白 token，前後加標記，
  轉成本地 int32 ID。回傳：本地詞表 list、本地 ID 陣列、每篇文章起點。
- `<s>` 固定為全域 ID 0，`</s>` 固定為 1。

### 4.3 ID 串流（counting.py）
- 主程序維護全域詞表 dict。對每個任務建「本地 ID → 全域 ID」的 numpy
  查表陣列，fancy indexing 重映射後 append 到串流。
- 存 `tokens.npy`（int32）、`vocab.txt`（一行一詞，行號即 ID）、
  `article_offsets.npy`（每篇文章在串流中的起點，用於切前綴子集）。

### 4.4 計數（counting.py）
- unigram：`np.bincount(tokens, minlength=V)`。
- bigram：`key = tokens[:-1].astype(int64) << 32 | tokens[1:]`，
  刪除 `tokens[:-1] == 1` 的位置，`np.unique(key, return_counts=True)`。
- trigram：對每個位置 `searchsorted(bigram_keys, key)` 取 bigram 序號，
  `key3 = bigram_idx << 32 | tokens[2:]`，刪除跨句位置
  （`tokens[:-2] == 1` 或 `tokens[1:-1] == 1`），`np.unique`。
  trigram 的第一欄指回 bigram 表，P(w₃|w₁,w₂) = count₃ / count₂ 可直接查。
- 「對一個 int64 陣列做 unique 計數」獨立成一個函式 `count_keys(keys)`。
  若記憶體不足，只改此函式為分片 unique 再合併。
- `meta.json` 記錄：N（含/不含標記）、V（含/不含標記）、B、T（相異 trigram）、
  各階段耗時、文章數、jieba 版本與字典、建立時間。

### 4.5 記憶體估算（以 3 億 token 計）

| 項目 | 大小 |
|---|---|
| tokens int32 | 1.2 GB |
| bigram key int64 | 2.4 GB |
| np.unique 排序副本 | 2.4 GB |
| trigram 的 searchsorted 結果 | 2.4 GB |
| 預估峰值 | 約 10 GB（機器 24 GB） |

### 4.6 計時
每個 CLI 指令結束時將耗時與輸入規模 append 到 `results/timings.json`。
Part 1 的 t 與第四題的成長數據都從此取。

## 5. 模型介面（model.py）

### 5.1 載入
`BigramModel.load(counts_dir)` 讀 vocab、unigram counts、bigram keys/counts，
建 `word → id` dict。`TrigramModel(BigramModel)` 再多載 trigram 表。

### 5.2 predict(w: str, k: int) -> list[tuple[str, float]]
- `searchsorted` 找 key 落在 `[w_id<<32, (w_id+1)<<32)` 的連續區間。
- 機率分母用該區間 count 總和（即 w 作為前詞的次數），不用 unigram
  count(w)，因為句尾 `</s>` 沒有後繼詞。報告說明此選擇。
- 依 count 降冪取前 k。w 不在詞表回傳空 list。

### 5.3 prob(sentence: str) -> ProbResult
- 與計數階段共用同一個分句、斷詞、加標記函式。
- 逐對查 bigram，累加 log₂ 機率。多句時每句獨立計算後相加。
- `ProbResult` 含：總 log 機率、token 列表、每步 (w₁, w₂, count, prob)、
  第一個未見詞對。任一對未見則總值為 −inf，明細仍保留。

### 5.4 generate(seed, n, mode, rng_seed) -> GenerateResult
- seed 經 jieba 斷詞取最後一個詞當起點；seed 為空從 `<s>` 開始。
- `greedy`：每步取 count 最高的後繼。`sample`：依 count 比例抽樣，
  使用 `numpy.random.default_rng(rng_seed)`。
- 候選排除 `<s>`。選到 `</s>` 時記錄句尾，從 `<s>` 重新開句，直到湊滿 n 個詞。
- greedy 另回傳「第一次回到已出現狀態的步數」與循環長度。
- trigram 版用 (w₁, w₂) 查 trigram 區間；起點不足兩個詞時退回 bigram。

### 5.5 CLI
```
python -m hw1 subset   --corpus <json> --n-articles N --out <json>
python -m hw1 count    --corpus <json> --out data/counts/<name> [--limit N] [--trigram]
python -m hw1 predict  <w> <k>       [--counts data/counts/full]
python -m hw1 prob     "<sentence>"  [--counts ...]
python -m hw1 generate --seed <s> --n N --mode greedy|sample [--trigram] [--rng-seed S]
python -m hw1 analyze  [--counts ...] [--out results/]
```

## 6. Part 3 與加分題（analysis.py）

每題一個函式，輸出 `results/qN.md` 與 `results/qN.json`。

- **Q1 貪婪生成與繞圈**：seed 取 台灣、他、的、1990年、`<s>` 五個；
  各跑 greedy 50 步一次、sample 50 步三次。記錄 greedy 首次重複狀態的
  步數與循環長度。trigram 同樣跑一次供比較。
- **Q2 「的」vs「台灣」**：各算後繼詞種類數、最高機率、前 10 名機率總和、
  條件熵 H(W|w) = −Σ p log₂ p。熵低者較好預測。
- **Q3 已見比例**：B / V²（V 不含標記），另附 B / V（平均每詞後繼數）。
- **Q4 B 的成長**：從 `tokens.npy` 以 `article_offsets` 切前綴，
  在 0.1%、1%、3%、10%、30%、100% 六個點重算 N、V、B。
  輸出表格與 log-log 圖，擬合 B 對 N 的成長指數。
- **Q5 自造句**：6 句（3 句百科體、2 句口語、1 句亂湊）。每句輸出 log₂ 機率、
  逐對明細、造成 −inf 的詞對。
- **Zipf 圖**：unigram 與 bigram 各一張 log-log rank-frequency PNG，
  對 top 1000 做線性擬合並標註斜率。
- **trigram 比較**：同組 seed 的 greedy/sample 輸出並列；代價表：相異數、
  計數時間、載入記憶體、Q5 六句中變成 −inf 的句數。
- **奇怪現象與遇到的問題**：不預設內容。各階段異常記錄到 `results/notes.md`，
  最後整理進報告。

## 7. 測試

1. **單元測試（pytest）**：玩具語料固定在 `tests/fixtures/toy.json`。
   覆蓋：分句、標記包裝、JSON 逐行讀取、ID 編碼與重映射、bigram 與
   trigram 計數、predict 與 prob 的手算值、generate 可重現性、文字 dump。
   先寫測試再寫實作。
2. **整合檢查（1% 子集）**：跑完整 CLI，把 token 串流 dump 成文字，
   `sort | uniq -c` 算 bigram，與 numpy 結果比對 B 與前 20 名。
   腳本放 `tests/integration/`，手動執行。
3. **全語料**：檢查 `meta.json` 四個數字合理，抽查 `predict("的", 10)`。

## 8. 開發順序與驗收

| 步驟 | 產物 | 驗收 |
|---|---|---|
| 1 環境 | pyproject、uv venv、jieba 與 dict.txt.big、玩具語料、1%/10% 子集 | `uv run pytest` 可執行 |
| 2 corpus + segment | 逐行讀、分句、斷詞、加標記 | 玩具語料 token 列表與手算一致 |
| 3 counting | ID 串流、bigram、trigram、存檔、timings | 玩具語料計數一致；1% 子集與 sort/uniq 一致；1% 在 1 分鐘內完成並推估全語料耗時 |
| 4 全語料計數 | `data/counts/full/` | meta.json 有 N、V、B、t |
| 5 model | predict、prob、generate | 玩具語料手算值；predict 機率和為 1（誤差 1e-9）；generate 可重現 |
| 6 analysis | 五題、Zipf、trigram 比較 | `results/` 檔案齊全，每題有數字或輸出 |
| 7 notebook + 報告 | showcase.ipynb、report.md | notebook 全跑通且不重跑計數；報告對照頁面清單逐項勾選 |

步驟 4 執行期間可用 1% 計數檔並行開發步驟 5。

## 9. 風險與對策

| 風險 | 對策 |
|---|---|
| 全語料斷詞太慢 | 步驟 3 用 1% 推估；十核超過 1 小時則換 jieba-rs，報告註明 |
| trigram 記憶體超標 | `count_keys` 改分片 unique 再合併 |
| jieba 切壞標記 | 標記不經過 jieba |
| 語料有空文章或異常行 | 跳過並計數，記入 notes.md |
| jieba 預設字典偏簡體 | 使用 dict.txt.big，報告註明 |

## 10. 報告內容清單（對照作業頁面）

- [ ] Part 1：N、V、B、t 與做法（含斷詞工具、分句規則、N/V 定義）
- [ ] Part 3：五題，每題有數據或輸出與解釋
- [ ] 一個奇怪的現象
- [ ] 遇到的問題與解法
- [ ] 加分：Zipf 圖
- [ ] 加分：trigram 比較
- [ ] 加分：最通順或最好笑的生成句
- [ ] 工具與語言說明
