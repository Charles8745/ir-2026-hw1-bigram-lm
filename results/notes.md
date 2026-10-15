# HW1 notes

## 1% 子集

子集：`data/subsets/p1.json`（前 13,230 篇，56,418,971 bytes）、`data/subsets/p10.json`（前 132,305 篇，219,213,324 bytes）。

指令：`uv run python -m hw1 count --corpus data/subsets/p1.json --out data/counts/p1 --workers 10 --trigram`

輸出：

```
N=12013994 (no markers 11161004) V=434999 (no markers 434997) B=4265006 T=8381788 t_segment=23.343s t_bigram=0.43s t_total=33.439s
```

（wall clock 34.8s，10 workers，另 t_trigram=9.634s。）

全語料斷詞時間推估：

- 依 plan 的 `time_segment_s × 100`：23.343 s × 100 = 2334 s，約 38.9 分鐘（< 1 小時，未超標）。
- 備註：前 1% 的篇數佔 1%，但位元組只佔全語料的 3.41%（56.4 MB / 1.65 GB），token 數約佔預估 3.3 億的 3.6%。前段文章平均較長，所以 ×100 是偏保守的上限；以位元組比例推估約 684 s，約 11.4 分鐘（10 workers）。

## sort | uniq -c 交叉驗證（1% 子集）

指令：`tests/integration/check_sort_uniq.sh data/counts/p1`

```
B numpy=4265006 sort|uniq=4265006
top-20 count mismatches: {}
OK
```

附帶檢查：dump 出來 11,587,499 行 = meta 的 `bigram_tokens_total`；`uniq -c` 計數總和同為 11,587,499。

## 全語料

指令：`uv run python -m hw1 count --corpus data/wiki_article_list_2023_tra.json --out data/counts/full --workers 10 --trigram`

```
N=352357784 (no markers 325610144) V=4401986 (no markers 4401984) B=55153238 T=157991994 t_segment=1000.706s t_bigram=59.833s t_total=1720.955s
real 1744.50s（/usr/bin/time -l 量到的是 uv 啟動器：peak memory footprint 15,942,136 bytes ≈ 15.9 MB；Python 計數行程本身的峰值記憶體未量測）
time_count_trigram_s=651.614, n_articles=1,323,055, n_sentences=13,373,820, skipped_bad_lines=0, skipped_empty_articles=0
```

執行期間機器記憶體吃緊：其他應用程式（AnyDesk、虛擬機、VS Code）佔約 13 GB，計數階段大量 swap；先前一個殘留的 numpy 基準測試行程（7.2 GB）被手動終止。

前 20 名 bigram：

```
B=55153238
13054207	。	</s>
756622	年	，
594759	日	，
550345	，	是
478604	，	在
478297	<s>	在
453874	，	但
386185	後	，
346684	月	，
323689	，	而
308220	，	並
296281	的	一個
294490	，	他
261499	中	，
235366	人	，
218787	，	為
214919	，	位於
213998	<s>	他
195449	時	，
190334	12	月
```
