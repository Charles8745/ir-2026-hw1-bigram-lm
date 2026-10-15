"""比對 `top-bigrams` 輸出與 `sort | uniq -c` 的結果。用法：compare_top.py top.txt uniq.txt"""
import sys
from collections import Counter

top_lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
B_numpy = int(top_lines[0].split("=")[1])
top_numpy = {(w1, w2): int(c) for c, w1, w2 in (l.split("\t") for l in top_lines[1:])}

uniq = Counter()
for line in open(sys.argv[2], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    count, rest = line.lstrip().split(" ", 1)
    w1, w2 = rest.split("\t")
    uniq[(w1, w2)] = int(count)

B_uniq = len(uniq)
top_uniq = dict(uniq.most_common(len(top_numpy)))
# 同分 tie 可能造成前 k 名集合不同，所以比對「numpy 前 k 名在 uniq 裡的次數」是否一致
mismatch = {k: (v, uniq.get(k)) for k, v in top_numpy.items() if uniq.get(k) != v}
print(f"B numpy={B_numpy} sort|uniq={B_uniq}")
print(f"top-{len(top_numpy)} count mismatches: {mismatch}")
ok = B_numpy == B_uniq and not mismatch
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
