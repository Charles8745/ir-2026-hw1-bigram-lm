#!/usr/bin/env bash
# 用法：tests/integration/check_sort_uniq.sh data/counts/p1
set -euo pipefail
COUNTS="$1"
TMP="${TMPDIR:-/tmp}/hw1_check"
mkdir -p "$TMP"
uv run python -m hw1 dump-bigrams --counts "$COUNTS" --out "$TMP/bigrams.txt"
LC_ALL=C sort "$TMP/bigrams.txt" | uniq -c > "$TMP/uniq.txt"
uv run python -m hw1 top-bigrams --counts "$COUNTS" --k 20 > "$TMP/top.txt"
uv run python tests/integration/compare_top.py "$TMP/top.txt" "$TMP/uniq.txt"
