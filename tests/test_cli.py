import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def run(*args, cwd):
    return subprocess.run([sys.executable, "-m", "hw1", *args], cwd=cwd, capture_output=True, text=True)


def test_subset_count_dump_top(tmp_path):
    sub = tmp_path / "sub.json"
    r = run("subset", "--corpus", str(FIXTURE), "--n-articles", "1", "--out", str(sub), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "1 articles" in r.stdout

    counts = tmp_path / "counts"
    r = run("count", "--corpus", str(FIXTURE), "--out", str(counts), "--workers", "1",
            "--timings", str(tmp_path / "t.json"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    meta = json.loads((counts / "meta.json").read_text())
    assert meta["n_bigrams"] == 9
    assert "N=18" in r.stdout and "V=8" in r.stdout and "B=9" in r.stdout

    dump = tmp_path / "bg.txt"
    r = run("dump-bigrams", "--counts", str(counts), "--out", str(dump), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(dump.read_text(encoding="utf-8").splitlines()) == 15

    r = run("top-bigrams", "--counts", str(counts), "--k", "2", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "B=9"
    assert lines[1].split("\t")[0] == "3"


def test_predict_prob_generate_cli(tmp_path):
    counts = tmp_path / "counts"
    r = run("count", "--corpus", str(FIXTURE), "--out", str(counts), "--workers", "1", "--trigram",
            "--timings", str(tmp_path / "t.json"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr

    r = run("predict", "我", "2", "--counts", str(counts), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0].startswith("喜歡\t0.6667")

    r = run("prob", "我喜歡你。", "--counts", str(counts), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "log2 P = -1.5850" in r.stdout

    r = run("generate", "--seed", "我", "--n", "4", "--mode", "greedy", "--counts", str(counts), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "喜歡你。 ｜" in r.stdout

    r = run("generate", "--seed", "我喜歡", "--n", "3", "--mode", "greedy", "--trigram",
            "--counts", str(counts), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "你。 ｜" in r.stdout
