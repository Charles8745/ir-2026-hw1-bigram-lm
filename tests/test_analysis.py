import json

from hw1.analysis import run_all


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
    assert tri["n_own_sentences"] == 6
    assert 0 <= tri["n_neg_inf_sentences"]["bigram"] <= 6
    assert 0 <= tri["n_neg_inf_sentences"]["trigram"] <= 6
    assert "−∞" in (out / "q5.md").read_text(encoding="utf-8")
    # q5 step detail carries w0 for trigram rows
    q5 = json.loads((out / "q5.json").read_text())
    tri_rows = [r for r in q5["rows"] if r["model"] == "trigram"]
    assert tri_rows and all(len(st) == 6 for r in tri_rows for st in r["steps"])


def test_analyze_cli_records_timing(toy_counts, tmp_path):
    from hw1.cli import main

    out = tmp_path / "res"
    main(["analyze", "--counts", str(toy_counts), "--out", str(out)])
    timings = json.loads((out / "timings.json").read_text())
    assert timings[-1]["command"] == "analyze"
    assert (out / "summary.json").exists()
