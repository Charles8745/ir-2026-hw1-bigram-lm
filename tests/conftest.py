from pathlib import Path

import pytest

from hw1.counting import run_count

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


@pytest.fixture(scope="session")
def toy_counts(tmp_path_factory):
    out = tmp_path_factory.mktemp("toy") / "counts"
    run_count(FIXTURE, out, workers=1, trigram=True, timings_path=out.parent / "t.json")
    return out
