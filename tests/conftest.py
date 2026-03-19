import pytest
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data():
    return np.load(Path(__file__).resolve().parent.parent / "doc" / "data.npz")

@pytest.fixture(scope="session")
def bces_result(test_data):
    import bces.bces as BCES
    d = test_data
    return BCES.bces(d['x'], d['errx'], d['y'], d['erry'], d['cov'])
