import pytest
import pandas as pd



@pytest.fixture
def test_dataFrame():
    return pd.DataFrame(
    {
        "A": [1, 2, None, 4, 5],
        "B": ["a", "b", None, "d", "d"],
        "C": ["x", "y", "z", None, "w"],
        "D": [None, 10, 20, 30, 40],
        "E": ["p", "q", "r", "r", None],
        "F": ["u", "v", None, "x", "y"],
    }
)