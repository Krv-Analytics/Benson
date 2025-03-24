"""Testing Benson's cleaning"""
import pytest
import pandas as pd

class TestClean:
    def test_shell(self,test_dataFrame):
        assert isinstance(test_dataFrame,pd.DataFrame)