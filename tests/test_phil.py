"""Testing Benson's cleaning"""
import pytest

class TestPhil:
    # Initialization Tests
    def test_init_with_default_parameters(self, mocker):
        from benson.phil import Phil
        from benson.magic import Magic
        from pydantic import BaseModel

        mock_config = mocker.Mock(spec=BaseModel)
        mock_magic = mocker.Mock(spec=Magic)
        mock_param_grid = mocker.Mock()

        mocker.patch.object(
            Phil, 
            '_configure_magic_method', 
            return_value=(mock_config, mock_magic)
        )
        mocker.patch.object(
            Phil, 
            '_configure_param_grid', 
            return_value=mock_param_grid
        )

        phil = Phil()

        Phil._configure_magic_method.assert_called_once_with(magic="ECT", config=None)
        Phil._configure_param_grid.assert_called_once_with("default")

        assert phil.samples == 30
        assert phil.param_grid == mock_param_grid
        assert phil.random_state is None
        assert phil.config == mock_config
        assert phil.magic == mock_magic
        assert phil.representations == []
        assert phil.magic_descriptors == []

    def test_init_with_invalid_magic_method(self, mocker):
        from benson.phil import Phil

        mocker.patch.object(
            Phil, 
            '_configure_magic_method', 
            side_effect=ValueError("Magic method 'INVALID_MAGIC' not found.")
        )

        with pytest.raises(ValueError) as excinfo:
            Phil(magic="INVALID_MAGIC")

        assert "Magic method 'INVALID_MAGIC' not found." in str(excinfo.value)
        Phil._configure_magic_method.assert_called_once_with(magic="INVALID_MAGIC", config=None)

    def test_init_with_custom_magic_method(self, mocker):
        from benson.phil import Phil
        from benson.magic import ECT, ECTConfig

        mock_config = ECTConfig(num_thetas=64, radius=1.0, resolution=64, scale=500, ect_fn="scaled_sigmoid", seed=42)
        mock_magic = ECT(config=mock_config)

        mocker.patch.object(Phil, '_configure_magic_method', return_value=(mock_config, mock_magic))
        mocker.patch.object(Phil, '_configure_param_grid', return_value={'some': 'params'})

        phil = Phil(magic="CustomMagic")

        Phil._configure_magic_method.assert_called_once_with(magic="CustomMagic", config=None)
        Phil._configure_param_grid.assert_called_once_with("default")

        assert phil.samples == 30
        assert phil.param_grid == {'some': 'params'}
        assert phil.random_state is None
        assert phil.representations == []
        assert phil.magic_descriptors == []
        assert phil.config == mock_config
        assert phil.magic == mock_magic

    # Imputation Tests
    def test_impute_empty_dataframe(self, mocker):
        import pandas as pd
        from benson.phil import Phil
        
        df = pd.DataFrame()
        phil = Phil()
        
        with pytest.raises(ValueError,match="No missing values found in the input DataFrame."):
            phil.impute(df)

        # Raises ValueError for DataFrames with no missing values
    def test_impute_no_missing_values(self):
        # Arrange
        import pandas as pd
        from benson.phil import Phil

        # Create a DataFrame with no missing values
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0],
            'num2': [5.0, 6.0, 7.0, 8.0],
            'cat1': ['a', 'b', 'c', 'd'],
            'cat2': ['w', 'x', 'y', 'z']
        })

        phil = Phil()

        # Act & Assert
        with pytest.raises(ValueError, match="No missing values found in the input DataFrame."):
            phil.impute(df, max_iter=10)

    def test_impute_with_missing_values(self, mocker):
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        df = pd.DataFrame({
            'num1': [1.0, np.nan, 3.0, 4.0],
            'num2': [np.nan, 6.0, 7.0, 8.0],
            'cat1': ['a', np.nan, 'c', 'd'],
            'cat2': ['x', 'y', np.nan, 'w']
        })

        phil = Phil()
        mock_identify = mocker.patch.object(phil, '_identify_column_types', return_value=(['cat1', 'cat2'], ['num1', 'num2']))
        mock_configure = mocker.patch.object(phil, '_configure_preprocessor', return_value=mocker.MagicMock())
        mock_create = mocker.patch.object(phil, '_create_imputers', return_value=[mocker.MagicMock()])
        mock_select = mocker.patch.object(phil, '_select_imputations', return_value=[mocker.MagicMock()])
        mock_apply = mocker.patch.object(phil, '_apply_imputations', return_value=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])])

        result = phil.impute(df, max_iter=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with("default", ['cat1', 'cat2'], ['num1', 'num2'])
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()

    def test_impute_mixed_data_types(self, mocker):
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        df = pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0],
            'num2': [5.0, np.nan, 7.0, 8.0],
            'cat1': ['a', 'b', np.nan, 'd'],
            'cat2': [np.nan, 'y', 'z', 'w']
        })

        phil = Phil()
        mock_identify = mocker.patch.object(phil, '_identify_column_types', return_value=(['cat1', 'cat2'], ['num1', 'num2']))
        mock_configure = mocker.patch.object(phil, '_configure_preprocessor', return_value=mocker.MagicMock())
        mock_create = mocker.patch.object(phil, '_create_imputers', return_value=[mocker.MagicMock()])
        mock_select = mocker.patch.object(phil, '_select_imputations', return_value=[mocker.MagicMock()])
        mock_apply = mocker.patch.object(phil, '_apply_imputations', return_value=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])])

        result = phil.impute(df, max_iter=15)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with("default", ['cat1', 'cat2'], ['num1', 'num2'])
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()

    # Column Identification Tests
    def test_identify_column_types(self, mocker):
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0],
            'num2': [5.0, np.nan, 7.0, 8.0],
            'cat1': ['a', 'b', 'c', 'd'],
            'cat2': ['x', 'y', 'z', 'w']
        })

        phil = Phil()
        mock_identify = mocker.patch.object(phil, '_identify_column_types', return_value=(['cat1', 'cat2'], ['num1', 'num2']))

        phil.impute(df)

        mock_identify.assert_called_once_with(df)
        # Handles the case when samples is larger than the number of available imputers
    def test_impute_samples_larger_than_imputers(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a DataFrame with missing values
        df = pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0],
            'num2': [5.0, np.nan, 7.0, 8.0],
            'cat1': ['a', 'b', np.nan, 'd'],
            'cat2': [np.nan, 'y', 'z', 'w']
        })

        # Mock internal methods to verify they're called correctly
        phil = Phil()
        phil.samples = 10  # Set samples larger than the number of imputers
        mock_identify = mocker.patch.object(phil, '_identify_column_types', return_value=(['cat1', 'cat2'], ['num1', 'num2']))
        mock_configure = mocker.patch.object(phil, '_configure_preprocessor', return_value=mocker.MagicMock())
        mock_create = mocker.patch.object(phil, '_create_imputers', return_value=[mocker.MagicMock() for _ in range(3)])
        mock_select = mocker.patch.object(phil, '_select_imputations', return_value=[mocker.MagicMock()])
        mock_apply = mocker.patch.object(phil, '_apply_imputations', return_value=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])])

        # Act
        result = phil.impute(df, max_iter=15)

        # Assert
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with("default", ['cat1', 'cat2'], ['num1', 'num2'])
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()
        
    
    
    def test_inverse_transform_numerical_columns(self, mocker):
        # Setup
        import numpy as np
        from benson.phil import Phil
        
        mock_transformer = mocker.Mock()
        mock_transformer.inverse_transform.return_value = np.array([[10.0, 20.0], [30.0, 40.0]])
    
        preprocessor = mocker.Mock()
        preprocessor.transformers = [('num_transformer', mock_transformer, ['col1', 'col2'])]
    
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        original_columns = ['col1', 'col2']
    
        # Execute
        from benson.phil import Phil
        result = Phil._inverse_preprocessing(preprocessor, X, original_columns)
    
        # Assert
        # Ensure the mock was called with the correct arguments
        called_args, _ = mock_transformer.inverse_transform.call_args
        np.testing.assert_array_equal(called_args[0], X)
        np.testing.assert_array_equal(result, np.array([[10.0, 20.0], [30.0, 40.0]]))
        
    def test_inverse_transform_categorical_columns_with_numerical_output(self, mocker):
        # Setup
        import numpy as np
        from benson.phil import Phil
        
        mock_transformer = mocker.Mock()
        mock_transformer.inverse_transform.return_value = np.array([[0, 1], [2, 3]])

        preprocessor = mocker.Mock()
        preprocessor.transformers = [('cat_transformer', mock_transformer, ['col1', 'col2'])]

        X = np.array([[0, 1], [2, 3]])
        original_columns = ['col1', 'col2']

        # Execute
        from benson.phil import Phil
        result = Phil._inverse_preprocessing(preprocessor, X, original_columns)

        # Assert
        called_args, _ = mock_transformer.inverse_transform.call_args
        np.testing.assert_array_equal(called_args[0], X)
        np.testing.assert_array_equal(result, X)
        
        # Returns a copy of the input array when all transformers are processed
    def test_inverse_preprocessing_all_transformers_processed(self, mocker):
        # Setup
        import numpy as np
        from benson.phil import Phil
        mock_num_transformer = mocker.Mock()
        mock_num_transformer.inverse_transform.return_value = np.array([[10.0, 20.0], [30.0, 40.0]])
    
        mock_cat_transformer = mocker.Mock()
        mock_cat_transformer.inverse_transform.return_value = np.array([['A', 'B'], ['C', 'D']])
    
        preprocessor = mocker.Mock()
        preprocessor.transformers = [
            ('num_transformer', mock_num_transformer, ['col1', 'col2']),
            ('cat_transformer', mock_cat_transformer, ['col3', 'col4'])
        ]
    
        X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        original_columns = ['col1', 'col2', 'col3', 'col4']

        # Execute
        result = Phil._inverse_preprocessing(preprocessor, X, original_columns)
    
        # Assert
        mock_num_transformer.inverse_transform.assert_called_once_with(X[:, [0, 1]])
        mock_cat_transformer.inverse_transform.assert_called_once_with(X[:, [2, 3]])
        expected_result = np.array([[10.0, 20.0, 'A', 'B'], [30.0, 40.0, 'C', 'D']])
        np.testing.assert_array_equal(result, expected_result)
    
    
