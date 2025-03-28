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
        
    # Successfully imputes missing values and returns a DataFrame with imputed values
    def test_fit_transform_returns_imputed_dataframe_numeric(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a test dataframe with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        imputed_df = pd.DataFrame(imputed_array, columns=['A', 'B'])
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=[np.array([0.1, 0.2])])
        mocker.patch.object(phil, '_select_representative', return_value=0)
      
        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', return_value=['A', 'B'])
        mock_pipeline.named_steps = {'imputer': mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Act
        result = phil.fit_transform(df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert list(result.columns) == ['A', 'B']
        phil.impute.assert_called_once_with(df, 5)
        assert phil.closest_index == 0
        
    
        # Successfully imputes missing values and returns a DataFrame with imputed values
    def test_fit_transform_imputes_missing_values_mixed_types(self, mocker):
        # Arrange
        from benson.phil import Phil
        import pandas as pd
        import numpy as np

        # Create a DataFrame with missing values
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, np.nan, 4.0],
            'cat_col': ['a', 'b', np.nan, 'd']
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0, 4.0],
            'cat_col': ['a', 'b', 'c', 'd']
        }).values

        mock_representations = [imputed_df]
        mock_descriptors = [np.array([0.1, 0.2, 0.3])]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=mock_representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=mock_descriptors)
        mocker.patch.object(phil, '_select_representative', return_value=0)

        # Create a mock imputer pipeline with get_feature_names_out method
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', return_value=['num_col', 'cat_col'])

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.named_steps = {'imputer': mock_imputer}

        phil.selected_imputers = [mock_pipeline]

        # Act
        result = phil.fit_transform(df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert list(result.columns) == ['num_col', 'cat_col']
        phil.impute.assert_called_once_with(df, 5)
        assert phil.closest_index == 0
       
       
        
    # Input DataFrame has no missing values (should raise ValueError in impute method)
    def test_fit_transform_raises_error_with_no_missing_values(self, mocker):
        # Arrange
        import pandas as pd
        import pytest
        from benson.phil import Phil
    
        # Create a test dataframe with no missing values
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8]
        })
    
        # Mock the impute method to raise ValueError as per its implementation
        phil = Phil()
        mocker.patch.object(phil, 'impute', side_effect=ValueError("No missing values found in the input DataFrame."))
    
        # Act & Assert
        with pytest.raises(ValueError, match="No missing values found in the input DataFrame."):
            phil.fit_transform(df)
        
        # Verify the impute method was called with correct parameters
        phil.impute.assert_called_once_with(df, 5)
        
    # Returns a DataFrame with the correct columns from the selected imputation
    def test_fit_transform_returns_dataframe_with_correct_columns(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a test dataframe with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=[np.array([0.1, 0.2])])
        mocker.patch.object(phil, '_select_representative', return_value=0)

        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', return_value=['A', 'B'])
        mock_pipeline.named_steps = {'imputer': mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Act
        result = phil.fit_transform(df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        
    # Selected imputer doesn't have imputed_columns_ attribute
    def test_fit_transform_raises_attribute_error_when_imputed_columns_missing(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a test dataframe with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=[np.array([0.1, 0.2])])
        mocker.patch.object(phil, '_select_representative', return_value=0)

        # Create a mock for selected_imputers without imputed_columns_ attribute
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', side_effect=AttributeError("Mocked error"))
        mock_pipeline.named_steps = {'imputer': mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Act & Assert
        with pytest.raises(AttributeError):
            phil.fit_transform(df)
            
    # Preserves column names in the returned DataFrame
    def test_fit_transform_preserves_column_names(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a test dataframe with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=[np.array([0.1, 0.2])])
        mocker.patch.object(phil, '_select_representative', return_value=0)

        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', return_value=['A', 'B'])
        mock_pipeline.named_steps = {'imputer': mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Act
        result = phil.fit_transform(df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        
    # Handles different max_iter values appropriately
    def test_fit_transform_with_various_max_iter(self, mocker):
        # Arrange
        import pandas as pd
        import numpy as np
        from benson.phil import Phil

        # Create a test dataframe with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, 'impute', return_value=representations)
        mocker.patch.object(phil, 'generate_descriptors', return_value=[np.array([0.1, 0.2])])
        mocker.patch.object(phil, '_select_representative', return_value=0)

        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, '_get_imputed_columns', return_value=['A', 'B'])
        mock_pipeline.named_steps = {'imputer': mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Act with different max_iter values
        result_5 = phil.fit_transform(df, max_iter=5)
        result_10 = phil.fit_transform(df, max_iter=10)

        # Assert
        assert isinstance(result_5, pd.DataFrame)
        assert result_5.shape == (4, 2)
        assert list(result_5.columns) == ['A', 'B']
    
        assert isinstance(result_10, pd.DataFrame)
        assert result_10.shape == (4, 2)
        assert list(result_10.columns) == ['A', 'B']
    
        assert phil.impute.call_count == 2
        phil.impute.assert_any_call(df, 5)
        phil.impute.assert_any_call(df, 10)
        
    # Correctly identifies numerical and categorical columns in a DataFrame with mixed data types, including boolean columns as numerical.
    def test_identifies_mixed_data_types_correctly_with_boolean(self):
        # Create a test DataFrame with mixed column types
        import pandas as pd
        from benson.phil import Phil
        data = {
            'string_col': ['apple', 'banana', 'cherry'],
            'category_col': pd.Series(['dog', 'cat', 'mouse']).astype('category'),
            'integer_col': [10, 20, 30],
            'float_col': [0.1, 0.2, 0.3],
            'boolean_col': [True, False, True]
        }
        df = pd.DataFrame(data)

        # Call the method under test
        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        # Assert that categorical columns are correctly identified
        assert set(categorical_cols) == {'string_col', 'category_col'}
        assert set(numerical_cols) == {'integer_col', 'float_col', 'boolean_col'}
        
    # Returns a tuple with two lists (categorical and numerical column names)
    def test_identifies_column_types_correctly(self):
        # Create a test DataFrame with mixed column types
        import pandas as pd
        from benson.phil import Phil
        data = {
            'string_col': ['apple', 'banana', 'cherry'],
            'category_col': pd.Series(['dog', 'cat', 'mouse']).astype('category'),
            'integer_col': [10, 20, 30],
            'float_col': [0.1, 0.2, 0.3]
        }
        df = pd.DataFrame(data)

        # Call the method under test
        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        # Assert that categorical columns are correctly identified
        assert set(categorical_cols) == {'string_col', 'category_col'}
        assert set(numerical_cols) == {'integer_col', 'float_col'}
        
    # Preserves column order within each type category
    def test_preserves_column_order_within_each_type_category(self):
        import pandas as pd
        from benson.phil import Phil
        # Create a test DataFrame with mixed column types
        data = {
            'int_col': [1, 2, 3],
            'object_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3],
            'category_col': pd.Series(['x', 'y', 'z']).astype('category')
        }
        df = pd.DataFrame(data)

        # Call the method under test
        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        # Assert that the order of columns is preserved within each type category
        assert categorical_cols == ['object_col', 'category_col']
        assert numerical_cols == ['int_col', 'float_col']
        
        # Generates descriptors for each imputed dataset in self.representations using a mocked _configure_magic_method
    def test_generates_descriptors_for_imputed_datasets_with_mocked_configure_magic_method(self, mocker):
        # Arrange
        from benson.phil import Phil
        import numpy as np

        # Create mock magic object
        mock_magic = mocker.Mock()
        mock_magic.generate.side_effect = lambda x: x + 1  # Simple transformation
    
        # Mock the _configure_magic_method to return a tuple of (mock_config, mock_magic)
        mock_config = mocker.Mock()
        mocker.patch.object(Phil, '_configure_magic_method', return_value=(mock_config, mock_magic))

        # Create test instance with mocked _configure_magic_method
        phil = Phil(magic="test_magic")

        # Set up test data
        test_representations = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        phil.representations = test_representations

        # Act
        result = phil.generate_descriptors()

        # Assert
        assert len(result) == len(test_representations)
        assert np.array_equal(result[0], test_representations[0] + 1)
        assert np.array_equal(result[1], test_representations[1] + 1)
        assert mock_magic.generate.call_count == len(test_representations)
        
        
    # Correctly calculates the mean descriptor from a list of descriptors
    def test_select_representative_finds_closest_to_mean(self):
        # Arrange
        import numpy as np
        from typing import List
    
        class TestClass:
            @staticmethod
            def _select_representative(descriptors: List[np.ndarray]) -> int:
                """Finds the descriptor closest to the mean representation."""
                avg_descriptor = np.mean(descriptors, axis=0)
                return np.argmin([np.linalg.norm(descriptor - avg_descriptor) for descriptor in descriptors])
    
        # Create test descriptors where the second one is closest to the mean
        descriptors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 2.0, 2.0]),  # This should be closest to mean
            np.array([3.0, 2.0, 1.0])
        ]
    
        # Calculate expected mean manually
        expected_mean = np.array([2.0, 2.0, 2.0])  # (1+2+3)/3, (2+2+2)/3, (3+2+1)/3
    
        # Act
        result = TestClass._select_representative(descriptors)
    
        # Assert
        assert result == 1, f"Expected index 1 but got {result}"
        # Verify the mean calculation is correct
        actual_mean = np.mean(descriptors, axis=0)
        np.testing.assert_array_almost_equal(actual_mean, expected_mean)
        
    # Empty list of descriptors
    def test_select_representative_with_empty_list(self):
        # Arrange
        import warnings
        import numpy as np
        import pytest
        from typing import List
    
        class TestClass:
            @staticmethod
            def _select_representative(descriptors: List[np.ndarray]) -> int:
                """Finds the descriptor closest to the mean representation."""
                avg_descriptor = np.mean(descriptors, axis=0)
                return np.argmin([np.linalg.norm(descriptor - avg_descriptor) for descriptor in descriptors])

        empty_descriptors = [] 
        # Act & Assert
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with pytest.raises(ValueError):
                TestClass._select_representative(empty_descriptors)
    
    def test_configure_param_grid_with_valid_string(self,mocker):
        """Handles valid string inputs like 'default', 'finance', 'healthcare'."""
        from benson.phil import Phil
        from benson import ImputationGrid
        from benson.gallery import GridGallery
        mock_grid = ImputationGrid(methods=["TestMethod"], modules=["test.module"], grids=[])
        mocker.patch.object(GridGallery, 'get', return_value=mock_grid)

        result = Phil._configure_param_grid("finance")

        GridGallery.get.assert_called_once_with("finance")
        assert result == mock_grid

    def test_configure_param_grid_with_invalid_inputs(self):
        """Ensures incorrect inputs raise ValueError."""
        from benson.phil import Phil
        from pydantic import BaseModel
        
        class MockInvalidBaseModel(BaseModel):
            field: str = "invalid"
        with pytest.raises(ValueError, match="Invalid parameter grid configuration."):
            Phil._configure_param_grid(MockInvalidBaseModel())

        with pytest.raises(ValueError, match="Invalid parameter grid configuration."):
            Phil._configure_param_grid({"invalid": "data"})

        with pytest.raises(ValueError, match="Invalid parameter grid type."):
            Phil._configure_param_grid(123)

    def test_configure_param_grid_with_valid_base_model(self,mocker):
        """Ensures valid BaseModel instances are converted to ImputationGrid."""
        from pydantic import BaseModel
        from benson.phil import Phil
        from benson import ImputationGrid
        from sklearn.model_selection import ParameterGrid
        from sklearn.compose import ColumnTransformer
        
        

        class MockValidBaseModel(BaseModel):
            methods: str = ["value"]
            modules: str = ["value"]
            grids: str = [ParameterGrid({})]
            
        mock_param_grid = MockValidBaseModel()
        
        result = Phil._configure_param_grid(mock_param_grid)

        assert isinstance(result, ImputationGrid)
        assert result.methods == ["value"]
        assert result.modules == ["value"]
        
    
        # Arrange
        from benson.phil import Phil
    
        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']
    
        # Act
        result = Phil._get_imputed_columns(mock_transformer)
    
        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()

    # Handles transformer that hasn't been fitted yet (should raise error)
    def test_raises_error_for_unfitted_transformer(self):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        import pytest
    
        # Create an unfitted ColumnTransformer
        transformer = ColumnTransformer(
            transformers=[
                ('imputer', SimpleImputer(), ['col1', 'col2'])
            ]
        )
    
        # Act & Assert
        with pytest.raises(ValueError):
            Phil._get_imputed_columns(transformer)

    # Correctly extracts column names from a properly configured transformer
    def test_extracts_column_names_from_transformer(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()

    # Returns a list of strings representing column names
    def test_returns_imputed_column_names(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()

    # Works with transformers that have been fitted on data
    def test_get_imputed_columns_with_fitted_transformer(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()

    # Handles empty transformer with no columns
    def test_empty_transformer_no_columns(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer with no columns
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = []

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == []
        mock_transformer.get_feature_names_out.assert_called_once()

    # Handles case when get_feature_names_out() method doesn't exist
    def test_handles_missing_get_feature_names_out_method(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer
    
        # Create a mock ColumnTransformer without get_feature_names_out method
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        del mock_transformer.get_feature_names_out
    
        # Act & Assert
        with pytest.raises(AttributeError):
            Phil._get_imputed_columns(mock_transformer)

    # Used in fit_transform method to get column names for DataFrame construction
    def test_get_imputed_columns_returns_correct_names(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()

    # Assumes transformer is a scikit-learn ColumnTransformer
    def test_get_imputed_columns_returns_feature_names(self, mocker):
        # Arrange
        from benson.phil import Phil
        from sklearn.compose import ColumnTransformer

        # Create a mock ColumnTransformer
        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = ['imputed_col1', 'imputed_col2']

        # Act
        result = Phil._get_imputed_columns(mock_transformer)

        # Assert
        assert result == ['imputed_col1', 'imputed_col2']
        mock_transformer.get_feature_names_out.assert_called_once()