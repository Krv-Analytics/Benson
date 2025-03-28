
import pytest
from benson.magic.base import Magic

class TestMagic:
    
    # Subclass implements configure method correctly
    def test_subclass_implements_configure_correctly(self):
        import numpy as np
        from abc import ABC, abstractmethod
    
        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs
        
            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X
    
        # Create an instance of the concrete subclass
        magic = ConcreteMagic()
    
        # Configure with some test parameters
        test_params = {"param1": 10, "param2": "test"}
        magic.configure(**test_params)
    
        # Verify the configuration was applied correctly
        assert hasattr(magic, "config"), "Configure method should store parameters"
        assert magic.config == test_params, "Configure method should store the provided parameters"
        
        # Instantiating the abstract base class directly raises TypeError
    def test_instantiating_abstract_base_class_raises_error(self):
        import numpy as np
        import pytest
        from abc import ABC, abstractmethod
    
        # Attempting to instantiate the abstract base class should raise TypeError
        with pytest.raises(TypeError) as excinfo:
            magic = Magic()
    
        # Verify the error message indicates it's due to abstract methods
        assert "abstract" in str(excinfo.value).lower(), "Error should mention abstract methods"
        
    # Subclass implements generate method correctly
    def test_subclass_implements_generate_correctly(self):
        import numpy as np
        from abc import ABC, abstractmethod

        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs
    
            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X * 2  # Example transformation
    
        # Create an instance of the concrete subclass
        magic = ConcreteMagic()

        # Test data
        test_data = np.array([1, 2, 3])

        # Generate the magic representation
        result = magic.generate(test_data)

        # Verify the generate method works correctly
        expected_result = test_data * 2
        assert np.array_equal(result, expected_result), "Generate method should correctly transform the input data"
        
    # Generate method returns numpy array of correct shape
    def test_generate_returns_correct_shape(self):
        import numpy as np
        from abc import ABC, abstractmethod
    
        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs
    
            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                # For testing, simply return the input array
                return X
    
        # Create an instance of the concrete subclass
        magic = ConcreteMagic()
    
        # Create a test input array
        test_input = np.random.rand(10, 3)
    
        # Generate the output using the generate method
        output = magic.generate(test_input)
    
        # Verify the output is a numpy array of the same shape as the input
        assert isinstance(output, np.ndarray), "Output should be a numpy array"
        assert output.shape == test_input.shape, "Output shape should match input shape"
        
    # Subclass fails to implement required abstract methods
    def test_subclass_without_abstract_methods(self):
        import pytest
        from abc import ABC, abstractmethod
    
        class IncompleteMagic(Magic):
            pass
    
        with pytest.raises(TypeError) as excinfo:
            incomplete_magic = IncompleteMagic()
        
    # Generate method called with empty array
    def test_generate_with_empty_array(self):
        import numpy as np
        from abc import ABC, abstractmethod
    
        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs
    
            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X
    
        # Create an instance of the concrete subclass
        magic = ConcreteMagic()
    
        # Call generate with an empty array
        empty_array = np.array([])
        result = magic.generate(empty_array)
    
        # Verify the result is an empty array
        assert isinstance(result, np.ndarray), "Generate should return a numpy array"
        assert result.size == 0, "Generate should return an empty array when input is empty"