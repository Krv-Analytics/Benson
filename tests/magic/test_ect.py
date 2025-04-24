import pytest
import numpy as np
import torch
import dect.ect_fn as ECT_FNs
from benson.magic.ect import ECT
from benson.magic.config import ECTConfig


class TestECT:
    # Initializes ECT object with valid ECTConfig
    def test_init_with_valid_config(self, mocker):
        # Arrange
        mock_configure = mocker.patch("benson.magic.ect.ECT.configure")
        config = ECTConfig(
            num_thetas=10,
            radius=1.0,
            resolution=100,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )

        # Act
        ect = ECT(config)

        # Assert
        assert ect.config == config
        mock_configure.assert_called_once_with(**config.model_dump())

    # Valid configuration keys including 'ect_fn' are updated in the ECT instance
    def test_valid_configuration_keys_are_updated_with_ect_fn(self, mocker):
        # Arrange
        mock_config = mocker.MagicMock()
        mock_config.num_thetas = 100
        mock_config.seed = 42
        mock_config.ect_fn = "scaled_sigmoid"
        mock_config.model_dump.return_value = {
            "num_thetas": 100,
            "seed": 42,
            "ect_fn": "scaled_sigmoid",
        }

        # Mock hasattr to return True for valid keys on the config object
        original_hasattr = hasattr

        def mock_hasattr(obj, attr):
            if obj is mock_config and attr in ["num_thetas", "seed", "ect_fn"]:
                return True
            return original_hasattr(obj, attr)

        mocker.patch("builtins.hasattr", side_effect=mock_hasattr)

        # Create instance with mocked config
        ect_instance = ECT(config=mock_config)

        # Mock _check_device to return 'cpu'
        mocker.patch.object(ect_instance, "_check_device", return_value="cpu")

        # Act
        ect_instance.configure(
            num_thetas=200, seed=123, ect_fn="scaled_sigmoid"
        )

        # Assert
        assert ect_instance.num_thetas == 200
        assert ect_instance.seed == 123
        assert ect_instance.ect_fn == ECT_FNs.scaled_sigmoid

    # Ensure the device is set to CPU when force_cpu is True
    def test_device_set_to_cpu_when_force_cpu_true(self, mocker):
        # Arrange
        mock_config = mocker.MagicMock()
        mock_config.model_dump.return_value = {"ect_fn": "scaled_sigmoid"}

        # Set up hasattr to return True for valid keys
        mocker.patch(
            "builtins.hasattr",
            side_effect=lambda obj, attr: (
                True if attr in ["ect_fn", "config"] else False
            ),
        )

        # Create instance with mocked config
        ect_instance = ECT(config=mock_config)

        # Mock _check_device to return 'cpu'
        mocker.patch.object(ect_instance, "_check_device", return_value="cpu")

        # Act
        ect_instance.configure(ect_fn="scaled_sigmoid")

        # Assert
        assert ect_instance.device == "cpu"

    # Test actual ECT generation with different dimensional inputs
    def test_generate_ect_2d(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(10, 2) for _ in range(5)]  # 2D point clouds

        # Act
        result = ect.generate(X)

        # Assert

        assert isinstance(result, list)

        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_ect_3d(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(10, 3) for _ in range(5)]  # 3D point clouds

        # Act
        result = ect.generate(X)

        # Assert
        assert isinstance(result, list)
        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_with_empty_input(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)
        X = [np.array([]).reshape(0, 2) for _ in range(2)]  # Empty point cloud

        # Act
        with pytest.raises(ValueError):
            ect.generate(X)

    def test_generate_with_single_point(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(1, 2)]  # List containing a single point

        # Act
        result = ect.generate(X)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_with_single_array_raises_error(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)
        X = np.random.rand(10, 2)  # Single numpy array, not in a list

        # Act & Assert
        with pytest.raises(
            ValueError, match="Input must be a list of numpy arrays"
        ):
            ect.generate(X)

    def test_different_ect_functions(self):
        # Test different ECT functions produce different outputs
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect_sigmoid = ECT(config)

        config.ect_fn = "indicator"
        ect_indicator = ECT(config)

        X = [np.random.rand(10, 2)]  # Input as list containing one array

        result_sigmoid = ect_sigmoid.generate(X)
        result_indicator = ect_indicator.generate(X)

        # Compare the first array in each list since generate returns List[np.ndarray]
        assert not np.array_equal(result_sigmoid[0], result_indicator[0])

    def test_consistent_output_with_seed(self):
        # Test that the same seed produces the same output
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect1 = ECT(config)
        ect2 = ECT(config)

        X = [np.random.rand(10, 2)]  # Input as list containing one array

        result1 = ect1.generate(X)
        result2 = ect2.generate(X)

        # Compare the first array in each list since generate returns List[np.ndarray]
        np.testing.assert_array_almost_equal(result1[0], result2[0])

    def test_convert_to_tensor(self):
        """Test converting numpy arrays to PyTorch tensor."""
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)

        # Test with 2D point clouds
        X_batch_2d = [np.random.rand(10, 2) for _ in range(3)]
        tensor_2d = ect._convert_to_tensor(X_batch_2d)
        assert isinstance(tensor_2d, torch.Tensor)
        assert tensor_2d.shape == (30, 2)  # (3*10, 2)
        assert tensor_2d.dtype == torch.float32

        # Test with 3D point clouds
        X_batch_3d = [np.random.rand(5, 3) for _ in range(2)]
        tensor_3d = ect._convert_to_tensor(X_batch_3d)
        assert tensor_3d.shape == (10, 3)  # (2*5, 3)

        # Verify data integrity
        start_idx = 0
        for x_np in X_batch_2d:
            chunk = tensor_2d[start_idx : start_idx + len(x_np)]
            np.testing.assert_array_almost_equal(x_np, chunk.numpy())
            start_idx += len(x_np)

    def test_convert_to_numpy(self):
        """Test converting PyTorch tensor to numpy arrays."""
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42,
        )
        ect = ECT(config)

        # Test with single batch
        input_tensor = torch.rand(1, 8, 10)  # (num_thetas, resolution)
        numpy_arrays = ect._convert_to_numpy(input_tensor)
        assert isinstance(numpy_arrays, list)
        assert len(numpy_arrays) == 1
        assert isinstance(numpy_arrays[0], np.ndarray)
        assert numpy_arrays[0].shape == (8, 10)

        # Test with multiple batches
        batch_tensor = torch.rand(
            3, 8, 10
        )  # (batch_size, num_thetas, resolution)
        batch_arrays = ect._convert_to_numpy(batch_tensor)
        assert len(batch_arrays) == 3
        for arr in batch_arrays:
            assert arr.shape == (8, 10)
            assert isinstance(arr, np.ndarray)

        # Verify data integrity
        reconstructed = torch.stack(
            [torch.from_numpy(arr) for arr in batch_arrays]
        )
        torch.testing.assert_close(batch_tensor, reconstructed)
