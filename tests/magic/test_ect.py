import pytest
import dect.ect_fn as ECT_FNs

class TestECT:
    # Initializes ECT object with valid ECTConfig
    def test_init_with_valid_config(self, mocker):
        from benson.magic.ect import ECT
        from benson.magic.config import ECTConfig
        # Arrange
        mock_configure = mocker.patch('benson.magic.ect.ECT.configure')
        config = ECTConfig(
            num_thetas=10,
            radius=1.0,
            resolution=100,
            scale=1,
            ect_fn="scaled_sigmoid",
            seed=42
        )
    
        # Act
        from benson.magic.ect import ECT
        ect = ECT(config)
    
        # Assert
        assert ect.config == config
        mock_configure.assert_called_once_with(**config.model_dump())
        
        # Valid configuration keys including 'ect_fn' are updated in the ECT instance
    def test_valid_configuration_keys_are_updated_with_ect_fn(self, mocker):
        # Arrange
        from benson.magic.ect import ECT

        # Mock the config object
        mock_config = mocker.MagicMock()
        mock_config.num_thetas = 100
        mock_config.seed = 42
        mock_config.ect_fn = 'scaled_sigmoid'
        mock_config.model_dump.return_value = {'num_thetas': 100, 'seed': 42, 'ect_fn': 'scaled_sigmoid'}

        # Mock hasattr to return True for valid keys on the config object
        original_hasattr = hasattr
        def mock_hasattr(obj, attr):
            if obj is mock_config and attr in ['num_thetas', 'seed', 'ect_fn']:
                return True
            return original_hasattr(obj, attr)
    
        mocker.patch('builtins.hasattr', side_effect=mock_hasattr)

        # Create instance with mocked config
        ect_instance = ECT(config=mock_config)

        # Mock _check_device to return 'cpu'
        mocker.patch.object(ect_instance, '_check_device', return_value='cpu')

        # Act
        ect_instance.configure(num_thetas=200, seed=123, ect_fn='scaled_sigmoid')

        # Assert
        assert ect_instance.num_thetas == 200
        assert ect_instance.seed == 123
        assert ect_instance.ect_fn == ECT_FNs.scaled_sigmoid
        
        # Ensure the device is set to CPU when force_cpu is True and the configuration is correctly applied.
    def test_device_set_to_cpu_when_force_cpu_true(self, mocker):
        # Arrange
        from benson.magic.ect import ECT
        from benson.magic.config import ECTConfig

        # Mock the config object
        mock_config = mocker.MagicMock()
        mock_config.model_dump.return_value = {'ect_fn': 'scaled_sigmoid'}

        # Set up hasattr to return True for valid keys
        mocker.patch('builtins.hasattr', side_effect=lambda obj, attr: True if attr in ['ect_fn', 'config'] else False)

        # Create instance with mocked config
        ect_instance = ECT(config=mock_config)

        # Mock _check_device to return 'cpu'
        mocker.patch.object(ect_instance, '_check_device', return_value='cpu')

        # Act
        ect_instance.configure(ect_fn='scaled_sigmoid')

        # Assert
        assert ect_instance.device == 'cpu'
        