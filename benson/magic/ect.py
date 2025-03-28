
import torch
from dect.directions import generate_uniform_directions
from dect.ect import compute_ect
import numpy as np
import dect.ect_fn as ECT_FNs

from benson.magic import Magic
from benson.magic.config import ECTConfig


class ECT(Magic):
    
    def __init__(self,config:ECTConfig):
        self.config = config
        self.configure(**config.model_dump())
    
    def configure(self, **kwargs):
        """
        Configures the ECTMagic with provided keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to update the configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self, 'config') and hasattr(self.config, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
            
        self.ect_fn = getattr(ECT_FNs, self.ect_fn, ECT_FNs.scaled_sigmoid)
        self.device = self._check_device(force_cpu=True)
        
            
    def generate(self,X:np.ndarray) -> np.ndarray:
        """
        Placeholder for the generate method.
        """
        dim = X.shape[1]
        self.directions = generate_uniform_directions(num_thetas=self.num_thetas, d=dim, seed=self.seed, device=self.device)

        ect = compute_ect(
            self._convert_to_tensor(X),
            v=self.directions,
            radius=self.radius,
            resolution=self.resolution,
            scale=self.scale,
            ect_fn=self.ect_fn,
        )

        return self._convert_to_numpy(ect)
    
    
    @staticmethod
    def _convert_to_tensor(X:np.ndarray):
        """
        Converts a DataFrame to a PyTorch tensor.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to be converted.

        Returns
        -------
        torch.Tensor
            Tensor representation of the DataFrame.
        """
        return torch.tensor(X, dtype=torch.float32)
    
    @staticmethod
    def _convert_to_numpy(tensor:torch.Tensor):
        """
        Converts a PyTorch tensor to a NumPy array.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to be converted.

        Returns
        -------
        numpy.ndarray
            NumPy array representation of the tensor.
        """
        return tensor.detach().squeeze().numpy().T
    
    
    @staticmethod
    def _check_device(force_cpu:bool=False):
        """
        Checks if the device is valid.

        Parameters
        ----------
        device : str
            Device to be checked.

        Raises
        ------
        ValueError
            If the device is not valid.
        """
        device = "cpu"
        if not force_cpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
        return device

        
        
