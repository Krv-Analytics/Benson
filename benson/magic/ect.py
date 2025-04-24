"""
Euler Characteristic Transform (ECT) implementation for topological data analysis.

This module implements the ECT magic method using the DECT library, providing
tools for analyzing the topological structure of imputed datasets.
"""

from typing import List
import torch
from dect.directions import generate_uniform_directions
from dect.ect import compute_ect_point_cloud
import numpy as np
import dect.ect_fn as ECT_FNs

from benson.magic import Magic
from benson.magic.config import ECTConfig


class ECT(Magic):
    """
    Euler Characteristic Transform implementation for representational analysis.

    This class implements the Magic interface using the Euler Characteristic
    Transform (ECT) from the DECT library. ECT captures the topological and
    geometric properties of point clouds by computing persistence-based
    signatures along different directions.

    Parameters
    ----------
    config : ECTConfig
        Configuration object specifying ECT parameters.

    Attributes
    ----------
    config : ECTConfig
        The configuration object containing ECT parameters.
    ect_fn : callable
        The ECT function to use (from dect.ect_fn).
    device : str
        The compute device ('cpu' or 'cuda').

    Methods
    -------
    configure(**kwargs)
        Update configuration parameters.
    generate(X)
        Generate ECT descriptor for input data.

    Examples
    --------
    >>> from benson.magic import ECT, ECTConfig
    >>> config = ECTConfig(
    ...     num_thetas=64,
    ...     radius=1.0,
    ...     resolution=100,
    ...     scale=500,
    ...     ect_fn="scaled_sigmoid",
    ...     seed=42
    ... )
    >>> ect = ECT(config)
    >>> X = np.random.randn(100, 3)  # Point cloud data
    >>> descriptor = ect.generate(X)

    Notes
    -----
    The ECT is particularly effective at capturing global shape characteristics
    while being stable to noise and deformations. It is used in Benson to
    compare different imputed versions of datasets to select a representative
    version.
    """

    def __init__(self, config: ECTConfig):
        """
        Initialize ECT with configuration.

        Parameters
        ----------
        config : ECTConfig
            Configuration object containing ECT parameters.
        """
        self.config = config
        self.configure(**config.model_dump())

    def configure(self, **kwargs):
        """
        Configure ECT parameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to update configuration.

        Raises
        ------
        ValueError
            If an invalid configuration key is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self, "config") and hasattr(self.config, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")

        self.device = self._check_device(force_cpu=True)

    def generate(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate ECT descriptor for input data.

        Parameters
        ----------
        X : List[np.ndarray]
            Input point cloud data as a list of arrays, where each array has
            shape (n_samples, n_features)

        Returns
        -------
        List[np.ndarray]
            ECT descriptor capturing topological features of the input(s).
            Each array in the list has shape (num_thetas, resolution).

        Raises
        ------
        ValueError
            If the input is empty, contains empty arrays, or is not a list of numpy arrays.
        """
        if not isinstance(X, list):
            raise ValueError("Input must be a list of numpy arrays")

        # Validate input
        if not X or any(x.size == 0 for x in X):
            raise ValueError("Input cannot be empty")

        try:
            # Convert to batched tensor format [B,N,D]
            batch_tensor = torch.stack([torch.from_numpy(x).float() for x in X])
        except RuntimeError as e:
            raise ValueError(f"Invalid input data format") from e

        dim = batch_tensor.shape[-1]
        directions = generate_uniform_directions(
            num_thetas=self.num_thetas,
            d=dim,
            device=self.device,
            seed=self.seed,
        )

        ect = compute_ect_point_cloud(
            x=batch_tensor,
            v=directions,
            radius=self.radius,
            resolution=self.resolution,
            scale=self.scale,
            normalize=self.config.normalize,
        )

        # Transpose each ECT to have shape [num_thetas, resolution]
        return [x.T.numpy() for x in ect]

    @staticmethod
    def _convert_to_tensor(X: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of numpy arrays to PyTorch tensor.

        Parameters
        ----------
        X : List[np.ndarray]
            Input data as a list of arrays, each with shape (N, d)

        Returns
        -------
        torch.Tensor
            PyTorch tensor with shape (B*N, d) where:
            - B is the number of arrays in the input list
            - N is the number of samples in each array
            - d is the dimension of each sample
        """
        tensors = [torch.from_numpy(x).float() for x in X]
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _check_device(force_cpu: bool = False) -> str:
        """
        Check available compute device.

        Parameters
        ----------
        force_cpu : bool, default=False
            If True, always use CPU regardless of GPU availability.

        Returns
        -------
        str
            'cuda' if GPU is available and not forced to CPU, 'cpu' otherwise.
        """
        if force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
