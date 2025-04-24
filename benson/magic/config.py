"""
Euler Characteristic Transform (ECT) configuration and implementations.

This module provides the configuration and implementation of ECT-based
methods used for comparing imputed datasets.
"""

from pydantic import BaseModel, Field


class ECTConfig(BaseModel):
    """
    Configuration for the Euler Characteristic Transform (ECT).

    This configuration class defines parameters for computing ECT using the DECT
    (Discrete Euler Characteristic Transform) library.

    Parameters
    ----------
    num_thetas : int
        Number of angles to sample for directional ECT computation.
        Controls the angular resolution of the transform.

    radius : float
        Maximum radius for the filtration computation.
        Determines the spatial scale of topological features considered.

    resolution : int
        Number of points to sample along each direction.
        Controls the granularity of the ECT computation.

    scale : int
        Scaling factor for the input point cloud.
        Adjusts the spatial scale of the analysis.

    ect_fn : str
        Name of the ECT function to use. Available options:
        - "scaled_sigmoid": Scaled sigmoid function (recommended)
        - "gaussian": Gaussian function
        - Other functions available in dect.ect_fn

    seed : int, default=0
        Random seed for reproducible direction sampling.

    Notes
    -----
    The ECT is a topological transform that captures the shape characteristics
    of point clouds by computing Euler characteristics along different directions.
    These parameters control the precision and computational cost of the transform.

    Examples
    --------
    >>> config = ECTConfig(
    ...     num_thetas=64,
    ...     radius=1.0,
    ...     resolution=100,
    ...     scale=500,
    ...     ect_fn="scaled_sigmoid",
    ...     seed=42
    ... )
    """

    num_thetas: int = Field(..., description="Number of angles to sample")
    radius: float = Field(..., description="Maximum radius for filtration")
    resolution: int = Field(..., description="Number of points per direction")
    scale: int = Field(..., description="Scaling factor for point cloud")
    ect_fn: str = Field(..., description="ECT function name")
    seed: int = Field(0, description="Random seed for reproducibility")
