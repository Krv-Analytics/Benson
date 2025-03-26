from abc import ABC, abstractmethod
import numpy as np

class Magic(ABC):
    """Base Magic class for generating low dimensional descriptors of point clouds."""
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Should configure the specific Magic generator with provided keyword arguments."""
        pass
    
    @abstractmethod
    def generate(self, X:np.ndarray, **kwargs) -> np.ndarray:
        """Should generate a magic representation of the provided vectorized data."""
        pass
    
    

