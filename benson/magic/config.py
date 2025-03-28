from pydantic import BaseModel


class ECTConfig(BaseModel):
    """Configuration for ECT (Euler Characteristic Transform) as using in DECT (https://github.com/aidos-lab/dect)."""
    num_thetas: int 
    radius: float 
    resolution: int 
    scale: int 
    ect_fn: str 
    seed: int = 0