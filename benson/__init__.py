from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Any
from typing import ClassVar
from sklearn.model_selection import ParameterGrid

class ImputationGrid(BaseModel):
    methods: List[str] = Field(
        default_factory=list,
        description="A list of sklearn model names, where the index maps to the corresponding ParameterGrid and module."
    )
    modules: List[str] = Field(
        default_factory=list,
        description="A list of sklearn module names, where the index maps to the corresponding model and ParameterGrid."
    )
    grids: List[ParameterGrid] = Field(
        default_factory=list,
        description="A list of ParameterGrid objects, where the index maps to the corresponding model and module."
    )

    # Annotate Config as ClassVar
    Config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

        
class PreprocessingConfig(BaseModel):
    method: str = Field(
        default="StandardScaler",
        description="The name of the sklearn model for which the preprocessing configuration is defined."
    )
    module: str = Field(
        default="sklearn.preprocessing",
        description="The name of the sklearn module that contains the specified model."
    )
    params: Dict = Field(
        default={},
        description="A parameter configuration, where keys are parameters for the method and values are the required value."
    )

    # Annotate Config as ClassVar
    Config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)