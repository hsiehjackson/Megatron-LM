
from dataclasses import dataclass
from .model_parallel_config import ModelParallelConfig
from .general_config import ModelGeneralConfig

@dataclass
class ArchitectureConfig(ModelParallelConfig, ModelGeneralConfig):
    def __post_init__(self):
        super().__post_init__()