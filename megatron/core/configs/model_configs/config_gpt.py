
from ..transformer_config import TransformerConfig
from dataclasses import dataclass
from megatron.core.transformer.spec_utils import ModuleSpec
from typing import Callable, Optional, Tuple, Literal

@dataclass
class GPTConfig(TransformerConfig):
   
    model_encoder_layer_spec: ModuleSpec = None
    """layer spec of encoders. ignored for gpt models"""

    model_decoder_layer_spec: ModuleSpec = None
    """layer spec of decoders."""

    position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute'

    rotary_percent: float = 1.0

    rotary_base=10000
    
    seq_len_interpolation_factor: Optional[float] = None