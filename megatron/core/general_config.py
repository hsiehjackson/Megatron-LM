from dataclasses import dataclass
from typing import Callable, ContextManager, Optional, Literal
from megatron.core.transformer.spec_utils import ModuleSpec
import torch

@dataclass
class ModelGeneralConfig:
    """
    General configuration for each architecture, excluding model parallel config
    """

    ###################
    # vocab size & train seq
    ###################
    vocab_size: int

    max_sequence_length: int

    ###################
    # Layer Spec
    ###################
    model_encoder_layer_spec: ModuleSpec = None
    """layer spec of encoders. ignored for gpt models"""

    model_decoder_layer_spec: ModuleSpec = None
    """layer spec of decoders."""

    ###################
    # pre/post process
    ###################
    pre_process: bool = True

    post_process: bool = True

    parallel_output: bool = True
    """Do not gather the outputs, keep them split across tensor parallel ranks"""

    share_embeddings_and_output_weights: bool = False



