from dataclasses import dataclass


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
    # pre/post process
    ###################
    pre_process: bool = True

    post_process: bool = True

    parallel_output: bool = True
    """Do not gather the outputs, keep them split across tensor parallel ranks"""

    share_embeddings_and_output_weights: bool = False



