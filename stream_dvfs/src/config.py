import math
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from math import log
from typing import Literal, Optional

BATCH_SIZE = 1


class ModelConfig(metaclass=ABCMeta):

    num_layer: int
    name: str
    batch_size: int
    type: Literal["FullModel", "SingleLayerModel", "AttentionHead", "FlashAttention"]

    @abstractmethod
    def to_single_layer_config(self) -> "ModelConfig": ...

    @property
    def prefill_size(self) -> int: ...

    @prefill_size.setter
    def prefill_size(self, value: int): ...

    @property
    def decode_size(self) -> int: ...

    @decode_size.setter
    def decode_size(self, value: int): ...

    @property
    def parameterized_name(self) -> str: ...

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class AttentionHeadConfig(ModelConfig):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        dim_k: int,
        dim_v: int,
        batch_size: int = 1,
        name: str = "AttentionHead",
        type: Literal["AttentionHead"] = "AttentionHead",
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.batch_size = batch_size
        self.name = name
        self.num_layer = 1  # Single layer
        self.type = type

    def to_single_layer_config(self) -> "ModelConfig":
        return deepcopy(self)  # Already single layer

    @property
    def prefill_size(self) -> int:
        return 1  # Not applicable

    @prefill_size.setter
    def prefill_size(self, value: int):
        pass  # Not applicable

    @property
    def decode_size(self) -> int:
        return 1  # Not applicable

    @decode_size.setter
    def decode_size(self, value: int):
        pass  # Not applicable

    @property
    def parameterized_name(self) -> str:
        return f"{self.name}_B={self.batch_size}_Seq={self.seq_len}_Embed={self.dim_k}"

class FlashAttentionConfig(ModelConfig):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        dim_k: int,
        dim_v: int,
        batch_size: int = 1,
        tile_Br: int = 16,
        tile_Bc: int = 16,
        name: str = "FlashAttention",
        type: Literal["FlashAttention"] = "FlashAttention",
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.batch_size = batch_size
        self.name = name
        self.tile_Br = tile_Br
        self.tile_Bc = tile_Bc
        self.num_layer = 1  # Single layer
        self.type = type

    def to_single_layer_config(self) -> "ModelConfig":
        return deepcopy(self)  # Already single layer

    @property
    def prefill_size(self) -> int:
        return 1  # Not applicable

    @prefill_size.setter
    def prefill_size(self, value: int):
        pass  # Not applicable

    @property
    def decode_size(self) -> int:
        return 1  # Not applicable

    @decode_size.setter
    def decode_size(self, value: int):
        pass  # Not applicable

    @property
    def parameterized_name(self) -> str:
        return f"{self.name}_B={self.batch_size}_Seq={self.seq_len}_Embed={self.dim_k}_TileBr={self.tile_Br}_TileBc={self.tile_Bc}"
    
class TransformerConfig(ModelConfig):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        dim_ff: int,
        num_head: int,
        num_layer: int,
        batch_size: int = 1,
        vocab_size: int = 1000,
        name: str = "",
        type: Literal["FullModel", "SingleLayerModel"] = "FullModel",
        # Automatically calculated
        head_size: Optional[int] = None,
        prefill_size: Optional[int] = None,
        decode_size: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.dim_ff = dim_ff
        self.num_head = num_head
        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.name = name
        self.type = type

        # Defaults
        self.head_size = head_size if head_size is not None else embedding_dim // num_head
        # Simulate prefill with half of the context window.
        self.__prefill_size = prefill_size if prefill_size is not None else seq_len // 2
        self.__decode_size = decode_size if decode_size is not None else seq_len // 2
        self.__decode_idx = self.__compute_decode_idx()

    def __compute_decode_idx(self):
        """Take the token halfway the decode sequence, as multiple of 2"""
        decode_idx = self.__prefill_size + self.__decode_size // 2
        rounded_to_two = 1 << int(log(decode_idx, 2))
        return rounded_to_two

    @property
    def prefill_size(self):
        return self.__prefill_size

    @prefill_size.setter
    def prefill_size(self, value: int):
        self.__prefill_size = value
        self.__decode_idx = self.__compute_decode_idx()

    @property
    def decode_size(self):
        return self.__decode_size

    @decode_size.setter
    def decode_size(self, value: int):
        self.__decode_size = value
        self.__decode_idx = self.__compute_decode_idx()

    @property
    def decode_idx(self):
        """To simulate the model in decode phase, only a single run (for a single) token is executed"""
        return self.__decode_idx

    @decode_idx.setter
    def decode_idx(self, value: int):
        """Manually override the to-be simulated token in the decode sequence"""
        self.__decode_idx = value

    @property
    def parameterized_name(self):
        return f"{self.name.replace('.', '_')}_B={self.batch_size}_FULL"

    @property
    def has_gate_layer(self):
        return "opt" in self.name.lower() or "llama" in self.name.lower()

    def to_single_layer_config(self):
        """Return a new TransformerConfig instance with only a single laker to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        return TransformerConfigSingleLayer(self)


class TransformerConfigSingleLayer(TransformerConfig):
    """Configuration with only a single layer and a all heads"""

    def __init__(self, full_config: TransformerConfig):
        assert full_config.num_layer > 1
        super().__init__(
            num_layer=1,
            seq_len=full_config.seq_len,
            embedding_dim=full_config.embedding_dim,
            dim_ff=full_config.dim_ff,
            num_head=full_config.num_head,
            batch_size=full_config.batch_size,
            vocab_size=full_config.vocab_size,
            name=full_config.name,
            head_size=full_config.head_size,
            prefill_size=full_config.prefill_size,
            decode_size=full_config.decode_size,
            type="SingleLayerModel",
        )
        self.num_layer_full = full_config.num_layer
    def to_single_layer_config(self):
        raise Exception("This already is a single layer configuration")

    @property
    def parameterized_name(self):
        return super().parameterized_name.replace("FULL", "SINGLELAYER")

    def get_post_simulation_multiplier(self, layer_name: str, amortize_within_batch: bool = True) -> float:
        """The model is simulated with reduced parameters i.e. only one layer. This function returns the factor with
        which the results for the given layer have to be multiplied in order to come to the result for the full model
        Moreover, the results are normalized to a single inference instead of a full batch
        @param amortize_within_batch if true, return the results for a single"""
        batch_factor = 1 / self.batch_size if amortize_within_batch else 1

        def name_contains(x: list[str]):
            return any([v in layer_name for v in x])

        # Special case: gate layer in Llama models
        if name_contains(["key_proj"]):
            return 4 * self.num_layer_full * batch_factor
        if name_contains(["query_proj", "value_proj", "out_proj"]):
            raise ValueError(
                "Only `key_proj` should be passed as argument, others are included in the factor of `key_proj`"
            )
        if name_contains(["up_proj"]) and self.has_gate_layer:
            return 2 * self.num_layer_full * batch_factor
        # For pre- and post-processing
        if name_contains(["embed", "final"]):
            return 1

        return self.num_layer_full * batch_factor


class QuantConfig:
    def __init__(self, weight_bits: int, act_bits: int, output_bits: Optional[int] = None):
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.intermediate_output_bits = output_bits if output_bits is not None else 2 * act_bits

    @property
    def name(self):
        return f"W{self.weight_bits}A{self.act_bits}"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name