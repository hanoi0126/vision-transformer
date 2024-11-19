from dataclasses import dataclass


@dataclass
class TrainConfig:
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    initializer_range: float
    image_size: int
    num_classes: int
    num_channels: int
    qkv_bias: bool
    use_faster_attention: bool

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.intermediate_size == 4 * self.hidden_size
        assert self.image_size % self.patch_size == 0
