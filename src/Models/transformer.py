import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

from src.Models.attention import SelfAttention
from src.Models.feed_forward import FeedForward


class Transformer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dim_head=64, num_heads=1, dropout_rate=0.2, num_layers=3, causal_masking=False):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.causal_masking = causal_masking

    def build(self, input_shape):
        self.attention_layers = [
            SelfAttention(
                self.dim_head, self.num_heads, self.dropout_rate, self.causal_masking
            ) for i in range(self.num_layers)
        ]
        self.feedforward_layers = [FeedForward(self.hidden_dim, self.dropout_rate) for i in range(self.num_layers)]
        self.layer_normalization = LayerNormalization(axis=-1)

    def call(self, x, **kwargs):
        for att, feedf in zip(self.attention_layers, self.feedforward_layers):
            y = self.layer_normalization(x)
            x = att(y) + x
            y = self.layer_normalization(x)
            x = feedf(y) + x
        return x

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({"hidden_dim": self.hidden_dim, "dim_head": self.dim_head, "num_heads": self.num_heads,
                       "dropout_rate": self.dropout_rate, "num_layers": self.num_layers})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
