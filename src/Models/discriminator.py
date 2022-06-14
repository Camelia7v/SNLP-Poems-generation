import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LayerNormalization, Dense

from einops.layers.tensorflow import Reduce

from src.Models.transformer import Transformer


class Discriminator(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            num_classes,
            dim_head=64,
            num_heads=1,
            dropout_rate=0.2,
            num_layers=3
    ):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_classes = num_classes

    def build(self, input_shape):
        self.transformer = Transformer(self.hidden_dim, self.dim_head, self.num_heads, self.dropout_rate, self.num_layers,
                                       causal_masking=True)

        self.reducer = Reduce('b n d -> b d', 'mean')
        self.layer_norm = LayerNormalization(axis=-1)
        self.to_logits = Dense(self.num_classes)

    def call(self, x):
        x = self.transformer(x)
        x = self.reducer(x)
        x = self.layer_norm(x)
        x = self.to_logits(x)
        return x

    def get_config(self, ):
        config = super(Discriminator, self).get_config()
        config.update({"hidden_dim": self.hidden_dim, "dim_head": self.dim_head,
                       "num_heads": self.num_heads, "dropout_rate": self.dropout_rate,
                       "num_layers": self.num_layers, "num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)