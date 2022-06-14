import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization
from tensorflow.random import normal

from src.Models.transformer import Transformer


class Generator(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            noise_shape,
            num_classes,
            query_dim=64,
            dim_head=64,
            num_heads=1,
            dropout_rate=0.2,
            num_layers=3
    ):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.noise_shape = noise_shape
        self.query_dim = query_dim

    def build(self, input_shape):
        self.to_query_dim = Dense(self.query_dim)
        self.transformer = Transformer(self.hidden_dim, self.dim_head, self.num_heads, self.dropout_rate, self.num_layers)
        self.layer_normalization = LayerNormalization(axis=-1)
        self.to_classes = Dense(self.num_classes)

    def call(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.noise_shape[-1]
        x = self.to_query_dim(x)
        x = self.layer_normalization(x)
        x = self.transformer(x)
        x = self.layer_normalization(x)
        x = self.to_classes(x)
        x = tf.nn.softmax(x, axis=-1)
        return x

    def get_config(self, ):
        config = super(Generator, self).get_config()
        config.update({"hidden_dim": self.hidden_dim, "dim_head": self.dim_head,
                       "num_heads": self.num_heads, "dropout_rate": self.dropout_rate,
                       "num_layers": self.num_layers, "num_classes": self.num_classes})
        return config

    def generate(self, batch_size=1, **kwargs):
        shape = [batch_size]
        shape.extend(list(self.noise_shape))
        z = normal(shape)
        z = self(z, **kwargs)
        return z

    @classmethod
    def from_config(cls, config):
        return cls(**config)
