import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax, Dropout
from tensorflow import einsum
from einops import rearrange

from src.Models.utils import create_triangular_mask


class SelfAttention(Layer):
    def __init__(self, dim_head=64, num_heads=1, dropout_rate=0.2, causal_masking=False):
        super(SelfAttention, self).__init__()
        self.inner_dim = dim_head * num_heads
        self.scale = dim_head ** (-0.5)
        self.softmax = Softmax(axis=-1)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causal_masking = causal_masking

    def build(self, input_shape):
        self.projection_q = Dense(self.inner_dim, use_bias=False)
        self.projection_k = Dense(self.inner_dim, use_bias=False)
        self.projection_v = Dense(self.inner_dim, use_bias=False)
        self.projection_out = Dense(input_shape[-1])
        self.dropout = Dropout(self.dropout_rate)

    def call(self, queries):
        # if context is None:
        context = queries
        q = self.projection_q(queries)
        k = self.projection_k(context)
        v = self.projection_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads), [q, k, v])

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if self.causal_masking:
            b, n, _ = sim.shape
            mask = create_triangular_mask(b, n)
            max_neg_value = -tf.experimental.numpy.finfo(sim.dtype).max
            sim = tf.where(~(mask), max_neg_value, sim)

        att = self.softmax(sim)
        att = self.dropout(att)

        out = einsum("b i j, b j d -> b i d", att, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.projection_out(out)

        return out

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"dim_head": self.dim_head, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate,
                       "num_layers": self.num_layers})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
