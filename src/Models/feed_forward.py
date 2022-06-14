from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras import Sequential


class FeedForward(Layer):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(FeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        dim = input_shape[-1]
        self.network = Sequential(
            layers=[
                Dense(self.hidden_dim, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(dim),
                Dropout(self.dropout_rate)
            ]
        )

    def call(self, x, **kwargs):
        return self.network(x)

    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
