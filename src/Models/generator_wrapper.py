import tensorflow as tf
from typing import Dict, Generator


class GeneratorWrapper(tf.keras.layers.Layer):
    def __init__(self, generator: Generator, word_dictionary: Dict):
        super(GeneratorWrapper, self).__init__()
        self.generator = generator
        self.word_dictionary = word_dictionary

    def generate(self):
        x = self.generator.generate()
        x = tf.math.argmax(x, axis=-1)
        # x=self.to_words(x)
        return x

    def to_words(self, x):
        # x=tf.math.argmax(x,axis=-2)
        x = tf.make_tensor_proto(x)
        x = tf.make_ndarray(x)
        x = [" ".join([self.word_dictionary[i] for i in b]) for b in x]

        return x

    def call(self, x):
        x = self.generator(x)
        x = tf.math.argmax(x, axis=-1)
        # x=self.to_words(x)
        return x

    def build(self, **kwargs):
        pass
