import tensorflow as tf
from tensorflow.python.ops.distributions.util import fill_triangular


def create_triangular_mask(batch_size, length):
    b = batch_size
    n = length

    mask = tf.ones([b, (n * (n + 1)) // 2], dtype=tf.dtypes.bool)
    mask = fill_triangular(mask)
    return mask
