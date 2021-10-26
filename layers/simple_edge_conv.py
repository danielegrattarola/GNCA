import tensorflow as tf
from spektral.layers import EdgeConv


class SimpleEdgeConv(EdgeConv):
    """
    An extension of EdgeConv that concatenates the difference and the norm of the
    difference to compute the messages.
    """

    def message(self, x, **kwargs):
        x_i = self.get_i(x)
        x_j = self.get_j(x)
        norm = tf.linalg.norm(x_i - x_j, axis=-1, keepdims=True)
        return self.mlp(tf.concat((-(x_j - x_i), norm), axis=-1))
