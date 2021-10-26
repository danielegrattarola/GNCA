import tensorflow as tf
from spektral.models.general_gnn import MLP, GeneralGNN

from layers.simple_edge_conv import SimpleEdgeConv


class GNNCASimpleBoids(tf.keras.Model):
    """
    GNCA that combines GeneralGNN and SimpleEdgeConv, designed to learn the
    Boids GCA.
    """

    def __init__(
        self,
        activation=None,
        message_passing=1,
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="sum",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.message_passing = message_passing
        self.batch_norm = batch_norm
        self.hidden = hidden
        self.hidden_activation = hidden_activation
        self.connectivity = connectivity
        self.aggregate = aggregate

    def build(self, input_shape):
        self.mp = GeneralGNN(
            2,
            activation="linear",
            message_passing=self.message_passing,
            pool=None,
            batch_norm=self.batch_norm,
            hidden=self.hidden,
            hidden_activation=self.hidden_activation,
            connectivity=self.connectivity,
            aggregate=self.aggregate,
        )

        self.mp_diff = SimpleEdgeConv(2, activation="linear", mlp_hidden=[self.hidden])

        self.limits_model = MLP(
            2, batch_norm=self.batch_norm, activation=self.hidden_activation
        )

    def call(self, inputs):
        x = inputs[0][:, :2]
        v = inputs[0][:, 2:]
        v_next = v + self.mp(inputs) + self.mp_diff(inputs[:2])
        v_next = self.limits_model(v_next)
        x_next = x + v_next
        output = tf.concat((x_next, v_next), axis=-1)

        return output

    @tf.function
    def steps(self, inputs, steps, **kwargs):
        x, a = inputs
        for _ in tf.range(steps):
            x = self([x, a], **kwargs)

        return x
