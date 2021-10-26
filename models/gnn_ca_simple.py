import tensorflow as tf
from spektral.models import GeneralGNN


class GNNCASimple(tf.keras.Model):
    """
    GNCA that uses You et al.'s GeneralGNN (with a single MP stage) to update the state.
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
            input_shape[0][-1],
            activation=self.activation,
            message_passing=self.message_passing,
            pool=None,
            batch_norm=self.batch_norm,
            hidden=self.hidden,
            hidden_activation=self.hidden_activation,
            connectivity=self.connectivity,
            aggregate=self.aggregate,
        )

    def call(self, inputs):
        x = self.mp(inputs)

        return x

    @tf.function
    def steps(self, inputs, steps, **kwargs):
        x, a = inputs
        for _ in tf.range(steps):
            x = self([x, a], **kwargs)

        return x
