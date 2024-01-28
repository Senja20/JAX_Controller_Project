import jax.numpy as jnp
from .GeneralController import GeneralController
from jax import random, nn, tree_map
from jax.random import KeyArray
from jax.scipy.special import logsumexp
from os import environ
from dotenv import load_dotenv


class NNController(GeneralController):
    """
    Neural network controller class
    AI - based
    Using a neural network to control the plant and update the parameters
    This implementation is based on the paper:
    "Neural Network Control of Nonlinear Dynamic Systems with Unknown Dead Zones":
    https://pdf.sciencedirectassets.com/272229/1-s2.0-S1568494614X00098/1-s2.0-S1568494614003640/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC0aCXVzLWVhc3QtMSJHMEUCIQCCFpEZXEwb8rVNLg5K9IOrVifJxI9AZB8Om2IQ%2F9CjtQIgStEke%2BuNmiyT%2BoDipaTLkHAwSdsNxE0kDcKUZKhfeZgquwUI1v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDD4pNermqVQSJ196NiqPBbVMVGoJtTwfjYEPqOWXae2LntbrGExqPwdiIxTjAMb3UHu85Ln%2FL5TfCp0m3cGLsShWmy8xDSuUND0Wua1dpLnwK%2F7%2BKpKoEJY1J%2Bb1TLDnUnWmZTMaf%2BPIdKHnFrrOAyMALGbznAKnDOENb1fAgqEgmZ%2BpAzDKOcMe5LhJ2UCtPfkhcG62%2B%2F%2BKZnJkNpOIrQ3GNNflvkqUcKBtKO3xemHrmUfz5eU%2Frfkx3ks56c%2BvNS2mvlOMqI6RPUH3gLzNfOnlmLl2mjUHGYYLKvKta46f91rMrKJul57Ni%2B%2BaMQkHQXfjhULThQVzBTaw9iC%2BXwcaVJT2A0fMaC6uTG2%2B1rN8UFVfOR%2BX71Gb9u%2F5w%2F2fHYnknQ23%2BO9pEOwz2VuLOOtFmd6JVCbHVXIDmJkzcivmaYeTRKUNozTeUIb%2BsqVcZKGhrBYZ8xnRtg6OB1BCsQYS%2BOi8z51y8fSFQO0ZyOguzhxUXKS0ipdR%2FEE7yiIT9Xotdb1CozpmLaEFm5%2BIWfbZvHDNRRno1sf3SRP2iZNU9hFXruAqZaTpFrJqj2jgqDpN0nRa4uyt24t5I%2BzXdBlerLoFAlB2BQacHbv43i1TYsjKe9kvxTwzkP8gZdzY%2BiUGAJKNiytl3Qs4gFK9mB4SBBa3CqqdjdLaTe%2FAt62JinGxwWhR%2FhVChF1EVqsIX0lZiZBEZBuG8r9JES07TDkKqFb1HLxGu5gNbXjyRLamLdcLarofnHBAFfwM1o4lYx1U3%2BOqMqTnG3Ka0fFDzFGNhNqLcHvmoHSo4kWbvDPRo2CNTdWy4b3tX%2BLB3sT223Hw%2FKkfvQdjlcT2Sksx1dLIbGymOHwnOY9cuizfb0xV%2BkjajGz%2BMH32EaG2CVMw87GkrQY6sQGXqbR4MY4tUiC7HKS0Bz6m0RNqm6H8pCQ9v5%2B2IEk%2B6XF5rjhNn3cKMqSXqDk1oSHofkHiaiHGjwfWapqw25fpeUl8ZX%2BSImrScFQOchi8ykDOu%2FbvnxmcnyO9lKu3PpExbDVaIheaLjhZXMakADj7NmqkDmKNeZNf5FiXVl5C8jlQCvphFCgl%2BKhGNr%2ButbZyd%2F1AYL9JSDLhzSghSLTOzQTKHHXqTYDUVDuD3S5UD7M%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240118T125001Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSVJJ6AM2%2F20240118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=62ea3894aa090a7f44c7a20c4e01fabf87a6ccbf9e15a55c35b03cfa5b81bede&hash=b2572aa4fd4a6a7097a30317dabc2a9b49a3c2052a10b4c2aa2483553f231f6a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1568494614003640&tid=spdf-01ca49ff-6f94-4f4f-9dc6-e07782937494&sid=a001a4c0998358465f1847d1dff9f86b5854gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140c5954505001575651&rr=8476f6139b3656c4&cc=no
    """

    # constructor
    def __init__(self, learning_rate: float):
        """Initialize the neural network controller"""
        load_dotenv()
        super().__init__(learning_rate)
        self.params = self.__init_network_params(
            [3, 1, 1],
            random.PRNGKey(0),
            (
                float(environ.get("WEIGHT_LOWER_BOUND")),
                float(environ.get("WEIGHT_UPPER_BOUND")),
            ),
            (
                float(environ.get("BIAS_LOWER_BOUND")),
                float(environ.get("BIAS_UPPER_BOUND")),
            ),
        )

        self.activation = nn.sigmoid
        # self.activation = nn.relu
        # self.activation = nn.tanh

    def __str__(self):
        """String representation of the neural network controller"""
        return "Neural_network_controller"

    # public method
    def update(
        self,
        params: dict,
        current_state: float,
        error_history: list[float],
        target_state: float,
    ) -> float:
        """
        Update the neural network controller
        :param params: the parameters of the neural network controller (list)
        :param current_state: the current state (float)
        :param error_history: the error history (list)
        :param target_state: the target state (float)
        :return: the output of the neural network controller (float)
        """
        self.error = target_state - current_state

        self.derivate = super()._calculate_derivative()
        self.integral = jnp.sum(jnp.array(error_history), dtype=jnp.float32)

        self.last_error = self.error

        input_layer = jnp.array(
            [self.error, self.derivate, self.integral], dtype=jnp.float32
        )

        # norm target state
        return self.__feedforward(params, input_layer)

    def update_params(self, grad: jnp.ndarray):
        """
        Update the parameters of neural network controller
        Traverse the parameters and gradients simultaneously, and update the parameters
        params has to be a list of tuples, where each tuple is (W, b)
        grad has to be a list of tuples, where each tuple is (dW, db)
        :param grad: the gradients (list)
        :return: None
        """

        grad_clipped = tree_map(lambda x: jnp.clip(x, -1, 1), grad)

        self.params = [
            (W - self.learning_rate * dW, b - self.learning_rate * db)
            for (W, b), (dW, db) in zip(self.params, grad_clipped)
        ]

    def visualization_params(self):
        """Visualization of the parameters of the neural network controller"""
        pass

    # private methods
    def __random_layer_params(
        self,
        input_size: int,
        output_size: int,
        key: KeyArray,
        weight_range: tuple[int, int],
        bias_range: tuple[int, int],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize the parameters of the neural network controller
        :param input_size: the input size (int)
        :param output_size: the output size (int)
        :param key: the random key (KeyArray)
        :param weight_range: the weight range (tuple)
        :param bias_range: the bias range (tuple)
        :return: the weight and bias values (tuple)
        """
        weight_key, bias_key = random.split(key, 2)
        weight_shape = (output_size, input_size)
        bias_shape = (output_size,)

        weight_vals = self.__gen_rand_vals(weight_key, weight_shape, weight_range)
        bias_vals = self.__gen_rand_vals(bias_key, bias_shape, bias_range)

        return weight_vals, bias_vals

    def __init_network_params(
        self,
        sizes: list[int],
        key: KeyArray,
        weight_range: tuple[int, int] = (-1, 1),
        bias_range: tuple[int, int] = (-1, 1),
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Initialize the parameters of the neural network controller
        :param sizes: the sizes of the neural network controller (list)
        :param key: the random key (KeyArray)
        :param weight_range: the weight range (tuple)
        :param bias_range: the bias range (tuple)
        :return: the parameters of the neural network controller (list)
        """

        keys = random.split(key, len(sizes))
        return [
            self.__random_layer_params(m, n, k, weight_range, bias_range)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def __gen_rand_vals(
        self, random_key: KeyArray, shape: tuple[int, int], value_range: tuple[int, int]
    ) -> jnp.ndarray:
        """
        Generate random values
        :param random_key: the random key (KeyArray)
        :param shape: the shape of the random values (tuple)
        :param value_range: the value range (tuple)
        :return: the random values (jnp.ndarray)
        """

        min_value, max_value = value_range
        return min_value + (max_value - min_value) * random.uniform(
            random_key, shape, dtype=jnp.float32
        )

    def __feedforward(self, params: dict, inputs_layer: jnp.ndarray) -> float:
        """Feedforward the neural network controller
        For each layer, multiply the inputs by the weights and add the bias
        Then, apply the activation function
        :param params: the parameters of the neural network controller (list)
        :param inputs_layer: the inputs of the neural network controller (list)
        :return: the output of the neural network controller (float)
        """

        step = jnp.array(inputs_layer)

        # remove the last layer
        hidden_layers = params[:-1]

        for W, b in hidden_layers:
            step = jnp.dot(W, step) + b
            step = self.activation(step)

        # output layer
        w_last, b_last = params[-1]
        logits = jnp.dot(w_last, step) + b_last

        return logits
