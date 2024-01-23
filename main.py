"""
1. Initialize the controllers parameters (Ω): the three k values for a standard PID controller and the
weights and biases for a neural-net-based controller
"""

import numpy as np
import jax.numpy as jnp
import jax
from os import environ
from dotenv import load_dotenv
import pprint
import random
import matplotlib.pyplot as plt

# controller imports
from controller import PIDController, NNController

# model imports
from plant import BathtubModel

# plot imports
from visualization import plot_error, plot_params

# hyperparameters
num_epochs = 100
num_timesteps = 10


class CONSYS:
    def __init__(self, controller, plant):
        """
        This class contains the controller, the plant and the target state.
        :param controller: the controller (model)
        :param plant: the plant (model)
        :param target_state: the target state (float)

        """
        self.controller = controller(
            float(environ.get("LEARNING_RATE")),
            random.uniform(
                float(environ.get("NOISE_LOWER_BOUND")),
                float(environ.get("NOISE_UPPER_BOUND")),
            ),
        )
        self.plant = plant()
        self.target = self.plant.target

    def run(self) -> list:
        """
        this method runs the system, and contains the loop for the epochs
        :return: error_history: the error history (list)
        """

        # initialize the error history
        # added two zeros to error_history to avoid error in mean_square_error
        error_history = []

        grad_func = jax.jit(jax.value_and_grad(self.run_epoch, argnums=0))

        for epoch in range(int(environ.get("NUMBER_OF_EPOCHS"))):
            # run the epoch
            error, grad = grad_func(self.controller.params, error_history)

            # track the error
            error_history.append(error)

            # (f) Update Ω based on the gradients.
            self.controller.update_params(grad)

            # print params every 10 epochs - for debugging
            if epoch % 10 == 0:
                print("Epoch: ", epoch)
                print("Error: ", error)

        # pass track_K_p, track_K_d, track_K_i to plot_params
        self.controller.visualization_params()

        plot_error(error_history, self.controller.__str__())

        return error_history

    def run_epoch(self, params: dict, error_history: list[float]) -> float:
        """
        This method runs the epoch, and contains the loop for the timesteps
        :param params: the parameters of the controller (list)
        :param error_history: the error history (list)
        :return: mean_square_error: the mean square error (float)
        """

        # reset the controller and the plant
        self.controller.reset()
        self.plant.reset()

        # noise

        try:
            self.controller.noise = random.uniform(
                float(environ.get("NOISE_LOWER_BOUND")),
                float(environ.get("NOISE_UPPER_BOUND")),
            )

        except TypeError as e:
            print(e)

        # re-initialize the history
        update_states = jnp.array([self.plant.initial_height])

        # initialize the error accumulator
        error_timestamp_acc = 0

        for s in range(int(environ.get("NUMBER_OF_TIMESTEPS"))):
            # Update the controller
            U = self.controller.update(
                params,
                update_states[-1],
                error_timestamp_acc,
                self.target,
            )
            # Update the plant
            new_state = self.plant.update(U, self.controller.noise)

            # save the values - used tp calculate the error in MSE function
            update_states = jnp.append(update_states, new_state)

            # save the error
            error_timestamp_acc += self.target - new_state

        # returns mean square error by using the difference between the states and the target
        return self.mean_square_error(update_states, self.target)

    def mean_square_error(self, predictions: list[float], target: float) -> float:
        """
        Loss
        We take the difference between the predictions and the targets, square it, and take the mean.
        The predictions are the states, and the targets are the goal height.
        :param predictions: the states (list)
        :param target: the target state (float)
        :return: the mean square error (float)
        """
        # (d) Compute MSE over the error history.
        return jnp.mean(jnp.square(target - predictions))


if __name__ == "__main__":
    load_dotenv()
    system = CONSYS(NNController, BathtubModel)
    error_history = system.run()
