"""
1. Initialize the controllers parameters (Ω): the three k values for a standard PID controller and the
weights and biases for a neural-net-based controller
"""

import numpy as np
import jax.numpy as jnp
import jax

import pprint
import random
import matplotlib.pyplot as plt

# controller imports
from controller.PIDController import PIDController
from controller.NNController import NNController

# model imports
from Plant.BathtubModel import BathtubModel

# plot imports
from visualization.params import plot_params
from visualization.error import plot_error

# hyperparameters
learning_rate = 0.01
noise_initial = random.uniform(-0.01, 0.01)

num_epochs = 100
num_timesteps = 10

# define the plant parameters - used by plant model
cross_sectional_area = 200.0
drain_cross_sectional_area = cross_sectional_area / 100.0
initial_height = 50.0
goal_height = 50.0


class CONSYS:
    def __init__(self, controller, plant, target_state):
        """
        This class contains the controller, the plant and the target state.
        :param controller: the controller (model)
        :param plant: the plant (model)
        :param target_state: the target state (float)

        """
        self.controller = controller(learning_rate, noise_initial)
        self.plant = plant(
            cross_sectional_area,
            drain_cross_sectional_area,
            initial_height,
            target_state,
        )
        self.target = target_state

    def run(self):
        """
        this method runs the system, and contains the loop for the epochs
        :return: error_history: the error history (list)
        """

        # initialize the error history
        # added two zeros to error_history to avoid error in mean_square_error
        error_history = []

        grad_func = jax.jit(jax.value_and_grad(self.run_epoch, argnums=0))

        for epoch in range(num_epochs):
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

    def run_epoch(self, params, error_history):
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
        self.controller.noise = random.uniform(-0.01, 0.01)

        # re-initialize the history
        update_states = jnp.array([initial_height])

        # initialize the error accumulator
        error_timestamp_acc = 0

        for s in range(num_timesteps):
            # Update the controller
            U = self.controller.update(
                params,
                update_states[-1],
                error_timestamp_acc,
                self.target,
            )
            # Update the plant
            new_height = self.plant.update(U, self.controller.noise)

            # save the values - used tp calculate the error in MSE function
            update_states = jnp.append(update_states, new_height)

            # save the error
            error_timestamp_acc += self.target - new_height

        # returns mean square error by using the difference between the states and the target
        return self.mean_square_error(update_states, self.target)

    def mean_square_error(self, predictions, target):
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
    system = CONSYS(NNController, BathtubModel, goal_height)
    error_history = system.run()
