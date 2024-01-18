"""
1. Initialize the controller’s parameters (Ω): the three k values for a standard PID controller and the
weights and biases for a neural-net-based controller
"""

import numpy as np
import jax.numpy as jnp
import jax

import pprint

import random

import matplotlib.pyplot as plt

from controller.PIDController import PIDController
from Plant.BathtubModel import BathtubModel

# plot imports
from visualization.params import plot_params
from visualization.error import plot_error


# 1. Init Controller parameters
learning_rate = 0.01
noise = random.uniform(-0.01, 0.01)
K_p = 0.1
K_d = 0.09
K_i = 0.3


num_epochs = 100
num_timesteps = 10

# Initialize the plant
cross_sectional_area = 200.0
drain_cross_sectional_area = cross_sectional_area / 100.0
initial_height = 50.0
goal_height = 50.0


params = {
    "K_p": K_p,  # blue
    "K_d": K_d,  # orange
    "K_i": K_i,  # green
}


class CONSYS:
    def __init__(self, controller, plant, target_state):
        self.controller = controller(params, learning_rate, noise)
        self.plant = plant
        self.target = target_state

    def run(self):
        # initialize the error history
        # added two zeros to error_history to avoid error in mean_square_error
        error_history = [0, 0]

        # initialize the states
        # states are the height of the water in the bathtub
        states = []
        states.append(initial_height)

        for _ in range(num_epochs):
            # run the epoch
            error, grad = jax.jit(jax.value_and_grad(self.run_epoch, argnums=0))(
                self.controller.params, states, error_history
            )

            print("-------------------")

            pprint.pprint("params")
            pprint.pprint(self.controller.params)
            pprint.pprint("grad")
            pprint.pprint(grad)

            # track the error
            error_history.append(error)
            states = states[-1:]

            self.controller.update_params(grad)

        # remove the first two zeros from error_history
        del error_history[:2]

        # pass track_K_p, track_K_d, track_K_i to plot_params
        self.controller.visualization_params()
        # pass error_history[2:] to plot_error to remove the first two zeros
        plot_error(error_history)

        return error_history

    def run_epoch(self, params, states, error_history):
        # Initialize the controller here in order for it to be traced by jax
        self.controller.reset()
        # Initialize the plant
        plant_instance = self.plant(
            cross_sectional_area,
            drain_cross_sectional_area,
            initial_height,
            self.target,
        )

        for _ in range(num_timesteps):
            # Update the controller
            U = self.controller.update(params, states[-1], error_history, self.target)
            # Update the plant
            states.append(plant_instance.update(U, noise))

        return self.mean_square_error(jnp.array(states), self.target)

    def mean_square_error(self, predictions, targets):
        """
        We take the difference between the predictions and the targets, square it, and take the mean.
        The predictions are the states, and the targets are the goal height.
        """
        return jnp.mean(jnp.square(predictions - targets))


if __name__ == "__main__":
    consys = CONSYS(PIDController, BathtubModel, goal_height)
    error_history = consys.run()
