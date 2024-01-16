"""
1. Initialize the controller’s parameters (Ω): the three k values for a standard PID controller and the
weights and biases for a neural-net-based controller
"""

import numpy as np
import jax.numpy as jnp
import jax

import random

from functools import partial
from jax import jit

import matplotlib.pyplot as plt

from controller.GeneralController import PIDController
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


class CONSYS:
    def __init__(self, controller, plant, target_state):
        self.controller = controller
        self.plant = plant
        self.target = target_state

    def run(self):
        error_history = [0, 0]

        track_K_p = []
        track_K_d = []
        track_K_i = []

        params = {
            "K_p": K_p,
            "K_d": K_d,
            "K_i": K_i,
        }
        states = []
        states.append(initial_height)

        for _ in range(num_epochs):
            # run the epoch
            error, grad = jax.jit(jax.value_and_grad(self.run_epoch, argnums=0))(
                params, states, error_history
            )

            print("params: ", params)
            print("grad: ", grad)

            # track the error
            error_history.append(error)
            states = states[-1:]

            # update the parameters
            for k, V in params.items():
                params[k] = V - learning_rate * grad[k]

            # for each epoch, track the parameters - used for plotting
            track_K_p.append(params["K_p"])
            track_K_d.append(params["K_d"])
            track_K_i.append(params["K_i"])

        # pass track_K_p, track_K_d, track_K_i to plot_params
        plot_params(track_K_p, track_K_d, track_K_i)
        # pass error_history[2:] to plot_error to remove the first two zeros
        self.plot_error(error_history[2:])

        return error_history

    def run_epoch(self, params, states, error_history):
        controller = self.controller()
        # Initialize the plant
        plant = self.plant(
            cross_sectional_area,
            drain_cross_sectional_area,
            initial_height,
            self.target,
        )

        for _ in range(num_timesteps):
            # Update the controller
            U = controller.update(params, states[-1], error_history, self.target)
            # Update the plant
            states.append(plant.update(U, noise))

        return self.mean_square_error(jnp.array(states), self.target)

    def mean_square_error(self, predictions, targets):
        return jnp.mean(jnp.square(predictions - targets))


if __name__ == "__main__":
    consys = CONSYS(PIDController, BathtubModel, goal_height)
    error_history = consys.run()
