"""
This is the main file of the project.
It contains the CONSYS class, which contains the controller, the plant and the target state.
"""

from os import environ
from time import sleep

import jax.numpy as jnp
from dotenv import load_dotenv
from jax import jit, value_and_grad
from tqdm import tqdm

# controller imports
from controller import NNController, PIDController

# model imports
from Plant import BathtubModel, CournotCompetition, HeatExchanger

# utils imports
from utils import generate_random_values

# plot imports
from visualization import plot_error, plot_params


class CONSYS:
    def __init__(self, controller, plant):
        """
        This class contains the controller, the plant and the target state.
        :param controller: the controller (model)
        :param plant: the plant (model)
        :param target_state: the target state (float)

        """
        print("Initializing the system...")
        self.controller = controller
        self.plant = plant
        try:
            self.controller_instance = self.controller(
                float(environ.get("LEARNING_RATE")),
            )
        except TypeError as e:
            print(e)
            # this is so that even if the values are not set in the .env file, the program will still run
            self.controller_instance = self.controller(0.01)

        self.plant_instance = self.plant()
        self.target = self.plant_instance.target

    def run(self) -> list:
        """
        this method runs the system, and contains the loop for the epochs
        :return: error_history: the error history (list)
        """

        # initialize the error history
        # added two zeros to error_history to avoid error in mean_square_error
        error_history = []

        # https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html
        grad_func = jit(value_and_grad(self.run_epoch, argnums=0))

        # https://tqdm.github.io/
        for epoch in tqdm(
            range(int(environ.get("NUMBER_OF_EPOCHS"))), desc="Training", unit="epoch"
        ):
            # run the epoch
            error, grad = grad_func(self.controller_instance.params)

            # track the error
            error_history.append(error)

            # (f) Update Ω based on the gradients.
            self.controller_instance.update_params(grad)

            # print params every 10 epochs - for debugging
            if epoch % 10 == 0:
                # https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write
                tqdm.write(f"\rEpoch: {epoch}, Error: {error}")

        # pass track_K_p, track_K_d, track_K_i to plot_params
        self.controller_instance.visualization_params()

        plot_error(
            error_history, (str(self.controller_instance)), (str(self.plant_instance))
        )

        return error_history

    def run_epoch(self, params: dict) -> float:
        """
        This method runs the epoch, and contains the loop for the timesteps
        :param params: the parameters of the controller (list)
        :param error_history: the error history (list)
        :return: mean_square_error: the mean square error (float)
        """

        # reset the controller and the plant
        # todo: reinitialize controller and plant
        self.controller_instance.reset()
        self.plant_instance.reset()

        # re-initialize the history
        update_states = jnp.array([self.plant_instance.initial_state])

        # initialize the error accumulator
        error_timestamp_acc = 0

        # generate noise list - new noise list for each epoch
        noise_list = generate_random_values(
            int(environ.get("NUMBER_OF_TIMESTEPS")),
            float(environ.get("NOISE_LOWER_BOUND")),
            float(environ.get("NOISE_UPPER_BOUND")),
        )

        for s in range(int(environ.get("NUMBER_OF_TIMESTEPS"))):
            # Update the controller
            U = self.controller_instance.update(
                params,
                update_states[-1],
                error_timestamp_acc,
                self.target,
            )

            # Update the plant
            new_state = self.plant_instance.update(U, noise_list[s])

            # save the values - used tp calculate the error in MSE function
            update_states = jnp.append(update_states, new_state)

            # save the error
            error_timestamp_acc += self.target - new_state

        # returns mean square error by using the difference between the states and the target
        error_ms = self.mean_square_error(update_states, self.target)
        return error_ms

    def mean_square_error(self, predictions: list[float], target: float) -> float:
        """
        Loss - Mean Square Error
        We take the difference between the predictions and the targets, square it, and take the mean.
        The predictions are the states, and the targets are the goal height.
        :param predictions: the states (list)
        :param target: the target state (float)
        :return: the mean square error (float)
        https://www.machinelearningnuggets.com/jax-loss-functions/
        https://medium.com/@sahinadirhan/simple-linear-regression-using-jax-5ef2eefb8cf4
        """
        # (d) Compute MSE over the error history.
        return jnp.mean(jnp.square(target - predictions))


if __name__ == "__main__":
    load_dotenv()

    system = CONSYS(NNController, BathtubModel)
    error_history = system.run()

# https://jax.readthedocs.io/en/latest/jax.numpy.html
# https://www.geeksforgeeks.org/python-os-environ-object/
