"""
1. Initialize the controllers parameters (Ω): the three k values for a standard PID controller and the
weights and biases for a neural-net-based controller
"""

import jax.numpy as jnp
from jax import value_and_grad, jit
from os import environ
from dotenv import load_dotenv


# controller imports
from controller import PIDController, NNController

# model imports
from Plant import BathtubModel, CournotCompetition, TemperatureModel

# plot imports
from visualization import plot_error, plot_params

# utils imports
from utils import generate_random_values


class CONSYS:
    def __init__(self, controller, plant):
        """
        This class contains the controller, the plant and the target state.
        :param controller: the controller (model)
        :param plant: the plant (model)
        :param target_state: the target state (float)

        """
        print("Initializing the system...")
        try:
            self.controller = controller(
                float(environ.get("LEARNING_RATE")),
            )
        except TypeError as e:
            print(e)
            # this is so that even if the values are not set in the .env file, the program will still run
            self.controller = controller(0.01)

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

        grad_func = jit(value_and_grad(self.run_epoch, argnums=0))

        for epoch in range(int(environ.get("NUMBER_OF_EPOCHS"))):
            # run the epoch
            error, grad = grad_func(self.controller.params)

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

        plot_error(error_history, (str(self.controller)), (str(self.plant)))

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
        self.controller.reset()
        self.plant.reset()

        # re-initialize the history
        update_states = jnp.array([self.plant.initial_state])

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
            U = self.controller.update(
                params,
                update_states[-1],
                error_timestamp_acc,
                self.target,
            )

            # Update the plant
            new_state = self.plant.update(U, noise_list[s])

            # save the values - used tp calculate the error in MSE function
            update_states = jnp.append(update_states, new_state)

            # save the error
            error_timestamp_acc += self.target - new_state

        # returns mean square error by using the difference between the states and the target
        return self.mean_square_error(update_states, self.target)

    def mean_square_error(self, predictions: list[float], target: float) -> float:
        """
        Loss - Mean Square Error
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
    system = CONSYS(PIDController, TemperatureModel)
    error_history = system.run()
