import jax.numpy as jnp
from controller.GeneralController import GeneralController
import matplotlib.pyplot as plt
import random
from os import environ
from dotenv import load_dotenv

# random initialization of the parameters
K_p = random.uniform(0, 1)
K_d = random.uniform(0, 1)
K_i = random.uniform(0, 1)


params_initial_PID = {
    "K_p": K_p,  # blue
    "K_d": K_d,  # orange
    "K_i": K_i,  # green
}


class PIDController(GeneralController):
    """PID controller class"""

    # constructor
    def __init__(self, learning_rate: float, noise_rate: float):
        """
        Initialize the PID controller - results in an instance of PID controller
        :param learning_rate: the learning rate (float)
        :param noise_rate: the noise rate (float)
        """

        load_dotenv()

        super().__init__(learning_rate, noise_rate)

        self.params = {
            "K_p": random.uniform(
                float(environ.get("KP_LOWER_BOUND")),
                float(environ.get("KP_UPPER_BOUND")),
            ),  # blue
            "K_d": random.uniform(
                float(environ.get("KD_LOWER_BOUND")),
                float(environ.get("KD_UPPER_BOUND")),
            ),  # orange
            "K_i": random.uniform(
                float(environ.get("KI_LOWER_BOUND")),
                float(environ.get("KI_UPPER_BOUND")),
            ),  # green
        }

        self.track_K_p = []
        self.track_K_i = []
        self.track_K_d = []

    def __str__(self):
        """String representation of the PID controller"""
        return "PID_controller"

    # public method
    def update(
        self,
        params: dict,
        current_state: float,
        error_history: list[float],
        target_state: float,
    ) -> float:
        """
        Update the PID controller
        :param params: the parameters of the PID controller (dict)
        :param current_state: the current state (float)
        :param error_history: the error history (list)
        :param target_state: the target state (float)
        :return: the output of the PID controller (float)
        """

        self.error = target_state - current_state

        self.predictions = params["K_p"] * self.error
        self.derivate = params["K_d"] * super()._calculate_derivative()
        self.integral = params["K_i"] * error_history

        self.last_error = self.error

        return self.predictions + self.derivate + self.integral

    def update_params(self, grad: dict) -> None:
        """¨
        Parameters update of the PID controller
        :param grad: the gradients (dict)
        :return: None
        """
        # update the parameters
        for k, V in self.params.items():
            self.params[k] = V - self.learning_rate * grad[k]

        # track the parameters for visualization
        self.__track_params()

    def __track_params(self):
        """
        Track the parameters by appending them to the lists
        :return: None
        """
        self.track_K_p.append(self.params["K_p"])
        self.track_K_i.append(self.params["K_i"])
        self.track_K_d.append(self.params["K_d"])

    def visualization_params(self):
        """
        Plot the parameters
        :return: None
        """
        plt.title("Control parameters")
        plt.plot(self.track_K_p, label="K_p - prediction", color="blue")
        plt.plot(self.track_K_d, label="K_d - derivative", color="orange")
        plt.plot(self.track_K_i, label="K_i - integral", color="green")
        plt.legend()
        plt.savefig("Control_parameters.png")
        plt.show()
