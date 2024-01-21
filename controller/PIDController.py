import jax.numpy as jnp
from controller.GeneralController import GeneralController
import matplotlib.pyplot as plt
import random

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

    def __init__(self, learning_rate, noise_rate):
        """Initialize the PID controller"""
        super().__init__(learning_rate, noise_rate)
        self.params = params_initial_PID

        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

        self.track_K_p = []
        self.track_K_i = []
        self.track_K_d = []

    def update(self, params, current_state, error_history, target_state):
        """Update the PID controller"""
        self.error = target_state - current_state

        self.derivate = params["K_d"] * self.__calculate_derivative()
        self.integral = params["K_i"] * jnp.sum(jnp.array(error_history))

        self.last_error = self.error

        return params["K_p"] * self.error + self.derivate + self.integral

    def __calculate_derivative(self):
        """Calculate the derivative of the PID controller"""
        return self.last_error - self.error

    def reset(self):
        """Reset the PID controller"""
        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

    def update_params(self, grad):
        # update the parameters
        for k, V in self.params.items():
            self.params[k] = V - self.learning_rate * grad[k]

        # track the parameters
        self.__track_params()

    def __track_params(self):
        """Track the parameters"""
        self.track_K_p.append(self.params["K_p"])
        self.track_K_i.append(self.params["K_i"])
        self.track_K_d.append(self.params["K_d"])

    def visualization_params(self):
        """Plot the parameters"""
        plt.title("Control parameters")
        plt.plot(self.track_K_p, label="K_p", color="blue")
        plt.plot(self.track_K_d, label="K_d", color="orange")
        plt.plot(self.track_K_i, label="K_i", color="green")
        plt.legend()
        plt.savefig("Control_parameters.png")
        plt.show()
