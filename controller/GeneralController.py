"""
This module contains the GeneralController class. 
This is controller which is inherited by PID and neural network controllers.
"""

import numpy as np
import jax.numpy as jnp
import jax


class GeneralController:
    """General controller class"""

    def __init__(self, learning_rate, noise_rate):
        """Initialize the controller"""
        self.learning_rate = learning_rate
        self.noise_rate = noise_rate


class PIDController(GeneralController):
    """PID controller class"""

    def __init__(self, learning_rate, noise_rate, Kp, Ki, Kd):
        """Initialize the PID controller"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0
        self.derivate = 0
        self.integral = 0

    def update(self, params, current_state, error_history, target_state):
        """Update the PID controller"""
        self.error = target_state - current_state

        error_array = jnp.array(error_history)
        self.derivate = params["K_d"] * ((self.error - error_array[-1]) / 2)
        self.integral = params["K_i"] * jnp.sum(error_array)

        return params["K_p"] * self.error + self.derivate + self.integral
