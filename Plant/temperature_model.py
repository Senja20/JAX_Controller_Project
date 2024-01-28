"""
This is the third model of the assignment. 
It is called the GrowthModel.
"""

import jax.numpy as jnp
from os import environ
from dotenv import load_dotenv
from .Plant import Plant


class TemperatureModel(Plant):
    """
    This is the GrowthModel class.
    The model is to describe the growth of a population in an environment with limited resources.
    dN/dt = rN(1-N/K) + D
    """

    load_dotenv()


    initial_state = float(environ.get("INITIAL_TEMPERATURE")) # T
    ambient_temperature = float(environ.get("AMBIENT_TEMPERATURE")) # T_a
    heat_transfer_coefficient = float(environ.get("HEAT_TRANSFER_COEFFICIENT")) # k
    target_temperature = float(environ.get("TARGET_TEMPERATURE")) # T_target

    def __init__(
        self,
        T: float = initial_state,
        T_a: float = ambient_temperature,
        k: float = heat_transfer_coefficient,
        target: float = target_temperature,
    ) -> None:
        """
        Initialize the temperature model class.
        :param T: initial temperature
        :param T_a: ambient temperature
        :param k: heat transfer coefficient
        :param target: target temperature
        """
        self.T = T  # initial temperature
        self.T_a = T_a  # ambient temperature
        self.k = k  # heat transfer coefficient
        self.target = target # target temperature

    def __str__(self) -> str:
        """
        String representation of the TemperatureModel class.
        :return: string
        """
        return "Temperature_model"

    def reset(self) -> None:
        """
        Reset the Temperature class.
        :return: None
        """
        self.T = self.initial_state

    def update(self, signal: float, noise: float = 0.0) -> float:
        """
        Update the temperature model.
        :param signal: the signal from the controller (float)
        :param noise: the noise (float)
        """
        # dT/dt = -k(T - T_a) + signal - noise

        temperature_change = - self.k * self.T + self.k * self.T_a

        # update the temperature

        self.T += temperature_change - noise + signal

        return self.T

