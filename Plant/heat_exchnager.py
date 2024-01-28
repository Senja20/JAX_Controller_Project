from os import environ

import jax.numpy as jnp
from dotenv import load_dotenv

from .Plant import Plant  # Assuming you have a Plant class that is being imported


class HeatExchanger(Plant):
    load_dotenv()

    initial_state = float(environ.get("INITIAL_TEMPERATURE"))
    target = float(environ.get("GOAL_TEMPERATURE"))
    heat_transfer_coefficient = float(environ.get("HEAT_TRANSFER_COEFFICIENT"))
    heat_capacity = float(environ.get("HEAT_CAPACITY"))
    heat_transfer_area = float(environ.get("HEAT_TRANSFER_AREA"))

    previous_temperature = initial_state

    def __init__(self, temperature: float = initial_state) -> None:
        """
        Initialize the HeatExchanger class.
        :param temperature: initial temperature of the system
        """
        self.temperature = temperature

    def __str__(self) -> str:
        """
        String representation of the HeatExchanger class.
        :return: string
        """
        return "HeatExchanger"

    def reset(self) -> None:
        """
        Reset the HeatExchanger class to its initial state.
        :return: None
        """
        self.temperature = self.initial_state

    def update(self, signal: float, noise: float = 0.0) -> float:
        """
        Update the HeatExchanger class.
        :param signal: input signal affecting the system
        :param noise: random noise affecting the system
        :return: current temperature of the system
        """
        # Calculate change in temperature
        # equation: Q = m * c * delta_T, solve for delta_T
        # https://www.geeksforgeeks.org/heat-transfer-formulas/
        delta_temperature = self.__calculate_delta_temperature(signal, noise)

        # Update temperature
        self.temperature += delta_temperature

        self.previous_temperature = self.temperature

        return self.temperature

    def __calculate_delta_temperature(self, signal: float, noise: float) -> float:
        """
        Calculate the change in temperature.
        equation: Q = m * c * delta_T, solve for delta_T
        https://www.geeksforgeeks.org/heat-transfer-formulas/
        :param signal: input signal affecting the system
        :param noise: random noise affecting the system
        :return: change in temperature
        """
        return (signal + noise - self.__calculate_heat_transfer()) / self.heat_capacity

    def __calculate_heat_transfer(self) -> float:
        """
        Calculate the heat transfer.
        equation: Q = h * A * delta_T
        Q: heat transfer
        h: heat transfer coefficient
        A: heat transfer area
        delta_T: change in temperature
        :return: heat transfer
        """
        return (
            self.heat_transfer_coefficient  # h
            * self.heat_transfer_area  # A
            * (self.previous_temperature - self.temperature)  # delta_T
        )
