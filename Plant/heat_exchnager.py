import jax.numpy as jnp
from os import environ
from dotenv import load_dotenv
from .Plant import Plant  # Assuming you have a Plant class that is being imported


class HeatExchanger(Plant):
    load_dotenv()

    initial_state = float(environ.get("INITIAL_TEMPERATURE"))
    target = float(environ.get("GOAL_TEMPERATURE"))
    heat_transfer_coefficient = float(environ.get("HEAT_TRANSFER_COEFFICIENT"))
    heat_capacity = float(environ.get("HEAT_CAPACITY"))
    heat_transfer_area = float(environ.get("HEAT_TRANSFER_AREA"))

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
        # Calculate heat transfer
        heat_transfer = (
            self.heat_transfer_coefficient
            * self.heat_transfer_area
            * (self.target - self.temperature)
        )

        # Calculate change in temperature
        # equation: Q = m * c * delta_T, solve for delta_T
        # https://www.geeksforgeeks.org/heat-transfer-formulas/
        delta_temperature = (signal + noise - heat_transfer) / self.heat_capacity

        # Update temperature
        self.temperature += delta_temperature

        return self.temperature

    def get_temperature(self) -> float:
        """
        Get the current temperature of the HeatExchanger.
        :return: current temperature
        """
        return self.temperature
