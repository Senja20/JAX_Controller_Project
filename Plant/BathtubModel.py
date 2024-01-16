"""
This is the first model of the assignment. 
It is called the BathtubModel as described in the assignment.
"""

import numpy as np
import jax.numpy as jnp
import jax


class BathtubModel:
    """
    This is the BathtubModel class.
    The bathtub is assumed to have a constant cross-sectional area (A) from top to bottom.
    It has a drain of cross-sectional area (C), which is typically a small fraction of A (e.g. C = A/100).
    The height of water in the bathtub is H, and the velocity (V) of water exiting through the drain is V = sqrt(2*g*H).
    where g is the gravitational constant (9.8m/sec^2).
    """

    g = 9.8

    def __init__(self, A: float, C: float, H_0: float) -> None:
        """
        Initialize the BathtubModel class.
        :param A: cross-sectional area of the bathtub
        :param C: cross-sectional area of the drain
        :param H_0: initial height of water in the bathtub
        """
        self.A = A  # cross-sectional area of the bathtub
        self.C = C  # cross-sectional area of the drain
        self.H = H_0  # initial height of water in the bathtub
        self.target = H_0

    def update(self, signal: float, noise: float = 0.01) -> float:
        """
        Update the BathtubModel class.
        :param H: height of water in the bathtub
        :return: height of water in the bathtub after one time step
        """
        # water exiting the drain
        water_exiting_drain = self.water_exiting_drain(self.target)
        # volume of water exiting the drain
        volume_exiting_drain = signal + noise - water_exiting_drain
        # height of water in the bathtub after one time step
        self.H = self.H + volume_exiting_drain / self.A

        return self.H

    def velocity(self, H: float) -> float:
        """
        Calculate the velocity of water exiting the drain.
        :param H: height of water in the bathtub
        :return: velocity of water exiting the drain
        """
        return jnp.sqrt(2 * self.g * H)

    def water_exiting_drain(self, H: float) -> float:
        """
        Calculate the water exiting the drain.
        :param H: height of water in the bathtub
        :return: water exiting the drain
        """
        return self.C * self.velocity(H)
