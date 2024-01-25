import jax.numpy as jnp
from os import environ
from dotenv import load_dotenv


class CournotCompetition:
    load_dotenv()

    c = float(environ.get("COST"))  # marginal cost
    p_max = float(environ.get("P_MAX"))  # maximum price
    target = float(environ.get("TARGET_PROFIT"))  # target profit

    initial_state = 0.0  # initial state

    def __init__(self, q1: float = initial_state, q2: float = initial_state) -> None:
        """
        Initialize the CournotCompetition class.
        :param q1: initial state of producer 1
        :param q2: initial state of producer 2
        """

        self.q1 = q1
        self.q2 = q2

    def __str__(self):
        return "Cournot_Competition"

    def update(self, signal: float, noise: float = 0.0) -> float:
        """
        Update the plant
        :param signal: the signal (float)
        :param noise: the noise (float)
        :return: the current state (float)
        """

        # 1. q_1 updates based on U
        self.q1 += signal
        # 2. q_2 updates based on D
        self.q2 += noise

        self.q1 = jnp.clip(self.q1, 0.0, 1.0)
        self.q2 = jnp.clip(self.q2, 0.0, 1.0)

        # 3. q = q_1 + q_2
        q = self.q1 + self.q2

        # 4. price: p(q) = p_max - q (assume p_max = 1)
        p = self.p_max - q

        # then on each timestep, producer 1's profit is: P_1 = p(q) * (q_1 - c_m)
        # is the marginal cost: the cost to produce each item, independent of the total number produced.
        profit_1 = p * self.q1 - p * self.c

        return profit_1

    def reset(self):
        """
        Reset the plant to its initial state after each epoch
        :return: None
        """
        self.q1 = 0.0
        self.q2 = 0.0
