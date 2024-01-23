class CournotCompetition:
    c = 0.1  # marginal cost
    p_max = 1  # maximum price

    target = 0.5  # target profit

    def __init__(self):
        pass

    def update(self, signal: float, noise: float = 0.01) -> float:
        # at each timestep:

        # 1. q_1 updates based on U

        # 2. q_2 updates based on D

        # 3. q = q_1 + q_2

        # 4. p(q) = p_max - q (assume p_max = 1)

        # then on each timestep, producer 1's profit is: P_1 = p(q) * (q_1 - c_m)
        # is the marginal cost: the cost to produce each item, independent of the total number produced.

        # In this model, the target value (T) denotes the goal profit for each timestep. Thus:
        # E = T - P_1

        # and the error (E) serves as input to the controller on each timestep.
        # In this model, cm can be a small fraction, such as 0.1, but you are free to experiment with different values
        # of it along with values of pmax and T

        pass
