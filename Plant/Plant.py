class Plant:
    def __str__(self):
        return f"Plants"

    def update(self, signal: float, noise: float = 0.0) -> float:
        """
        Update the plant.
        :param signal: the signal from the controller (float)
        :param noise: the noise (float)
        :return: the new state (float)
        """
        raise NotImplementedError("Plant.update()")

    def reset(self):
        """
        Reset the plant to its initial state after each epoch
        :return: None
        """
        raise NotImplementedError("Plant.reset()")
