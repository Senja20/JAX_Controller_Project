"""
This module contains the GeneralController class. 
This is controller which is inherited by PID and neural network controllers.
"""


class GeneralController:
    """General controller class"""

    def __init__(self, learning_rate, noise_rate):
        """Initialize the controller"""
        self.learning_rate = learning_rate
        self.noise_rate = noise_rate
