"""
This module contains the GeneralController class. 
This is controller which is inherited by PID and neural network controllers.
"""


class GeneralController:
    """General controller class"""

    # constructor
    def __init__(self, learning_rate: float):
        """¨
        Initialize the controller
        :param learning_rate: the learning rate (float)
        """
        self.learning_rate = learning_rate

        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

    def __str__(self):
        """String representation of the controller"""
        return "General controller"

    # public method
    def reset(self):
        """
        Reset the PID controller"
        :return: None
        """
        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

    # protected method
    def _calculate_derivative(self):
        """¨
        Calculate the derivative of the neural network controller
        :return: the derivative (float)
        """
        return self.last_error - self.error
