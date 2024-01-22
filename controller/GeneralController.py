"""
This module contains the GeneralController class. 
This is controller which is inherited by PID and neural network controllers.
"""


class GeneralController:
    """General controller class"""

    # constructor
    def __init__(self, learning_rate, noise_rate):
        """Initialize the controller"""
        self.learning_rate = learning_rate
        self.noise_rate = noise_rate

        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

    def __str__(self):
        """String representation of the controller"""
        return "General controller"

    # public method
    def reset(self):
        """Reset the PID controller"""
        self.error = 0
        self.derivate = 0
        self.integral = 0
        self.last_error = 0

    # protected method
    def _calculate_derivative(self):
        """Calculate the derivative of the neural network controller"""
        return self.last_error - self.error
