"""This file contains the function to plot the error history of the neural network."""

import matplotlib.pyplot as plt


def plot_error(error_history, controller_name):
    """
    The function to plot the error history of the neural network.
    This makes a plot which is displayed to the user.

    :param error_history: the error history (list)
    :param controller_name: the name of the controller (string)

    :return: None
    """
    plt.title(f"Learning progress - {controller_name}")
    plt.plot(error_history, label="error", color="black")
    plt.legend()
    plt.savefig(f"Learning_progress_{controller_name}.png")
    plt.show()
