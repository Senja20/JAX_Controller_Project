"""This file contains the function to plot the error history of the neural network."""

import matplotlib.pyplot as plt


def plot_error(error_history, controller_name):
    plt.title(f"Learning progress - {controller_name}")
    plt.plot(error_history, label="error", color="black")
    plt.legend()
    plt.savefig(f"Learning_progress_{controller_name}.png")
    plt.show()
