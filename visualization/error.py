"""This file contains the function to plot the error history of the neural network."""

import matplotlib.pyplot as plt


def plot_error(error_history):
    plt.title("Learning progress")
    plt.plot(error_history, label="error", color="black")
    plt.legend()
    plt.savefig("Learning_progress.png")
    plt.show()
