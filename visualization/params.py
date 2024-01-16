"""
This file contains the function to plot the control parameters.
"""

import matplotlib.pyplot as plt


def plot_params(track_K_p, track_K_d, track_K_i):
    """
    Function to plot the control parameters.
    """
    plt.title("Control parameters")
    plt.plot(track_K_p, label="K_p", color="blue")
    plt.plot(track_K_d, label="K_d", color="orange")
    plt.plot(track_K_i, label="K_i", color="green")
    plt.legend()
    plt.savefig("Control_parameters.png")
    plt.show()
