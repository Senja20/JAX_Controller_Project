"""
This file contains the function to plot the control parameters.
"""

import matplotlib.pyplot as plt


def plot_params(track_K_p: list[float], track_K_d: list[float], track_K_i: list[float]):
    """
    Function to plot the control parameters.
    This makes a plot which is displayed to the user.

    :param track_K_p: the K_p values (list)
    :param track_K_d: the K_d values (list)
    :param track_K_i: the K_i values (list)

    :return: None
    """
    plt.title("Control parameters")
    plt.plot(track_K_p, label="K_p", color="blue")
    plt.plot(track_K_d, label="K_d", color="orange")
    plt.plot(track_K_i, label="K_i", color="green")
    plt.legend()
    plt.savefig("Control_parameters.png")
    plt.show()
