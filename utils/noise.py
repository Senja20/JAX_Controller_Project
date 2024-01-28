import jax.numpy as jnp
from jax.random import PRNGKey, uniform


def generate_random_values(size, lower_bound, upper_bound):
    key = PRNGKey(0)

    # Generate random values between 0 and 1
    random_values = uniform(key, shape=(size,), dtype=jnp.float32)

    # Scale and shift the values to the desired range
    scaled_values = lower_bound + random_values * (upper_bound - lower_bound)

    return scaled_values
