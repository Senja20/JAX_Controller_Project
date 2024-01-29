import jax.numpy as jnp
from jax.random import PRNGKey, uniform


def generate_random_values(size, lower_bound, upper_bound):
    key = PRNGKey(0)

    # Generate random values between 0 and 1
    return uniform(
        key, shape=(size,), dtype=jnp.float32, minval=lower_bound, maxval=upper_bound
    )
