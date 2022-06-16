import jax.numpy as jnp
from jax import random, device_put, devices, grad, jit, vmap
from timeit_custom import timeit

# init rand
key = random.PRNGKey(0)

# init variables
size = 7000
x = random.normal(key, (size, size), dtype=jnp.float32)

timeit(lambda: jnp.dot(x, x.T), nr_runs=10)

# runs on the GPU
timeit(
    lambda: jnp.dot(x, x.T).block_until_ready(),
    nr_runs=10,
)

x = device_put(x)  # store on the gpu
# runs on the GPU
timeit(
    lambda: jnp.dot(x, x.T).block_until_ready(),
    nr_runs=10,
)


def matmul(x):
    return jnp.dot(x, x.T)


matmul_jit = jit(matmul)  # just-in time compile the function


# runs on the GPU, but cached
timeit(
    lambda: matmul_jit(x),
    nr_runs=10,
)
