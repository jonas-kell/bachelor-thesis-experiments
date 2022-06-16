import jax.numpy as jnp
from jax import random, device_put, devices, grad, jit, vmap
from timeit_custom import timeit

# init rand
key = random.PRNGKey(0)

# init variables
size = 7000
x = random.normal(key, (size, size), dtype=jnp.float32)

# time for runs
time_runs = 100

timeit(lambda x: jnp.dot(x, x.T), time_runs)(x=x)

# runs on the GPU
timeit(lambda x: jnp.dot(x, x.T).block_until_ready(), time_runs)(x=x)

x = device_put(x)  # store on the gpu
# runs on the GPU, but tnesor is already stored there
timeit(lambda x: jnp.dot(x, x.T).block_until_ready(), time_runs)(x=x)


def matmul(x):
    return jnp.dot(x, x.T)


matmul_jit = jit(matmul)  # just-in time compile the function


# runs on the GPU, but cached
# -> the timeit must also be wrapped in jit. Otherwise only the second time "matmul_jit" gests invoked "ON THIS LEVEL" it actually is faster. (that it is invoked multiple times INSIDE timit doesn't help)
# https://jax.readthedocs.io/en/latest/faq.html#:~:text=JAX%20code%20is%20Just%2DIn%2DTime%20(JIT)%20compiled.&text=jit()%20on%20your%20outer,functions%20are%20also%20JIT%20compiled.
@jit
def wrap(x):
    timeit(matmul_jit, time_runs)(x=x)


wrap(x)
