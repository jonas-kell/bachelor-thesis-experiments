import jax.numpy as jnp
from jax import random, device_put, devices, grad, jit, vmap
from timeit import timeit

key = random.PRNGKey(0)

size = 7000
x = random.normal(key, (size, size), dtype=jnp.float32)
print(timeit("jnp.dot(x, x.T)", "from __main__ import jnp, x", number=100))
print(
    timeit(
        "jnp.dot(x, x.T).block_until_ready()",
        "from __main__ import jnp, x",
        number=100,
    )
)  # runs on the GPU

x = device_put(x)  # store on the gpu
print(
    timeit(
        "jnp.dot(x, x.T).block_until_ready()",
        "from __main__ import jnp, x",
        number=100,
    )
)  # runs on the GPU


def matmul(x):
    return jnp.dot(x, x.T)


matmul_jit = jit(matmul)  # just-in time compile the function
print(
    timeit(
        "matmul_jit(x).block_until_ready()",
        "from __main__ import matmul_jit, x",
        number=100,
    )
)  # runs on the GPU, but cached
