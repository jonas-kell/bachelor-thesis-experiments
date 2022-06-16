import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2.0, 3.0, 4.0])


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1 : i + 2], w))
    return jnp.array(output)


print(convolve(x, w))


xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

auto_batch_convolve = jax.vmap(convolve)
print(auto_batch_convolve(xs, ws))
