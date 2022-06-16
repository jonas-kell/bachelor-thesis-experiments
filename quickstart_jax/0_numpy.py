import jax.numpy as jnp

y = jnp.array([1, 2, 3])


def jax_in_place_modify(x):
    return x.at[0].set(123)


print(jax_in_place_modify(y))
print(y)
