import jax
import jax.numpy as jnp

# input
x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])


# !grad-transformation


def sum_of_squares(x):
    return jnp.sum(x**2)


def sum_squared_error(x, y):
    return jnp.sum((x - y) ** 2)


# get gradient function
sum_of_squares_dx = jax.grad(sum_of_squares)
print(sum_of_squares(x))
print(sum_of_squares_dx(x))

# !can differentiate with respect to multiple variables
sum_squared_error_dx = jax.grad(sum_squared_error)  # this is in respect to x
print(sum_squared_error_dx(x, y))

print(jax.grad(sum_squared_error, argnums=(0, 1))(x, y))  # Find gradient wrt both x & y

# !can do value and grad
print(jax.value_and_grad(sum_squared_error)(x, y))
