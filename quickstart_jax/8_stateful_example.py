from typing import NamedTuple
import jax
import jax.numpy as jnp


class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray


def init(rng) -> Params:
    """Returns the initial model params."""
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)


def loss(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least squares error of the model's predictions on x against y."""
    pred = params.weight * x + params.bias
    return jnp.mean((pred - y) ** 2)


LEARNING_RATE = 0.005


@jax.jit
def update(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> Params:
    """Performs one SGD update step on params using the given data."""
    grad = jax.grad(loss)(params, x, y)

    # If we were using Adam or another stateful optimizer,
    # we would also do something like
    # ```
    # updates, new_optimizer_state = optimizer(grad, optimizer_state)
    # ```
    # and then use `updates` instead of `grad` to actually update the params.
    # (And we'd include `new_optimizer_state` in the output, naturally.)

    new_params = jax.tree_map(lambda param, g: param - g * LEARNING_RATE, params, grad)

    return new_params


import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(42)

# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
x_rng, noise_rng = jax.random.split(rng)
xs = jax.random.normal(x_rng, (128, 1))
noise = jax.random.normal(noise_rng, (128, 1)) * 0.5
ys = xs * true_w + true_b + noise

# Fit regression
params = init(rng)
for _ in range(1000):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, params.weight * xs + params.bias, c="red", label="Model Prediction")
plt.legend()
plt.waitforbuttonpress()
