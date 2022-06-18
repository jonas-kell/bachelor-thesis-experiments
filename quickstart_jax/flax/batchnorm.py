import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax


class BiasAdderWithRunningMean(nn.Module):
    decay: float = 0.99

    @nn.compact
    def __call__(self, x):
        # easy pattern to detect if we're initializing via empty variable tree
        is_initialized = self.has_variable("batch_stats", "mean")
        ra_mean = self.variable(
            "batch_stats", "mean", lambda s: jnp.zeros(s), x.shape[1:]
        )
        mean = ra_mean.value  # This will either get the value or trigger init
        bias = self.param("bias", lambda rng, shape: jnp.zeros(shape), x.shape[1:])
        if is_initialized:
            ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(
                x, axis=0, keepdims=True
            )

        return x - ra_mean.value + bias


key1, key2 = random.split(random.PRNGKey(0), 2)
x = jnp.ones((10, 5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print("initialized variables:\n", variables)
y, updated_state = model.apply(variables, x, mutable=["batch_stats"])
print("updated state:\n", updated_state)

for val in [1.0, 2.0, 3.0]:
    x = val * jnp.ones((10, 5))
    y, updated_state = model.apply(variables, x, mutable=["batch_stats"])
    old_state, params = variables.pop("params")
    variables = freeze({"params": params, **updated_state})
    print("updated state:\n", updated_state)  # Shows only the mutable part


def update_step(tx, apply_fn, x, opt_state, params, state):
    def loss(params):
        y, updated_state = apply_fn(
            {"params": params, **state}, x, mutable=list(state.keys())
        )
        l = ((x - y) ** 2).sum()
        return l, updated_state

    (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, state


x = jnp.ones((10, 5))
variables = model.init(random.PRNGKey(0), x)
state, params = variables.pop("params")
del variables
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for _ in range(3):
    opt_state, params, state = update_step(tx, model.apply, x, opt_state, params, state)
    print("Updated state: ", state)
