import jax
import jax.numpy as jnp

example_trees = [
    [1, "a", object()],
    (1, (2, 3), ()),
    [1, {"k1": 2, "k2": (3, 4)}, 5],
    {"a": 2, "b": (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
    leaves = jax.tree_leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

# Map tree
list_of_lists = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
print(jax.tree_map(lambda x: x * 2, list_of_lists))

another_list_of_lists = list_of_lists
print(jax.tree_map(lambda x, y: x + y, list_of_lists, another_list_of_lists))

#! example ml parameters

import numpy as np


def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                biases=np.ones(shape=(n_out,)),
            )
        )
    return params


params = init_mlp_params([1, 128, 128, 1])

print(jax.tree_map(lambda x: x.shape, params))


def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


LEARNING_RATE = 0.0001


@jax.jit
def update(params, x, y):

    grads = jax.grad(loss_fn)(params, x, y)
    # Note that `grads` is a pytree with the same structure as `params`.
    # `jax.grad` is one of the many JAX functions that has
    # built-in support for pytrees.

    # This is handy, because we can apply the SGD update using tree utils:
    return jax.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


import matplotlib.pyplot as plt

xs = np.random.normal(size=(128, 1))
ys = xs**2

for _ in range(1000):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label="Model prediction")
plt.legend()
plt.waitforbuttonpress()
