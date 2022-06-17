import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn


class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel",
            self.kernel_init,  # Initialization function
            (inputs.shape[-1], self.features),
        )  # shape info.
        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
        )  # TODO Why not jnp.dot?
        bias = self.param("bias", self.bias_init, (self.features,))
        y = y + bias
        return y


key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4, 4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print("initialized parameters:\n", params)
print("output:\n", y)
