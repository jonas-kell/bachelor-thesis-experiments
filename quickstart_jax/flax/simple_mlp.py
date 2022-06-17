import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn


class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
            # providing a name is optional though!
            # the default autonames would be "Dense_0", "Dense_1", ...
        return x


key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4, 4))

model = SimpleMLP(features=[3, 4, 5])
params = model.init(key2, x)
y = model.apply(params, x)

print("initialized parameter shapes:\n", jax.tree_map(jnp.shape, unfreeze(params)))
print("output:\n", y)
