from jax import random

key = random.PRNGKey(42)

print(key)

print(random.normal(key))
print(random.normal(key))

# !example
print("old key", key)
new_key, subkey = random.split(key)
del key  # The old key is discarded -- we must never use it again.
normal_sample = random.normal(subkey)
print(r"    \---SPLIT --> new key   ", new_key)
print(r"             \--> new subkey", subkey, "--> normal", normal_sample)
del subkey  # The subkey is also discarded after use.

# Note: you don't actually need to `del` keys -- that's just for emphasis.
# Not reusing the same values is enough.

key = new_key  # If we wanted to do this again, we would use new_key as the key.

# !easier
key, subkey = random.split(key)
key, *forty_two_subkeys = random.split(key, num=43)

random.normal(subkey, shape=(3,))
