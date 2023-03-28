import jax
import jax.numpy as jnp


def get_initial(depth, T=1.0):
    dt = T / depth
    r = (jnp.arange(depth) + 1) / depth
    gammas = r * dt
    betas = (1 - r + 1 / depth) * dt
    return jnp.stack((betas, gammas)).T


def fwht(x):
    d = x.shape[0]
    h = 2
    while h <= d:
        hf = h // 2
        x = x.reshape((-1, h))
        half_1, half_2 = x[:, :hf], x[:, hf:]
        x = jnp.hstack((half_1 + half_2, half_1 - half_2))
        h = 2 * h

    return (x / jnp.sqrt(d)).reshape((d,))


def build_mixer(n):
    a = jnp.array([-0.5, 0.5], dtype=jnp.float32)
    ret = jnp.array([0], dtype=jnp.float32)
    for _ in range(n):
        ret = ret[:, None] + a[None, :]
        ret = ret.ravel()
    return ret


def get_layer(objective, mixer):
    @jax.jit
    def layer(x, params):
        x = x * jnp.exp(-1j * objective * params[1])
        x = fwht(x)
        x = x * jnp.exp(-1j * mixer * params[0])
        x = fwht(x)
        return x, jnp.abs(x) ** 2

    return layer


def qaoa(objective, steps: int = 1, T: float = 1.0):
    initial = jnp.ones_like(objective, dtype=jnp.float32) / jnp.sqrt(objective.shape[0])
    mixer = build_mixer(int(jnp.round(jnp.log2(objective.shape[0]))))
    layer = get_layer(objective, mixer)

    params = get_initial(steps, T)

    x = initial
    xs = []
    for p in params:
        x, y = layer(x, p)
        xs.append(y)

    return x, jnp.array(xs).T.real
