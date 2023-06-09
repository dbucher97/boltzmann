import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from qaoa import qaoa, get_initial
from qiskit_qaoa import qaoa as qiskit_qaoa
from annealing_exact import get_svs

N = 6


def obj(x):
    return np.sin(x) * np.exp(-(x**2) / 20) / 2 + 0.5


def get_bm(T):
    bm = np.exp(-f[:, None] / T[None, :])
    return bm / bm.sum(axis=0)[None, :]


def plot_anim(bm, name, rvar, n=None, running="temp"):
    fig = plt.figure()
    ax = plt.axes(ylim=(0, np.max(bm[:, -1])))
    bars = ax.bar(x, [0] * len(x), width= 9 / bm.shape[0])
    tex = ax.text(2, 0.9 * np.max(bm[:, -1]), "text")
    ax2 = ax.twinx()
    ax2.set_ylim(-0.2, 1.2)
    ax2.plot(x, f, color="tab:orange")
    ax2.set_ylabel("objective function")
    ax.set_ylabel("probability")
    ax.set_xlabel("x")
    ax.set_title(name)

    if n is not None:
        step = bm.shape[1] // n
        bm = bm[:, ::step]
        rvar = rvar[::step]

    def init():
        tex.set_text(f"{running} = {rvar[0]:.3f}")
        return tex,

    def animate(i):
        for rect, y in zip(bars, bm[:, i]):
            rect.set_height(y)
        tex.set_text(f"{running} = {rvar[i]:.3f}")
        return tex,

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=bm.shape[1], interval=20, blit=True
    )

    anim.save(f"{name}.mp4", fps=30, extra_args=["-vcodec", "libx264"])


if __name__ == "__main__":
    x = np.linspace(-5, 5, 2**N)
    f = obj(x)

    # T = np.logspace(2, -2, 100)
    # bm = get_bm(T)
    # plot_anim(bm, "boltzmann", T, running="temp")

    # Tfinal = 100
    # steps = 100
    # y = qiskit_qaoa(f, steps, Tfinal)
    # rvar = np.linspace(0, Tfinal, steps)
    # plot_anim(np.array(y), "qiskit_qaoa", rvar, n=100, running="time")

    # p = get_initial(steps, Tfinal)
    #
    # plt.plot(p[:, 1], label="beta")
    # plt.plot(p[:, 0], label="gamma")
    #
    # plt.savefig("schedule.png", dpi=300)

    n = 100
    res = get_svs(f, n, T=0)
    rvar = np.linspace(0, 1, n)
    plot_anim(res, "exact_diagonalization_T=0", rvar, n=n, running="s")
