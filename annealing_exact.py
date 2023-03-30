import scipy.sparse as sp
import numpy as np

sx = np.array([[0, 1], [1, 0]])

def get_x_at(N, i):
    if i == 0:
        return np.kron(sx, np.eye(2 ** (N - 1)))
    elif i == N - 1:
        return np.kron(np.eye(2 ** (N - 1)), sx)
    else:
        return np.kron(np.kron(np.eye(2 ** i), sx), np.eye(2 ** (N - i - 1)))

def get_x_sum(n):
    N = int(np.round(np.log2(n)))
    return sum(get_x_at(N, i) for i in range(N))


def get_hamiltonian(f, s):
    return -(1-s) * get_x_sum(len(f)) + s * np.diag(f)


def get_svs(f, n=100, T=1e-3):
    s = np.linspace(0, 1, n)
    ves = []

    for si in s:
        h = get_hamiltonian(f, si)
        v, w = np.linalg.eigh(h)
        vmin = np.min(v)
        logz = -vmin / T + np.log(np.exp((-v + vmin) / T).sum())
        p = -v / T - logz
        p = np.exp(p)
        vec = w.dot(np.sqrt(p)).flatten()
        ves.append(vec ** 2)
    return np.array(ves).T
