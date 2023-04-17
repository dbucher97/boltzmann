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

    # h0 = get_hamiltonian(f, 1)
    # v, w = np.linalg.eigh(h0)
    # x = np.argmin(v)
    # print(v[x])
    # print(w[:, x].conj().T @ h0 @ w[:, x])

    ves = []

    for si in s:
        h = get_hamiltonian(f, si)
        v, w = np.linalg.eigh(h)
        vmin = np.min(v)
        if T == 0:
            vec = w[:, np.argmin(v)]
            # vec /= np.sqrt(np.sum(np.abs(vec) ** 2))
            vec = vec.flatten()
        else:
            logz = -vmin / T + np.log(np.exp((-v + vmin) / T).sum())
            p = -v / T - logz
            p = np.exp(p)
            vec = w.dot(np.sqrt(p)).flatten()
        ves.append(np.abs(vec) ** 2)
    return np.array(ves).T
