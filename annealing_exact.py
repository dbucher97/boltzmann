import scipy.sparse as sp
import numpy as np

def get_hamiltonian(f, s):
    return -(1-s) * np.flip(sp.identity(len(f)).tolil(), axis=1) + s * sp.diags(f)


def get_svs(f, n=100, T=1e-3):
    s = np.linspace(0, 1, n)
    ves = []

    for si in s:
        h = get_hamiltonian(f, si)
        v, w = np.linalg.eigh(h.todense())
        vmin = np.min(v)
        logz = -vmin / T + np.log(np.exp((-v + vmin) / T).sum())
        p = -v / T - logz
        p = np.exp(p)
        vec = w.dot(np.sqrt(p)).A.flatten()
        ves.append(vec ** 2)
    return np.array(ves).T
