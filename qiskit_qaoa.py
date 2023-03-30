from qiskit import QuantumCircuit, Aer, QuantumRegister
import numpy as np

N = 8

def obj(x):
    return np.sin(x) * np.exp(-(x**2) / 20) / 2 + 0.5

def get_initial(depth, T=1.0):
    dt = T / depth
    r = (np.arange(depth) + 1) / depth
    gammas = r * dt
    betas = (1 - r + 1 / depth) * dt
    return np.stack((betas, gammas)).T


x = np.linspace(-5, 5, 2**N)
f = obj(x)


def sig(x):
    return 1/(1 + np.exp(-x))

def qaoa(f, steps=100, T=10):
    N = int(np.log2(f.shape[0]))

    qr = QuantumRegister(N)
    qc = QuantumCircuit(qr)
    backend = Aer.get_backend("aer_simulator")

    qc.h(qr)

    for i in range(steps):
        s = i / steps
        k = 5
        # s = np.array(1 - np.exp(-k * s)) / np.array(1 - np.exp(-k))
        gamma = s * T / steps
        beta = (1-s) * T / steps
        qc.diagonal(np.exp(-1j * f * gamma).tolist(), qr)
        qc.rx(-beta, qr)

        qc.save_statevector(f"{i}")

    result = backend.run(qc).result().data()

    statevectors = []
    for i in range(steps):
        statevectors.append(np.abs(result[f"{i}"].data) ** 2)

    return np.array(statevectors).T
