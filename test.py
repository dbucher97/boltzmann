import numpy as np

h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

h1 = np.kron(np.eye(4), h)
h2 = np.kron(np.kron(np.eye(2), h), np.eye(2))
h3 = np.kron(h, np.eye(4))

print(h1 @ h2 @ h3)


print(h1)
