import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
import cvxpy as cp
import glpk


def naive_randomization(n, N):
    return


def main():
    N = 1000
    n = 20
    np.random.seed(0)
    a = np.random.rand(n, n)
    g = np.tril(a) + np.tril(a, -1).T
    L = laplacian(g)

    plt.figure(figsize=(12, 8))

    # Naive randomization
    naive_rand = np.random.choice([-1., 1.], (n, N))
    naive_rand_objective = np.diag(naive_rand.T @ L @ naive_rand)
    plt.plot(naive_rand_objective, 'o', label="Naive randomization")

    # SDP relaxation
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i, i] == 1 for i in range(n)
    ]
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X)),
                      constraints)
    prob.solve(solver=cp.CVXOPT)
    plt.axhline(y=prob.value, color='r', linestyle='-', label="SDP relaxation")

    # Goemans-Williamson
    vals = []
    V = np.linalg.cholesky(X.value)
    x = np.zeros(n)
    for i in range(N):
        xi = np.random.randn(n)
        xi /= np.linalg.norm(xi)
        for j in range(len(x)):
            x[j] = 1. if V[:, j] @ xi > 0 else -1.
        vals.append((x.T @ L @ x).item())
    plt.plot(vals, 'o', label="Goemans-Williamson")
    plt.axhline(y=np.mean(vals), color='g', linestyle='-', label="Goemans-Williamson mean value")

    plt.legend()
    plt.savefig("max_cut.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
