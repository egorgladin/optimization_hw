import numpy as np
from statistics import stdev
from operator import itemgetter
from copy import deepcopy
import matplotlib.pyplot as plt


def nelder_mead(X0, f, tol, params):
    alpha, gamma, rho, sigma = itemgetter('alpha', 'gamma', 'rho', 'sigma')(params)
    X = X0
    F = [f(x) for x in X]
    trajectory = []

    while stdev(F) > tol:

        # sort
        F, X = (list(t) for t in zip(*sorted(zip(F, X))))
        trajectory.append(deepcopy(X))

        # centroid
        x_o = np.mean(np.vstack(X[:-1]), axis=0)

        # reflection
        x_r = x_o + alpha * (x_o - X[-1])
        f_r = f(x_r)
        if F[0] <= f_r < F[-2]:
            X[-1] = x_r
            F[-1] = f_r
            continue

        # expansion
        if f_r < F[0]:
            x_e = x_o + gamma * (x_r - x_o)
            f_e = f(x_e)
            if f_e < f_r:
                X[-1] = x_e
                F[-1] = f_e
            else:
                X[-1] = x_r
                F[-1] = f_r
            continue

        # contraction
        x_c = x_o + rho * (X[-1] - x_o)
        f_c = f(x_c)
        if f_c < F[-1]:
            X[-1] = x_c
            F[-1] = f_c
            continue

        # shrink
        for i in range(len(X) - 1):
            X[i + 1] = X[0] + sigma * (X[i + 1] - X[0])
            F[i + 1] = f(X[i + 1])

    F, X = zip(*sorted(zip(F, X)))
    trajectory.append(deepcopy(X))
    return X[0], F[0], trajectory


def bird_function(x):
    if np.linalg.norm(x + 5) >= 5:
        return 1e12
    sin = np.sin(x[1])
    cos = np.cos(x[0])
    exp1 = (1 - cos)**2
    exp2 = (1 - sin)**2
    return sin * np.exp(exp1) + cos * np.exp(exp2) + (x[0] - x[1])**2


def bird_function_array(X, Y):
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = bird_function(x)
    return Z


def init_simplex(x, step):
    X = [x]
    for i in range(len(x)):
        x_i = x.copy()
        x_i[i] += step
        X.append(x_i)
    return X


def main():
    x0 = -6 * np.ones(2)
    X0 = init_simplex(x0, 3)
    params = {'alpha': 1., 'gamma': 2., 'rho': 0.5, 'sigma': 0.5}
    x_opt, f_opt, trajectory = nelder_mead(X0, bird_function, 1e-2, params)

    X, Y = np.meshgrid(np.linspace(-10, 0, 100), np.linspace(-10, 0, 100))
    Z = bird_function_array(X, Y)
    vals = set(list(Z.flatten()))
    vals.remove(1e12)
    vals = list(vals)
    min_, max_ = min(vals), max(vals)
    labels = [min_ + (max_ - min_) * i / 10 for i in range(1, 10)]

    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contour(X, Y, Z, labels)

    for X in trajectory:
        xs = [x[0] for x in X]
        ys = [x[1] for x in X]
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys, color='r', linewidth=1)

    plt.savefig("levels.png", bbox_inches='tight')


if __name__=="__main__":
    main()
