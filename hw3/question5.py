import numpy as np
import matplotlib.pyplot as plt


def grad_descent(compute_grad, choose_step, x, tol=1e-3):
    trajectory = [x.copy()]
    go_on = True

    while go_on:
        grad = compute_grad(x)
        if np.linalg.norm(grad) < tol:
            return trajectory
        alpha = choose_step(x, grad)
        x -= alpha * grad
        trajectory.append(x.copy())


def oracle(x, grad=True):
    x1, x2 = x[0, 0], x[1, 0]
    if grad:
        f1 = 8 * x1**3 + 4 * x1 + x2 - 3
        f2 = 12 * x2**3 + 8 * x2 + x1 - 2
        return np.array([[f1],
                         [f2]])
    x1, x2 = x[0, 0], x[1, 0]
    return 2 * x1**4 + 3 * x2**4 + 2 * x1**2 + 4 * x2 ** 2 + x1 * x2 - 3 * x1 - 2 * x2


def armijo(f, grad, x, t, alpha, gamma):
    """
    Compute stepsize according to Armijo rule.

    :param f: method that takes argument x and returns f(x)
    :param grad: gradient of f in x
    :param x: argument at the current step
    :param t: initial stepsize (positive number)
    :param alpha: number in range (0, 1)
    :param gamma: number in range (0, 1), decay rate
    """
    while f(x - t * grad) > f(x) - alpha * t * np.linalg.norm(grad) ** 2:
        t *= gamma
    return t


def plot_f():
    # create a grid and choose function's levels to plot
    X, Y = np.meshgrid(np.linspace(-0.2, 0.9, 200), np.linspace(-0.2, 0.6, 200))
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([[X[i, j]],
                          [Y[i, j]]])
            Z[i, j] = oracle(x, grad=False)
    vals = list(Z.flatten())
    min_, max_ = min(vals), max(vals)
    labels = [min_ + (max_ - min_) * i**3 / 20**3 for i in range(1, 20)]

    # plot function
    fig, ax = plt.subplots(figsize=(11, 8))
    cs = ax.contour(X, Y, Z, labels)


def main():
    x0 = np.zeros((2, 1))
    f = lambda x: oracle(x, grad=False)
    t, alpha, gamma = 1., 0.1, 0.9
    choose_step = lambda x, grad: armijo(f, grad, x, t, alpha, gamma)
    trajectory = grad_descent(oracle, choose_step, x0)

    plot_f()
    for j, x in enumerate(trajectory):
        if j < len(trajectory) - 1:
            dx = trajectory[j+1] - x
            plt.arrow(x[0, 0], x[1, 0], dx[0, 0], dx[1, 0],
                      head_width=0.015, length_includes_head=True)
    plt.savefig("plots/task5.png", bbox_inches='tight')
    print(f"# of iteration = {len(trajectory) - 1}, final point: {trajectory[-1].flatten()}")


if __name__ == "__main__":
    main()
