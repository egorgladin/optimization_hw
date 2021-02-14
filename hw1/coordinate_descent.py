import numpy as np
import matplotlib.pyplot as plt
from hw1.nelder_mead import bird_function, plot_bird_function
import random


def bisection(g, y0, step, n_steps):
    a, b = y0 - step, y0 + step
    c = (a + b) / 2
    gc = g(c)
    for i in range(n_steps):
        d = (a + c) / 2
        e = (c + b) / 2
        gd, ge = g(d), g(e)
        if gd < gc:
            c = d
            b = c
            gc = gd
        elif ge < gc:
            a = c
            c = e
            gc = ge
        else:
            a = d
            b = e
    return c


def coordinate_descent(x0, step, f, n_bisection, tol):
    x = x0
    trajectory = [x.copy()]
    n = len(x0)
    iter = 0
    stationary = 0
    evals = [0]

    while stationary < 2 * n:
        random.seed(iter)
        i = random.randint(0, 1)
        iter += 1

        def g(y):
            x_ = x.copy()
            x_[i] = y
            return f(x_)

        y = bisection(g, x[i], step, n_bisection)  # 1 + 2*n_bisection evals
        if abs(x[i] - y) < tol:
            stationary += 1
        else:
            stationary = 0
        x[i] = y
        trajectory.append(x.copy())
        evals.append(evals[-1] + 1 + 2*n_bisection)

    return x, trajectory, evals


def main():
    EXPERIMENT_NUMBER = 1

    # initialize
    if EXPERIMENT_NUMBER == 1:
        x0 = -6 * np.ones(2)  # np.array([-5, -7])

    elif EXPERIMENT_NUMBER == 2:
        x0 = -4 * np.ones(2)

    elif EXPERIMENT_NUMBER == 3:
        x0 = -6 * np.ones(2)

    else:
        raise ValueError("Unknown experiment number")

    # run coordinate descent
    tol = 1e-2
    n_bisection = 6
    step = 3.
    x_opt, trajectory, evals = coordinate_descent(x0, step, bird_function, n_bisection, tol)

    plot_bird_function()

    # plot trajectory
    xs = [x[0] for x in trajectory]
    ys = [x[1] for x in trajectory]
    plt.plot(xs, ys, color='r', linewidth=1)
    plt.savefig(f"experiment{EXPERIMENT_NUMBER + 3}.png", bbox_inches='tight')

    fsize = 15
    fig = plt.figure(figsize=(10, 5))
    plt.plot(evals, [bird_function(x) for x in trajectory])
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r"# of $f$ evaluations", fontsize=fsize)
    plt.ylabel(r"$f(x)$", fontsize=fsize)
    plt.savefig(f"experiment{EXPERIMENT_NUMBER + 3}_evals.png", bbox_inches='tight')
    print(x_opt)


if __name__=="__main__":
    main()
