import numpy as np
import matplotlib.pyplot as plt

from hw1.coordinate_descent import bisection


def grad_descent(compute_grad, choose_step, x, N, tol=1e-6):
    """
    Minimize f via gradient descent.

    :param compute_grad: method that takes argument x and returns df/dx
    :param choose_step: method such that choose_step(x, grad) returns stepsize
    :param x: starting point
    :param N: maximal number of iterations
    """
    trajectory = [x.copy()]

    for i in range(N):
        grad = compute_grad(x)
        alpha = choose_step(x, grad)
        x -= alpha * grad
        trajectory.append(x.copy())

        if np.linalg.norm(x - trajectory[-2]) < tol:
            print(f"Stopping criterion is satisfied after {i} iterations")
            return trajectory

    return trajectory


def oracle8(A, b, x, grad=False):
    """
    Compute f(x)=(Ax,x)+(b,x) or f'(x)=2Ax+b.

    :param A: square matrix
    :param b: vector
    :param x: query point
    :param grad: if True, return df/dx, else return f(x).
    """
    if grad:
        return 2 * A @ x + b
    return (x.T @ A @ x + b.T @ x).item()


def f8_array(A, b, X, Y):
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = oracle8(A, b, x)
    return Z


def plot_f8(A, b):
    # create a grid and choose function's levels to plot
    X, Y = np.meshgrid(np.linspace(-1, 11, 200), np.linspace(-1, 11, 200))
    Z = f8_array(A, b, X, Y)
    vals = list(Z.flatten())
    min_, max_ = min(vals), max(vals)
    labels = [min_ + (max_ - min_) * i**3 / 20**3 for i in range(1, 20)]

    # plot function
    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contour(X, Y, Z, labels)


def get_step_func(f, line_search):
    def choose_step(x, grad):
        g = lambda alpha: f(x - alpha * grad)
        return line_search(g)
    return choose_step


def task_8(initial_stepsize, n_bisection, n_steps):
    A = np.array([[1., 0.5],
                  [0.5, 10]])
    b = np.array([[-5.],
                  [-22.]])
    f = lambda x: oracle8(A, b, x)
    compute_grad = lambda x: oracle8(A, b, x, grad=True)
    line_search = lambda g: bisection(g, initial_stepsize, initial_stepsize, n_bisection)
    choose_step = get_step_func(f, line_search)

    plot_f8(A, b)

    init_points = [np.array([[1.], [10]]),
                   np.array([[10.], [10]]),
                   np.array([[10.], [1]])]
    for i, x0 in enumerate(init_points):
        trajectory = grad_descent(compute_grad, choose_step, x0, n_steps)

        colors = ['r', 'g', 'b']
        for j, x in enumerate(trajectory):
            if j < len(trajectory) - 1:
                dx = trajectory[j+1] - x
                plt.arrow(x[0, 0], x[1, 0], dx[0, 0], dx[1, 0],
                          color=colors[i], head_width=0.08, length_includes_head=True)

        for x in trajectory:
            print(f"{x[0, 0]:.2f} & {x[1, 0]:.2f} & {f(x):.2f} \\")
        print("="*20)

    plt.savefig("plots/task8.png", bbox_inches='tight')


def oracle9(x, grad=False):
    """
    Compute f(x) or f'(x), where f is the function from the task 9

    :param x: query point
    :param grad: if True, return df/dx, else return f(x).
    """
    vector = (2 * x**2 - np.roll(x, -1) - 1)
    vector[-1, 0] = 0
    if grad:
        dfdx1 = 8 * vector * x
        dfdx1[0, 0] += (x[0, 0] - 1) / 2
        dfdx2 = 2 * np.roll(vector, 1)
        dfdx2[0, 0] = 0
        return dfdx1 - dfdx2
    return (x[0, 0] - 1)**2 / 4 + (vector**2).sum()  # .item()


def task_9(initial_stepsize, n_bisection, n_steps):
    compute_grad = lambda x: oracle9(x, grad=True)
    line_search = lambda g: bisection(g, initial_stepsize, initial_stepsize, n_bisection)

    for n in [3, 10]:
        print(f"{'='*10} n={n} {'='*10}")
        x0 = np.ones((n, 1))
        x0[0] = -1.5

        for alpha in [None, 0.1, 0.5, 1.]:
            choose_step = get_step_func(oracle9, line_search) if alpha is None\
                else lambda x, grad: alpha
            # if n == 3 and alpha is None:
            #     n_steps = 1000
            trajectory = grad_descent(compute_grad, choose_step, x0.copy(), n_steps)

            fsize = 15
            fig = plt.figure(figsize=(10, 5))
            plt.plot([oracle9(x) for x in trajectory])
            plt.xticks(fontsize=fsize)
            plt.yticks(fontsize=fsize)
            plt.yscale('log')
            plt.xlabel(r"# of iterations", fontsize=fsize)
            plt.ylabel(r"$f(x)$", fontsize=fsize)
            plt.savefig(f"plots/task9_n{n}_alpha{'_argmin' if alpha is None else alpha}.png", bbox_inches='tight')


def main():
    # Task 8
    initial_stepsize = 5.
    n_bisection = 8
    n_steps = 15
    task_8(initial_stepsize, n_bisection, n_steps)

    # Task 9
    task_9(initial_stepsize, n_bisection, n_steps)


if __name__ == "__main__":
    main()
