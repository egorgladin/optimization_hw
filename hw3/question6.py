import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, hilbert

from hw3.question5 import armijo


def method(compute_grad, choose_step, x, stopping_criterion,
           momentum=0., nesterov=False):
    trajectory = [x.copy(), x.copy()]

    while not stopping_criterion(trajectory):
        y = x + momentum * (x - trajectory[-2])
        grad = compute_grad(y) if nesterov else compute_grad(x)
        alpha = choose_step(x, grad)
        x = y - alpha * grad
        trajectory.append(x.copy())

    return trajectory[1:]


def oracle(A, x, grad=True):
    if grad:
        return 2 * A @ x
    return (x.T @ A @ x).item()


def grad_descent_w_const_stepsize(L, x0, compute_grad, stopping_criterion):
    choose_step = lambda x, grad: 0.5 / L
    trajectory = method(compute_grad, choose_step, x0, stopping_criterion)
    return trajectory


def grad_descent_w_armijo(L, x0, f, compute_grad, stopping_criterion):
    t, alpha, gamma = 1. / L, 0.4, 0.5
    choose_step = lambda x, grad: armijo(f, grad, x, t, alpha, gamma)
    trajectory = method(compute_grad, choose_step, x0, stopping_criterion)
    return trajectory


def steepest_descent(A, x0, compute_grad, stopping_criterion):
    choose_step = lambda x, grad: grad.T @ grad / (2 * grad.T @ A @ grad)
    trajectory = method(compute_grad, choose_step, x0, stopping_criterion)
    return trajectory


def heavy_ball(stepsize, momentum, x0, compute_grad, stopping_criterion):
    choose_step = lambda x, grad: stepsize
    trajectory = method(compute_grad, choose_step, x0, stopping_criterion, momentum=momentum)
    return trajectory


def fgm(stepsize, momentum, x0, compute_grad, stopping_criterion):
    choose_step = lambda x, grad: stepsize
    trajectory = method(compute_grad, choose_step, x0, stopping_criterion, momentum=momentum, nesterov=True)
    return trajectory


def tune_params(f, L, mu, x0, compute_grad, stopping_criterion):
    theor_stepsize = 4. / (np.sqrt(L) + np.sqrt(mu))**2
    theor_momentum = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)))**2
    print(f"Theoretical stepsize and momentum for heavy ball: alpha={theor_stepsize:.3f}, beta={theor_momentum:.3f}")

    theor_stepsize_fgm = 1. / L
    theor_momentum_fgm = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    print(f"Theoretical stepsize and momentum for FGM: alpha={theor_stepsize_fgm:.3f}, beta={theor_momentum_fgm:.3f}")

    stepsizes = [theor_stepsize * 1.5**(2-i) for i in range(10)]
    momentums = [theor_momentum * 1.5**(2-j) for j in range(10)]

    stepsizes_fgm = [theor_stepsize_fgm * 1.5**(2-i) for i in range(10)]
    momentums_fgm = [theor_momentum_fgm * 1.5**(2-j) for j in range(10)]

    best = None
    best_fgm = None
    for i, stepsize in enumerate(stepsizes):
        for j, momentum in enumerate(momentums):
            x_opt = heavy_ball(stepsize, momentum, x0, compute_grad, stopping_criterion)[-1]
            if best is None or f(x_opt) < best[0]:
                best = (f(x_opt), stepsize, momentum)

            stepsize_fgm, momentum_fgm = stepsizes_fgm[i], momentums_fgm[j]
            x_opt = fgm(stepsize_fgm, momentum_fgm, x0, compute_grad, stopping_criterion)[-1]
            if best_fgm is None or f(x_opt) < best_fgm[0]:
                best_fgm = (f(x_opt), stepsize_fgm, momentum_fgm)

    print(f"Best stepsize and momentum for heavy ball: alpha={best[1]:.3f}, beta={best[2]:.3f}")
    print(f"Best stepsize and momentum for FGM: alpha={best_fgm[1]:.3f}, beta={best_fgm[2]:.3f}")
    return best[1], best[2], best_fgm[1], best_fgm[2]


def main():
    n = 5
    A = hilbert(n)
    eigvals = eigh(A, eigvals_only=True)
    L, mu = eigvals[-1], eigvals[0]

    n_steps = 100
    stopping_criterion = lambda traj: len(traj) > n_steps
    compute_grad = lambda x: oracle(A, x)
    f = lambda x: oracle(A, x, grad=False)

    np.random.seed(0)
    x0 = np.random.randn(n, 1)

    stepsize, momentum, stepsize_fgm, momentum_fgm = tune_params(f, L, mu, x0, compute_grad, stopping_criterion)

    methods = ['GD w/ const stepsize',
               'GD w/ Armijo',
               'Steepest Descent',
               'Heavy Ball',
               'FGM']
    trajectories = [grad_descent_w_const_stepsize(L, x0, compute_grad, stopping_criterion),
                    grad_descent_w_armijo(L, x0, f, compute_grad, stopping_criterion),
                    steepest_descent(A, x0, compute_grad, stopping_criterion),
                    heavy_ball(stepsize, momentum, x0, compute_grad, stopping_criterion),
                    fgm(stepsize_fgm, momentum_fgm, x0, compute_grad, stopping_criterion)]

    fsize = 15
    fig = plt.figure(figsize=(10, 5))
    for trajectory, method in zip(trajectories, methods):
        plt.plot([f(x) for x in trajectory], label=method)

    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.yscale('log')
    plt.xlabel(r"# of iterations", fontsize=fsize)
    plt.ylabel(r"$f(x)$", fontsize=fsize)
    plt.legend()
    plt.savefig(f"plots/task6.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
