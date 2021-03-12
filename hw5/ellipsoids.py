import numpy as np
import matplotlib.pyplot as plt


def ellipsoid_function(X, Y, P_inv, c):
    Z = np.empty((len(Y), len(X)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            xy = np.array([[x],
                           [y]]) - c
            Z[j, i] = (xy.T @ P_inv @ xy).item()
    return Z


def plot_ellipse(P, c, X, Y, color, w=None):
    P_inv = np.linalg.inv(P)
    Z = ellipsoid_function(X, Y, P_inv, c)
    plt.contour(X, Y, Z - 1, [0], colors=color)
    if w is not None:
        xx, yy = get_segment(w, X, Y, P_inv)
        plt.plot(xx, yy, color=color)


def get_segment(w, X, Y, P_inv):
    xx, yy = [], []
    if w[1] == 0:
        x = 0.
        for y in Y:
            xy = np.array([[x],
                           [y]])
            if (xy.T @ P_inv @ xy).item() <= 1:
                xx.append(x)
                yy.append(y)

    else:
        for x in X:
            y = - (w[0] * x / w[1]).item()
            xy = np.array([[x],
                           [y]])
            if (xy.T @ P_inv @ xy).item() <= 1:
                xx.append(x)
                yy.append(y)

    return xx, yy


def main():
    X = np.linspace(-2, 2, 400)
    Y = np.linspace(-3, 3, 600)

    # First figure
    P_0 = np.array([[1., 0],
                    [0, 4]])
    c0 = np.zeros((2, 1))

    P_1 = np.array([[4/9, 0],
                    [0, 16/3]])
    c1 = np.array([[-1/3],
                   [0]])
    w1 = np.array([[1.],
                   [0]])

    plt.figure(figsize=(6, 9))
    plot_ellipse(P_0, c0, X, Y, 'r', w1)
    plot_ellipse(P_1, c1, X, Y, 'g')

    plt.savefig("ellipsoids1.png", bbox_inches='tight')

    # Second figure
    P_2 = np.array([[13, 8],
                    [8., 28]]) * 4 / 45
    c2 = np.array([[-1],
                   [4]]) / (3 * np.sqrt(5))
    w2 = np.array([[0.5],
                   [-0.5]])

    plt.figure(figsize=(6, 9))
    plot_ellipse(P_0, c0, X, Y, 'r', w2)
    plot_ellipse(P_2, c2, X, Y, 'b')
    plt.savefig("ellipsoids2.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
