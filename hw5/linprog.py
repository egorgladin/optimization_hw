from scipy.optimize import linprog
import numpy as np


def roll_n_stack(a):
    tup = (a, np.roll(a, 4), np.roll(a, 8))
    return np.vstack(tup)


def main():
    c = np.tile([310, 380, 350, 285], [3])

    available = np.tile(np.eye(4), [1, 3])
    weight_constr = roll_n_stack(np.array([1.]*4 + [0]*8))
    a = [480., 650, 580, 390]
    vol_constr = roll_n_stack(np.array(a + [0]*8))
    A_ub = np.vstack((
        available, weight_constr, vol_constr
    ))

    limits = [18., 15, 23, 12]
    weight_cap = [10, 16, 8]
    vol_cap = [6800, 8700, 5300]
    b_ub = np.array(limits + weight_cap + vol_cap)

    A_eq = np.array([
        [4]*4 + [0]*4 + [-5]*4,
        [0]*4 + [1]*4 + [-2]*4,
    ])
    b_eq = np.zeros(2)

    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
    np.set_printoptions(precision=3)
    print("Solution: ", res.x)
    print("Optimal value: ", -res.fun)

    for i, lim in enumerate(limits):
        for factor, mode in zip([0.9, 1.1], ['Decreasing', 'Increasing']):
            print(f"\n{mode} available weight of C{i+1} by 10%")
            limits[i] = factor * lim
            b_ub = np.array(limits + weight_cap + vol_cap)

            res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
            np.set_printoptions(precision=3)
            print("Solution: ", res.x)
            print("Optimal value: ", -res.fun)
        limits[i] = lim


if __name__ == "__main__":
    main()
