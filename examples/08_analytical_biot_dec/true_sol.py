import numpy as np
import sympy as sp

## FLOW

# 1 Darcy in terms of m2
K = 1 / 1.01325e12

visc = 1e-3 * 0.5

x = sp.symbols("x y z")

p = sp.sin(sp.pi * x[0]) ** 2 * sp.sin(sp.pi * x[1]) ** 2 * sp.sin(sp.pi * x[2]) ** 2
q = [-K / visc * sp.diff(p, x_j) for x_j in x]

source = sum([sp.diff(q_i, x_i) for (q_i, x_i) in zip(q, x)])

# Functions
p_lamb = sp.lambdify(x, p, "numpy")
q_lamb = sp.lambdify(x, q, "numpy")
source_lamb = sp.lambdify(x, source, "numpy")


def p_func(x):
    return np.array(p_lamb(*x))


def q_func(x):
    return np.array(q_lamb(*x))


def source_func(x):
    return np.array(source_lamb(*x))


## MECHANICS


mu = 0.01
labda = 1

x = sp.symbols("x y z")
phi = sp.sin(sp.pi * x[0]) ** 2 * sp.sin(sp.pi * x[1]) ** 2 * sp.sin(sp.pi * x[2]) ** 2
# phi = sum([x_i * (1 - x_i) for x_i in x])

u = [
    sp.diff(phi, x[1]) - sp.diff(phi, x[2]),
    sp.diff(phi, x[2]) - sp.diff(phi, x[0]),
    sp.diff(phi, x[0]) - sp.diff(phi, x[1]),
]


del_u = np.array([[sp.diff(u_i, x_j) for x_j in x] for u_i in u])
sym_u = 0.5 * (del_u + del_u.T)
asym_u = del_u - del_u.T

r = [asym_u[1, 2], asym_u[2, 0], asym_u[0, 1]]

div_u = sum([sp.diff(u_i, x_j) for (u_i, x_j) in zip(u, x)])

stress = 2 * mu * sym_u + labda * div_u * np.eye(3)

div_stress = [
    sum([sp.diff(s_i, x_j) for (s_i, x_j) in zip(stress_i, x)]) for stress_i in stress
]

# Functions
u_lamb = sp.lambdify(x, u, "numpy")
r_lamb = sp.lambdify(x, r, "numpy")
body_force_lamb = sp.lambdify(x, div_stress, "numpy")


def u_func(x):
    return np.array(u_lamb(*x))


def body_force_func(x):
    return np.array(body_force_lamb(*x))


def r_func(x):
    return mu * np.array(r_lamb(*x))


pass
