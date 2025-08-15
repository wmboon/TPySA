import numpy as np
import matplotlib.pyplot as plt


def get_convs():
    conv_56 = [
        6.85e-01,
        7.76e-01,
        4.90e-01,
        3.87e-01,
        2.14e-01,
        2.99e-01,
        6.06e-01,
        6.28e-02,
        3.38e-02,
        3.75e-02,
        3.17e-02,
        1.34e-02,
        9.76e-03,
        8.48e-03,
        3.62e-03,
        3.10e-03,
        2.31e-03,
        1.44e-03,
        7.49e-04,
        5.92e-04,
        1.11e-03,
        3.08e-04,
        1.91e-04,
        1.45e-04,
        2.02e-04,
        1.68e-04,
        6.58e-05,
        5.27e-05,
        5.43e-05,
        4.40e-05,
        3.90e-05,
        3.57e-05,
    ]

    conv_37 = [
        6.76e-01,
        4.82e-01,
        2.85e-01,
        1.99e-01,
        1.01e-01,
        6.13e-02,
        1.20e-01,
        3.41e-02,
        1.59e-02,
        1.07e-02,
        8.17e-03,
        5.04e-03,
        6.38e-03,
        2.38e-03,
        3.34e-03,
        1.10e-03,
        2.92e-03,
        8.46e-04,
        4.62e-04,
        1.92e-04,
        9.49e-05,
        1.16e-04,
        3.80e-05,
        2.80e-05,
        2.37e-05,
        7.82e-06,
    ]

    conv_25 = [
        6.59e-01,
        2.61e-01,
        1.52e-01,
        8.58e-02,
        6.10e-02,
        5.49e-02,
        1.47e-02,
        1.70e-02,
        9.78e-03,
        5.77e-03,
        3.69e-03,
        6.66e-02,
        4.71e-04,
        4.01e-04,
        6.98e-04,
        2.67e-04,
        5.23e-05,
        2.38e-05,
    ]

    conv_16 = [
        6.18e-01,
        1.35e-01,
        6.16e-02,
        2.59e-02,
        2.04e-02,
        7.74e-03,
        6.20e-03,
        4.24e-03,
        2.00e-03,
        3.97e-04,
        1.22e-04,
        6.02e-05,
        2.81e-05,
        1.67e-05,
    ]
    return [conv_16, conv_25, conv_37, conv_56]


def get_range(x):
    return np.arange(len(x)) + 1


convs = get_convs()

for conv in convs:
    plt.semilogy(get_range(conv), conv)

# plt.semilogy(get_range(conv), np.full_like(conv, 1e-4), "--")
plt.grid(True, which="both", ls="-", color="0.65")

plt.legend(("n = 16", "n = 25", "n = 37", "n = 56"))
ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Residual")

plt.show()
