import numpy as np


def from_Lame_to_YoungPoisson(labda, mu):
    pratio = labda / (2 * (labda + mu))
    ymodulus = mu * (3 * labda + 2 * mu) / (labda + mu)
    return ymodulus, pratio


def from_YoungPoisson_to_Lame(ymodulus, pratio):
    mu = ymodulus / (2 * (1 + pratio))
    labda = ymodulus * pratio / (1 + pratio) / (1 - 2 * pratio)
    return labda, mu


if __name__ == "__main__":
    labda, mu = np.random.rand(2, 5)

    labda_out, mu_out = from_YoungPoisson_to_Lame(*from_Lame_to_YoungPoisson(labda, mu))

    assert np.allclose(mu_out, mu)
    assert np.allclose(labda_out, labda)

    labda_out, mu_out = from_Lame_to_YoungPoisson(*from_YoungPoisson_to_Lame(labda, mu))

    assert np.allclose(mu_out, mu)
    assert np.allclose(labda_out, labda)

    E, nu = from_Lame_to_YoungPoisson(1e9, 1e9)
    print("{:.5e}".format(E * 1e-9))
    print("{:.5e}".format(nu))

    labda, mu = from_YoungPoisson_to_Lame(E, nu)
    print("{:.5e}".format(labda * 1e-9))
    print("{:.5e}".format(mu * 1e-9))
    pass
