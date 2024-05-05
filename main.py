import scipy.sparse as sps
import os
import numpy as np
import porepy as pp
import pygeon as pg

from TPSA import TPSA

mdg = pg.unit_grid(3, 0.1)
mdg.compute_geometry()
sd = mdg.subdomains()[0]

mu = np.ones(sd.num_cells)
l_squared = np.ones(sd.num_cells)
labda = np.ones(sd.num_cells)

disc = TPSA()
A = disc.assemble(sd, mu, l_squared, labda)

rhs = np.zeros(A.shape[0])
rhs[(sd.dim - 1) * sd.num_cells : sd.dim * sd.num_cells] = sd.cell_volumes

x = sps.linalg.spsolve(A, rhs)

# Primal variables
indices = sd.num_cells * np.cumsum([sd.dim, disc.dim_r(sd)])
u, r, p = np.split(x, indices)

# Dual variables
y = disc.sigma @ x
indices = sd.num_faces * np.cumsum([sd.dim, disc.dim_r(sd)])
sigma, tau, v = np.split(y, indices)

if sd.dim == 2:
    # we need to add the z component for the exporting
    u = np.hstack((u, np.zeros(sd.num_cells)))
else:
    r = r.reshape((3, -1))

u = u.reshape((3, -1))


folder = os.path.dirname(os.path.abspath(__file__))
save = pp.Exporter(sd, "sol_TPSA", folder_name=folder)
save.write_vtu([("u", u), ("r", r), ("p", p)])
