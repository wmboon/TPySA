import os

import numpy as np
import porepy as pp
import pygeon as pg
import scipy.sparse as sps

from tpysa import TPSA

# Grid the unit cube
sd = pg.unit_grid(3, 0.1, as_mdg=False)
sd.compute_geometry()

# Material parameters
mu = np.full(sd.num_cells, 1.0)
l_squared = np.full(sd.num_cells, 1.0)
labda = np.full(sd.num_cells, 1.0)

# Initiate the discretization class and assemble the lhs
disc = TPSA()
A = disc.discretize(sd, mu, l_squared, labda)

# Assemble the rhs
rhs = np.zeros(A.shape[0])

# Set a (gravity-like) force on the z-component
indices_uz = slice((sd.dim - 1) * sd.num_cells, sd.dim * sd.num_cells)
rhs[indices_uz] = sd.cell_volumes

# Solve the system with a direct solve
x = sps.linalg.spsolve(A, rhs)

# Extract the primary (cell-based) variables
indices = sd.num_cells * np.cumsum([sd.dim, disc.dim_r(sd)])
u, r, p = np.split(x, indices)

# Extract the secondary (face_based) variables
y = disc.sigma @ x
indices = sd.num_faces * np.cumsum([sd.dim, disc.dim_r(sd)])
sigma, tau, v = np.split(y, indices)

if sd.dim == 2:
    # Add the z component for vtk
    u = np.hstack((u, np.zeros(sd.num_cells)))
else:
    # The rotation is a vector in 3D
    r = r.reshape((3, -1))
u = u.reshape((3, -1))

# Export
folder = os.path.dirname(__file__)
save = pp.Exporter(sd, "sol_TPSA", folder_name=folder)
save.write_vtu([("u", u), ("r", r), ("p", p)])
