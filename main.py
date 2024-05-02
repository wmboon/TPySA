import scipy.sparse as sps
import numpy as np
import pygeon as pg

from TPSA import TPSA

mdg = pg.unit_grid(2, 0.5)
mdg.compute_geometry()
sd = mdg.subdomains()[0]

mu = np.ones(sd.num_cells)
l_squared = np.ones(sd.num_cells)

disc = TPSA(sd, mu)


pass
