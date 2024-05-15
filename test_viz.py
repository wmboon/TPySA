import pygeon as pg
import porepy as pp
import numpy as np

mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
grids = [
    # pp.mdg_library.square_with_orthogonal_fractures("simplex", mesh_args, [1]),
    pg.unit_grid(2, 0.15),
]

# for g in grids:
#     mdg, _ = g
#     pg.convert_from_pp(mdg)
#     mdg.compute_geometry()

mdg = pg.as_mdg(grids[0])
pg.convert_from_pp(mdg)
mdg.compute_geometry()

tree = pg.SpanningTree(mdg)
tree.visualize_2d(mdg, "tree_1.pdf")

sd = mdg.subdomains()[0]
# start = np.argmax(sd.tags["domain_boundary_faces"])
start = np.where(sd.tags["domain_boundary_faces"])[0]
start = np.split(start, 2)[0]
tree = pg.SpanningTree(mdg, start)
tree.visualize_2d(mdg, "tree_2.pdf")
