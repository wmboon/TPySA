We define the solid pressure as
$$
p_s = \lambda(\nabla \cdot u) - \alpha p_f
$$
which rewrites the mass conservation law
$$
\nabla \cdot q + \partial_t (\alpha \lambda^{-1} p_s + (\eta + \alpha^2 \lambda ^{-1})p_f) = f
$$
In a fixed-stress splitting scheme, we lag the solid pressure by one iteration, thus moving the term with $p_s$ to the right-hand side. The remaining equation then has the same structure as the original mass conservation eqaution with a modified storativity term:
$$
\eta \rightarrow \eta + \alpha^2 \lambda^{-1}
$$
This "additional compressibility" of $\alpha^2 \lambda^{-1}$ is implemented by including a new keyword `ROCKBIOT` in opm-common and opm-simulators.
