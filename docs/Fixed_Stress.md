## Solving the Biot poroelasticity using fixed stress iterations

Let us briefly recall the equations. First, the mechanics of the porous medium influence the mass balance equation

$$
    \nabla \cdot q + \partial_t (\alpha \nabla \cdot u + \eta p_f) = f_p
$$

Second, the fluid pressure acts as an isotropic stress in the momentum balance equation

$$
    \nabla \cdot(2 \mu \varepsilon(u) + (\lambda \nabla \cdot u - \alpha p_f)I) = f_u
$$

We define the solid pressure as

$$
    p_s = \lambda(\nabla \cdot u) - \alpha p_f
$$

which rewrites these conservation laws as

$$
\begin{align}
    \nabla \cdot q + \partial_t (\alpha \lambda^{-1} p_s + (\eta + \alpha^2 \lambda^{-1})p_f) &= f_p \\
    \nabla \cdot(2 \mu \varepsilon(u) + p_s I) &= f_u
\end{align}
$$

A TPSA discretization of these equations includes the face variables $(\sigma, \tau, v, q)$ and yields the system:

$$
\begin{align}
    \nabla \cdot 
    \begin{bmatrix}
        \sigma \\\ \tau \\\ v \\\ q
    \end{bmatrix}
    - 
    \begin{bmatrix}
        0 \\\ & \mu^{-1} \\\ & & \lambda^{-1} & \alpha \lambda^{-1} \\\ & & - \alpha \lambda^{-1} \partial_t & -(\eta + \alpha^2 \lambda^{-1}) \partial_t
    \end{bmatrix}
    \begin{bmatrix}
        u \\\ r \\\ p_s \\\ p_f
    \end{bmatrix}
    =
    \begin{bmatrix}
        f_u \\\ 0 \\\ 0 \\\ f_p
    \end{bmatrix}
\end{align}
$$

In a fixed-stress splitting scheme, we solve these equations iteratively. Letting a superscript denote the iteration index, we obtain

$$
\begin{align}
    \nabla \cdot 
    \begin{bmatrix}
        \sigma \\\ \tau \\\ v \\\ q
    \end{bmatrix}^{n + 1}
    - 
    \begin{bmatrix}
        0 \\\ & \mu^{-1} \\\ & & \lambda^{-1} & \\\ & &  & -(\eta + \alpha^2 \lambda^{-1}) \partial_t
    \end{bmatrix}
    \begin{bmatrix}
        u \\\ r \\\ p_s \\\ p_f
    \end{bmatrix}^{n + 1}
    =
    \begin{bmatrix}
        f_u \\\ 0 \\\ 0 \\\ f_p
    \end{bmatrix}
    +
    \begin{bmatrix}
       0 \\\ 0 \\\ \alpha \lambda^{-1} p_f^n \\\ - \alpha \lambda^{-1} \partial_t p_s^{n + 1}
    \end{bmatrix}
\end{align}
$$

Note that the first three equations are exactly TPSA for elasticity. The final equation can be solved using OPM Flow. 
For that, we needed to make two adjustments. First, the right-hand side needs to be adjusted using the keyword `SOURCE`. 
Second, the "additional compressibility" of $\alpha^2 \lambda^{-1}$ is implemented by including a new keyword `ROCKBIOT`.

Unfortunately, we cannot directly start iterating at a time step because I haven't figured out a way to revert OPM Flow to a previous time step, using Python commands. Nevertheless, there are two straightforward iteration schemes that we can now consider:
- Lagged: At time step $t_i$, we set $p_f^n = p_f(t_i)$, instead of the unkown $p_f(t_{i + 1})$. This effectively lags the solid pressure by one iteration.
- Time-loop: Solve for an entire simulation and save the source terms $- \alpha \lambda^{-1} \partial_t p_s^{n + 1}$ at each time step. Since we cannot reinstantiate a Flow Simulator (something to do with MPI), this involves writing to file and re-running the Python script.