# Optimization Methods for Advanced Understanding

## Optimization model

Let

```math
\mathcal T=\mathbb R^{n_1\times\cdots\times n_d}
```

be the ambient tensor space. An optimization-based decomposition uses a
parameter manifold ``\mathcal M`` and a reconstruction map

```math
\Phi:\mathcal M\longrightarrow\mathcal T.
```

For a parameter point ``p\in\mathcal M`` and target tensor ``A\in\mathcal T``,
the solver minimizes

```math
\min_{p\in\mathcal M} f(p),
\qquad
f(p)
=\frac12\left\|
  \Phi(p)-A
 \right\|_F^2.
```

Thus the optimizer updates ``p`` on ``\mathcal M``; it does not optimize
the entries of the reconstructed tensor independently.

For example, a rank-``R`` CP model uses

```math
\Phi_{\mathrm{CP}}(p)
=\sum_{r=1}^{R}\lambda_r
  u_r^{(1)}\otimes\cdots\otimes u_r^{(d)},
```

with

```math
p
=\left(\lambda,U^{(1)},\ldots,U^{(d)}\right),
\qquad
U^{(m)}
=\left[u_1^{(m)},\ldots,u_R^{(m)}\right].
```

The selected geometry determines the precise representation of
``\mathcal M`` and its metric, while the reconstruction objective remains the
same. See [Join models](join.md) for the product-manifold, image-map, and
differential formulation of sums of structured components.

## Residual, gradient, and optimization steps

Define the residual tensor and its vectorization by

```math
\mathcal R(p)=\Phi(p)-A,
\qquad
\rho(p)=\operatorname{vec}(\mathcal R(p))\in\mathbb R^N,
\qquad
N=\prod_{m=1}^{d}n_m.
```

The differential ``D\Phi(p)`` maps a tangent direction
``\xi\in T_p\mathcal M`` to the corresponding first-order change in the
reconstructed tensor. The directional derivative of the objective is

```math
Df(p)[\xi]
=\left\langle
  \mathcal R(p),D\Phi(p)[\xi]
 \right\rangle_F.
```

The Riemannian gradient is the tangent vector characterized by

```math
g_p\!\left(\operatorname{grad}f(p),\xi\right)
=Df(p)[\xi]
\qquad
\text{for every }\xi\in T_p\mathcal M,
```

where ``g_p`` is the metric selected for the parameter manifold.

To express the residual differential as a matrix, choose an orthonormal tangent
basis ``\mathcal B(p)=(E_1,\ldots,E_q)``, where ``q=\dim\mathcal M``. The
coordinate Jacobian is

```math
J(p)\in\mathbb R^{N\times q},
\qquad
J(p)_{:,j}=D\rho(p)[E_j]
=\operatorname{vec}\!\left(D\Phi(p)[E_j]\right).
```

Because the basis is orthonormal, the tangent-coordinate representation of the
Riemannian gradient is ``J(p)^\top\rho(p)``. Different geometries can therefore
use the same reconstruction objective while producing different tangent bases,
metrics, and optimization steps.

A Riemannian gradient step first chooses a tangent direction and then maps it
back to the manifold with a retraction ``\operatorname{Retr}``:

```math
\xi_k=-\alpha_k\operatorname{grad}f(p_k),
\qquad
p_{k+1}=\operatorname{Retr}_{p_k}(\xi_k).
```

The step size ``\alpha_k`` may be fixed or selected by line search. RCG and
L-BFGS use the same tangent-space gradient but also reuse information from
earlier iterations to construct the search direction.

Levenberg--Marquardt instead uses the residual Jacobian. In the same orthonormal
tangent basis, it solves the damped Gauss--Newton system

```math
\left(J(p)^\top J(p)+\mu_{\mathrm{LM}}I\right)s
=-J(p)^\top\rho(p),
\qquad
s\in\mathbb R^q.
```

The coordinate vector ``s`` defines the tangent step
``\xi=\sum_{j=1}^q s_jE_j``, and the next iterate is
``p^+=\operatorname{Retr}_p(\xi)``. The parameter ``\mu_{\mathrm{LM}}>0``
controls the damping. Smaller damping approaches a
Gauss--Newton step, while larger damping produces a more conservative,
gradient-like step. Consequently, LM requires both residual evaluation and
Jacobian actions; decomposition routes without those operations cannot use the
LM solver.

## Methods by decomposition

### CPD

- `:als` -- alternating least squares, a fast structured baseline.
- `:rgd` -- Riemannian gradient descent with line search; the default
  refinement method.
- `:rgd_fixed` -- Riemannian gradient descent with a fixed step, mainly for
  controlled experiments.
- `:rcg` -- Riemannian conjugate gradient.
- `:lbfgs` -- limited-memory Riemannian quasi-Newton refinement.
- `:lm` -- Levenberg--Marquardt using residual and Jacobian information.

### BTD

- `:als` -- alternating updates of the Tucker blocks.
- `:rgd` -- Riemannian gradient descent with line search; the default
  refinement method.
- `:rgd_fixed` -- fixed-step Riemannian gradient descent.
- `:rcg` -- Riemannian conjugate gradient.
- `:lbfgs` -- limited-memory Riemannian quasi-Newton refinement.
- `:btd_tsd` -- BTD-specific blockwise tangent-subspace descent.

BTD does not currently support `:lm`.

### Generic Join approximation

- `:rgd`
- `:rgd_fixed`
- `:rcg`
- `:lbfgs`
- `:lm`

Generic Join approximation does not provide ALS. When `approx` routes a uniform
Segre or Tucker collection to CPD or BTD, it inherits that specialized method
set.

### Tucker decomposition

Tucker uses direct decomposition routines rather than these optimization solver
objects:

- `tucker(...; method=:sthosvd)` -- sequentially truncated HOSVD.
- `tucker(...; method=:hooi)` -- iterative higher-order orthogonal iteration.
- `thosvd(...)` -- classical truncated HOSVD through its standalone function;
  `:thosvd` is not currently a `tucker` method symbol.

The high-level entry-point docstrings remain the source of truth for current
support and defaults: [`cpd`](@ref), [`btd`](@ref), [`approx`](@ref), and
[`tucker`](@ref).

## Initialization

CPD and BTD are nonconvex. Their default non-ALS paths use an ALS warm start so
the refinement solver begins from a structured fit. For reproducible comparisons,
set the relevant random seed or provide an explicit starting point.

## Stopping and diagnostics

Common controls include:

- `maxiter`: maximum number of refinement iterations;
- `tol`: stopping tolerance;
- `stepsize`: initial or fixed step size for applicable first-order methods;
- `verbose`: progress output.

Solver-specific constructor docstrings below list every advanced control and
its current default. When a solver is selected by symbol, pass those controls
as keywords to the high-level entry point. Passing a constructed solver object
is useful when the same configuration is reused.

Optimization-based result types provide:

```julia
rel_error(result)
iterations(result)
converged(result)
solver(result)
solver_info(result)
```

Use `rel_error(A, result)` when you want to recompute error from an explicit
reconstruction. Compare solver settings on held-out or application-relevant data
rather than selecting them from training error alone.

## Solver API and defaults

The constructor signatures and defaults below are inserted directly from source
docstrings:

```@docs
ALSSolver
RGDSolver
RGDFixedSolver
RCGSolver
LBFGSSolver
LMSolver
```

## Nonnegative CPD

`nncpd` handles nonnegative components through the CPD pipeline. Its standard
configuration is intended to hide geometry details, while advanced experiments
can compare alternative nonnegative geometries. The supported names and current
defaults are maintained in the [`nncpd`](@ref) docstring.
