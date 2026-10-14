# Optimization Methods

## The common problem

An optimization-based decomposition has two parts:

- a parameter point ``p`` that stores factors, cores, or weights;
- a reconstruction map ``\Phi(p)`` that turns those parameters into a tensor.

For a target tensor ``A``, the solver minimizes

```math
f(p)=\frac12\left\|\Phi(p)-A\right\|_F^2.
```

The parameters lie on a manifold ``\mathcal M``. A manifold describes valid
parameter values, such as unit-length factor vectors or orthonormal factor
matrices. The solver changes ``p`` while keeping it on this valid set.

## Residual, gradient, and retraction

The residual is the difference between the reconstruction and the target:

```math
\mathcal R(p)=\Phi(p)-A.
```

For a small valid parameter change ``\xi``, the differential
``D\Phi(p)[\xi]`` is the corresponding first-order change in the reconstructed
tensor. The objective changes by

```math
Df(p)[\xi]
=\left\langle\mathcal R(p),D\Phi(p)[\xi]\right\rangle_F.
```

The Riemannian gradient is the tangent vector that represents this derivative
under the manifold metric. A gradient step has the form

```math
p_{k+1}
=\operatorname{Retr}_{p_k}\!\left(-\alpha_k\operatorname{grad}f(p_k)\right).
```

Here, ``\alpha_k`` is the step size and `Retr` is a retraction: a map that
returns a tangent-space step to the manifold.

## Gauss--Newton and Levenberg--Marquardt

Write the flattened residual as ``\rho(p)=\operatorname{vec}(\mathcal R(p))``
and let ``J(p)`` be its Jacobian in tangent coordinates. A damped
Gauss--Newton step solves

```math
\left(J(p)^\mathsf TJ(p)+\mu I\right)s
=-J(p)^\mathsf T\rho(p).
```

The damping ``\mu>0`` keeps the system well defined when the local problem is
poorly conditioned. Levenberg--Marquardt (LM) uses residual and Jacobian
operations while adapting its step. GN can consume a gradient and normal
action directly, without storing residuals or Jacobians.

## Damped Gauss--Newton details

Run damped Riemannian Gauss--Newton on a decomposition model. The
step solves

```math
(J(p)^*J(p)+\mu I)\eta=-\operatorname{grad}f(p).
```

`linear_solver=:cg` uses the analytic matrix-free normal action and tangent
conjugate gradients. `linear_solver=:dense` constructs the small intrinsic
normal matrix in an orthonormal tangent basis and solves it directly; it is a
reference/benchmark option, not the scalable path.
The dense matrix is assembled once at each outer point and reused across
rejected trials; each trial applies its own diagonal damping shift to a copy.

`GaussNewtonSolver()` defaults to `:cg`, adaptive inner residual tolerance with
minimum `1e-10`, and initial damping `1e-6`. The damping policy allows at most
eight trials per outer iteration. `solve` defaults to `maxiter=100` and
`tol=1e-8`; `grad_tol` defaults to `tol`. Only `grad_tol` establishes
convergence. A small accepted step ends the run with reason `:small_step`.
GN uses the unnormalized objective and requires `gradient_mode=:riemannian`.
It currently rejects objective normalization, iteration callbacks, and a
custom vector transport. When passing a solver object to `symcpd`, set its
inner options and damping on that object rather than repeating those keywords.

For a trial step `eta`, the Levenberg--Marquardt acceptance ratio is

```math
\operatorname{pred}(\eta)
=-\langle g,\eta\rangle
-\tfrac12\langle\eta,J^*J\eta\rangle,
\qquad
\rho=\frac{f(p)-f(R_p(\eta))}{\operatorname{pred}(\eta)}.
```

A trial is accepted only when its predicted reduction is positive and `rho`
exceeds `damping.acceptance_threshold`. Poor trials increase the damping and good trials
decrease it. The predicted reduction intentionally uses the undamped
Gauss--Newton model; damping controls the step rather than redefining the
reported model agreement.

With `inner=InnerSolveOptions(tolerance=AdaptiveResidualTolerance(...))`, the
inner relative residual tolerance is the forcing term

```math
\xi_k=\operatorname{clamp}
\left(c\|\operatorname{grad}f(p_k)\|^\theta,
\xi_{\min},\xi_{\max}\right),
```

where `maximum` is ``\xi_{\max}``, `minimum` is ``\xi_{\min}``, and `scale`
and `power` define the forcing schedule. Thus early linear systems may be
solved approximately while the requested accuracy tightens near a stationary
point. `RelativeResidualTolerance` selects a fixed relative threshold and
`AbsoluteResidualTolerance` selects an absolute residual threshold.
In finite precision, both forcing bounds are at least `eps(T)`.

This relative-residual forcing condition follows R. S. Dembo, S. C. Eisenstat,
and T. Steihaug, "Inexact Newton Methods," *SIAM Journal on Numerical
Analysis* 19(2) (1982), 400--408, doi:10.1137/0719025; the power-law schedule
above is TensorKitchen's bounded specialization for the damped GN system.

An inexact CG direction may be used before the inner residual reaches its
tolerance. `solver_info.inner` records the convergence flag, iteration count,
requested tolerance, achieved residual, and termination reason for every CG
trial. Dense solves record their residuals and status, with iteration counts
and tolerances set to `nothing`. `solve_count` includes every attempted linear
solve, including rejected damping trials. Outer histories record damping,
predicted and actual reduction, acceptance ratio, and step acceptance.

The product-of-Veronese Riemannian GN formulation follows Khouja, Khalil, and
Mourrain (2022), doi:10.1016/j.laa.2021.12.008. Their published TensorDec
implementation assembles a dense normal matrix. The `:cg` option instead uses
the operator/Krylov pattern established for tensor GN by Sorber, Van Barel, and
De Lathauwer (2013), doi:10.1137/120868323, and Singh et al. (2021),
doi:10.1137/20M1344561.

## Solver summary

| Method | Main idea |
| --- | --- |
| ALS | Update one factor or block while the others are fixed. |
| RGD | Move along the negative Riemannian gradient. |
| Fixed-step RGD | Use a user-specified step size at every iteration. |
| RCG | Combine the current gradient with the previous search direction. |
| L-BFGS | Use a short history of changes to approximate curvature. |
| LM | Solve a damped least-squares model of the residual. |
| GN-CG | Solve the damped Gauss--Newton system with conjugate gradients. |

## Available methods

| Decomposition | Solver symbols |
| --- | --- |
| CPD | `:als`, `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, `:lm` |
| BTD | `:als`, `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, `:lm`, `:btd_tsd` |
| Symmetric CPD | `:cls`, `:rgd`, `:rcg`, `:lbfgs`, `:gn_cg`, `:gn_dense` |
| General Join | `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, `:lm` |

`approx` can route compatible Segre and Tucker components to the CPD and BTD
implementations. In that case, the specialized solver list applies.

Tucker decomposition uses direct factorization methods instead:

- `tucker(...; method=:sthosvd)` uses sequentially truncated HOSVD;
- `tucker(...; method=:hooi)` uses higher-order orthogonal iteration;
- `thosvd(...)` computes classical truncated HOSVD.

See [Symmetric CPD methods](symcpd.md) for its initialization, target storage,
weight refitting, and variable-projection options.

## Stopping and results

Common controls are:

- `maxiter`: maximum number of iterations;
- `tol`: stopping tolerance;
- `stepsize`: initial or fixed step size when the method uses it;
- `verbose`: whether to print progress.

Inspect an optimization result with:

```julia
rel_error(result)
iterations(result)
converged(result)
solver(result)
solver_info(result)
```

`rel_error(A, result)` recomputes the relative reconstruction error from an
explicit target.

## Solver API

The source docstrings below contain the current constructor arguments and
defaults.

```@docs
ALSSolver
RGDSolver
RGDFixedSolver
RCGSolver
LBFGSSolver
LMSolver
GaussNewtonSolver
```

## Nonnegative CPD

`nncpd` constrains weights and factors to be nonnegative. Its source docstring
lists the available geometries, solvers, and defaults.
