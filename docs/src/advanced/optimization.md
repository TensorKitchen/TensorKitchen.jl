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
poorly conditioned. Levenberg--Marquardt (LM) uses the same residual and
Jacobian information while adapting its step. These methods require a model
that supplies residual and Jacobian operations.

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
```

## Nonnegative CPD

`nncpd` constrains weights and factors to be nonnegative. Its source docstring
lists the available geometries, solvers, and defaults.
