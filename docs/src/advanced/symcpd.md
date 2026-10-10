# Symmetric CPD methods

This page explains the main choices in [`symcpd`](@ref). See
[Symmetric CP decomposition](../symcpd.md) for the model definition and its
gradient and Gauss--Newton equations.

## The model

A symmetric CP decomposition writes a symmetric order-``D`` tensor as

```math
\widehat A=\sum_{r=1}^{R}\lambda_r x_r^{\otimes D},
\qquad \|x_r\|_2=1.
```

Each ``x_r`` is one direction that is repeated in all ``D`` modes.
``\lambda_r`` is its signed weight. The rank ``R`` is the number of terms.

## Default call

```julia
result = symcpd(A, R)
```

The array API uses these defaults:

| Choice | Default | Meaning |
|:--|:--|:--|
| solver | `:gn_cg` | matrix-free Gauss--Newton |
| initialization | `:auto` | SS-HOPM for ``R=1``; random for ``R>1`` |
| target storage | `:dense` | keep and contract the dense tensor |
| maximum iterations | `500` | outer solver limit |
| tolerance | `1e-6` | outer stopping tolerance |

An explicit starting point `p0` overrides `init`.

## Choosing a solver

| Solver | Use |
|:--|:--|
| `:gn_cg` | Default for general problems. It applies the normal operator without storing its matrix. |
| `:gn_dense` | Check GN-CG on a small problem. It forms and solves the dense normal matrix. |
| `:rgd` | Simple gradient-descent reference. |
| `:rcg` | First-order method that reuses the previous search direction. |
| `:lbfgs` | First-order method that stores a short history of gradient information. |
| `:cls` | Normalized conditional least squares without Riemannian refinement. |

### Why GN-CG is matrix-free

Let ``J`` describe how a small change in the parameters changes the fitted
tensor. A damped Gauss--Newton step solves

```math
(J^*J+\mu I)\eta=-\operatorname{grad}f.
```

`solver=:gn_dense` builds the matrix ``J^*J``. Its storage grows quadratically
with the number of parameters.

`solver=:gn_cg` uses conjugate gradients. It only needs products of the form

```math
X\longmapsto J^*JX.
```

TensorKitchen computes this product directly from the factors. It does not
store ``J``, ``J^*J``, or a full residual tensor.

The damping ``\mu`` is changed after each trial step. A successful step can
reduce it; a rejected step increases it. Inner CG stops when its relative
residual reaches `cg_tol`, or an adaptive tolerance between `cg_min_tol` and
`cg_tol`.

## Choosing an initialization

### Automatic initialization

`init=:auto` uses

```math
\operatorname{init}(R)=
\begin{cases}
\text{SS-HOPM}, & R=1,\\
\text{random point}, & R>1.
\end{cases}
```

The rank-one case has a specialized tensor eigenvector iteration. For higher
rank, the random start has a smaller setup cost.

### Random initialization

```julia
result = symcpd(A, R; init=:random)
```

Symmetric CPD is a nonconvex problem, so different starts can reach different
solutions. For an experiment, report the seed, number of starts, and success
rate.

### SS-HOPM initialization

```julia
result = symcpd(A, R; init=:sshopm, solver=:gn_cg)
```

SS-HOPM searches for tensor eigenvectors and uses them as initial directions.
For fixed directions, it computes the weights from a small linear system. This
costs more than one random start but gives a structure-based initial point.

For a rank-one problem, the specialized interface is shorter:

```julia
target = DenseSymmetricTarget(A)
result = best_symmetric_rank1(target; starts=8)
```

### CLS initialization

```julia
result = symcpd(A, R; init=:cls, solver=:gn_cg)
```

This runs a short normalized conditional least-squares fit and passes its best
point to GN-CG. Configure it with [`NormalizedCLSInit`](@ref):

```julia
init = NormalizedCLSInit(
    sweeps=10,
    base_init=:random,
    damping=1e-10,
    patience=3,
)

result = symcpd(A, R; init=init, solver=:gn_cg)
```

### Explicit starting point

Use the same `p0` when comparing solvers:

```julia
result_gn = symcpd(target, R; solver=:gn_cg, p0=deepcopy(p0))
result_rcg = symcpd(target, R; solver=:rcg, p0=deepcopy(p0))
```

This removes initialization as a source of difference.

## Exact weight refit

If the unit directions ``x_1,\ldots,x_R`` are fixed, only the weights remain
unknown. They solve the ``R\times R`` system

```math
K\lambda=c,
\qquad
K_{rs}=(x_r^\top x_s)^D,
\qquad
c_r=\langle A,x_r^{\otimes D}\rangle.
```

```julia
target = DenseSymmetricTarget(A)
lambda = refit_symcpd_weights(target, X)
```

The solve uses a pseudoinverse when the directions are linearly dependent or
nearly dependent. `damping` adds regularization when needed.

## Normalized conditional least squares

For the direction matrix ``X=[x_1,\ldots,x_R]``, one CLS sweep forms

```math
M_{:,r}=A(I,x_r,\ldots,x_r),
\qquad
H=(X^\top X)^{\circ(D-1)},
\qquad
B=MH^\dagger.
```

The columns of ``B`` are normalized, and the weights are refitted from
``K\lambda=c``. These operations use tensor contractions; they do not build a
tensor unfolding or a Khatri--Rao matrix.

After normalization, a CLS sweep is not guaranteed to lower the original
symmetric objective. TensorKitchen therefore returns the best iterate it saw.

Use CLS by itself with

```julia
result = symcpd(A, R; solver=:cls)
```

or construct the solver:

```julia
method = SymmetricCLS(damping=1e-10, patience=3)
result = symcpd(A, R; solver=method)
```

## Variable projection

The weights are linear variables, so they can be solved exactly for every
direction matrix ``X``:

```math
\lambda^\star(X)=K(X)^\dagger c(X).
```

Variable projection optimizes only the directions:

```math
\widetilde f(X)=f(\lambda^\star(X),X).
```

```julia
result = symcpd(
    A,
    R;
    variable_projection=true,
    solver=:lbfgs, # or :rcg
)
```

This removes ``R`` weight variables, but each objective and gradient
evaluation must solve an ``R\times R`` system. Reduced Gauss--Newton is not
implemented; use `:lbfgs` or `:rcg` with this option.

## Choosing target storage

| Target | Storage | Use |
|:--|:--|:--|
| `DenseSymmetricTarget` | ``N^D`` tensor entries | The dense tensor fits in memory. This is the array-input default. |
| `CompressedSymmetricTarget` | ``\binom{N+D-1}{D}`` symmetric coordinates | Dense storage is too large and symmetric coordinates are available. |
| `FunctionalSymmetricTarget` | functions and ``\|A\|_F^2`` | The application can evaluate contractions without storing the tensor. |

Compressed storage saves memory, but it is not always faster. For a small
tensor, dense contraction can have less overhead.

A functional target must provide three consistent quantities:

```julia
target = FunctionalSymmetricTarget(
    N,
    D,
    normA2;
    evaluate = x -> target_polynomial(x),
    contract = x -> target_contraction(x),
)
```

- `normA2` is ``\|A\|_F^2``;
- `evaluate(x)` is ``\langle A,x^{\otimes D}\rangle``;
- `contract(x)` is ``A(x,\ldots,x,\mathord\cdot)``.

## Checking a result

For an exact synthetic tensor, use the relative reconstruction error

```math
\varepsilon_{\mathrm{rel}}
=\frac{\|A-\widehat A\|_F}{\|A\|_F}.
```

For noisy data, compare the error with the noise level or with a held-out
metric. `converged(result)` only reports that the stopping rule was met; it
does not prove that the factors are the true factors.

Useful output includes

```julia
rel_error(result)
grad_norm(result)
iterations(result)
converged(result)
solver_info(result)
```

For GN-CG, `solver_info(result)` also records CG iterations, CG residuals,
damping values, accepted-step ratios, and rejected steps.

When two directions become nearly parallel,
``|x_r^\top x_s|\approx1``, the decomposition is poorly conditioned. In that
case, small changes in the data may cause large changes in the individual
components even when the reconstructed tensor changes little.

## References

The implementation uses the symmetric tensor geometry of Khouja, Khalil, and
Mourrain (2022), SS-HOPM of Kolda and Mayo (2011), conditional least squares
of Favier, Kibangou, and Bouilloc (2012), and variable projection of Golub and
Pereyra (1973). Full citations are listed in [References](../references.md).
