# Symmetric CP decomposition

`symcpd(A, r)` approximates an order-``D`` symmetric tensor with equal mode
size ``N`` as

```math
\widehat A
=\sum_{k=1}^{r}\lambda_k x_k^{\otimes D},
\qquad \|x_k\|_2=1.
```

Each component is stored intrinsically as ``p_k=(\lambda_k,x_k)`` on a
Veronese manifold. The factor ``x_k`` therefore appears in every tensor mode;
this is not an ordinary CPD with unrelated factor matrices.

## Component and join structure

TensorKitchen represents one symmetric rank-one term by
`SymmetricRankOne(N, D)`. It plays the same architectural role as a Segre
rank-one component in ordinary CPD:

```math
\begin{array}{c|c|c}
& \text{ordinary CPD} & \text{symmetric CPD} \\
\hline
\text{rank-one map}
& a_1\otimes\cdots\otimes a_D
& \lambda x^{\otimes D} \\
\text{geometry} & \text{Segre} & \text{Veronese} \\
\text{rank-}R\text{ model}
& \operatorname{Join}_R(\text{Segre})
& \operatorname{Join}_R(\text{Veronese})
\end{array}
```

The explicit construction is

```julia
component = SymmetricRankOne(N, D)
target = DenseSymmetricTarget(A)
model = JoinModel(component, R, target)
```

`symcpd(A, R)` is the convenient public pipeline for the same construction.
`SymCPDModel(target, R)` remains as a compatibility constructor and also
returns this specialized `JoinModel`; it is not a separate decomposition
model hierarchy.

## Basic use

```julia
using TensorKitchen

result = symcpd(A, 3; solver=:lbfgs, maxiter=300)
lambda = weights(result)       # length 3
X = factors(result)            # N x 3; column k is x_k
A_fit = reconstruct(result)    # allocates the full N^D tensor on request
```

The decomposition itself remains factorized. `reconstruct` is an explicit
output operation and can be expensive when ``N^D`` is large.

## What “matrix-free” means here

TensorKitchen expands the least-squares objective

```math
\begin{aligned}
f(p)
&=\frac12\left\|A-\sum_r\lambda_rx_r^{\otimes D}\right\|_F^2\\
&=\frac12\|A\|_F^2
-\sum_r\lambda_r\langle A,x_r^{\otimes D}\rangle
+\frac12\sum_{r,s}\lambda_r\lambda_s(x_r^\top x_s)^D.
\end{aligned}
```

The model side uses only the Veronese kernel
``(x_r^\top x_s)^D``. It does not materialize the predicted tensor, a residual
tensor, or compressed Veronese coordinates. The target side is separate:
``\langle A,x^{\otimes D}\rangle`` and
``A(x,\ldots,x,\mathord\cdot)`` are evaluated by the selected storage backend.

This distinction matters: compressed symmetric storage reduces the target
representation, while matrix-free optimization avoids model-side ambient
objects. Either can be used without requiring the other.

## Target storage

```julia
# Keep the dense tensor and contract it directly.
dense_result = symcpd(A, 3; target_backend=:dense)

# Compress the symmetric target once. Model-model terms remain kernelized.
compressed_result = symcpd(A, 3; target_backend=:compressed)
```

`target_backend=:dense` is the default. The two backends represent the same
Frobenius inner product and should give the same objective and gradient up to
roundoff.

For a target that is available only through tensor-vector contractions, use a
functional backend:

```julia
target = FunctionalSymmetricTarget(
    N,
    D,
    normA2;
    evaluate = x -> target_polynomial(x),
    contract = x -> one_mode_contraction(x),
)
result = symcpd(target, 3; solver=:gn_cg)
```

Every target backend implements the same three operations:

- `target_norm2(target)` returns ``\|A\|_F^2``;
- `evaluate(target, x)` returns ``\langle A,x^{\otimes D}\rangle``;
- `contract(target, x)` returns ``A(x,\ldots,x,\mathord\cdot)``.

The functional backend stores the supplied callables and scalar norm, not the
tensor itself. The caller is responsible for making the two operators and
`normA2` describe the same symmetric tensor.

## Intrinsic gradient

Let

```math
a_r=\langle A,x_r^{\otimes D}\rangle,
\qquad
b_r=A(x_r,\ldots,x_r,\mathord\cdot),
\qquad
c_{rs}=x_r^\top x_s.
```

The coordinate derivatives are

```math
\partial_{\lambda_r}f
=-a_r+\sum_s\lambda_sc_{rs}^D,
```

```math
\nabla_{x_r}f
=-D\lambda_rb_r
+D\lambda_r\sum_s\lambda_sc_{rs}^{D-1}x_s.
```

The factor derivative is projected to ``x_r^\perp``. TensorKitchen then
applies the inverse induced metric

```math
g_{(\lambda,x)}((\nu,u),(\xi,v))
=\nu\xi+D\lambda^2u^\top v
```

to obtain the Riemannian gradient used by the solver.

## Analytic Gauss--Newton operator

For ``X_r=(\nu_r,u_r)`` and ``Z_s=(\xi_s,v_s)``, define
``c=x_r^\top x_s``. The cross-component normal bilinear form is

```math
\begin{aligned}
B_{rs}(X_r,Z_s)={}&
\nu_r\xi_s c^D
+\nu_r\lambda_sD c^{D-1}x_r^\top v_s\\
&+\lambda_r\xi_sD c^{D-1}u_r^\top x_s
+\lambda_r\lambda_sD c^{D-1}u_r^\top v_s\\
&+\lambda_r\lambda_sD(D-1)c^{D-2}
(u_r^\top x_s)(x_r^\top v_s).
\end{aligned}
```

`normal_operator!` applies the resulting ``J^*J`` action directly. It does not
form ``J``, ``J^*J``, the full tensor, or the compressed feature vector.

Two Gauss--Newton routes are available:

- `solver=:gn_cg` applies `normal_operator!` inside tangent conjugate gradients.
  This is the scalable path.
- `solver=:gn_dense` constructs the intrinsic ``rN\times rN`` normal matrix
  from operator columns and solves it directly. Use it for small validation or
  timing comparisons.

```julia
result = symcpd(
    A,
    3;
    solver=:gn_cg,
    damping=1e-6,
    cg_tol=1e-8,
    maxiter=100,
)
```

For a small problem, `dense_normal_matrix(model, p; reference=true)` explicitly
forms a compressed-coordinate Jacobian and returns ``J^\top J``. This is a
test oracle: the analytic operator-derived matrix should agree with it to
machine precision. It is not used by `:gn_cg`.

The damped Gauss--Newton step is globalized by a Riemannian
Levenberg--Marquardt acceptance ratio. For a trial tangent ``\eta``, define

```math
\operatorname{pred}(\eta)
=-\langle \operatorname{grad}f,\eta\rangle
-\frac12\langle\eta,J^*J\eta\rangle,
```

```math
\operatorname{ared}(\eta)=f(p)-f(R_p(\eta)),
\qquad
\rho=\frac{\operatorname{ared}(\eta)}{\operatorname{pred}(\eta)}.
```

The step is accepted when `pred` is positive and ``\rho`` is at least
`acceptance_ratio`. Ratios below `poor_step_ratio` increase the damping;
ratios above `good_step_ratio` decrease it. `solver_info` records
`predicted_reduction_history`, `actual_reduction_history`, `rho_history`,
`damping_history`, `step_accepted_history`, and the accepted/rejected counts.

GN-CG is an inexact Gauss--Newton method: an inner CG solve may provide a
useful direction before reaching `cg_tol`. The returned `solver_info` records
`cg_converged_history`, `cg_iterations_history`, and `cg_failed_count` so this
is visible. `termination_reason=:small_step` denotes stagnation and does not
set `converged=true`; outer convergence requires the gradient tolerance.

## Sign-equivalent representatives

For odd ``D``, ``(\lambda,x)`` and ``(-\lambda,-x)`` represent the same
tensor. For even ``D``, changing ``x`` to ``-x`` leaves the component unchanged
without changing ``\lambda``. Factor comparisons should account for these
equivalent representatives.

## Method references

- The product-of-Veronese Riemannian Newton and Gauss--Newton formulation:
  Khouja, Khalil, and Mourrain (2022),
  [doi:10.1016/j.laa.2021.12.008](https://doi.org/10.1016/j.laa.2021.12.008).
- Matrix-free nonlinear least squares for CPD/BTD: Sorber, Van Barel, and De
  Lathauwer (2013),
  [doi:10.1137/120868323](https://doi.org/10.1137/120868323).
- Implicit normal products in GN-CG for CPD: Singh, Ma, Yang, and Solomonik
  (2021), [doi:10.1137/20M1344561](https://doi.org/10.1137/20M1344561).
- The intrinsic warped Segre--Veronese metric: Jacobsson, Swijsen, Van der
  Veken, and Vannieuwenhoven (2026),
  [doi:10.1137/25M1790099](https://doi.org/10.1137/25M1790099).

```@docs
symcpd
AbstractJoinComponent
SymmetricRankOne
JoinModel
SymCPDModel
SymmetricCPDBackend
DenseSymmetricTarget
CompressedSymmetricTarget
FunctionalSymmetricTarget
target_norm2
evaluate
contract
component_inner
data_inner
pushforward!
pullback
normal_operator!
normal_operator
dense_normal_matrix
SymCPDResult
SymCPDComponent
compressed_coordinates
compress_symmetric_tensor
expand_symmetric_tensor
symmetric_multiindices
multinomial_multiplicity
```
