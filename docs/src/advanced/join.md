# Join Models

A Join model approximates a target by adding structured components:

```math
\Phi(p)
=\sum_{r=1}^{R}\phi_r(p_r),
\qquad
p=(p_1,\ldots,p_R),
\quad p_r\in\mathcal M_r.
```

Here, ``p_r`` stores the parameters of component ``r``. The map ``\phi_r``
turns those parameters into a tensor or vector with the same shape as the
target.

## Parameter space and output space

The component parameters belong to manifolds
``\mathcal M_1,\ldots,\mathcal M_R``. Together they form the product manifold

```math
\mathcal M
=\mathcal M_1\times\cdots\times\mathcal M_R.
```

This product is the **parameter space**: it is where the solver moves. The
reconstructed tensor belongs to the **output space**

```math
\mathcal T
=\mathbb R^{n_1\times\cdots\times n_d}.
```

The image map ``\phi_r:\mathcal M_r\to\mathcal T`` connects the two spaces.
For example, a Segre component produces a rank-one tensor,

```math
\phi_r\!\left(\lambda_r,u_r^{(1)},\ldots,u_r^{(d)}\right)
=\lambda_r u_r^{(1)}\otimes\cdots\otimes u_r^{(d)},
```

while a Tucker component produces

```math
\phi_r\!\left(\mathcal G_r,U_r^{(1)},\ldots,U_r^{(d)}\right)
=\mathcal G_r\times_1U_r^{(1)}\cdots\times_dU_r^{(d)}.
```

Every component map must produce the same output shape so that the components
can be added.

## Least-squares problem

For a target ``A\in\mathcal T``, TensorKitchen minimizes

```math
f(p)
=\frac12\left\|\Phi(p)-A\right\|_F^2
=\frac12\left\|\sum_{r=1}^{R}\phi_r(p_r)-A\right\|_F^2.
```

The residual is

```math
\mathcal R(p)=\Phi(p)-A.
```

A tangent vector ``\xi=(\xi_1,\ldots,\xi_R)`` describes a small change to all
components. Its first-order effect on the output is

```math
D\Phi(p)[\xi]
=\sum_{r=1}^{R}D\phi_r(p_r)[\xi_r].
```

The derivative of the objective is therefore

```math
Df(p)[\xi]
=\left\langle\mathcal R(p),D\Phi(p)[\xi]\right\rangle_F.
```

TensorKitchen uses these component derivatives to form gradients and Jacobian
actions without changing the mathematical model.

Different parameter points can represent the same output because components
may be permuted, rescaled, or cancel one another. A small reconstruction error
therefore does not imply unique component parameters.

## Input forms

The same two-component problem can be written in several ways:

```julia
using TensorKitchen, Manifolds

target = [1.2, 0.4]
circle = Sphere(1)

# A tuple or vector of component manifolds
r1 = approx((circle, circle), target; verbose = false)
r2 = approx([circle, circle], target; verbose = false)

# A product manifold
product = ProductManifold(circle, circle)
r3 = approx(product, target; verbose = false)

# Repeat one component manifold twice
r4 = approx(circle, 2, target; verbose = false)

# Use an existing model
model = JoinModel((circle, circle), target)
r5 = approx(model; verbose = false)
```

Omit the component count for a one-component problem:

```julia
result = approx(Sphere(2), [1.2, 0.4, -0.3]; verbose = false)
```

## Automatic routing

With `dispatch=:auto`, `approx` chooses a specialized implementation when all
components have a recognized structure:

| Components | Route | Result type |
| --- | --- | --- |
| compatible Segre manifolds | CPD | `CPDResult` |
| compatible Tucker manifolds | BTD | `BTDResult` |
| mixed or other manifolds | general Join | `ApproxResult` |

Use `dispatch=:generic` to keep the general Join route. Use `dispatch=:cpd` or
`:btd` to request a compatible specialized route explicitly. An incompatible
forced route raises `ArgumentError`.

```julia
A = reshape(collect(1.0:24.0), 4, 3, 2)
segres = (Manifolds.Segre(size(A)), Manifolds.Segre(size(A)))

cp_result = approx(
    segres,
    A;
    solver = :als,
    init = :tucker,
    maxiter = 3,
    verbose = false,
)

generic_result = approx(
    segres,
    A;
    dispatch = :generic,
    solver = :rgd_fixed,
    init = :deterministic,
    stepsize = 1e-3,
    maxiter = 3,
    verbose = false,
)
```

## Results

```julia
fitted = components(generic_result)
target_approx = reconstruct(generic_result)
error = rel_error(A, generic_result)
```

The source docstring below lists the current signatures, routing choices,
initializers, solvers, and defaults.

```@docs
approx
```
