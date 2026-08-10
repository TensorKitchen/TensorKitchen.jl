# Join Models for Advanced Understanding

Join decomposition approximates a target as a sum of structured components
generated from one or more manifolds. Its basic mathematical form is

```math
\Phi(p)
=\phi_1(p_1)+\phi_2(p_2)+\cdots+\phi_R(p_R)
=\sum_{r=1}^{R}\phi_r(p_r),
\qquad
p=(p_1,\ldots,p_R),
\quad p_r\in\mathcal M_r.
```

where ``\mathcal M_r`` specifies the allowed parameter structure of component
``r`` and ``\phi_r`` maps that parameter to the ambient target space. The
sections below make this parameter-space and image-space distinction precise.

## Mathematical model

Let the target tensor have shape ``n_1 \times \cdots \times n_d``. Its ambient
space can be viewed either as a tensor space or as its flattened vector space,

```math
\mathcal T
= \mathbb R^{n_1\times\cdots\times n_d}
\cong \mathbb R^N,
\qquad
N=\prod_{m=1}^{d}n_m.
```

The Frobenius norm of a tensor is the Euclidean norm of its flattened vector,
so these two views give the same least-squares problem.

### Parameter space and product manifold

Component ``r`` is described by a parameter manifold ``\mathcal M_r``. The
complete parameter space is the product manifold

```math
\mathcal M
= \mathcal M_1 \times \cdots \times \mathcal M_R,
\qquad
p=(p_1,\ldots,p_R)\in\mathcal M.
```

This ``\mathcal M`` is the domain on which the optimizer moves. In
TensorKitchen, `ProductManifold(M1, ..., MR)` represents this parameter space;
it is not the set of reconstructed tensors.

At ``p``, the tangent space and product metric are

```math
T_p\mathcal M
= T_{p_1}\mathcal M_1 \oplus \cdots \oplus T_{p_R}\mathcal M_R,
```

```math
g_p(\xi,\eta)
= \sum_{r=1}^{R}g^{(r)}_{p_r}(\xi_r,\eta_r).
```

Thus a tangent direction ``\xi=(\xi_1,\ldots,\xi_R)`` updates all components
while respecting the geometry of each component manifold.

### Component image maps

A parameter point is converted into an ambient component by an image map

```math
\phi_r:\mathcal M_r\longrightarrow\mathcal T,
\qquad
x_r=\phi_r(p_r).
```

For a Segre component this map produces a rank-one tensor,

```math
\phi_r\!\left(\lambda_r,u_r^{(1)},\ldots,u_r^{(d)}\right)
= \lambda_r
  u_r^{(1)}\otimes\cdots\otimes u_r^{(d)},
```

whereas a Tucker component maps its core and factors to

```math
\phi_r\!\left(G_r,U_r^{(1)},\ldots,U_r^{(d)}\right)
=G_r\times_1U_r^{(1)}\times_2\cdots\times_dU_r^{(d)}.
```

For a manifold already embedded in the target vector space, such as the sphere
example below, ``\phi_r`` is the usual ambient embedding. Every component map
must have the same ambient output size as the flattened target.

### Join map and least-squares objective

The joint image map adds the ambient components,

```math
\Phi:\mathcal M\longrightarrow\mathcal T,
\qquad
\Phi(p)=\sum_{r=1}^{R}\phi_r(p_r).
```

Given a target ``A\in\mathcal T``, the generic Join path solves

```math
\min_{p\in\mathcal M} f(p),
\qquad
f(p)
=\frac12\left\|\Phi(p)-A\right\|_F^2
=\frac12\left\|
  \sum_{r=1}^{R}\phi_r(p_r)-A
 \right\|_F^2.
```

The residual tensor and residual vector are

```math
\mathcal R(p)=\Phi(p)-A,
\qquad
\rho(p)=\mathrm{vec}(\mathcal R(p))\in\mathbb R^N.
```

The differential of the Join map is

```math
D\Phi(p)[\xi]
=\sum_{r=1}^{R}D\phi_r(p_r)[\xi_r],
```

and therefore

```math
Df(p)[\xi]
=\left\langle \mathcal R(p),D\Phi(p)[\xi]\right\rangle_F.
```

The component Riemannian gradients are characterized by

```math
g^{(r)}_{p_r}\!\left(\mathrm{grad}_r f,\xi_r\right)
=\left\langle
  \mathcal R(p),D\phi_r(p_r)[\xi_r]
 \right\rangle_F
\quad
\text{for every }\xi_r\in T_{p_r}\mathcal M_r.
```

TensorKitchen evaluates the component embeddings and their pushforwards, sums
the pushforwards to obtain ``D\Phi(p)[\xi]``, and converts the resulting
component gradients to the corresponding tangent spaces. This same
differential supplies the Jacobian action required by the LM solver.

### Parameter space versus image geometry

The representable tensors form the image set

```math
\mathcal J
=\Phi(\mathcal M)
=\left\{
  \sum_{r=1}^{R}\phi_r(p_r)
  : p_r\in\mathcal M_r
 \right\}\subseteq\mathcal T.
```

The smooth optimization domain is ``\mathcal M``; the image ``\mathcal J``
need not be a smooth manifold everywhere. The number of parameter-space
directions that are visible to first order in the ambient space at ``p`` is
``\mathrm{rank}D\Phi(p)``, with

```math
\mathrm{rank}D\Phi(p)
\leq
\min\!\left(
  N,\sum_{r=1}^{R}\dim\mathcal M_r
\right).
```

At regular points this rank agrees with the local image dimension. At critical
or singular points, the image can require a more careful local analysis.

Different parameter points can produce the same tensor because of component
permutations, scaling or gauge symmetries, coincident components, or
cancellation. Consequently, Join optimization identifies a good reconstructed
tensor but does not by itself guarantee a unique component parameterization.

## Supported input forms

The examples below use a common target and component manifold:

```julia
using TensorKitchen, Manifolds

target = [1.2, 0.4]
circle = Sphere(1)
```

### Existing `JoinModel`

Use this form when the target and component structure have already been packed
into a model:

```julia
model = JoinModel((circle, circle), target)
model_result = approx(model; init = :deterministic, maxiter = 20, verbose = false)
```

### Manifold collection and target

A tuple or vector lists each component manifold explicitly:

```julia
tuple_result = approx(
    (circle, circle),
    target;
    init = :deterministic,
    maxiter = 20,
    verbose = false,
)

vector_result = approx(
    [circle, circle],
    target;
    init = :deterministic,
    maxiter = 20,
    verbose = false,
)
```

### Product manifold and target

If the components are already stored in a `ProductManifold`, its factors are
used directly:

```julia
product = ProductManifold(circle, circle)
product_result = approx(
    product,
    target;
    init = :deterministic,
    maxiter = 20,
    verbose = false,
)
```

### Repeated base manifold

Pass a base manifold and component count to repeat the same component type:

```julia
repeated_result = approx(
    circle,
    2,
    target;
    init = :deterministic,
    maxiter = 20,
    verbose = false,
)
```

### Single component

Omit the component count for a one-component join:

```julia
single_target = [1.2, 0.4, -0.3]
single_result = approx(
    Sphere(2),
    single_target;
    init = :target,
    maxiter = 20,
    verbose = false,
)
```

## Automatic routing

With `dispatch=:auto`:

- uniform `Manifolds.Segre` components route to `cpd`;
- uniform compatible `Manifolds.Tucker` components route to `btd`;
- mixed or other component families use the general Join path.

The specialized routes return `CPDResult` or `BTDResult`; the general path
returns `ApproxResult`. Use `dispatch=:generic`, `:cpd`, or `:btd` to request a
specific route when the component family is compatible.

### Segre components: CPD routing

```julia
A = reshape(collect(1.0:24.0), 4, 3, 2)
segres = (Manifolds.Segre(size(A)), Manifolds.Segre(size(A)))

cp_auto = approx(
    segres,
    A;
    solver = :als,
    init = :tucker,
    maxiter = 3,
    verbose = false,
)

cp_forced = approx(
    segres,
    A;
    dispatch = :cpd,
    solver = :als,
    init = :tucker,
    maxiter = 3,
    verbose = false,
)
```

Both calls return a `CPDResult`.

### Tucker components: BTD routing

```julia
tuckers = (
    Manifolds.Tucker(size(A), (2, 2, 1)),
    Manifolds.Tucker(size(A), (2, 2, 1)),
)

btd_auto = approx(
    tuckers,
    A;
    solver = :als,
    init = :sthosvd,
    maxiter = 3,
    verbose = false,
)
```

This call returns a `BTDResult`. `dispatch=:btd` can be used to request and
validate the same route explicitly.

### Forcing the generic path

Even a uniform Segre collection can be treated as a generic join when the
specialized CPD result type or CPD-specific pipeline is not wanted:

```julia
generic_result = approx(
    segres,
    A;
    dispatch = :generic,
    init = :deterministic,
    solver = :rgd_fixed,
    stepsize = 1e-3,
    maxiter = 3,
    verbose = false,
)
```

This returns an `ApproxResult`. Forcing `:cpd` or `:btd` with an incompatible
manifold family throws `ArgumentError` rather than silently changing the model.

## Inspecting the result

```julia
fitted = components(generic_result)
target_approx = reconstruct(generic_result)
error = rel_error(A, generic_result)
```

For ordinary CPD or BTD models, prefer the specialized high-level functions
unless you specifically need Join routing or mixed component families.

## API, options, and defaults

The general path does not assume that arbitrary components expose ALS factor
updates. The canonical source docstring below supplies the current signatures,
routing choices, initializers, solver list, and defaults automatically:

```@docs
approx
```

Every component must embed into the same flattened target length. A forced CPD
or BTD route additionally requires compatible component dimensions and ranks.
