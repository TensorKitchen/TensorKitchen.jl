# Advanced Join Routing

A Join model represents

```math
\hat{x}=x_1+\cdots+x_R,
\qquad x_r\in\mathcal M_r,
```

with each component constrained to a selected manifold.

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
