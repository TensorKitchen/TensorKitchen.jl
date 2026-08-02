# Join Decomposition

Join decomposition approximates a target as a sum of components chosen from
one or more manifolds. Use it when CPD, Tucker, or BTD does not describe the
component structure you need.

Its basic mathematical form is

```math
\hat{x} = x_1 + x_2 + \cdots + x_R,
\qquad x_r \in \mathcal M_r,
```

where each manifold ``\mathcal M_r`` specifies the allowed structure of one
component.

If your model is an ordinary CPD or BTD, the specialized [`cpd`](@ref) and
[`btd`](@ref) functions are usually simpler.

## Inputs

- one manifold, or a collection of manifolds, describing the allowed components;
- `target`: the vector or tensor to approximate.

Each fitted component must lie on its corresponding manifold, and all
components must embed into the same target shape.

## Example

The following example approximates a vector with two components on a circle:

```julia
using TensorKitchen
using Manifolds

target = [1.2, 0.4]
circle = Sphere(1)
result = approx((circle, circle), target; verbose = false)
```

Here, `(circle, circle)` requests two components, each constrained to the unit
circle. `approx` adjusts those components so that their sum approximates
`target`.

## Output

For a general join, `result` is an `ApproxResult`.

```julia
fitted_components = components(result)
target_approx = reconstruct(result)
error = rel_error(target, result)
```

`target_approx` has the same dimensions as `target`. A smaller relative error
means that the fitted components reproduce the target more closely.

TensorKitchen may return a specialized `CPDResult` or `BTDResult` when all
components match one of those decomposition families.

See [Advanced Join routing](advanced/join.md) for supported input forms and
automatic routing behavior.

## API reference

```@docs
ApproxResult
components(::ApproxResult)
reconstruct(::ApproxResult)
```
