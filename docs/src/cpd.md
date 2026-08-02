# CP Decomposition

CP decomposition represents a tensor as a sum of rank-one components. It is a
good starting point when the same number of components should describe every
mode of the tensor.

For a ``d``-way tensor, a rank-``R`` CP approximation is

```math
\hat{\mathcal A}
= \sum_{r=1}^{R} \lambda_r
  u_r^{(1)} \otimes u_r^{(2)} \otimes \cdots \otimes u_r^{(d)}.
```

Each outer product is one rank-one component. The scalar ``\lambda_r`` is its
weight, and ``u_r^{(k)}`` is its factor vector for mode ``k``.

## Inputs

- `A`: the numerical tensor to approximate.
- `rank`: the number of rank-one components to keep.

Larger ranks can improve reconstruction accuracy, but they also use more
storage and take longer to fit.

## Example

```julia
using TensorKitchen

A = randn(20, 15, 10)
result = cpd(A, 5; verbose = false)
```

## Output

`result` is a `CPDResult`. It stores one weight per component and one factor
matrix per tensor mode.

```julia
component_weights = weights(result)
factor_matrices = factors(result)
A_approx = reconstruct(result)
error = rel_error(A, result)
```

`A_approx` has the same dimensions as `A`. Relative error is zero for an exact
reconstruction, and smaller values indicate a closer approximation.

## Nonnegative data

Use `nncpd` when both the input data and the fitted components should be
nonnegative:

```julia
A_nonnegative = abs.(A)
result = nncpd(A_nonnegative, 5; verbose = false)
```

See [Advanced optimization methods](advanced/optimization.md) for solver,
initialization, and nonnegative-geometry details.

## API reference

```@docs
CPDResult
weights(::CPDResult)
factors(::CPDResult)
reconstruct(::CPDResult)
```

The complete method and option reference is on the
[Advanced CPD methods](advanced/cpd.md) page.
