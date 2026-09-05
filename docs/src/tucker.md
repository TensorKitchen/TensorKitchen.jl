# Tucker Decomposition

Tucker decomposition compresses a tensor into a smaller core tensor together
with one factor matrix for each mode. It is useful when different modes need
different compression levels.

For a ``d``-way tensor, the Tucker approximation is

```math
\hat{\mathcal A}
= \mathcal G
  \times_1 U^{(1)}
  \times_2 U^{(2)}
  \cdots
  \times_d U^{(d)}.
```

The core ``\mathcal G`` has size ``(r_1,\ldots,r_d)``, and each factor matrix
``U^{(k)}`` maps the compressed mode of size ``r_k`` back to the original mode.

## Inputs

- `A`: the numerical tensor to compress.
- `ranks`: one retained rank for each mode of `A`.

For a tensor of size `(20, 15, 10)`, ranks `(5, 4, 3)` produce a core of size
`(5, 4, 3)`.

## Example

```julia
using TensorKitchen

A = randn(20, 15, 10)
result = tucker(A, (5, 4, 3))
```

## Output

`result` is a `TuckerResult`.

```julia
compressed = core(result)
factor_matrices = factors(result)
A_approx = reconstruct(result)
error = rel_error(A, result)
```

The core is the compressed representation. `A_approx` has the same dimensions
as `A`, and a smaller relative error means a closer reconstruction.

## Large tensors

For a large tensor, the randomized backend can reduce memory use and runtime
when the requested ranks are much smaller than the input dimensions:

```julia
result = tucker(
    A,
    (5, 4, 3);
    method = :sthosvd,
    svd_backend = :randomized,
)
```

The randomized result is approximate and may vary slightly between runs. Use
`rel_error(A, result)` to evaluate it. `error_bound` is available only when the
decomposition records the full singular-value information.

See [Advanced Tucker methods](advanced/tucker.md) for ST-HOSVD, randomized
sketching, and HOOI details.

## API reference

```@docs
TuckerResult
core(::TuckerResult)
factors(::TuckerResult)
multilinear_rank(::TuckerResult)
factor_dims(::TuckerResult)
reconstruct(::TuckerResult)
```
