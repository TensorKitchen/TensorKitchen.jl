# Choosing a Decomposition

TensorKitchen provides several decompositions with similar workflows. The main
difference is the structure used to represent the input tensor.

## Quick comparison

| Use case | Function | Main size parameter | Output |
| --- | --- | --- | --- |
| Shared rank-one components across all modes | `cpd(A, rank)` | Number of components | `CPDResult` |
| Nonnegative data and components | `nncpd(A, rank)` | Number of components | `CPDResult` |
| A compact core with a separate rank per mode | `tucker(A, ranks)` | Rank tuple | `TuckerResult` |
| A sum of several Tucker blocks | `btd(A, blocks, ranks)` | Blocks and rank tuple | `BTDResult` |
| A custom collection of component types | `approx(manifolds, target)` | Component manifolds | `ApproxResult` |

## Common workflow

All result types can be inspected and reconstructed in a similar way:

```julia
using TensorKitchen

A = randn(20, 15, 10)
result = tucker(A, (5, 4, 3))

A_approx = reconstruct(result)
error = rel_error(A, result)
```

`A_approx` has the same dimensions as `A`. Relative error is zero for an exact
reconstruction, and smaller values indicate a closer approximation.

## Choosing a rank

There is no single best rank for every dataset. A practical approach is:

1. Start with a small rank.
2. Fit the decomposition.
3. Measure the reconstruction error.
4. Increase the rank until the improvement is no longer worth the additional
   storage or runtime.

For Tucker and BTD, each entry of the rank tuple corresponds to one tensor
mode. For example, `(8, 3, 5)` retains different amounts of information along
the three modes.

## Understanding the outputs

- `CPDResult`: component weights and factor matrices.
- `TuckerResult`: a core tensor and factor matrices.
- `BTDResult`: a collection of Tucker blocks.
- `ApproxResult`: components from a custom join model.

Use `reconstruct(result)` when you need the approximation in the original
tensor shape. For large data, remember that the reconstructed array can be much
larger than the compact decomposition result.

## Next steps

- [CP decomposition](cpd.md)
- [Tucker decomposition](tucker.md)
- [Block term decomposition](btd.md)
- [Join decomposition](join.md)
- [Saving and loading results](utils.md)
- [Advanced methods](advanced/index.md)
