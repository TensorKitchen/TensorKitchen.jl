# Block Term Decomposition

Block term decomposition (BTD) represents a tensor as a sum of Tucker blocks.
It can be useful when one Tucker decomposition is not flexible enough to
describe the data.

With ``B`` blocks, a BTD approximation is

```math
\hat{\mathcal A}
= \sum_{b=1}^{B}
  \mathcal G_b
  \times_1 U_b^{(1)}
  \times_2 U_b^{(2)}
  \cdots
  \times_d U_b^{(d)}.
```

Each summand is a Tucker decomposition with its own core and factor matrices.

## Inputs

- `A`: the numerical tensor to approximate.
- `blocks`: the number of Tucker blocks.
- `ranks`: the multilinear rank used for each block.

TensorKitchen currently uses the same `ranks` for every block.

## Example

```julia
using TensorKitchen

A = randn(20, 15, 10)
result = btd(A, 3, (5, 4, 3); verbose = false)
```

## Output

`result` is a `BTDResult`.

```julia
terms = blocks(result)
first_core = core(terms[1])
first_factors = factors(terms[1])

A_approx = reconstruct(result)
error = rel_error(A, result)
```

`terms` contains the fitted Tucker blocks. `A_approx` has the same dimensions
as `A`, and a smaller relative error means a closer reconstruction.

## Choosing blocks and ranks

Start with a small number of blocks and small ranks. Increase them only when
the reconstruction error is too large for your application. More blocks and
larger ranks can improve the fit, but they require more memory and computation.

For large tensors, fitting the compact BTD factors avoids storing a full
reconstruction during most of the calculation. Calling `reconstruct(result)`
still creates a full-size tensor.

See [Advanced BTD methods](advanced/btd.md) for initialization, solver, and
refinement controls.

## API reference

```@docs
BTDResult
blocks
reconstruct(::BTDResult)
```
