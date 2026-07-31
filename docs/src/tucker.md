# Tucker Decomposition 

Approximating `A` by a Tucker decomposition
```math
\hat A = C \times_1 U \times_2 V \times_3 W
```
with multilinear rank `mlrank` can be computed as follows.

```julia-repl
julia> mlrank = (5, 4, 3)
julia> tucker_res = tucker(A, mlrank)
TuckerResult{Float64, 3}
  Original size:    (20, 15, 10)
  Core size:        (5, 4, 3)
  Multilinear rank: (5, 4, 3)
  Compression:      12.0x
```

The core $C$ and the factor matrices $(U, V, W)$ of the decomposition can be accessed as follows.

```julia
core(tucker_res)
factors(tucker_res)
```

## Large tensors without explicit unfoldings

For a fixed multilinear rank, ST-HOSVD can use an implicit randomized backend:

```julia
using Random

tucker_res = tucker(
    A,
    (100, 40, 60);
    method = :sthosvd,
    svd_backend = :randomized,
    processing_order = [2, 3, 1],
    oversampling = 16,
    power_iterations = 1,
    block_columns = 65_536,
    rng = MersenneTwister(0),
)
```
Instead of constructing ``A_{(k)}``, the backend evaluates randomized projections
and subspace iterations as tensor contractions. The Gaussian sketch is generated in 
bounded blocks, and the input is not copied before its first mode projection. This
substantially reduces memory when the tensor is large and the requested ranks are
small relative to the mode dimensions.
The exact backend remains the default and is generally preferable for small tensors,
near-full ranks. Randomized results do not support `error_bound`, because discarded
singular values are not computed.

## Tucker Docs

```@docs
tucker
TuckerResult
core(::TuckerResult)
factors(::TuckerResult)
multilinear_rank(::TuckerResult)
factor_dims(::TuckerResult)
reconstruct(::TuckerResult)
```
