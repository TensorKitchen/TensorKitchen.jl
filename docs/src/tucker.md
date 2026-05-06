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
