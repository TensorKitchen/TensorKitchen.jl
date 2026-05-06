# Block Term Decomposition 

A block term decomposition (BTD) with `r` blocks writes
```math
\hat A = \sum_{i=1}^r A_i,
```
where each block $A_i$ is represented as a Tucker decomposition. At present, only homogeneous BTDs are supported, that is, all blocks must have the same multilinear rank.

To compute a block term decomposition of A with 10 blocks, each of multilinear rank (5, 4, 3), use

```julia-repl
julia> r = 10
julia> mlrank = (5, 4, 3)
julia> btd_res = btd(A, r, mlrank)
BTDResult{Float64}
  Blocks:       10
  Rel. error:   0.2551559591470521
```

The blocks of `btd_res` can be obtained as follows:

```julia
blocks = blocks(btd_res)
```

Each block is represented as a Tucker decomposition, so we can access its core and factor matrices via:

```julia
blk = blocks[1]
core(blk)
factors(blk)
```

## BTD Docs

```@docs
btd
BTDResult
blocks
reconstruct(::BTDResult)

```
