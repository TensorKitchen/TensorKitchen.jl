# Preparing Data

TensorKitchen can store observations in one number type and compute in another.
For example, an `Int16` tensor can remain in integer storage while the
decomposition uses `Float32` arithmetic.

```julia
counts = rand(Int16(0):Int16(100), 200, 150, 80)
```

The two main options are:

- `compute_type`: floating-point type used by the decomposition;
- `materialize`: whether to create a full converted copy of the input.

## CPD and nonnegative CPD

With `materialize=false`, CPD reads the original observations in bounded pieces
and converts them as needed:

```julia
using TensorKitchen

result = cpd(
    counts,
    10;
    compute_type = Float32,
    materialize = false,
    solver = :als,
    init = :random,
    verbose = false,
)
```

This is an exact CP-ALS computation over all observations. It is not a random
sample or sketch. The gradient-based solvers `:rgd`, `:rgd_fixed`, `:rcg`, and
`:lbfgs` also support this input path.

For lazy input, `init=:auto` uses a random initializer. Explicit
`RandomInit()`, `PointInit(...)`, and an ALS warm start based on random
initialization are also supported. Tucker-based CP initializers and
`solver=:lm` require `materialize=true`.

Nonnegative CPD uses the same storage options:

```julia
result = nncpd(
    counts,
    10;
    compute_type = Float32,
    materialize = false,
    solver = :als,
    verbose = false,
)
```

Before fitting, `nncpd` checks for nonfinite and negative observations without
creating a full converted copy.

## Tucker decomposition

Exact ST-HOSVD and HOOI require materialized compute storage. Randomized
ST-HOSVD can use lazy converted input:

```julia
result = tucker(
    counts,
    (20, 15, 10);
    compute_type = Float32,
    materialize = false,
    method = :sthosvd,
    svd_backend = :randomized,
    verbose = false,
)
```

Use `materialize=true` for exact ST-HOSVD or HOOI.

## Block term decomposition

Lazy BTD uses projected HOOI contractions instead of a full residual tensor:

```julia
result = btd(
    counts,
    2,
    (5, 5, 5);
    compute_type = Float32,
    materialize = false,
    solver = :als,
    verbose = false,
)
```

For lazy input, `init=:auto` selects
[`BTDProjectedMultistartInit`](@ref). Its contractions use all observations.
Lazy BTD requires `block_method=:hooi`; HOSVD initialization and
`block_method=:sthosvd` require `materialize=true`.

## Meaning of `materialize=false`

When preprocessing returns a [`ComputeArray`](@ref), `materialize=false` means
that TensorKitchen does not allocate an input-sized copy in `compute_type`.
Compact factors, cores, and temporary workspaces are still allocated.

Unsupported combinations raise `ArgumentError`. TensorKitchen does not
silently materialize the input or replace an explicitly requested method.

Calling `reconstruct(result)` is separate from input preparation: it creates a
full tensor with the original dimensions.

## Inspect prepared data

Most calls can pass preprocessing options directly. Use the lower-level API to
inspect or reuse a prepared tensor:

```julia
A = prepare_tensor(counts; compute_type = Float32, materialize = false)
stats = observation_stats(A)

storage_type(A)  # Int16
compute_type(A)  # Float32
stats.norm2
stats.has_nonfinite
stats.has_negative
```

```@docs
ComputeArray
prepare_tensor
materialize_tensor
observation_norm2
observation_stats
```
