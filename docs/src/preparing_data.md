# Preparing data without an unnecessary full copy

TensorKitchen separates the type used to store observations from the
floating-point type used for numerical work. This matters when, for example, a
large tensor is stored as `Int16` but the decomposition should run in
`Float32`.

Start with the cheapest exact option. Avoid a full converted copy first; choose
a randomized algorithm only when the approximation reduces enough work to be
useful.

```text
Is converting the whole input the memory problem?
│
├─ no  → use the usual decomposition API
│
└─ yes → keep native storage and choose compute_type
         │
         ├─ CP or nonnegative CP → exact ALS or manifold optimization
         │                          with implicit contractions
         │
         └─ Tucker → randomized ST-HOSVD when approximation is acceptable
```

## Exact CP decomposition from integer storage

The public API prepares real-valued input automatically. With
`materialize=false`, TensorKitchen keeps the original storage and converts
bounded pieces as exact kernels read them:

```julia
using TensorKitchen

counts = rand(Int16(0):Int16(100), 200, 150, 80)

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

This CP-ALS path uses all observations. It does not construct a dense unfolding
or Khatri--Rao matrix and is not a randomized approximation.

The same lazy input can be used by the gradient-based manifold solvers. The
default `init=:auto` selects a random initializer for a lazy converted input so
the basic public call remains observation-preserving:

```julia
result = cpd(
    counts,
    10;
    compute_type = Float32,
    materialize = false,
    solver = :lbfgs, # also :rgd, :rgd_fixed, or :rcg
    verbose = false,
)
```

An explicit `RandomInit()`, `PointInit(...)`, or
`ALSWarmStartInit(...; base_init=RandomInit())` is also safe. Explicit
structured initializers such as `TuckerInit()` and `TuckerDiagInit()` are
rejected unless the input is materialized.

For rank two and above, objective and gradient evaluations use exact implicit
MTTKRP contractions. Rank-one models use exact tensor-vector contractions.
Their norm, component trace, and final-error diagnostics also preserve lazy
storage. The target norm is computed once and reused across initialization,
optimization, and diagnostics. These methods still inspect all observations;
they are not sketches.

For nonnegative data, use the same storage options with `nncpd`:

```julia
result = nncpd(
    counts,
    10;
    compute_type = Float32,
    solver = :als,
    verbose = false,
)
```

`nncpd` checks the input for nonfinite and negative observations in a streaming
pass before fitting. Its lazy path supports the same `:als`, `:rgd`,
`:rgd_fixed`, `:rcg`, and `:lbfgs` solver choices.

## Tucker decomposition

Exact ST-HOSVD and HOOI currently require materialized compute storage. When a
randomized approximation is acceptable, randomized ST-HOSVD can consume a lazy
converted tensor without constructing complete mode unfoldings:

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

Set `materialize=true` when you deliberately want an exact ST-HOSVD or HOOI
run:

```julia
result = tucker(
    counts,
    (20, 15, 10);
    compute_type = Float32,
    materialize = true,
)
```

## What `materialize=false` guarantees

When preprocessing produces a [`ComputeArray`](@ref), `materialize=false`
means that TensorKitchen does not allocate an input-sized copy containing every
observation in compute precision. Decomposition factors, projected Tucker
cores, and bounded workspaces are still allocated.

Unsupported lazy combinations fail with an `ArgumentError` and explain which
option must change. TensorKitchen does not silently materialize the input or
silently replace an explicitly requested initializer or exact method. The
automatic CP/NNCP initialization policy selects `RandomInit()` for lazy inputs;
explicit structured CP initializers and `solver=:lm` still require a
materialized input. LM constructs an input-sized ambient residual.

BTD is not connected to the lazy input path yet. To use BTD with integer data,
make the conversion explicit:

```julia
A = materialize_tensor(counts, Float32)
result = btd(A, 2, (5, 5, 5))
```

## Inspect preprocessing directly

Most users can pass preprocessing keywords directly to `cpd`, `nncpd`, or
`tucker`. The lower-level functions are useful when inspecting or reusing a
prepared tensor:

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
