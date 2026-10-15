# TensorKitchen without the tensor math

This page is for people who want to use TensorKitchen before learning the
mathematics of tensor decomposition. You only need to know what each dimension
of your array means and what you want to do with the result.

## Start with your data

A tensor is simply a numerical array with two or more dimensions. For example:

- rows × columns: a matrix;
- height × width × color: one image;
- height × width × time: a video or image sequence;
- sensor × time × trial: an experiment;
- user × item × date: repeated interaction data.

TensorKitchen accepts Julia arrays directly. Keep the dimensions in an order
that is meaningful for your application, and record what each dimension means.

```julia
using TensorKitchen

# Example: sensor × time × trial
A = rand(Float32, 24, 1_000, 40)
size(A)
```

Missing values, strings, and categorical labels must be handled before calling
a decomposition. `nncpd` additionally requires every observation to be finite
and nonnegative.

## Choose a decomposition by your goal

You do not need to choose an optimization algorithm first. Choose the model
that best matches what you want from the data.

| Your goal | Start with | Main setting |
| --- | --- | --- |
| Compress every dimension into a smaller representation | `tucker` | one rank per dimension |
| Find a shared set of components across all dimensions | `cpd` | number of components |
| Find nonnegative components in counts or intensities | `nncpd` | number of components |
| Represent several different low-dimensional groups | `btd` | number of blocks and one rank per dimension |

If you are unsure, start with Tucker for compression and CPD for component
discovery. NNCPD is useful only when negative values have no meaning. BTD is a
more flexible model and usually needs more tuning.

## Four runnable recipes

The examples below use the same tensor so that only the model choice changes.

### Tucker: compress the tensor

```julia
A = rand(Float32, 24, 100, 40)

# Keep 6 sensor patterns, 12 time patterns, and 5 trial patterns.
result = tucker(A, (6, 12, 5))

compressed_data = core(result)
mode_patterns = factors(result)
fit_error = rel_error(A, result)
```

The three ranks correspond to the three dimensions of `A`, in the same order.
The core is the compact representation. Smaller ranks give more compression;
larger ranks usually give a closer reconstruction.

### CPD: find shared components

```julia
A = rand(Float32, 24, 100, 40)

# Fit eight components shared by sensors, time, and trials.
result = cpd(A, 8; verbose = false)

component_strength = weights(result)
mode_patterns = factors(result)
fit_error = rel_error(A, result)
```

`mode_patterns[1]`, `mode_patterns[2]`, and `mode_patterns[3]` describe the
sensor, time, and trial sides of the same eight components. Column `j` in every
factor matrix belongs to component `j`.

### NNCPD: keep components nonnegative

```julia
counts = rand(Float32, 24, 100, 40)

result = nncpd(counts, 8; solver = :als, verbose = false)

component_strength = weights(result)
mode_patterns = factors(result)
fit_error = rel_error(counts, result)
```

Use NNCPD for values such as counts, concentrations, or intensities when
negative fitted components would be hard to interpret. Do not take the absolute
value of signed data merely to make it compatible with NNCPD; use `cpd` instead.

### BTD: fit a sum of compact blocks

```julia
A = rand(Float32, 24, 100, 40)

# Three blocks; every block has multilinear rank (4, 8, 3).
result = btd(A, 3, (4, 8, 3); solver = :als, verbose = false)

fitted_blocks = blocks(result)
first_block_core = core(fitted_blocks[1])
first_block_patterns = factors(fitted_blocks[1])
fit_error = rel_error(A, result)
```

Use BTD when one global Tucker model is too restrictive and you expect several
different groups or regimes. TensorKitchen currently uses the same rank tuple
for every block.

## Read the result without reconstructing it

All decomposition results store compact factors. These accessors are the normal
way to inspect them:

| Result | Useful accessors |
| --- | --- |
| CPD or NNCPD | `weights(result)`, `factors(result)` |
| Tucker | `core(result)`, `factors(result)` |
| BTD | `blocks(result)`, then `core(block)` and `factors(block)` |
| Any of the above | `rel_error(A, result)` |

Use `reconstruct(result)` only when you actually need a full tensor-shaped
approximation:

```julia
A_approx = reconstruct(result)
size(A_approx) == size(A)
```

This allocation can be as large as the original tensor. For large data, prefer
the compact factors and summary diagnostics.

## Choose ranks with a small experiment

There is no universally correct rank. Start small, fit several candidates, and
compare reconstruction error together with whether the patterns are useful for
your task.

```julia
candidate_ranks = (2, 4, 8)
errors = map(candidate_ranks) do rank
    result = cpd(A, rank; solver = :als, verbose = false)
    rel_error(A, result)
end
```

A lower error is a closer fit, but a very large model can fit noise and is
harder to interpret. Prefer the smallest model after which increasing the rank
provides little practical improvement. For Tucker and BTD, each rank must not
exceed the size of its corresponding dimension.

## Use large integer or low-precision data without a full converted copy

Large measurements are often stored as integers even though decomposition
needs floating-point arithmetic. TensorKitchen can keep the original storage
and convert bounded pieces while computing:

```julia
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

The two important options have simple meanings:

- `compute_type=Float32` selects the arithmetic precision;
- `materialize=false` avoids making a full Float32 copy during preprocessing.

This is exact observation-preserving computation: the solver still uses all
observations. It is not sketching, and it does not make reading all of the data
free. Factors, small projected arrays, and bounded workspaces are still
allocated.

The same storage options are available for `cpd`, `nncpd`, `btd`, and `tucker`,
but the supported algorithms differ:

| Decomposition | Practical large-data starting point |
| --- | --- |
| CPD | `solver=:als`, `materialize=false` |
| NNCPD | `solver=:als`, `materialize=false` |
| BTD | `solver=:als`, `materialize=false` |
| Tucker | randomized ST-HOSVD with `materialize=false` |

For a lazy Tucker approximation, use:

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

Randomized Tucker intentionally trades a small amount of accuracy for lower
cost. CPD, NNCPD, and projected BTD in the examples above avoid an unnecessary
input copy but still perform exact contractions over the observations.

Set `materialize=true` only when you deliberately want a full floating-point
copy, for example to use an exact Tucker SVD or a structured initializer that
requires dense compute storage. TensorKitchen reports an error instead of
silently materializing an unsupported lazy path.

See [Preparing large data](../preparing_data.md) for the complete storage
contract and supported solver combinations.

## A safe first workflow

1. Check `size(A)` and write down what every dimension represents.
2. Decide whether signed values are meaningful.
3. Choose Tucker for compression, CPD/NNCPD for shared components, or BTD for
   several compact groups.
4. Start with small ranks and `verbose=false` once the call works.
5. Inspect `rel_error`, factors, weights, or cores before reconstructing.
6. Compare a few nearby ranks instead of trusting one run.
7. For a large non-floating input, add `compute_type=Float32` and
   `materialize=false`.

## Common surprises

**The result changes between runs.** Initial points and randomized algorithms
can vary. Compare several runs or set Julia's random seed before fitting when
you need a repeatable experiment.

```julia
using Random
Random.seed!(2026)
```

**The error is not close to zero.** Tensor decompositions are usually compact
approximations, not exact copies. Increase the rank gradually and compare the
gain against runtime and interpretability.

**The fit is slow even with `materialize=false`.** That option avoids one large
converted copy; it does not skip observations. Try smaller ranks or fewer
iterations. For Tucker, use the randomized backend when approximation is
acceptable.

**Memory grows when calling `reconstruct`.** Reconstruction creates the full
approximated tensor. Work with factors, weights, cores, and blocks when a full
array is unnecessary.

**NNCPD rejects the input.** Check for negative, `NaN`, or infinite values.
Choose CPD if negative observations are meaningful.

## Where to go next

- [CP decomposition](../cpd.md)
- [Tucker decomposition](../tucker.md)
- [Block term decomposition](../btd.md)
- [Preparing large data](../preparing_data.md)
- [Choosing a decomposition](../PIPELINE.md)
- [Advanced guide](../advanced/index.md)
