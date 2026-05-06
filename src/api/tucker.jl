export tucker

"""
    tucker(A, ranks; method = :sthosvd, kwargs...) returns a TuckerResult

Computes a Tucker decomposition of `A` with multilinear rank `ranks`.

## Main Options

* `method = :sthosvd`: Sets the Tucker decomposition algorithm. Possible options are:
    - `:sthosvd` (default): Sequentially Truncated HOSVD. A direct one-pass decomposition, mainly used as a fast standalone Tucker approximation or as the default initializer for :hooiFast, deterministic, and usually a good initial point.
    - `:hooi`: High-Order Orthogonal Iteration. Iteratively refines the Tucker factors, initialized by ST-HOSVD by default.

## Extended Options

For `method = :hooi`:

* `maxiter = 50`: Maximum number of HOOI iterations.
* `tol = 1e-8`: Convergence tolerance based on change in relative reconstruction error.
* `init = :sthosvd`
    - `:sthosvd`: Uses ST-HOSVD to initialize the Tucker factors.
    - `TuckerResult`: Uses an existing Tucker decomposition as the initial point.

## Example

```julia-repl
julia> using Random
julia> Random.seed!(0)
julia> A = randn(20, 15, 10); ranks = (5, 4, 3)
julia> res = tucker(A, ranks; verbose = false)
TuckerResult{Float64, 3}
  Original size:    (20, 15, 10)
  Core size:        (5, 4, 3)
  Multilinear rank: (5, 4, 3)
  Compression:      12.0x

```
The core tensor and factor matrices can be accessed by

```julia-repl
core(res)
factors(res)
```
A tensor approximation can be reconstructed by

```julia-repl
reconstruct(res)
```
or equivalently

```julia-repl
reconstruct_tucker(core(res), factors(res))
```
## Notes

* `:sthosvd` is not an iterative solver. It directly returns a `TuckerResult` and does not expose solver-style outputs such as iteration counts or convergence diagnostics.
* `:hooi` is the iterative refinement method in the current Tucker implementation.
* Important distinction: For the current Tucker implementation, do not use `solver = :rgd`, `init = :auto`, or manifold solvers like RGD, RCG, LBFGS, etc.

"""
function tucker(A, ranks; method::Symbol = :sthosvd, kwargs...)
    fn = (sthosvd = sthosvd, hooi = hooi)
    f = get(fn, method) do
        throw(ArgumentError("Unknown method=$method. Use :sthosvd or :hooi."))
    end
    return f(A, ranks; kwargs...)
end
