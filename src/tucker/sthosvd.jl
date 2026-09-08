export reconstruct, sthosvd, thosvd, optimal_mode_order, error_bound

"""
    ST-HOSVD (Sequentially Truncated HOSVD)

    Sequentially Truncated Higher-Order Singular Value Decomposition (ST-HOSVD)
from:

    N. Vannieuwenhoven, R. Vandebril, K. Meerbergen,
    "A New Truncation Strategy for the Higher-Order Singular Value Decomposition",
    SIAM J. Sci. Comput., 34(2), A1027–A1052, 2012.
    DOI: 10.1137/110836067

The ST-HOSVD computes a Tucker decomposition A ≈ S ×₁ U₁ ×₂ U₂ ⋯ ×_d U_d
by sequentially computing truncated SVDs mode by mode, projecting (shrinking) the tensor after each mode.
"""

# TuckerResult struct and show live in core/types.jl. Uses unfold_mode, mode_n_product from core/tensor_ops.

"""
    reconstruct(td::TuckerResult)

Reconstruct the dense tensor represented by a Tucker decomposition result.

For a Tucker result with core `S` and factors `U_1, ..., U_d`, this returns
`S ×_1 U_1 ×_2 U_2 ... ×_d U_d`.
"""
function reconstruct(td::TuckerResult{T,N}) where {T,N}
    A = td.core
    for k = 1:N
        A = mode_n_product(A, td.factors[k], k)
    end
    return A
end

# rel_error / relative_error(A, td::TuckerResult) — defined in results/rel_error.jl.
# Processing-order heuristics

"""
    optimal_mode_order(dims, ranks)
    optimal_mode_order(dims)

Heuristic for choosing a good processing order for ST-HOSVD.

With known ranks, modes are sorted by decreasing `dims[k] / ranks[k]`
(equivalently, increasing `ranks[k] / dims[k]`). This prioritizes the largest
fractional reduction of the working tensor.

Without ranks, modes are sorted by increasing mode dimension. This is the
size-only compact-SVD heuristic proposed in section 6.4 of Vannieuwenhoven,
Vandebril, and Meerbergen (2012).

Both are inexpensive greedy heuristics, not globally optimal order solvers for
runtime or approximation error. Supply `processing_order` explicitly when
domain knowledge or benchmarking identifies a better order.
"""
function optimal_mode_order(dims::NTuple{N,Int}, ranks::NTuple{N,Int}) where {N}
    # Rank-aware storage-reduction heuristic: strongest fractional shrink first.
    ratios = [dims[k] / ranks[k] for k = 1:N]
    return sortperm(ratios; rev = true)
end

function optimal_mode_order(dims::NTuple{N,Int}) where {N}
    # When ranks are unknown, process smallest dimensions first
    return sortperm(collect(dims))
end


# ST-HOSVD  —  Algorithm 1 from the paper


"""
    sthosvd(A, ranks; processing_order=optimal_mode_order(size(A), ranks),
        svd_backend=:exact, oversampling=16, power_iterations=1,
        block_columns=65_536, rng=Random.default_rng(), verbose=false)
    Computes the rank-(r₁,…,r_d) Sequentially Truncated HOSVD (ST-HOSVD).

# Exact algorithm (Definition 6.1, Algorithm 1)
Given tensor A ∈ ℝ^{n₁×⋯×n_d} and target multilinear rank (r₁,…,r_d):

1. Set Ŝ₀ = A
2. For k = 1,…,d (in processing order p):
   a. Compute the mode-p[k] unfolding of Ŝ_{k-1}
   b. Compute the compact SVD; truncate to rank r_{p[k]}
   c. Set Û_{p[k]} = first r_{p[k]} left singular vectors
   d. Ŝₖ = Ŝ_{k-1} ×_{p[k]} Û_{p[k]}ᵀ   (project and shrink)
3. Return core Ŝ_d and factor matrices Û₁,…,Û_d

# Implicit randomized backend

For the current sequential core `S`, let its conceptual mode-`k` unfolding be
`A_(k) ∈ ℝ^(n_k × N_k)`, where `N_k = ∏_{j≠k} n_j`. For target rank `r_k`, set
the sketch size to `ℓ = min(r_k + oversampling, n_k, N_k)`. The randomized
backend then performs:

1. Gaussian range finding: `Y = A_(k) Ω_k`, where
   `Ω_k ∈ ℝ^(N_k × ℓ)` has independent standard-normal entries.
2. Basis construction: `Q = qr(Y)`, retaining `ℓ` orthonormal columns.
3. Optional power iteration: repeatedly approximate the range of
   `A_(k) A_(k)ᵀ Q`, with re-orthogonalization after every iteration.
4. Projection: `B = Qᵀ A_(k)`, stored with the shape of a tensor whose mode `k`
   has length `ℓ`.
5. Rayleigh-Ritz refinement: eigendecompose the small Gram matrix
   `B Bᵀ ∈ ℝ^(ℓ × ℓ)`, retain its leading `r_k` eigenvectors in `R`, and set
   `U_k = Q R` and `S_new = Rᵀ B`.

This is a Gaussian projection of all conceptual unfolding columns, not random
column selection. Neither `A_(k)` nor the complete `Ω_k` is constructed. The
products in steps 1, 3, and 4 are evaluated as blockwise tensor contractions.
The implementation materializes only the small sketch and basis, the small Gram
matrix, and the progressively compressed tensor `B`/`S_new`.

The power loop orthonormalizes after each complete `A_(k) A_(k)ᵀ`
application. It does not orthonormalize between the individual transpose and
forward products as in fully stabilized randomized subspace iteration. Keep
`power_iterations` small and verify reconstruction error when singular values
span a wide numerical range.

# References
- N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions," *SIAM Review*, 53(2), 217–288, 2011.
  DOI: `10.1137/090771806`.

The reference provides the Gaussian range-finder and power-iteration framework
applied here to each conceptual mode unfolding. Evaluating those matrix products
through blockwise tensor contractions is TensorKitchen's implementation strategy
for efficient computation.

# Arguments
- `A::AbstractArray{T,N}`: floating-point input tensor
- `ranks::NTuple{N,Int}`: target multilinear rank (r₁,…,r_d)

# Keyword arguments
- `processing_order::Vector{Int}`: order in which to process modes (default: heuristic)
- `svd_backend::Symbol`: `:exact` materializes each unfolding and uses a conventional
  SVD; `:randomized` computes a truncated left singular subspace from implicit tensor
  contractions without materializing the unfolding (default: `:exact`)
- `oversampling::Int`: additional randomized sketch dimensions (default: `16`)
- `power_iterations::Int`: randomized subspace power iterations (default: `1`)
- `block_columns::Int`: approximate maximum number of implicit unfolding columns per
  random-projection block (default: `65_536`)
- `rng::AbstractRNG`: random-number generator used by `:randomized` (default:
  `Random.default_rng()`)
- `verbose::Bool`: print progress information (default: false)

# Returns
`TuckerResult` with core tensor and factor matrices.

# Example
```julia
A = randn(50, 40, 30)
td = sthosvd(A, (5, 4, 3))
rel_err = relative_error(A, td)
```
"""

function sthosvd(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int};
    processing_order::Vector{Int} = optimal_mode_order(size(A), ranks),
    svd_backend::Symbol = :exact,
    oversampling::Int = 16,
    power_iterations::Int = 1,
    block_columns::Int = 65_536,
    rng::AbstractRNG = Random.default_rng(),
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    dims = size(A)
    d = N

    # Validate inputs
    for k = 1:d
        @assert 1 <= ranks[k] <= dims[k] "Rank r[$k]=$(ranks[k]) must be in [1, $(dims[k])]"
    end
    @assert sort(processing_order) == 1:d "processing_order must be a permutation of 1:$d"
    svd_backend in (:exact, :randomized) || throw(
        ArgumentError("svd_backend must be :exact or :randomized; received $svd_backend"),
    )
    oversampling >= 0 || throw(ArgumentError("oversampling must be nonnegative"))
    power_iterations >= 0 || throw(ArgumentError("power_iterations must be nonnegative"))
    block_columns >= 1 || throw(ArgumentError("block_columns must be positive"))

    progress =
        d > 0 ? make_sthosvd_progress(d; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    # The randomized backend keeps a reference to an mmap/lazy input until the
    # first projection instead of creating a tensor-sized input copy.
    S = svd_backend === :exact ? copy(A) : A
    factors = Vector{Matrix{T}}(undef, d)
    singular_vals = Vector{Vector{T}}(undef, d)

    for (step, k) in enumerate(processing_order)
        rk = ranks[k]

        if svd_backend === :exact
            Sk_unfold = unfold_mode(S, k)
            F = svd(Sk_unfold)
            rk_actual = min(rk, length(F.S))
            Uk = Matrix(@view F.U[:, 1:rk_actual])
            singular_vals[k] = F.S
            factors[k] = Uk
            S = mode_n_product(S, Uk', k)
        else
            Uk, S, _ = _randomized_implicit_mode_step(
                S,
                k,
                rk,
                rng;
                oversampling,
                power_iterations,
                block_columns,
            )
            factors[k] = Uk
            # The implicit range finder does not compute the complete discarded
            # spectrum needed by `error_bound`.
            singular_vals[k] = T[]
        end

        verbose && update_progress!(
            progress,
            step;
            showvalues = Any[
                ("Mode", k),
                ("Target rank", rk),
                ("SVD backend", svd_backend),
                ("Core size", size(S)),
            ],
        )
    end

    verbose &&
        finish_progress!(progress; showvalues = Any[("Status", "Completed"), ("Steps", d)])

    return TuckerResult{T,N}(S, factors, processing_order, singular_vals)
end

@inline function _orthonormal_columns(Y::AbstractMatrix{T}, count::Int) where {T}
    Q = qr(Y).Q
    return Matrix{T}(Q[:, 1:count])
end

"""
    _randomized_implicit_mode_step(A, mode, rank, rng;
        oversampling, power_iterations, block_columns)

Approximate the leading rank-`rank` left singular subspace of the conceptual
mode-`mode` unfolding `A_(mode)` without materializing that unfolding.

With `ℓ = rank + oversampling` (bounded by the unfolding dimensions), the method
computes the Gaussian sketch `Y = A_(mode) Ω`, obtains `Q = qr(Y)`, and stores the
projection `B = Qᵀ A_(mode)` in tensor form. Each power iteration applies the
equivalent of `A_(mode) A_(mode)ᵀ Q` using `_implicit_mode_cross` and
`_implicit_mode_product`. It then diagonalizes the small matrix `B Bᵀ`, rotates
`Q` by the leading eigenvectors, and applies the same rotation to `B` to produce
the sequentially reduced core.

`_implicit_mode_sketch` generates `Ω` in blocks containing approximately at most
`block_columns` conceptual unfolding columns. Consequently, neither the full
unfolding nor the full Gaussian matrix is allocated. The returned singular values
are Rayleigh-Ritz approximations for the retained subspace only; discarded
singular values are unavailable.

This mode step adapts the randomized range-finding and power-iteration framework
of N. Halko, P. G. Martinsson, and J. A. Tropp, *SIAM Review* 53(2), 217–288,
2011, DOI: `10.1137/090771806`.
"""
function _randomized_implicit_mode_step(
    A::AbstractArray{T,N},
    mode::Int,
    rank::Int,
    rng::AbstractRNG;
    oversampling::Int,
    power_iterations::Int,
    block_columns::Int,
) where {T<:AbstractFloat,N}
    column_count = div(length(A), size(A, mode))
    sketch_rank = min(rank + oversampling, size(A, mode), column_count)
    retained_rank = min(rank, sketch_rank)

    Y = _implicit_mode_sketch(A, mode, sketch_rank, rng; block_columns)
    Q = _orthonormal_columns(Y, sketch_rank)
    projected = _implicit_mode_product(A, transpose(Q), mode; block_columns)

    # Apply A_(mode) * A_(mode)' through tensor contractions. Neither the mode
    # unfolding nor its transpose is materialized.
    for _ = 1:power_iterations
        Y = _implicit_mode_cross(A, projected, mode; block_columns)
        Q = _orthonormal_columns(Y, sketch_rank)
        projected = _implicit_mode_product(A, transpose(Q), mode; block_columns)
    end

    # Rayleigh-Ritz refinement within the randomized range. `projected` is
    # Q' * A_(mode), stored with its tensor shape, so its small left Gram matrix
    # can also be obtained without an unfolding.
    gram = _implicit_mode_cross(projected, projected, mode; block_columns)
    gram = (gram + transpose(gram)) / T(2)
    decomposition = eigen(Symmetric(gram))
    order = sortperm(decomposition.values; rev = true)
    selected = order[1:retained_rank]
    rotation = Matrix{T}(decomposition.vectors[:, selected])
    eigenvalues = max.(decomposition.values[selected], zero(T))
    approximate_singular_values = sqrt.(eigenvalues)

    factor = Q * rotation
    core = _implicit_mode_product(projected, transpose(rotation), mode; block_columns)
    return factor, core, approximate_singular_values
end


"""
    sthosvd(A, tol; processing_order=optimal_mode_order(size(A)), verbose=false)

Tolerance-based ST-HOSVD: automatically determine ranks to achieve
    ‖A - Â‖_F ≤ tol · ‖A‖_F

Uses the error bound from Theorem 6.4: the squared error decomposes as a sum of
squared errors from each truncation step. By distributing the error budget equally
across modes (ϵₖ = ϵ/√d for each mode), the final error is bounded by ϵ.

# Arguments
- `A::AbstractArray{T,N}`: floating-point input tensor
- `tol::Float64`: requested relative Frobenius-error tolerance

# Keyword arguments
- `processing_order::Vector{Int}`: order in which to process modes (default: smallest first)
- `verbose::Bool`: print progress information (default: false)

This overload uses exact SVDs to choose the ranks and returns a
[`TuckerResult`](@ref). The randomized backend is available only for the
fixed-rank overload.
"""
function sthosvd(
    A::AbstractArray{T,N},
    tol::Float64;
    processing_order::Vector{Int} = optimal_mode_order(size(A)),
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    dims = size(A)
    d = N
    normA = norm(A)

    @assert sort(processing_order) == 1:d "processing_order must be a permutation of 1:$d"

    progress =
        d > 0 ? make_sthosvd_progress(d; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    # Distribute error budget equally across modes (Theorem 6.4)
    # ‖A - Â‖²_F = Σₖ ‖Ŝₖ₋₁ ×_{p[k]} (I - ÛₖÛₖᵀ)‖²_F
    # To ensure total relative error ≤ tol, allow tol/√d per mode
    tol_per_mode = tol * normA / sqrt(d)

    S = copy(A)
    factors = Vector{Matrix{T}}(undef, d)
    singular_vals = Vector{Vector{T}}(undef, d)
    ranks = zeros(Int, d)

    for (step, k) in enumerate(processing_order)
        Sk_unfold = unfold_mode(S, k)
        F = svd(Sk_unfold)

        # Determine rank: find smallest rk such that
        #   √(σ²_{rk+1} + σ²_{rk+2} + ⋯) ≤ tol_per_mode
        # i.e., the discarded singular values have small enough energy
        sigma = F.S
        cum_sq = reverse(cumsum(reverse(sigma .^ 2)))  # cum_sq[j] = Σ_{i≥j} σᵢ²

        rk = length(sigma)  # default: keep all
        for j in eachindex(sigma)
            tail_energy = j < length(cum_sq) ? cum_sq[j+1] : 0.0
            if sqrt(tail_energy) <= tol_per_mode
                rk = j
                break
            end
        end
        rk = max(rk, 1)  # keep at least rank 1

        if verbose
            discarded = rk < length(sigma) ? sqrt(sum(sigma[(rk+1):end] .^ 2)) : 0.0
            update_progress!(
                progress,
                step;
                showvalues = Any[
                    ("Mode", k),
                    ("Rank", "$rk/$(size(S, k))"),
                    ("Discarded energy", discarded),
                ],
            )
        end

        Uk = Matrix(@view F.U[:, 1:rk])
        singular_vals[k] = sigma
        factors[k] = Uk
        ranks[k] = rk

        S = mode_n_product(S, Uk', k)
    end

    if verbose
        finish_progress!(
            progress;
            showvalues = Any[
                ("Status", "Completed"),
                ("Final core size", size(S)),
                ("Multilinear rank", Tuple(ranks)),
            ],
        )
    end

    return TuckerResult{T,N}(S, factors, processing_order, singular_vals)
end


# T-HOSVD (classical truncated HOSVD) for comparison

"""
    thosvd(A, ranks; verbose=false)

Classical Truncated HOSVD (T-HOSVD).

Computes all factor matrices from the *original* tensor (no sequential truncation),
then computes the core tensor at the end.

ST-HOSVD often requires fewer operations because it shrinks the working tensor
between modes. Approximation quality can depend on the tensor and processing
order, so compare reconstruction errors when the distinction matters.

`A` must be a floating-point tensor, and `ranks` must contain one valid rank per
mode. Returns a [`TuckerResult`](@ref).
"""
function thosvd(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int};
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    dims = size(A)
    d = N

    for k = 1:d
        @assert 1 <= ranks[k] <= dims[k] "Rank r[$k]=$(ranks[k]) must be in [1, $(dims[k])]"
    end

    progress =
        d > 0 ? make_thosvd_progress(d; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    factors = Vector{Matrix{T}}(undef, d)
    singular_vals = Vector{Vector{T}}(undef, d)

    # Step 1: Compute all factor matrices from the ORIGINAL tensor
    for k = 1:d
        Ak = unfold_mode(A, k)
        F = svd(Ak)
        factors[k] = Matrix(@view F.U[:, 1:ranks[k]])
        singular_vals[k] = F.S

        verbose && update_progress!(
            progress,
            k;
            showvalues = Any[("Mode", k), ("Shape", size(Ak)), ("Rank", ranks[k])],
        )
    end

    # Step 2: Compute core tensor  S = A ×₁ U₁ᵀ ×₂ U₂ᵀ ⋯ ×_d Udᵀ
    S = copy(A)
    for k = 1:d
        S = mode_n_product(S, factors[k]', k)
    end
    verbose && finish_progress!(
        progress;
        showvalues = Any[("Status", "Completed"), ("Core size", size(S))],
    )

    return TuckerResult{T,N}(S, factors, collect(1:d), singular_vals)
end


# Error analysis utilities (Theorem 6.4)
"""
    error_bound(td::TuckerResult)

Return the square root of the summed discarded singular-value energy stored in
`td`. For an exact ST-HOSVD result, Theorem 6.4 gives

`‖A - Â‖²_F = Σₖ Σ_{j > rₖ} σ²_{k,j}`,

so this is the absolute Frobenius reconstruction error. The interpretation is
specific to the spectra generated by exact ST-HOSVD. Use `rel_error(A, td)` for
HOOI, T-HOSVD, or other Tucker results. Randomized ST-HOSVD does not store the
complete discarded spectra, and this function throws `ArgumentError` for such a
result.
"""
function error_bound(td::TuckerResult{T,N}) where {T,N}
    sq_error = zero(T)
    for k = 1:N
        rk = size(td.core, k)
        sigma = td.singular_values[k]
        isempty(sigma) && throw(
            ArgumentError(
                "error_bound requires complete per-mode singular spectra; " *
                "they are unavailable for randomized ST-HOSVD results",
            ),
        )
        if rk < length(sigma)
            sq_error += sum(sigma[(rk+1):end] .^ 2)
        end
    end
    return sqrt(sq_error)
end


# Convenience: vector/tuple input for ranks

function sthosvd(A::AbstractArray{T,N}, ranks::Vector{Int}; kwargs...) where {T,N}
    @assert length(ranks) == N
    return sthosvd(A, NTuple{N,Int}(ranks); kwargs...)
end

function thosvd(A::AbstractArray{T,N}, ranks::Vector{Int}; kwargs...) where {T,N}
    @assert length(ranks) == N
    return thosvd(A, NTuple{N,Int}(ranks); kwargs...)
end
