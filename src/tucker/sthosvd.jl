export reconstruct, sthosvd, thosvd, optimal_mode_order, error_bound

"""
    ST-HOSVD (Sequentially Truncated HOSVD)

Implementation of the Sequentially Truncated Higher-Order Singular Value Decomposition (ST-HOSVD)
from:

    N. Vannieuwenhoven, R. Vandebril, K. Meerbergen,
    "A New Truncation Strategy for the Higher-Order Singular Value Decomposition",
    SIAM J. Sci. Comput., 34(2), A1027–A1052, 2012.
    DOI: 10.1137/110836067

The ST-HOSVD computes a Tucker decomposition A ≈ S ×₁ U₁ ×₂ U₂ ⋯ ×_d U_d
by sequentially computing truncated SVDs mode by mode, projecting (shrinking) the
tensor after each mode. Uses `unfold_mode` and `mode_n_product` from the Tucker module.
"""

# TuckerResult struct and show live in core/types.jl. Uses unfold_mode, mode_n_product from core/tensor_ops.

"""
    reconstruct(td::TuckerResult) -> Array

Reconstruct the full tensor from a Tucker decomposition:
    A ≈ S ×₁ U₁ ×₂ U₂ ⋯ ×_d U_d
"""
function reconstruct(td::TuckerResult{T,N}) where {T,N}
    A = td.core
    for k = 1:N
        A = mode_n_product(A, td.factors[k], k)
    end
    return A
end

# rel_error / relative_error(A, td::TuckerResult) — defined in results/rel_error.jl.
# Processing order heuristics (Section 6.3 of the paper)

"""
    optimal_mode_order(dims::Tuple, ranks::Tuple) -> Vector{Int}

Heuristic for choosing a good processing order for ST-HOSVD.

Following Section 6.3 of Vannieuwenhoven et al., a good heuristic is to process
modes in increasing order of nₖ / rₖ (i.e., process the most compressible modes
first). When ranks are not known a priori, processing from smallest to largest
dimension is a reasonable default.
"""
function optimal_mode_order(dims::NTuple{N,Int}, ranks::NTuple{N,Int}) where {N}
    # Sort modes by compression ratio nk/rk (ascending = most compressible first)
    ratios = [dims[k] / ranks[k] for k = 1:N]
    return sortperm(ratios)  # most compressible first
end

function optimal_mode_order(dims::NTuple{N,Int}) where {N}
    # When ranks are unknown, process smallest dimensions first
    return sortperm(collect(dims))
end


# ST-HOSVD  —  Algorithm 1 from the paper


"""
    sthosvd(A, ranks; [processing_order], [verbose]) -> TuckerResult

Compute the rank-(r₁,…,r_d) Sequentially Truncated HOSVD (ST-HOSVD).

# Algorithm (Definition 6.1, Algorithm 1)

Given tensor A ∈ ℝ^{n₁×⋯×n_d} and target multilinear rank (r₁,…,r_d):

1. Set Ŝ₀ = A
2. For k = 1,…,d (in processing order p):
   a. Compute the mode-p[k] unfolding of Ŝ_{k-1}
   b. Compute the compact SVD; truncate to rank r_{p[k]}
   c. Set Û_{p[k]} = first r_{p[k]} left singular vectors
   d. Ŝₖ = Ŝ_{k-1} ×_{p[k]} Û_{p[k]}ᵀ   (project and shrink)
3. Return core Ŝ_d and factor matrices Û₁,…,Û_d

# Arguments
- `A::Array{T,N}`: input tensor
- `ranks::NTuple{N,Int}`: target multilinear rank (r₁,…,r_d)

# Keyword arguments
- `processing_order::Vector{Int}`: order in which to process modes (default: heuristic)
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
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    dims = size(A)
    d = N

    # Validate inputs
    for k = 1:d
        @assert 1 <= ranks[k] <= dims[k] "Rank r[$k]=$(ranks[k]) must be in [1, $(dims[k])]"
    end
    @assert sort(processing_order) == 1:d "processing_order must be a permutation of 1:$d"

    progress =
        d > 0 ? make_sthosvd_progress(d; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    # Initialize
    S = copy(A)  # Ŝ₀ = A (will be progressively truncated)
    factors = Vector{Matrix{T}}(undef, d)
    singular_vals = Vector{Vector{T}}(undef, d)

    for (step, k) in enumerate(processing_order)
        rk = ranks[k]

        # Mode-k unfolding of current (partially truncated) core tensor
        Sk_unfold = unfold_mode(S, k)

        # Compact SVD, truncated to rank rk
        F = svd(Sk_unfold)
        rk_actual = min(rk, length(F.S))  # handle rank-deficient case

        Uk = F.U[:, 1:rk_actual]           # nk × rk  (orthonormal columns)
        singular_vals[k] = F.S             # store full spectrum for error bound

        # Store factor matrix
        factors[k] = Uk

        # Project: Ŝₖ = Ŝ_{k-1} ×_k Ûₖᵀ  (shrinks mode k from nk to rk)
        S = mode_n_product(S, Uk', k)

        verbose && update_progress!(
            progress,
            step;
            showvalues = Any[("Mode", k), ("Target rank", rk), ("Core size", size(S))],
        )
    end

    verbose &&
        finish_progress!(progress; showvalues = Any[("Status", "Completed"), ("Steps", d)])

    return TuckerResult{T,N}(S, factors, processing_order, singular_vals)
end


"""
    sthosvd(A, tol; [processing_order], [verbose]) -> TuckerResult

Tolerance-based ST-HOSVD: automatically determine ranks to achieve
    ‖A - Â‖_F ≤ tol · ‖A‖_F

Uses the error bound from Theorem 6.4: the squared error decomposes as a sum of
squared errors from each truncation step. By distributing the error budget equally
across modes (ϵₖ = ϵ/√d for each mode), the final error is bounded by ϵ.

# Arguments
- `A::Array{T,N}`: input tensor
- `tol::Float64`: relative error tolerance

# Keyword arguments
- `processing_order::Vector{Int}`: order in which to process modes (default: smallest first)
- `verbose::Bool`: print progress information (default: false)
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
        for j = 1:length(sigma)
            tail_energy = j < length(cum_sq) ? cum_sq[j+1] : 0.0
            if sqrt(tail_energy) <= tol_per_mode
                rk = j
                break
            end
        end
        rk = max(rk, 1)  # keep at least rank 1

        if verbose
            discarded = rk < length(sigma) ? sqrt(sum(sigma[rk+1:end] .^ 2)) : 0.0
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

        Uk = F.U[:, 1:rk]
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
    thosvd(A, ranks; [verbose]) -> TuckerResult

Classical Truncated HOSVD (T-HOSVD).

Computes all factor matrices from the *original* tensor (no sequential truncation),
then computes the core tensor at the end. This is the standard approach from
De Lathauwer et al. (2000).

The ST-HOSVD is generally preferred: it requires fewer operations and typically
yields a better approximation.
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
        factors[k] = F.U[:, 1:ranks[k]]
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
    error_bound(td::TuckerResult) -> Float64

Compute the a posteriori error bound from Theorem 6.4:
    ‖A - Â‖²_F = Σₖ Σ_{j > rₖ} σ²_{k,j}
where σ_{k,j} are the singular values at step k of the ST-HOSVD.

This is exact (not just a bound) for the ST-HOSVD.
"""
function error_bound(td::TuckerResult{T,N}) where {T,N}
    sq_error = zero(T)
    for k = 1:N
        rk = size(td.core, k)
        sigma = td.singular_values[k]
        if rk < length(sigma)
            sq_error += sum(sigma[rk+1:end] .^ 2)
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
