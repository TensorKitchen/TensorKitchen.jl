export tucker_hosvd, reconstruct_tucker

"""
Tucker decomposition via HOSVD and reconstruction utilities.
Uses unfold_mode and mode_n_product from core/tensor_ops.
"""

"""
    tucker_hosvd(A, ranks) -> (core, factors)

Compute Tucker decomposition via Higher-Order SVD (HOSVD).

For each mode, computes SVD of the unfolded tensor and takes the top `ranks[mode]`
left singular vectors as the factor matrix. The core tensor is computed by projecting
the original tensor onto these factor spaces.

# Arguments
- `A`: Input tensor
- `ranks`: Tuple of target ranks for each mode

# Returns
- `core`: Core tensor
- `factors`: Vector of factor matrices (one per mode)
"""
function tucker_hosvd(A::AbstractArray{T}, ranks::NTuple{N,Int}) where {T<:AbstractFloat,N}
    dims = size(A)
    factors = Vector{Matrix{T}}(undef, N)
    for mode = 1:N
        A_mode = unfold_mode(A, mode)
        U, _, _ = svd(A_mode)
        r = min(ranks[mode], size(U, 2))
        factors[mode] = U[:, 1:r]
    end
    core = A
    for mode = 1:N
        core = mode_n_product(core, factors[mode]', mode)
    end
    return core, factors
end

"""
    reconstruct_tucker(core, factors) -> Array

Reconstruct the tensor from Tucker decomposition: core ×₁ factors[1] ×₂ factors[2] ⋯
"""
function reconstruct_tucker(core::AbstractArray{T}, factors) where {T<:AbstractFloat}
    A = core
    for mode = 1:length(factors)
        A = mode_n_product(A, factors[mode], mode)
    end
    return A
end

# reconstruction_error(A, core, factors) — defined in results/rel_error.jl (alias of rel_error).
