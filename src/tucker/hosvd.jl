export tucker_hosvd, reconstruct_tucker, reconstruct_tucker!

"""
    tucker_hosvd(A, ranks) computes (core, factors)

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
        factors[mode] = Matrix(@view U[:, 1:r])
    end
    core = A
    for mode = 1:N
        core = mode_n_product(core, factors[mode]', mode)
    end
    return core, factors
end

"""
    reconstruct_tucker(core, factors) reconstructs the tensor from Tucker decomposition

    A = core ×₁ factors[1] ×₂ factors[2] ⋯ ×_d factors[d]
"""
function reconstruct_tucker(core::AbstractArray{T}, factors) where {T<:AbstractFloat}
    A = core
    for mode in eachindex(factors)
        A = mode_n_product(A, factors[mode], mode)
    end
    return A
end

"""
    reconstruct_tucker!(out, core, factors)

Reconstruct a Tucker tensor directly into `out`.

This low-memory kernel avoids allocating the full reconstructed tensor before
copying it into `out`. It is useful in workspace-based paths where avoiding
large temporary tensors is more important than using BLAS-heavy mode products.
"""
function reconstruct_tucker!(
    out::AbstractArray{T,N},
    core::AbstractArray{T,N},
    factors,
) where {T<:AbstractFloat,N}
    length(factors) == N || throw(
        DimensionMismatch(
            "reconstruct_tucker!: got $(length(factors)) factors for an order-$N core.",
        ),
    )
    @inbounds for mode = 1:N
        size(out, mode) == size(factors[mode], 1) || throw(
            DimensionMismatch(
                "reconstruct_tucker!: output mode $mode has size $(size(out, mode)), " *
                "expected $(size(factors[mode], 1)).",
            ),
        )
        size(core, mode) == size(factors[mode], 2) || throw(
            DimensionMismatch(
                "reconstruct_tucker!: core mode $mode has size $(size(core, mode)), " *
                "expected $(size(factors[mode], 2)).",
            ),
        )
    end

    @inbounds for I in CartesianIndices(out)
        acc = zero(T)
        for J in CartesianIndices(core)
            val = core[J]
            for mode = 1:N
                val *= factors[mode][I[mode], J[mode]]
            end
            acc += val
        end
        out[I] = acc
    end
    return out
end

# reconstruction_error(A, core, factors) — defined in results/rel_error.jl (alias of rel_error).
