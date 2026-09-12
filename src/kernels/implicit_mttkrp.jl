# kernels/implicit_mttkrp.jl — observation-preserving CP contraction

export implicit_mttkrp, implicit_mttkrp!

function _validate_implicit_mttkrp(A, U, mode, rank)
    N = ndims(A)
    1 <= mode <= N || throw(ArgumentError("mode must be in 1:$N; received $mode"))
    length(U) == N ||
        throw(DimensionMismatch("expected $N factor matrices, got $(length(U))"))
    @inbounds for m = 1:N
        size(U[m], 1) == size(A, m) || throw(
            DimensionMismatch("U[$m] has $(size(U[m], 1)) rows, expected $(size(A, m))"),
        )
        size(U[m], 2) == rank ||
            throw(DimensionMismatch("all factors must have the same column count"))
    end
    return nothing
end

"""
    implicit_mttkrp!(out, A, U, mode; block_length=65_536)

Compute the observation-preserving mode-`mode` MTTKRP while allowing the
stored element type of `A` to differ from the floating-point compute type of
`out` and `U`. Every observation is accumulated directly into the small
`size(A, mode) × rank` output; no unfolding, Khatri--Rao matrix, or converted
copy of `A` is materialized.
"""
function implicit_mttkrp!(
    out::AbstractMatrix{TC},
    A::AbstractArray{TA,N},
    U::AbstractVector{<:AbstractMatrix{TC}},
    mode::Int;
    block_length::Int = 65_536,
) where {TA<:Real,TC<:AbstractFloat,N}
    rank = size(out, 2)
    size(out, 1) == size(A, mode) ||
        throw(DimensionMismatch("out has $(size(out, 1)) rows, expected $(size(A, mode))"))
    _validate_implicit_mttkrp(A, U, mode, rank)
    fill!(out, zero(TC))
    cartesian_indices = CartesianIndices(A)
    foreach_compute_block(A; compute_type = TC, block_length) do values, linear_indices
        @inbounds for (block_index, linear_index) in enumerate(linear_indices)
            I = cartesian_indices[linear_index]
            value = values[block_index]
            for component = 1:rank
                product = value
                for m = 1:N
                    m == mode && continue
                    product *= U[m][I[m], component]
                end
                out[I[mode], component] += product
            end
        end
    end
    return out
end

"""
    implicit_mttkrp(A, U, mode; block_length=65_536)

Allocate the small mode-`mode` output and evaluate the observation-preserving
MTTKRP implemented by [`implicit_mttkrp!`](@ref). The stored element type of
`A` may differ from the floating-point element type shared by the factors.
"""
function implicit_mttkrp(
    A::AbstractArray{<:Real},
    U::AbstractVector{<:AbstractMatrix{TC}},
    mode::Int;
    block_length::Int = 65_536,
) where {TC<:AbstractFloat}
    isempty(U) && throw(ArgumentError("factor list is empty"))
    out = similar(U[1], TC, size(A, mode), size(U[1], 2))
    return implicit_mttkrp!(out, A, U, mode; block_length)
end
