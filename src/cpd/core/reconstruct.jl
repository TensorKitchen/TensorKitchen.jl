# cpd/core/reconstruct.jl — CP reconstruction (TensorOperations)
export reconstruct_cp_rank1,
    reconstruct_cp_rankr,
    reconstruct_cpd_rankr,
    embed_point_rank1,
    embed_point_rankr,
    embed_point_rank1_nn,
    embed_point_rankr_nn

using TensorOperations
using LinearAlgebra

"""Rank-1 CP: λ · u₁ ⊗ u₂ ⊗ ... ⊗ u_d via ncon."""
function reconstruct_cp_rank1(
    λ::T,
    U::AbstractVector{<:AbstractVector{T}},
) where {T<:AbstractFloat}
    isempty(U) && throw(ArgumentError("reconstruct_cp_rank1: empty factor vectors"))
    N = length(U)
    # A[i1,...,iN] = λ * u1[i1] * u2[i2] * ... * uN[iN] — outer product scaled by λ
    tensors = [vec(U[m]) for m = 1:N]
    return T(λ) .* ncon(tensors, [[-m] for m = 1:N])
end

"""Rank-r CP from (λ, U). Batched broadcast + sum over r."""
function reconstruct_cpd_rankr(λ::Vector{T}, U::Vector{Matrix{T}}) where {T<:AbstractFloat}
    (isempty(U) || isempty(λ)) &&
        throw(ArgumentError("reconstruct_cpd_rankr: empty factors or λ"))
    r, N = length(λ), length(U)
    for m = 1:N
        size(U[m], 2) == r || throw(
            DimensionMismatch(
                "reconstruct_cpd_rankr: U[$m] has $(size(U[m],2)) cols, expected r=$r",
            ),
        )
    end
    dims = ntuple(m -> size(U[m], 1), N)
    M = U[1] .* reshape(λ, 1, :)
    @inbounds for m = 2:N
        shape_M = (ntuple(i -> dims[i], m - 1)..., 1, r)
        shape_U = (ntuple(_ -> 1, m - 1)..., dims[m], r)
        M = reshape(M, shape_M) .* reshape(U[m], shape_U)
    end
    return dropdims(sum(M; dims = N + 1); dims = N + 1)
end

"""Rank-r CP from components. Validates structure, delegates to (λ,U)."""
function reconstruct_cpd_rankr(
    components::AbstractVector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    isempty(components) && throw(ArgumentError("reconstruct_cpd_rankr: empty components"))
    r, N = length(components), length(components[1].vectors)
    for k = 2:r
        length(components[k].vectors) == N || throw(
            DimensionMismatch(
                "reconstruct_cpd_rankr: component $k has $(length(components[k].vectors)) modes",
            ),
        )
        for m = 1:N
            length(components[k].vectors[m]) == length(components[1].vectors[m]) || throw(
                DimensionMismatch(
                    "reconstruct_cpd_rankr: component $k mode $m incompatible",
                ),
            )
        end
    end
    λ = [c.λ for c in components]
    U = [hcat((c.vectors[m] for c in components)...) for m = 1:N]
    return reconstruct_cpd_rankr(λ, U)
end

reconstruct_cp_rankr(λ::Vector{T}, U::Vector{Matrix{T}}) where {T<:AbstractFloat} =
    reconstruct_cpd_rankr(λ, U)

embed_point_rank1(p, dims::NTuple{N,Int}) where {N} =
    reconstruct_cp_rank1(unpack_point_rank1(p, dims)...)
embed_point_rankr(p, dims::NTuple{N,Int}, r::Int) where {N} =
    reconstruct_cpd_rankr(unpack_point_rankr(p, dims, r)...)

function embed_point_rank1_nn(p, dims::NTuple{N,Int}) where {N}
    λ̃, Ũ = unpack_point_rank1(p, dims)
    return reconstruct_cp_rank1(λ̃ .^ 2, [Ũ[m] .^ 2 for m in eachindex(Ũ)])
end

"""
    embed_point_rankr_nn(p, dims, r)

Embed nonnegative rank-r squaring-parameterized point into ambient tensor space.
"""
function embed_point_rankr_nn(p, dims::NTuple{N,Int}, r::Int) where {N}
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    return reconstruct_cpd_rankr(λ̃ .^ 2, [Ũ[m] .^ 2 for m in eachindex(Ũ)])
end
