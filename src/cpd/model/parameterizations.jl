# cpd/model/parameterizations.jl — CP point parameterization helpers

abstract type AbstractCPParameterization end

struct NativeCPEmbedding <: AbstractCPParameterization end
struct CanonicalCPEmbedding <: AbstractCPParameterization end
struct SquaredNonnegativeCPEmbedding <: AbstractCPParameterization end
struct SoftplusNonnegativeCPEmbedding <: AbstractCPParameterization end

@inline _cp_softplus_encode_value(x::T) where {T<:AbstractFloat} =
    _invsoftplus(max(x, eps(T)))

function _cp_rank1_decode_factors(::NativeCPEmbedding, dims, p)
    return unpack_point_rank1(p, dims)
end

function _cp_rank1_decode_factors(::SquaredNonnegativeCPEmbedding, dims, p)
    λ̃, Ũ = unpack_point_rank1(p, dims)
    return λ̃^2, [Ũ[m] .^ 2 for m in eachindex(Ũ)]
end

function _cp_rank1_decode_factors(::SoftplusNonnegativeCPEmbedding, dims, p)
    λ̃, Ũ = unpack_point_rank1(p, dims)
    return _softplus_value(λ̃), [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
end

function _cp_rank1_encode_point(::NativeCPEmbedding, λ, U)
    return pack_point_rank1_segre(λ, U)
end

function _cp_rank1_encode_point(::SquaredNonnegativeCPEmbedding, λ, U)
    return pack_point_rank1(
        sqrt(max(λ, zero(λ))),
        [sqrt.(max.(u, zero(eltype(u)))) for u in U],
    )
end

function _cp_rank1_encode_point(::SoftplusNonnegativeCPEmbedding, λ, U)
    T = typeof(λ)
    return pack_point_rank1(
        _cp_softplus_encode_value(max(λ, zero(T))),
        [_cp_softplus_encode_value.(max.(u, zero(eltype(u)))) for u in U],
    )
end

function _cp_rank1_seed_point(::NativeCPEmbedding, λ, U)
    return pack_point_rank1_segre(λ, U)
end

function _cp_rank1_seed_point(::SquaredNonnegativeCPEmbedding, λ, U)
    T = typeof(λ)
    return pack_point_rank1(
        sqrt(max(abs(λ), eps(T))),
        [sqrt.(max.(abs.(u), eps(eltype(u)))) for u in U],
    )
end

function _cp_rank1_seed_point(::SoftplusNonnegativeCPEmbedding, λ, U)
    T = typeof(λ)
    return pack_point_rank1(
        _invsoftplus(max(abs(λ), eps(T))),
        [_invsoftplus.(max.(abs.(u), eps(eltype(u)))) for u in U],
    )
end

function _cp_rank1_embed_tensor(embedding::AbstractCPParameterization, dims, p)
    λ, U = _cp_rank1_decode_factors(embedding, dims, p)
    return reconstruct_cp_rank1(λ, U)
end

function _cp_rank1_tangent_tensorvec!(
    out::AbstractVector{T},
    λ::T,
    U::AbstractVector{<:AbstractVector{T}},
    λ̇::T,
    U̇::AbstractVector{<:AbstractVector{T}},
) where {T<:AbstractFloat}
    comp = ([λ], U...)
    xcomp = ([λ̇], U̇...)
    copyto!(out, _segre_tangent_tensorvec(comp, xcomp))
    return out
end

function _cp_rank1_decode_tangent_factors(::NativeCPEmbedding, dims, p, X)
    λ, U = unpack_point_rank1(p, dims)
    λ̇, U̇ = unpack_point_rank1(X, dims)
    return λ, U, λ̇, U̇
end

function _cp_rank1_decode_tangent_factors(::SquaredNonnegativeCPEmbedding, dims, p, X)
    λ̃, Ũ = unpack_point_rank1(p, dims)
    λ̇̃, U̇̃ = unpack_point_rank1(X, dims)
    λ = λ̃^2
    U = [Ũ[m] .^ 2 for m in eachindex(Ũ)]
    λ̇ = 2 * λ̃ * λ̇̃
    U̇ = [2 .* Ũ[m] .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end

function _cp_rank1_decode_tangent_factors(::SoftplusNonnegativeCPEmbedding, dims, p, X)
    λ̃, Ũ = unpack_point_rank1(p, dims)
    λ̇̃, U̇̃ = unpack_point_rank1(X, dims)
    λ = _softplus_value(λ̃)
    U = [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
    λ̇ = _softplus_derivative(λ̃) * λ̇̃
    U̇ = [_softplus_derivative.(Ũ[m]) .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end

function _cp_rankr_decode_factors(::NativeCPEmbedding, dims, r, p)
    return unpack_rankr_native(p, dims, r)
end

function _cp_rankr_decode_factors(::CanonicalCPEmbedding, dims, r, p)
    return unpack_rankr_canonical(p, dims, r)
end

function _cp_rankr_decode_factors(::SquaredNonnegativeCPEmbedding, dims, r, p)
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    return λ̃ .^ 2, [Ũ[m] .^ 2 for m in eachindex(Ũ)]
end

function _cp_rankr_decode_factors(::SoftplusNonnegativeCPEmbedding, dims, r, p)
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    return _softplus_value.(λ̃), [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
end

function _cp_rankr_encode_point(::NativeCPEmbedding, λ, U, r)
    return pack_rankr_native(λ, U, r)
end

function _cp_rankr_encode_point(::CanonicalCPEmbedding, λ, U, r)
    return pack_rankr_canonical(λ, U, r)
end

function _cp_rankr_encode_point(::SquaredNonnegativeCPEmbedding, λ, U, r)
    T = eltype(λ)
    return pack_point_rankr(
        sqrt.(max.(λ, zero(T))),
        [sqrt.(max.(F, zero(eltype(F)))) for F in U],
        r,
    )
end

function _cp_rankr_encode_point(::SoftplusNonnegativeCPEmbedding, λ, U, r)
    T = eltype(λ)
    return pack_point_rankr(
        _cp_softplus_encode_value.(max.(λ, zero(T))),
        [_cp_softplus_encode_value.(max.(F, zero(eltype(F)))) for F in U],
        r,
    )
end

function _cp_rankr_seed_point(::NativeCPEmbedding, λ, U, r)
    return pack_rankr_native(λ, U, r)
end

function _cp_rankr_seed_point(::CanonicalCPEmbedding, λ, U, r)
    return pack_rankr_canonical(λ, U, r)
end

function _cp_rankr_seed_point(::SquaredNonnegativeCPEmbedding, λ, U, r)
    T = eltype(λ)
    return pack_point_rankr(
        sqrt.(max.(abs.(λ), eps(T))),
        [sqrt.(max.(abs.(F), eps(eltype(F)))) for F in U],
        r,
    )
end

function _cp_rankr_seed_point(::SoftplusNonnegativeCPEmbedding, λ, U, r)
    T = eltype(λ)
    return pack_point_rankr(
        _invsoftplus.(max.(abs.(λ), eps(T))),
        [_invsoftplus.(max.(abs.(F), eps(eltype(F)))) for F in U],
        r,
    )
end

function _cp_rankr_embed_tensor(embedding::AbstractCPParameterization, dims, r, p)
    λ, U = _cp_rankr_decode_factors(embedding, dims, r, p)
    return reconstruct_cpd_rankr(λ, U)
end

function _cp_rankr_tangent_tensorvec!(
    out::AbstractVector{T},
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    λ̇::AbstractVector{T},
    U̇::Vector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    r = length(λ)
    d = length(U)
    Ucols = Vector{AbstractVector{T}}(undef, d)
    U̇cols = Vector{AbstractVector{T}}(undef, d)
    @inbounds for k = 1:r
        for m = 1:d
            Ucols[m] = @view U[m][:, k]
            U̇cols[m] = @view U̇[m][:, k]
        end
        out .+= _segre_tangent_tensorvec(([λ[k]], Ucols...), ([λ̇[k]], U̇cols...))
    end
    return out
end

function _cp_rankr_decode_tangent_factors(::NativeCPEmbedding, dims, r, p, X)
    λ, U = unpack_rankr_native(p, dims, r)
    xparts = parts_tuple(X)
    length(xparts) == r || throw(
        DimensionMismatch("expected $r native tangent components, got $(length(xparts))"),
    )
    T = eltype(λ)
    d = length(dims)
    λ̇ = similar(λ)
    U̇ = [zeros(T, dims[m], r) for m = 1:d]
    @inbounds for k = 1:r
        xk_λ, xk_U = unpack_point_rank1(xparts[k], dims)
        λ̇[k] = xk_λ
        for m = 1:d
            U̇[m][:, k] .= xk_U[m]
        end
    end
    return λ, U, λ̇, U̇
end

function _cp_rankr_decode_tangent_factors(::CanonicalCPEmbedding, dims, r, p, X)
    λ, U = unpack_rankr_canonical(p, dims, r)
    λ̇, U̇ = unpack_rankr_canonical(X, dims, r)
    return λ, U, λ̇, U̇
end

function _cp_rankr_decode_tangent_factors(::SquaredNonnegativeCPEmbedding, dims, r, p, X)
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    λ̇̃, U̇̃ = unpack_point_rankr(X, dims, r)
    λ = λ̃ .^ 2
    U = [Ũ[m] .^ 2 for m in eachindex(Ũ)]
    λ̇ = 2 .* λ̃ .* λ̇̃
    U̇ = [2 .* Ũ[m] .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end

function _cp_rankr_decode_tangent_factors(::SoftplusNonnegativeCPEmbedding, dims, r, p, X)
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    λ̇̃, U̇̃ = unpack_point_rankr(X, dims, r)
    λ = _softplus_value.(λ̃)
    U = [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
    λ̇ = _softplus_derivative.(λ̃) .* λ̇̃
    U̇ = [_softplus_derivative.(Ũ[m]) .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end
