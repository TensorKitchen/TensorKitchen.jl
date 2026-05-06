# core/pack_points.jl — Internal canonical point packing helpers

@inline _components_to_lambda_U(cs::Vector{RankOneTensor{T}}) where {T} =
    ([c.λ for c in cs], factors_from_components(cs), length(cs))

_rankr_components_from_λU(λ, U) =
    [RankOneTensor(λ[k], [Vector(U[m][:, k]) for m = 1:length(U)]) for k = 1:length(λ)]

function pack_point_rank1_segre(λ::T, U::Vector{Vector{T}}) where {T<:AbstractFloat}
    λ_abs = abs(λ)
    λ_abs < eps(T) && (λ_abs = one(T))
    s = λ >= 0 ? one(T) : -one(T)
    parts = Vector{Vector{T}}(undef, length(U) + 1)
    parts[1] = T[λ_abs]
    @inbounds for i = 1:length(U)
        parts[i+1] = i == 1 ? s .* U[1] : U[i]
    end
    return parts
end

function pack_tangent_rank1_segre(ν::T, Udot::Vector{Vector{T}}) where {T<:AbstractFloat}
    parts = Vector{Vector{T}}(undef, length(Udot) + 1)
    parts[1] = T[ν]
    @inbounds for i = 1:length(Udot)
        parts[i+1] = Udot[i]
    end
    return parts
end

function pack_rankr_native(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    d = length(U)
    length(λ) == r || throw(DimensionMismatch("length(λ)=$(length(λ)) must equal r=$r"))
    for m = 1:d
        size(U[m], 2) == r ||
            throw(DimensionMismatch("U[$m] has $(size(U[m],2)) columns, expected r=$r"))
    end

    comps = Vector{Vector{Vector{T}}}(undef, r)
    @inbounds for k = 1:r
        λk = λ[k]
        Uk = Vector{Vector{T}}(undef, d)
        for m = 1:d
            u = Vector{T}(U[m][:, k])
            λk = _normalize_column_into_lambda!(u, λk)
            Uk[m] = u
        end
        comps[k] = pack_point_rank1_segre(λk, Uk)
    end
    return ArrayPartition(comps...)
end

function pack_rankr_native_tuple(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    ap = pack_rankr_native(λ, U, r)
    return parts_tuple(ap)
end

function pack_rankr_native(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_rankr_native(λ, U, r)
end

function pack_rankr_native_tuple(
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_rankr_native_tuple(λ, U, r)
end

function pack_rankr_canonical(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    return pack_rankr_canonical_tuple(λ, U, r)
end

function pack_rankr_canonical_tuple(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    d = length(U)
    length(λ) == r || throw(DimensionMismatch("length(λ)=$(length(λ)) must equal r=$r"))
    for m = 1:d
        size(U[m], 2) == r ||
            throw(DimensionMismatch("U[$m] has $(size(U[m],2)) columns, expected r=$r"))
    end

    λn = Vector{T}(undef, r)
    mode_cols = [Vector{Vector{T}}(undef, r) for _ = 1:d]
    @inbounds for k = 1:r
        λk = λ[k]
        for m = 1:d
            u = Vector{T}(U[m][:, k])
            λk = _normalize_column_into_lambda!(u, λk)
            mode_cols[m][k] = u
        end
        λn[k] = λk
    end
    mode_parts = map(mc -> (mc...,), mode_cols)
    return (λn, mode_parts...)
end

function pack_rankr_join_tuple(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    d = length(U)
    length(λ) == r || throw(DimensionMismatch("length(λ)=$(length(λ)) must equal r=$r"))
    for m = 1:d
        size(U[m], 2) == r ||
            throw(DimensionMismatch("U[$m] has $(size(U[m],2)) columns, expected r=$r"))
    end
    parts = Vector{Vector{T}}(undef, r * (d + 1))
    idx = 1
    @inbounds for k = 1:r
        parts[idx] = T[λ[k]]
        idx += 1
        for m = 1:d
            parts[idx] = Vector{T}(U[m][:, k])
            idx += 1
        end
    end
    return (parts...,)
end

function pack_rankr_canonical(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_rankr_canonical(λ, U, r)
end

function pack_rankr_canonical_tuple(
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_rankr_canonical_tuple(λ, U, r)
end

function pack_rankr_join_tuple(
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_rankr_join_tuple(λ, U, r)
end
