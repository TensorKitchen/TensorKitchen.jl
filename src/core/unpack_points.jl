# core/unpack_points.jl — Point unpacking and legacy/vector interop
export pack_point_rank1,
    unpack_point_rank1, pack_point_rankr, unpack_point_rankr, unpack_point_rankr_components

function unpack_rankr_native(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = normalize_rankr_native_point(p, dims, r)

    T = eltype(parts[1][1])
    d = length(dims)
    proto = parts[1][2]
    λ = similar(proto, T, r)
    U = [similar(proto, T, dims[m], r) for m = 1:d]

    @inbounds for k = 1:r
        comp = parts[k]
        length(comp) == d + 1 || throw(
            DimensionMismatch(
                "native component $k must have $(d+1) parts, got $(length(comp)).",
            ),
        )
        λ[k] = comp[1][1]
        for m = 1:d
            U[m][:, k] .= comp[m+1]
        end
    end
    return λ, U
end

function unpack_rankr_native_components(p, dims::NTuple{N,Int}, r::Int) where {N}
    return _rankr_components_from_λU(unpack_rankr_native(p, dims, r)...)
end

function unpack_rankr_canonical(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = normalize_rankr_canonical_point(p, dims, r)
    d = length(dims)
    λ = parts[1]
    T = eltype(λ)
    U = Vector{Matrix{T}}(undef, d)
    @inbounds for m = 1:d
        mode_m = parts[m+1]
        proto = mode_m[1]
        Um = similar(proto, T, dims[m], r)
        for k = 1:r
            uk = mode_m[k]
            length(uk) == dims[m] || throw(
                DimensionMismatch(
                    "mode $m, component $k has length $(length(uk)), expected $(dims[m])",
                ),
            )
            Um[:, k] .= uk
        end
        U[m] = Um
    end
    return λ, U
end

function unpack_rankr_join(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = normalize_rankr_join_point(p, dims, r)
    T = eltype(parts[1])
    proto = parts[2]
    λ = similar(proto, T, r)
    U = [similar(proto, T, dims[m], r) for m = 1:N]
    idx = 1
    @inbounds for k = 1:r
        λ[k] = parts[idx][1]
        idx += 1
        for m = 1:N
            U[m][:, k] .= parts[idx]
            idx += 1
        end
    end
    return λ, U
end

function pack_point_rank1(λ::T, U::Vector{Vector{T}}) where {T<:AbstractFloat}
    parts = Vector{Vector{T}}(undef, length(U) + 1)
    parts[1] = T[λ]
    @inbounds for i = 1:length(U)
        parts[i+1] = U[i]
    end
    return ArrayPartition(parts...)
end

function unpack_point_rank1(p, dims::NTuple{N,Int}) where {N}
    parts = point_parts(p)
    λ = parts[1][1]
    U = [
        begin
            pi = parts[i]
            if pi isa AbstractVector
                v = Vector{eltype(pi)}(undef, length(pi))
                copyto!(v, pi)
                v
            else
                pi
            end
        end for i = 2:(length(dims)+1)
    ]
    return λ, U
end

function pack_point_rankr(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    return pack_rankr_join_tuple(λ, U, r)
end

function pack_point_rankr_ap(
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    r::Int,
) where {T<:AbstractFloat}
    return ArrayPartition(pack_rankr_join_tuple(λ, U, r)...)
end

function unpack_point_rankr(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = point_parts(p)
    d, nparts = length(dims), length(parts)
    nparts == d + 1 && return unpack_rankr_canonical(parts, dims, r)
    nparts == r && return unpack_rankr_native(parts, dims, r)
    nparts == r * (d + 1) && return unpack_rankr_join(parts, dims, r)
    throw(
        DimensionMismatch(
            "unpack_point_rankr: unknown layout with $nparts parts. Expected $(d + 1), $r, or $(r * (d + 1)).",
        ),
    )
end

function pack_point_rankr(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_point_rankr(λ, U, r)
end

function pack_point_rankr_ap(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat}
    λ, U, r = _components_to_lambda_U(components)
    return pack_point_rankr_ap(λ, U, r)
end

function unpack_point_rankr_components(p, dims::NTuple{N,Int}, r::Int) where {N}
    return _rankr_components_from_λU(unpack_point_rankr(p, dims, r)...)
end

function unpack_point_rank1(
    p::AbstractVector{T},
    dims::NTuple{N,Int},
) where {T<:AbstractFloat,N}
    hasproperty(p, :x) && return unpack_point_rank1(point_parts(p), dims)
    λ̃ = p[1]
    U = Vector{Vector{T}}(undef, length(dims))
    idx = 2
    for m = 1:length(dims)
        n = dims[m]
        U[m] = p[idx:(idx+n-1)]
        idx += n
    end
    return λ̃, U
end

function unpack_point_rankr(
    p::AbstractVector{T},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    hasproperty(p, :x) && return unpack_point_rankr(point_parts(p), dims, r)
    d = length(dims)
    block = 1 + sum(dims)
    λ = similar(p, T, r)
    U = [similar(p, T, dims[m], r) for m = 1:d]
    for k = 1:r
        b = (k - 1) * block
        λ[k] = p[b+1]
        idx = b + 2
        for m = 1:d
            n = dims[m]
            U[m][:, k] = p[idx:(idx+n-1)]
            idx += n
        end
    end
    return λ, U
end

function pack_point_rank1_to_vector(λ̃::T, U::Vector{Vector{T}}) where {T}
    return vcat([λ̃], U...)
end

function pack_point_rankr_to_vector(λ::Vector{T}, U::Vector{Matrix{T}}, r::Int) where {T}
    return vcat((vcat(λ[k], (U[m][:, k] for m = 1:length(U))...) for k = 1:r)...)
end
