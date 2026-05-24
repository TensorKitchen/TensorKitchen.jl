# cpd/core/cp_points.jl — Canonical CP point container + strict CP point normalization/validation
export CPDPoint, lambda

# ========== Canonical CPDPoint ==========

"""
    CPDPoint{T}

Explicit CP representation storing the component weights and factor matrices.
This is a normalization-friendly container independent of any manifold layout.

`CPDPoint` is primarily a backend/intermediate representation used by the CPD
pipeline for normalization, diagnostics, and postprocessing. It is not the
native optimization point used by manifold solvers. To convert between solver
points and this canonical CP representation, use [`cpd_point`](@ref) and
[`pack_cpd_point`](@ref).
"""
struct CPDPoint{T<:AbstractFloat}
    lambda::Vector{T}
    factors::Vector{Matrix{T}}
end

lambda(point::CPDPoint) = point.lambda
weights(point::CPDPoint) = lambda(point)
factors(point::CPDPoint) = point.factors

"""
    CPDPoint(lambda, factors)

Construct a layout-independent CP representation from explicit weights and
factor matrices.

This constructor is useful for backend utilities such as normalization or
diagnostics, and for advanced user workflows that want a canonical CP
representation without committing to a particular solver/manifold layout.
"""
function CPDPoint(
    lambda::AbstractVector{T},
    factors::Vector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    isempty(factors) &&
        throw(ArgumentError("CPDPoint requires at least one factor matrix."))
    r = length(lambda)
    mats = Vector{Matrix{T}}(undef, length(factors))
    @inbounds for m in eachindex(factors)
        size(factors[m], 2) == r || throw(
            DimensionMismatch("factor $m has $(size(factors[m], 2)) columns, expected $r"),
        )
        mats[m] = Matrix{T}(factors[m])
    end
    return CPDPoint{T}(collect(lambda), mats)
end

"""
    cpd_point(result::CPDResult)

Extract a [`CPDPoint`](@ref) from a completed CPD result.

This is the user-facing shortcut for obtaining the backend canonical CP
representation of a finished decomposition.
"""
cpd_point(r::CPDResult{T}) where {T<:AbstractFloat} = CPDPoint(weights(r), factors(r))

# ========== Strict internal point normalization ==========

@inline function _normalize_column_into_lambda!(u::AbstractVector{T}, λk::T) where {T}
    nu = norm(u)
    nu <= eps(T) && (fill!(u, zero(T)); u[1] = one(T); nu = one(T))
    λk *= nu
    u ./= nu
    return λk
end

"""
    normalize_rank1_segre_point(p, dims)

Normalize a rank-1 Segre point into strict internal layout `Vector{Vector}`:
`[[λ], u₁, ..., u_d]`.
"""
function normalize_rank1_segre_point(p, dims::NTuple{N,Int}) where {N}
    parts = parts_tuple(p)
    length(parts) == N + 1 ||
        throw(DimensionMismatch("expected $(N+1) Segre parts, got $(length(parts))"))
    λpart0 = _unwrap_part(parts[1])
    λpart0 isa AbstractVector ||
        throw(DimensionMismatch("Segre λ part must be a vector, got $(typeof(λpart0))"))
    length(λpart0) == 1 ||
        throw(DimensionMismatch("Segre λ part must have length 1, got $(length(λpart0))"))
    T = eltype(λpart0)
    out = Vector{Vector{T}}(undef, N + 1)
    λ = isfinite(λpart0[1]) ? λpart0[1] : zero(T)
    flip_first = λ < 0
    out[1] = T[abs(λ)]
    @inbounds for m in eachindex(dims)
        u0 = _unwrap_part(parts[m+1])
        u0 isa AbstractVector || throw(
            DimensionMismatch("Segre mode $m part must be a vector, got $(typeof(u0))"),
        )
        length(u0) == dims[m] || throw(
            DimensionMismatch("Segre mode $m length $(length(u0)) != dims[$m]=$(dims[m])"),
        )
        u = Vector{T}(u0)
        (m == 1 && flip_first) && (u .*= -one(T))
        nu = norm(u)
        (!isfinite(nu) || nu <= eps(T)) && (fill!(u, zero(T)); u[1] = one(T))
        out[m+1] = u
    end
    return out
end

"""
    normalize_rankr_native_point(p, dims, r)

Normalize a rank-r point on `ProductManifold(Manifolds.Segre(...), ...)`
into strict tuple layout:
`(p₁, ..., pᵣ)` where each `pₖ = [[λₖ], u₁ₖ, ..., u_dₖ]`.
"""
function normalize_rankr_native_point(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = parts_tuple(p)
    length(parts) == r ||
        throw(DimensionMismatch("expected $r Segre components, got $(length(parts))"))
    comps = [normalize_rank1_segre_point(parts[k], dims) for k in eachindex(parts)]
    return (comps...,)
end

"""
    normalize_rankr_canonical_point(p, dims, r)

Normalize a canonical rank-r point into strict tuple layout:
`(λ, mode₁, ..., mode_d)` where each `mode_m` is a tuple
`(u_{m,1}, ..., u_{m,r})`.
"""
function normalize_rankr_canonical_point(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = parts_tuple(p)
    length(parts) == N + 1 ||
        throw(DimensionMismatch("expected $(N+1) canonical parts, got $(length(parts))"))

    λ0 = _unwrap_part(parts[1])
    λ0 isa AbstractVector ||
        throw(DimensionMismatch("canonical λ part must be a vector, got $(typeof(λ0))"))
    length(λ0) == r || throw(DimensionMismatch("expected λ length $r, got $(length(λ0))"))
    λ = collect(λ0)
    T = eltype(λ)

    mode_parts = ntuple(
        m -> begin
            mode_m0 = _unwrap_part(parts[m+1])
            mode_m = mode_m0 isa Tuple ? mode_m0 : Tuple(mode_m0)
            length(mode_m) == r || throw(
                DimensionMismatch("mode $m has $(length(mode_m)) vectors, expected $r"),
            )
            cols = Vector{Vector{T}}(undef, r)
            for k in eachindex(cols)
                uk0 = _unwrap_part(mode_m[k])
                uk0 isa AbstractVector || throw(
                    DimensionMismatch(
                        "mode $m, component $k must be a vector, got $(typeof(uk0))",
                    ),
                )
                length(uk0) == dims[m] || throw(
                    DimensionMismatch(
                        "mode $m, component $k has length $(length(uk0)), expected $(dims[m])",
                    ),
                )
                cols[k] = Vector{T}(uk0)
            end
            return Tuple(cols)
        end,
        N,
    )
    return (λ, mode_parts...)
end

"""
    normalize_rankr_join_point(p, dims, r)

Normalize a join rank-r point into strict flattened tuple layout:
`([λ₁], u₁₁, ..., u_d₁, [λ₂], ..., u_dᵣ)`.
"""
function normalize_rankr_join_point(p, dims::NTuple{N,Int}, r::Int) where {N}
    parts = parts_tuple(p)
    expected = r * (N + 1)
    length(parts) == expected ||
        throw(DimensionMismatch("expected $expected join parts, got $(length(parts))"))

    λ1 = _unwrap_part(parts[1])
    λ1 isa AbstractVector ||
        throw(DimensionMismatch("join λ[1] must be a vector, got $(typeof(λ1))"))
    length(λ1) == 1 ||
        throw(DimensionMismatch("join λ[1] must have length 1, got $(length(λ1))"))
    T = eltype(λ1)

    out = Vector{Vector{T}}(undef, expected)
    idx = 1
    @inbounds for k in eachindex(Base.OneTo(r))
        λ0 = _unwrap_part(parts[idx])
        λ0 isa AbstractVector ||
            throw(DimensionMismatch("join λ[$k] must be a vector, got $(typeof(λ0))"))
        length(λ0) == 1 ||
            throw(DimensionMismatch("join λ[$k] must have length 1, got $(length(λ0))"))
        out[idx] = Vector{T}(λ0)
        idx += 1
        for m in eachindex(dims)
            u0 = _unwrap_part(parts[idx])
            u0 isa AbstractVector || throw(
                DimensionMismatch(
                    "join mode $m, component $k must be a vector, got $(typeof(u0))",
                ),
            )
            length(u0) == dims[m] || throw(
                DimensionMismatch(
                    "join mode $m, component $k has length $(length(u0)), expected $(dims[m])",
                ),
            )
            out[idx] = Vector{T}(u0)
            idx += 1
        end
    end
    return (out...,)
end
