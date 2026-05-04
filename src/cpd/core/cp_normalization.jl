# cpd/core/cp_normalization.jl — Normalization utilities for CP factor matrices and CPDPoint containers.
export AbstractNormalizationPolicy,
    NoNormalization,
    SeparateLambdaNormalization,
    LastModeNormalization,
    EvenDistributionNormalization,
    normalize_components!,
    normalize_components

abstract type AbstractNormalizationPolicy end

struct NoNormalization <: AbstractNormalizationPolicy end
struct SeparateLambdaNormalization <: AbstractNormalizationPolicy end
struct LastModeNormalization <: AbstractNormalizationPolicy end
struct EvenDistributionNormalization <: AbstractNormalizationPolicy end

_normalization_policy(::NoNormalization) = NoNormalization()
_normalization_policy(::SeparateLambdaNormalization) = SeparateLambdaNormalization()
_normalization_policy(::LastModeNormalization) = LastModeNormalization()
_normalization_policy(::EvenDistributionNormalization) = EvenDistributionNormalization()
_normalization_policy(::Nothing) = NoNormalization()

function _normalization_policy(policy::Symbol)
    policy === :none && return NoNormalization()
    policy === :lambda_separate && return SeparateLambdaNormalization()
    policy === :last_mode && return LastModeNormalization()
    policy === :distribute_evenly && return EvenDistributionNormalization()
    throw(
        ArgumentError(
            "Unknown normalization=$policy. Use :none, :lambda_separate, :last_mode, or :distribute_evenly.",
        ),
    )
end

function _normalization_policy(policy)
    throw(
        ArgumentError(
            "Unsupported normalization specification $(typeof(policy)). " *
            "Use a normalization policy object or symbol.",
        ),
    )
end

@inline _sign_or_zero(x::T) where {T<:AbstractFloat} = iszero(x) ? zero(T) : sign(x)

@inline function _safe_column_norm!(u::AbstractVector{T}) where {T<:AbstractFloat}
    nu = norm(u)
    if !isfinite(nu) || nu <= eps(T)
        fill!(u, zero(T))
        u[1] = one(T)
        return one(T)
    end
    return nu
end

"""
    normalize_components!(factors, λ, policy=SeparateLambdaNormalization())

Normalize CP factor matrices and weights in place according to `policy`, preserving
the represented CP tensor.
"""
function normalize_components!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    policy::AbstractNormalizationPolicy = SeparateLambdaNormalization(),
) where {T<:AbstractFloat}
    isempty(factors) && return factors
    r = length(lambda)
    d = length(factors)
    @inbounds for m = 1:d
        size(factors[m], 2) == r || throw(
            DimensionMismatch("factor $m has $(size(factors[m], 2)) columns, expected $r"),
        )
    end

    if policy isa NoNormalization
        return factors
    elseif policy isa SeparateLambdaNormalization
        @inbounds for k = 1:r
            scale = lambda[k]
            for m = 1:d
                col = @view factors[m][:, k]
                scale = _normalize_column_into_lambda!(col, scale)
            end
            lambda[k] = scale
        end
        return factors
    elseif policy isa LastModeNormalization
        last_mode = d
        @inbounds for k = 1:r
            total_scale = lambda[k]
            for m = 1:d
                col = @view factors[m][:, k]
                nu = _safe_column_norm!(col)
                col ./= nu
                total_scale *= nu
            end
            mag = abs(total_scale)
            factors[last_mode][:, k] .*= mag
            lambda[k] = _sign_or_zero(total_scale)
        end
        return factors
    elseif policy isa EvenDistributionNormalization
        @inbounds for k = 1:r
            total_scale = lambda[k]
            for m = 1:d
                col = @view factors[m][:, k]
                nu = _safe_column_norm!(col)
                col ./= nu
                total_scale *= nu
            end
            mag = abs(total_scale)
            scale = mag <= eps(T) ? zero(T) : mag^(inv(T(d)))
            for m = 1:d
                factors[m][:, k] .*= scale
            end
            lambda[k] = _sign_or_zero(total_scale)
        end
        return factors
    end

    throw(ArgumentError("Unsupported normalization policy $(typeof(policy))."))
end

normalize_components!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    policy,
) where {T<:AbstractFloat} =
    normalize_components!(factors, lambda, _normalization_policy(policy))

function normalize_components(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    policy::AbstractNormalizationPolicy = SeparateLambdaNormalization(),
) where {T<:AbstractFloat}
    lambda_copy = copy(lambda)
    factor_copy = [copy(F) for F in factors]
    normalize_components!(factor_copy, lambda_copy, policy)
    return factor_copy, lambda_copy
end

normalize_components(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    policy,
) where {T<:AbstractFloat} =
    normalize_components(factors, lambda, _normalization_policy(policy))

"""
    normalize_components!(point::CPDPoint, policy=SeparateLambdaNormalization())

Normalize a canonical [`CPDPoint`](@ref) in place.

This is the preferred backend entry point for normalization, since it is
independent of solver/manifold layout. To normalize an actual optimization
iterate, combine it with [`cpd_point`](@ref) and [`pack_cpd_point`](@ref), or
use [`post_step!`](@ref).
"""
normalize_components!(
    point::CPDPoint{T},
    policy::AbstractNormalizationPolicy = SeparateLambdaNormalization(),
) where {T<:AbstractFloat} =
    (normalize_components!(point.factors, point.lambda, policy); point)

normalize_components!(point::CPDPoint{T}, policy) where {T<:AbstractFloat} =
    normalize_components!(point, _normalization_policy(policy))

"""
    normalize_components(point::CPDPoint, policy=SeparateLambdaNormalization())

Return a normalized copy of a canonical [`CPDPoint`](@ref).
"""
function normalize_components(
    point::CPDPoint{T},
    policy::AbstractNormalizationPolicy = SeparateLambdaNormalization(),
) where {T<:AbstractFloat}
    q = CPDPoint(copy(point.lambda), [copy(F) for F in point.factors])
    normalize_components!(q, policy)
    return q
end

normalize_components(point::CPDPoint{T}, policy) where {T<:AbstractFloat} =
    normalize_components(point, _normalization_policy(policy))

"""
    normalize_factors!(U, λ)

Backward-compatible alias for full lambda separation.
"""
normalize_factors!(U::Vector{Matrix{T}}, λ::Vector{T}) where {T<:AbstractFloat} =
    normalize_components!(U, λ, SeparateLambdaNormalization())
