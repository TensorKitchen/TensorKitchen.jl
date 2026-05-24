# cpd/core/cp_normalization.jl — Normalization utilities for CP factor matrices and CPDPoint containers.
export AbstractNormalizationPolicy,
    NoNormalization,
    SeparateLambdaNormalization,
    NonnegativeSeparateLambdaNormalization,
    normalize_components!,
    normalize_components

abstract type AbstractNormalizationPolicy end

struct NoNormalization <: AbstractNormalizationPolicy end
struct SeparateLambdaNormalization <: AbstractNormalizationPolicy end
struct NonnegativeSeparateLambdaNormalization <: AbstractNormalizationPolicy end

_normalization_policy(::NoNormalization) = NoNormalization()
_normalization_policy(::SeparateLambdaNormalization) = SeparateLambdaNormalization()
_normalization_policy(::NonnegativeSeparateLambdaNormalization) =
    NonnegativeSeparateLambdaNormalization()
_normalization_policy(::Nothing) = NoNormalization()

function _normalization_policy(policy::Symbol)
    policy === :none && return NoNormalization()
    policy === :lambda_separate && return SeparateLambdaNormalization()
    policy === :nn_lambda_separate && return NonnegativeSeparateLambdaNormalization()
    throw(
        ArgumentError(
            "Unknown normalization=$policy. Use :none, :lambda_separate, or :nn_lambda_separate.",
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
    @inbounds for m in eachindex(factors)
        size(factors[m], 2) == r || throw(
            DimensionMismatch("factor $m has $(size(factors[m], 2)) columns, expected $r"),
        )
    end

    return _normalize_components_policy!(factors, lambda, policy)
end

_normalize_components_policy!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    ::NoNormalization,
) where {T<:AbstractFloat} = factors

"""
    NonnegativeSeparateLambdaNormalization

Gauge fixing for nonnegative CPD/NNCPD manifold refinement.

Applied in decoded nonnegative coordinates (`cpd_point` → normalize → `pack_cpd_point`):
column norms are absorbed into `λ` while preserving the represented tensor. Intended
for `:softplus_metric` / squaring geometries where latent scale can drift without
changing the tensor much.
"""
function _normalize_components_policy!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    ::NonnegativeSeparateLambdaNormalization,
) where {T<:AbstractFloat}
    r = length(lambda)
    d = length(factors)
    @inbounds for k in eachindex(lambda)
        lambda[k] = max(lambda[k], zero(T))
        for m in eachindex(factors)
            col = @view factors[m][:, k]
            col .= max.(col, zero(T))
        end
    end
    return _normalize_components_policy!(factors, lambda, SeparateLambdaNormalization())
end

function _normalize_components_policy!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    ::SeparateLambdaNormalization,
) where {T<:AbstractFloat}
    r = length(lambda)
    d = length(factors)
    @inbounds for k in eachindex(lambda)
        scale = lambda[k]
        for m in eachindex(factors)
            col = @view factors[m][:, k]
            scale = _normalize_column_into_lambda!(col, scale)
        end
        lambda[k] = scale
    end
    return factors
end

function _normalize_components_policy!(
    factors::Vector{Matrix{T}},
    lambda::Vector{T},
    policy::AbstractNormalizationPolicy,
) where {T<:AbstractFloat}
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
