# Policies shared by inexact tangent-space linear solves.
export AbstractInnerTolerance,
    RelativeResidualTolerance,
    AdaptiveResidualTolerance,
    AbsoluteResidualTolerance,
    LMInnerOptions

abstract type AbstractInnerTolerance end

"""A fixed relative linear residual tolerance in `(0, 1)`."""
struct RelativeResidualTolerance <: AbstractInnerTolerance
    tolerance::Float64
    function RelativeResidualTolerance(tolerance::Real = 1e-2)
        0 < tolerance < 1 || throw(ArgumentError("Relative tolerance must lie in (0, 1)."))
        new(Float64(tolerance))
    end
end

"""Bounded forcing `clamp(scale * ‖grad f‖^power, minimum, maximum)`.

LM uses the unnormalized objective gradient to make this policy independent
of `normalized_objective`. The effective lower bound is at least `eps(T)`.
"""
struct AdaptiveResidualTolerance <: AbstractInnerTolerance
    maximum::Float64
    minimum::Float64
    scale::Float64
    power::Float64
    function AdaptiveResidualTolerance(;
        maximum::Real = 1e-2,
        minimum::Real = 1e-8,
        scale::Real = 1,
        power::Real = 0.5,
    )
        0 < minimum <= maximum < 1 ||
            throw(ArgumentError("Require 0 < minimum <= maximum < 1."))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("Forcing scale must be finite and positive."))
        0 < power <= 1 || throw(ArgumentError("Forcing power must lie in (0, 1]."))
        new(Float64(maximum), Float64(minimum), Float64(scale), Float64(power))
    end
end

"""Absolute linear residual tolerance, primarily for controlled comparisons."""
struct AbsoluteResidualTolerance <: AbstractInnerTolerance
    tolerance::Float64
    function AbsoluteResidualTolerance(tolerance::Real = 1e-16)
        isfinite(tolerance) && tolerance > 0 ||
            throw(ArgumentError("Absolute tolerance must be finite and positive."))
        new(Float64(tolerance))
    end
end

@inline _bounded_forcing(g, maximum, minimum, scale, power) =
    clamp(scale * g^power, minimum, maximum)

_inner_tolerance(policy::RelativeResidualTolerance, g::T) where {T} =
    max(T(policy.tolerance), eps(T))
_inner_tolerance(policy::AbsoluteResidualTolerance, g::T) where {T} = T(policy.tolerance)
function _inner_tolerance(policy::AdaptiveResidualTolerance, g::T) where {T}
    return _bounded_forcing(
        g,
        max(T(policy.maximum), eps(T)),
        max(T(policy.minimum), eps(T)),
        T(policy.scale),
        T(policy.power),
    )
end

"""Configure the LM tangent CR solve. Each solve starts from zero.

`maxiter=nothing` uses `max(20*dim(M), 200)` as a safety cap. Accuracy is
controlled by `tolerance`, independently of the outer stopping criteria.
"""
struct LMInnerOptions{P<:AbstractInnerTolerance}
    tolerance::P
    maxiter::Union{Nothing,Int}
    function LMInnerOptions(;
        tolerance::P = AdaptiveResidualTolerance(),
        maxiter::Union{Nothing,Integer} = nothing,
    ) where {P<:AbstractInnerTolerance}
        isnothing(maxiter) ||
            maxiter > 0 ||
            throw(ArgumentError("Inner maxiter must be positive."))
        new{P}(tolerance, isnothing(maxiter) ? nothing : Int(maxiter))
    end
end
