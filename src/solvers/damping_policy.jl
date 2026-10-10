# Damping configuration shared by nonlinear least-squares solver backends.
export DampingPolicy

"""Multiplicative damping update and acceptance policy.

The solver backend decides how damping enters its linear system. This policy
only owns bounds, update factors, acceptance thresholds, and the maximum
number of trial steps. `reduction_threshold=Inf` disables damping reduction.
"""
struct DampingPolicy
    initial::Float64
    minimum::Float64
    maximum::Float64
    increase_factor::Float64
    reduction_factor::Float64
    acceptance_threshold::Float64
    increase_threshold::Float64
    reduction_threshold::Float64
    max_trials::Int
end

function DampingPolicy(;
    initial::Real,
    minimum::Real = 0,
    maximum::Real = Inf,
    increase_factor::Real,
    reduction_factor::Real,
    acceptance_threshold::Real,
    increase_threshold::Real,
    reduction_threshold::Real,
    max_trials::Integer = 1,
)
    initial, minimum, maximum = Float64(initial), Float64(minimum), Float64(maximum)
    increase_factor, reduction_factor = Float64(increase_factor), Float64(reduction_factor)
    acceptance_threshold, increase_threshold, reduction_threshold =
        Float64(acceptance_threshold),
        Float64(increase_threshold),
        Float64(reduction_threshold)
    isfinite(initial) && isfinite(minimum) ||
        throw(ArgumentError("Initial and minimum damping must be finite."))
    0 <= minimum <= initial <= maximum ||
        throw(ArgumentError("Require 0 <= minimum <= initial <= maximum."))
    isfinite(increase_factor) && increase_factor > 1 ||
        throw(ArgumentError("Damping increase factor must be finite and exceed one."))
    0 < reduction_factor <= 1 ||
        throw(ArgumentError("Damping reduction factor must lie in (0, 1]."))
    0 <= acceptance_threshold <= increase_threshold <= 1 ||
        throw(ArgumentError("Acceptance threshold must not exceed the increase threshold."))
    reduction_threshold == Inf ||
        increase_threshold <= reduction_threshold <= 1 ||
        throw(
            ArgumentError("Reduction threshold must be in [increase_threshold, 1] or Inf."),
        )
    max_trials > 0 || throw(ArgumentError("Damping max_trials must be positive."))
    return DampingPolicy(
        Float64(initial),
        Float64(minimum),
        Float64(maximum),
        Float64(increase_factor),
        Float64(reduction_factor),
        Float64(acceptance_threshold),
        Float64(increase_threshold),
        Float64(reduction_threshold),
        Int(max_trials),
    )
end

function _damping_bounds(policy::DampingPolicy, ::Type{T}) where {T<:AbstractFloat}
    lower = T(policy.minimum)
    lower < policy.minimum && (lower = nextfloat(lower))
    upper = min(T(policy.maximum), floatmax(T))
    upper > policy.maximum && (upper = prevfloat(upper))
    # Preserve positive damping down to the smallest representable value,
    # without imposing a problem-independent epsilon scale.
    lower = max(lower, nextfloat(zero(T)))
    isfinite(lower) && lower <= upper ||
        throw(ArgumentError("Damping bounds contain no positive finite value of $T."))
    return lower, upper
end

function _increase_damping(policy::DampingPolicy, damping::T) where {T<:AbstractFloat}
    lower, upper = _damping_bounds(policy, T)
    return clamp(damping * T(policy.increase_factor), lower, upper)
end

function _update_accepted_damping(
    policy::DampingPolicy,
    damping::T,
    ratio::T,
) where {T<:AbstractFloat}
    lower, upper = _damping_bounds(policy, T)
    if ratio > T(policy.reduction_threshold)
        return clamp(damping * T(policy.reduction_factor), lower, upper)
    elseif ratio < T(policy.increase_threshold)
        return _increase_damping(policy, damping)
    end
    return clamp(damping, lower, upper)
end
