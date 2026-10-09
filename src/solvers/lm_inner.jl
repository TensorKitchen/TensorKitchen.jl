# The only layer that depends on Manopt's CR lifecycle and stopping state.
mutable struct _LMInnerStopping{T,P} <: Manopt.StoppingCriterion
    policy::P
    relative::Manopt.StopWhenRelativeResidualLess{T}
    objective_scale::T
    maxiter::Int
    tolerance::T
    initial_norm::T
    final_norm::T
    at_iteration::Int
    reason::Symbol
end

function _LMInnerStopping(
    ::Type{T},
    options::InnerSolveOptions,
    dimension,
    objective_scale,
) where {T}
    cap = _inner_maxiter(options, dimension)
    return _LMInnerStopping(
        options.tolerance,
        Manopt.StopWhenRelativeResidualLess(one(T), T(1e-2)),
        T(objective_scale),
        cap,
        zero(T),
        zero(T),
        zero(T),
        -1,
        :not_started,
    )
end

function (stop::_LMInnerStopping{T})(problem, state, k) where {T}
    TpM = Manopt.get_manifold(problem)
    M, p = ManifoldsBase.base_manifold(TpM), ManifoldsBase.base_point(TpM)
    stop.final_norm = norm(M, p, state.r)
    if k == 0
        # With zero start, the already initialized residual equals -b. Avoid
        # re-evaluating the vector field merely to initialize relative stopping.
        stop.initial_norm = stop.final_norm
        stop.tolerance =
            _inner_tolerance(stop.policy, T(stop.initial_norm * stop.objective_scale))
        stop.relative.c = stop.initial_norm
        stop.relative.ε = stop.tolerance
        stop.relative.at_iteration = -1
        stop.at_iteration = -1
        stop.reason = :running
    end
    stop.relative.norm_r = stop.final_norm
    reason = if !isfinite(stop.final_norm)
        :nonfinite_residual
    elseif iszero(stop.final_norm)
        :zero_residual
    elseif stop.policy isa AbsoluteResidualTolerance && stop.final_norm <= stop.tolerance
        :absolute_tolerance
    elseif !(stop.policy isa AbsoluteResidualTolerance) &&
           k > 0 &&
           stop.relative(problem, state, k)
        :relative_tolerance
    elseif k >= stop.maxiter
        :maxiter
    else
        :running
    end
    stop.reason = reason
    if reason != :running
        stop.at_iteration = k
        return true
    end
    return false
end

Manopt.get_reason(stop::_LMInnerStopping) =
    stop.at_iteration < 0 ? "" :
    "LM inner solve stopped with $(stop.reason) after $(stop.at_iteration) iterations.\n"
Manopt.indicates_convergence(stop::_LMInnerStopping) =
    stop.reason in (:zero_residual, :absolute_tolerance, :relative_tolerance)

mutable struct _LMInnerTrace{T}
    iterations::Vector{Int}
    tolerance::Vector{T}
    initial_residual::Vector{T}
    final_residual::Vector{T}
    reason::Vector{Symbol}
    seconds::Vector{Float64}
    started::UInt64
    timed::Bool
end
_LMInnerTrace(::Type{T}, timed) where {T} =
    _LMInnerTrace(Int[], T[], T[], T[], Symbol[], Float64[], UInt64(0), timed)

function _lm_cr_state(M, p, objective, options::InnerSolveOptions, objective_scale, trace)
    T = _scalar_eltype(p)
    TpM = TangentSpace(_lm_subproblem_manifold(M), p)
    stop = _LMInnerStopping(T, options, manifold_dimension(M), objective_scale)
    callbacks = Dict{Symbol,Function}(
        :BeforeInit =>
            (problem, state, k) ->
                (trace.started = trace.timed ? time_ns() : UInt64(0)),
        :Stop => function (problem, state, k)
            push!(trace.iterations, k)
            push!(trace.tolerance, stop.tolerance)
            push!(trace.initial_residual, stop.initial_norm)
            push!(trace.final_residual, stop.final_norm)
            push!(trace.reason, stop.reason)
            push!(trace.seconds, trace.timed ? (time_ns() - trace.started) / 1e9 : 0.0)
        end,
    )
    # Provide ALL buffers explicitly: constructor defaults otherwise evaluate
    # the objective before the actual CR initialization, even with X=0.
    z() = ManifoldsBase.zero_vector(M, p)
    state = Manopt.ConjugateResidualState(
        TpM,
        objective;
        X = z(),
        r = z(),
        d = z(),
        Ar = z(),
        Ad = z(),
        α = zero(T),
        β = zero(T),
        warm_start = false,
        stopping_criterion = stop,
        callbacks,
    )
    return Manopt.DefaultManoptProblem(TpM, objective), state
end

function _lm_inner_info(trace::_LMInnerTrace, options::InnerSolveOptions, maxiter::Int)
    convergence = [
        reason in (:zero_residual, :absolute_tolerance, :relative_tolerance) for
        reason in trace.reason
    ]
    return (
        inner = _inner_history_info(
            trace.iterations,
            trace.tolerance,
            trace.initial_residual,
            trace.final_residual,
            trace.reason;
            solver = :cr,
            policy = options.tolerance,
            maxiter,
            converged = convergence,
            seconds = trace.timed ? trace.seconds : nothing,
        ),
    )
end
