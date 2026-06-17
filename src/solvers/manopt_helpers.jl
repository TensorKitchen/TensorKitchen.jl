
######################## 
# This file contains helper functions for communicating with Manopt
########################

# Sink Manopt's built-in debug text while still letting debug actions run.
struct _SolverDebugSink <: IO end
Base.isopen(::_SolverDebugSink) = true
Base.write(::_SolverDebugSink, ::UInt8) = 1
Base.write(::_SolverDebugSink, s::Union{String,SubString{String}}) = sizeof(s)
Base.unsafe_write(::_SolverDebugSink, ::Ptr{UInt8}, n::UInt) = Int(n)

# Shared no-op IO used by Manopt debug groups.
const _SOLVER_DEBUG_SINK = _SolverDebugSink()

# Stop when both the relative cost change and Riemannian gradient norm are small.
mutable struct StopWhenCostRelChangeAndGradientLess{T<:Real} <: Manopt.StoppingCriterion
    tol_cost::T
    tol_grad::T
    prev_cost::T
    last_cost_rel_change::T
    last_grad_norm::T
    at_iteration::Int
end

# Initialize the dual cost-change/gradient stopping rule with empty history.
function StopWhenCostRelChangeAndGradientLess(tol_cost::T, tol_grad::T) where {T<:Real}
    return StopWhenCostRelChangeAndGradientLess{T}(
        tol_cost,
        tol_grad,
        T(Inf),
        T(Inf),
        T(Inf),
        -1,
    )
end

# Update the dual stopping rule from the current Manopt problem/state.
function (c::StopWhenCostRelChangeAndGradientLess)(problem, state, i)
    if i == 0
        c.prev_cost = Manopt.get_cost(problem, Manopt.get_iterate(state))
        c.last_cost_rel_change = oftype(c.tol_cost, Inf)
        c.last_grad_norm = oftype(c.tol_grad, Inf)
        c.at_iteration = -1
        return false
    end
    M = Manopt.get_manifold(problem)
    p = Manopt.get_iterate(state)
    cost_val = Manopt.get_cost(problem, p)
    grad_val = Manopt.get_gradient(problem, p)
    grad_norm = norm(M, p, grad_val)
    rel_change = abs(c.prev_cost - cost_val) / max(abs(c.prev_cost), one(cost_val))
    c.prev_cost = cost_val
    c.last_cost_rel_change = rel_change
    c.last_grad_norm = grad_norm
    if rel_change < c.tol_cost && grad_norm < c.tol_grad
        c.at_iteration = i
        return true
    end
    return false
end

# Explain why the dual stopping rule stopped, for Manopt status reporting.
function Manopt.get_reason(c::StopWhenCostRelChangeAndGradientLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the relative cost change ($(c.last_cost_rel_change)) " *
               "is below $(c.tol_cost) and the gradient norm ($(c.last_grad_norm)) " *
               "is below $(c.tol_grad).\n"
    end
    return ""
end

# Summarize the current dual stopping-rule state for Manopt displays.
function Manopt.status_summary(c::StopWhenCostRelChangeAndGradientLess)
    has_stopped = c.at_iteration >= 0
    status = has_stopped ? "reached" : "not reached"
    return "cost rel change < $(c.tol_cost) and |grad f| < $(c.tol_grad): $status"
end

# Mark the dual stopping rule as convergence, not failure or exhaustion.
Manopt.indicates_convergence(::StopWhenCostRelChangeAndGradientLess) = true

# Print the dual stopping rule compactly in Manopt diagnostics.
function Base.show(io::IO, c::StopWhenCostRelChangeAndGradientLess)
    return print(
        io,
        "StopWhenCostRelChangeAndGradientLess($(c.tol_cost), $(c.tol_grad))\n    $(Manopt.status_summary(c))",
    )
end


# Extract the final iterate across Manopt versions and nested state wrappers.
function _tk_get_solver_result(state)
    try
        return Manopt.get_solver_result(state)
    catch
    end
    while hasproperty(state, :state)
        state = state.state
    end
    for key in (:p, :x, :point)
        hasproperty(state, key) && return getproperty(state, key)
    end
    throw(
        ArgumentError(
            "Cannot extract point from state $(typeof(state)). Properties: $(propertynames(state))",
        ),
    )
end


# Match gradients/tangents to the point container Manopt is currently using.
@inline _align_layout_like_point(p, x) =
    hasproperty(p, :x) ?
    (hasproperty(x, :x) ? x : (x isa Tuple ? ArrayPartition(x...) : x)) :
    (hasproperty(x, :x) ? Tuple(getproperty(x, :x)) : x)


# Recursively convert tuple-like product points to ArrayPartition layout.
function _to_array_partition(x)
    if x isa ArrayPartition
        return ArrayPartition(map(_to_array_partition, x.x)...)
    elseif hasproperty(x, :x)
        return ArrayPartition(map(_to_array_partition, getproperty(x, :x))...)
    elseif x isa Tuple
        return ArrayPartition(map(_to_array_partition, x)...)
    end
    return x
end


# Adapt an initial point to the layout expected by the solver manifold.
function _solver_point(M, p0)
    M2 = _unwrap_solver_manifold(M)
    return M2 isa ProductManifold ? _to_array_partition(p0) : p0
end


# Detect pullback nonnegative geometries that need conservative line search.
function _contains_sqeuclidean_manifold(M)
    M2 = _unwrap_solver_manifold(M)
    if M2 isa SqEuclidean || M2 isa SoftplusEuclidean
        return true
    elseif M2 isa ProductManifold
        return any(_contains_sqeuclidean_manifold, M2.manifolds)
    elseif hasproperty(M2, :native) && (
        getproperty(M2, :native) isa SqEuclidean ||
        getproperty(M2, :native) isa SoftplusEuclidean
    )
        return true
    end
    return false
end

# Detect strict squaring geometries that require extra Armijo safeguards.
function _contains_strict_sqeuclidean_manifold(M)
    M2 = _unwrap_solver_manifold(M)
    if M2 isa SqEuclidean
        return true
    elseif M2 isa ProductManifold
        return any(_contains_strict_sqeuclidean_manifold, M2.manifolds)
    elseif hasproperty(M2, :native) && (getproperty(M2, :native) isa SqEuclidean)
        return true
    end
    return false
end


# Compute how many Armijo contractions are needed before alpha_min is reached.
function _armijo_max_decreases(initial_stepsize::Real, contraction::Real, alpha_min::Real)
    initial_stepsize <= alpha_min && return 0
    (contraction <= 0 || contraction >= 1) && return 1000
    n = floor(Int, log(alpha_min / initial_stepsize) / log(contraction))
    return max(n, 0)
end


# Estimate a guarded initial RGD step from a one-sided curvature probe.
function _adaptive_initial_stepsize(
    M,
    p0,
    model_grad,
    retraction_method,
    base_stepsize::T;
    alpha_min::T,
    scale_c::T = one(T),
    clamp_low_factor::T = T(0.1),
    clamp_high_factor::T = T(10),
    delta_scale::T = T(1e-3),
) where {T<:AbstractFloat}
    g0 = model_grad(M, p0)
    d = -copy(g0)
    dnorm = norm(M, p0, d)
    (!isfinite(dnorm) || dnorm <= sqrt(eps(T))) && return base_stepsize
    δ = delta_scale / max(dnorm, one(T))
    q = try
        retract(M, p0, δ .* d, retraction_method)
    catch
        return base_stepsize
    end
    _all_finite(q) || return base_stepsize
    gq = model_grad(M, q)
    _all_finite(gq) || return base_stepsize
    κ_num = inner(M, p0, gq .- g0, d)
    κ_den = δ * dnorm^2
    (!isfinite(κ_num) || !isfinite(κ_den) || κ_den <= eps(T)) && return base_stepsize
    κ = max(κ_num / κ_den, eps(T))
    α_raw = scale_c / κ
    α_low = max(alpha_min, clamp_low_factor * base_stepsize)
    α_high = clamp_high_factor * base_stepsize
    α = clamp(α_raw, α_low, α_high)
    if !isfinite(α)
        return base_stepsize
    end
    return α
end

# Check recursively that points, tangents, and nested manifold containers are finite.
function _all_finite(x)
    if x isa Number
        return isfinite(x)
    elseif hasproperty(x, :x)
        return all(_all_finite, getproperty(x, :x))
    elseif x isa AbstractArray
        return all(isfinite, x)
    elseif x isa Tuple
        return all(_all_finite, x)
    elseif x isa Manifolds.TuckerPoint
        return _all_finite(x.hosvd.core) && all(_all_finite, x.hosvd.U)
    elseif x isa Manifolds.TuckerTangentVector
        return _all_finite(getproperty(x, :Ċ)) && all(_all_finite, getproperty(x, :U̇))
    end
    try
        return all(isfinite, x)
    catch
        return false
    end
end


# Wrap a cost so invalid points return Inf instead of poisoning line search.
function _safe_cost_function(model_cost)
    return function (M, p)
        _all_finite(p) || return Inf
        c = model_cost(M, p)
        return isfinite(c) ? c : Inf
    end
end


# Wrap a gradient so its container layout follows the queried point layout.
function _layout_adapt_gradient(model_grad)
    return function (M, p)
        g = model_grad(M, p)
        return _align_layout_like_point(p, g)
    end
end

# Infer the scalar element type from nested points/tangents used by solvers.
function _scalar_eltype(p)
    if hasproperty(p, :x) || p isa AbstractVector || p isa Tuple
        parts = point_parts(p)
        isempty(parts) && return Float64
        return _scalar_eltype(first(parts))
    elseif p isa Manifolds.TuckerPoint
        return eltype(p.hosvd.core)
    elseif p isa Manifolds.TuckerTangentVector
        return eltype(getproperty(p, :Ċ))
    elseif p isa Real
        return typeof(p)
    else
        return eltype(p)
    end
end

# Read Manopt's cached gradient when the current state exposes it.
@inline function _solver_gradient(state)
    try
        return Manopt.get_gradient(state)
    catch
        return nothing
    end
end

# Scale cost and gradient when using relative error  
@inline _scale_solver_tangent(x::Number, scale::Real) = x * scale
_scale_solver_tangent(x::AbstractArray, scale::Real) = x .* scale
_scale_solver_tangent(x::ArrayPartition, scale::Real) =
    ArrayPartition(map(part -> _scale_solver_tangent(part, scale), x.x)...)
_scale_solver_tangent(x::Tuple, scale::Real) =
    map(part -> _scale_solver_tangent(part, scale), x)
function _scale_solver_tangent(x, scale::Real)
    try
        return x .* scale
    catch
        return scale * x
    end
end
function _relative_solver_functions(model_cost, model_grad, scale::Real)
    scale > 0 || return model_cost, model_grad, false
    scale == one(scale) && return model_cost, model_grad, false
    inv_scale = inv(scale)
    return (
        (M, p) -> model_cost(M, p) * inv_scale,
        (M, p) -> _scale_solver_tangent(model_grad(M, p), inv_scale),
        true,
    )
end

# Query Manopt's convergence flag through one local compatibility point.
@inline _solver_has_converged(state) = Manopt.has_converged(state)

# Recover the stopping iteration across Manopt versions and wrapped states.
function _solver_iterations(state, maxiter::Int)
    if isdefined(Manopt, :stopped_at)
        try
            k = Manopt.stopped_at(state)
            return k > 0 ? Int(k) : maxiter
        catch
        end
    end
    while hasproperty(state, :state)
        state = state.state
    end
    return hasproperty(state, :stop) && hasproperty(state.stop, :at_iteration) ?
           state.stop.at_iteration : maxiter
end

# Report which Manopt iteration API was used for solver metadata.
@inline function _solver_iteration_source()
    return isdefined(Manopt, :stopped_at) ? :stopped_at : :stop_at_iteration_fallback
end

# Convert a squared residual value to the public relative-error diagnostic.
_solver_rel_error(cost_for_error, ::Nothing, ::Type) = sqrt(cost_for_error)
_solver_rel_error(cost_for_error, normA2::Real, ::Type{T}) where {T} =
    _relative_error_frob_sq(cost_for_error, T(normA2))

# Build common solver result stats, with optional ||A||^2 for relative scaling.
function _solver_stats(
    model_cost,
    model_grad,
    M,
    p_opt,
    state,
    normA2::Union{Nothing,Real};
    tol_T,
    maxiter::Int,
    solver::Symbol,
    tiny_grad_tol = nothing,
    solver_info = (;),
    use_state_gradient::Bool = true,
)
    T = typeof(tol_T)
    final_cost = model_cost(M, p_opt)
    cost_for_error = max(T(0), T(2) * final_cost)
    rel_error = _solver_rel_error(cost_for_error, normA2, T)
    grad_state = use_state_gradient ? _solver_gradient(state) : nothing
    grad_from_state = !isnothing(grad_state)
    grad_final =
        isnothing(grad_state) ? model_grad(M, p_opt) :
        _align_layout_like_point(p_opt, grad_state)
    grad_norm = norm(M, p_opt, grad_final)
    iterations = _solver_iterations(state, maxiter)
    converged_grad =
        grad_norm < tol_T || (!isnothing(tiny_grad_tol) && grad_norm < tiny_grad_tol)
    converged_state = _solver_has_converged(state)
    solver_info = merge(
        solver_info,
        (
            gradient_source = grad_from_state ? :state : :recomputed,
            has_converged_state = converged_state,
            converged_by_gradient_threshold = converged_grad,
            iteration_source = _solver_iteration_source(),
        ),
    )
    return (
        point = p_opt,
        cost = final_cost,
        rel_error = rel_error,
        grad_norm = grad_norm,
        iterations = iterations,
        converged = converged_state,
        solver = solver,
        solver_info = solver_info,
    )
end

# Collect only active Manopt debug callbacks, dropping omitted hooks.
_solver_debug_callbacks(callbacks...) = Any[cb for cb in callbacks if !isnothing(cb)]

# Allow callers to pass `nothing` (e.g., when verbose/debug is omitted)
# Build debug actions when the verbose flag was omitted.
_solver_debug_actions(::Nothing, callbacks...) = _solver_debug_callbacks(callbacks...)

# Build Manopt debug actions and attach TensorKitchen callback hooks.
function _solver_debug_actions(verbose::Bool, callbacks...)
    callback_actions = _solver_debug_callbacks(callbacks...)
    if verbose
        io = _SOLVER_DEBUG_SINK
        init_group = Manopt.DebugGroup([
            Manopt.DebugDivider("Initial "; io, at_init = true),
            Manopt.DebugCost(; io, format = "f(x): %.6e", at_init = true),
            Manopt.DebugGradientNorm(; io, format = "|grad f(p)|:%.6e", at_init = true),
            Manopt.DebugDivider("\n"; io, at_init = true),
        ])
        iter_group = Manopt.DebugEvery(
            Manopt.DebugGroup([
                Manopt.DebugIteration(; io, format = "# %-6d"),
                Manopt.DebugDivider(" "; io, at_init = true),
                Manopt.DebugCost(; io, format = "f(x): %.6e", at_init = true),
                Manopt.DebugGradientNorm(; io, format = "|grad f(p)|:%.6e", at_init = true),
                Manopt.DebugDivider("\n"; io, at_init = true),
            ]),
            100,
        )
        iteration_actions = Any[iter_group]
        append!(iteration_actions, callback_actions)
        return Any[:Start=>Any[init_group], :Iteration=>iteration_actions]
    end
    return callback_actions
end

# Create a Manopt iteration callback that updates TensorKitchen progress output.
function _solver_progress_callback(
    progress,
    model_cost,
    model_grad,
    M;
    normA2 = nothing,
    diagnostics_recorder = nothing,
)
    progress isa NoMethodProgress && return nothing
    has_relative_scale = !isnothing(normA2) && normA2 > 0
    target_norm = has_relative_scale ? sqrt(normA2) : nothing
    return function (problem, state, k)
        k <= 0 && return nothing
        p = get_iterate(state)
        c = model_cost(M, p)
        g = model_grad(M, p)
        gnorm = norm(M, p, g)
        c_display = has_relative_scale ? sqrt(max(2 * c, zero(c))) : c
        gnorm_display = has_relative_scale ? gnorm * target_norm : gnorm
        showvalues = Any[("Iter", k), ("Cost", c_display), ("Grad norm", gnorm_display)]
        if !isnothing(diagnostics_recorder)
            step = diagnostics_recorder.accepted_stepsize_history
            trials = diagnostics_recorder.line_search_trial_history
            !isempty(step) && push!(showvalues, ("Accepted α", step[end]))
            !isempty(trials) && push!(showvalues, ("Line-search trials", trials[end]))
        end
        update_progress!(progress, k; showvalues)
        return nothing
    end
end

# Accumulate line-search, step-size, and evaluation diagnostics during solving.
mutable struct _SolverDiagnosticsRecorder
    first_accepted_stepsize::Float64
    min_accepted_stepsize::Float64
    first_line_search_trials::Int
    line_search_trial_count::Int
    function_evaluations::Int
    gradient_evaluations::Int
    prev_function_evaluations::Int
    prev_gradient_evaluations::Int
    line_search_enabled::Bool
    fallback_stepsize::Float64
    accepted_stepsize_history::Vector{Float64}
    line_search_trial_history::Vector{Int}
end

# Initialize a diagnostics recorder for solvers with or without line search.
function _SolverDiagnosticsRecorder(;
    line_search_enabled::Bool,
    fallback_stepsize::Real = NaN,
)
    return _SolverDiagnosticsRecorder(
        NaN,
        Inf,
        0,
        0,
        0,
        0,
        0,
        0,
        line_search_enabled,
        Float64(fallback_stepsize),
        Float64[],
        Int[],
    )
end

# Read Manopt objective evaluation counters defensively across configurations.
function _solver_eval_count(problem, sym::Symbol)
    try
        count = get_count(get_objective(problem), sym)
        return count < 0 ? 0 : Int(count)
    catch
        return 0
    end
end

# Create a Manopt callback that records per-iteration solver diagnostics.
function _solver_diagnostics_callback(recorder::_SolverDiagnosticsRecorder)
    return function (problem, state, k)
        fe = _solver_eval_count(problem, :Cost)
        ge = _solver_eval_count(problem, :Gradient)
        if k == 0
            recorder.function_evaluations = fe
            recorder.gradient_evaluations = ge
            recorder.prev_function_evaluations = fe
            recorder.prev_gradient_evaluations = ge
            return nothing
        end
        step =
            recorder.line_search_enabled ? get_last_stepsize(problem, state, k) :
            recorder.fallback_stepsize
        delta_fe = max(fe - recorder.prev_function_evaluations, 0)
        step_f = Float64(step)
        ls_trials = recorder.line_search_enabled ? max(delta_fe - 1, 0) : 0
        if isnan(recorder.first_accepted_stepsize)
            recorder.first_accepted_stepsize = step_f
            recorder.first_line_search_trials = ls_trials
        end
        recorder.min_accepted_stepsize = min(recorder.min_accepted_stepsize, step_f)
        recorder.line_search_trial_count += ls_trials
        push!(recorder.accepted_stepsize_history, step_f)
        push!(recorder.line_search_trial_history, ls_trials)
        recorder.function_evaluations = fe
        recorder.gradient_evaluations = ge
        recorder.prev_function_evaluations = fe
        recorder.prev_gradient_evaluations = ge
        return nothing
    end
end

# Convert recorded diagnostics to the public solver_info named tuple.
function _solver_info(recorder::_SolverDiagnosticsRecorder, iterations::Int)
    return (
        total_iterations = iterations,
        first_accepted_stepsize = recorder.first_accepted_stepsize,
        min_accepted_stepsize = recorder.min_accepted_stepsize,
        first_line_search_trials = recorder.first_line_search_trials,
        line_search_trial_count = recorder.line_search_trial_count,
        function_evaluations = recorder.function_evaluations,
        gradient_evaluations = recorder.gradient_evaluations,
        accepted_stepsize_history = recorder.accepted_stepsize_history,
        line_search_trial_history = recorder.line_search_trial_history,
    )
end

# Create a callback that normalizes/postprocesses iterates after each solver step.
function _solver_post_step_callback(
    model::AbstractDecompositionModel,
    M,
    normalization::AbstractNormalizationPolicy,
    solver_sym::Symbol,
)
    normalization isa NoNormalization && return nothing
    return function (problem, state, k)
        p_old = get_iterate(state)
        # Backend postprocessing (for example normalization) runs in canonical
        # CP coordinates and then packs back into the solver's current layout.
        p_new = post_step!(
            model,
            p_old;
            normalization,
            solver = solver_sym,
            problem,
            state,
            iteration = k,
        )
        p_new = _align_layout_like_point(p_old, p_new)
        p_new === p_old && return nothing
        set_iterate!(state, M, p_new)
        if solver_sym == :rcg && hasproperty(state, :X)
            get_gradient!(problem, state.X, get_iterate(state))
            if hasproperty(state, :δ)
                state.δ = -copy(M, get_iterate(state), state.X)
            end
            hasproperty(state, :β) && (state.β = zero(typeof(state.β)))
            if hasproperty(state, :coefficient) && hasproperty(state.coefficient, :storage)
                update_storage!(state.coefficient.storage, problem, state)
            end
        end
        return nothing
    end
end
