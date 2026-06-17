# solvers/rgd.jl — Riemannian Gradient Descent
export RGDSolver, RGDFixedSolver
using Manopt

struct _SolverDebugSink <: IO end
Base.isopen(::_SolverDebugSink) = true
Base.write(::_SolverDebugSink, ::UInt8) = 1
Base.write(::_SolverDebugSink, s::Union{String,SubString{String}}) = sizeof(s)
Base.unsafe_write(::_SolverDebugSink, ::Ptr{UInt8}, n::UInt) = Int(n)

const _SOLVER_DEBUG_SINK = _SolverDebugSink()

mutable struct StopWhenCostRelChangeAndGradientLess{T<:Real} <: Manopt.StoppingCriterion
    tol_cost::T
    tol_grad::T
    prev_cost::T
    last_cost_rel_change::T
    last_grad_norm::T
    at_iteration::Int
end

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

function Manopt.get_reason(c::StopWhenCostRelChangeAndGradientLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the relative cost change ($(c.last_cost_rel_change)) " *
               "is below $(c.tol_cost) and the gradient norm ($(c.last_grad_norm)) " *
               "is below $(c.tol_grad).\n"
    end
    return ""
end

function Manopt.status_summary(c::StopWhenCostRelChangeAndGradientLess)
    has_stopped = c.at_iteration >= 0
    status = has_stopped ? "reached" : "not reached"
    return "cost rel change < $(c.tol_cost) and |grad f| < $(c.tol_grad): $status"
end

Manopt.indicates_convergence(::StopWhenCostRelChangeAndGradientLess) = true

function Base.show(io::IO, c::StopWhenCostRelChangeAndGradientLess)
    return print(
        io,
        "StopWhenCostRelChangeAndGradientLess($(c.tol_cost), $(c.tol_grad))\n    $(Manopt.status_summary(c))",
    )
end


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


@inline _align_layout_like_point(p, x) =
    hasproperty(p, :x) ?
    (hasproperty(x, :x) ? x : (x isa Tuple ? ArrayPartition(x...) : x)) :
    (hasproperty(x, :x) ? Tuple(getproperty(x, :x)) : x)


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


function _solver_point(M, p0)
    M2 = _unwrap_solver_manifold(M)
    return M2 isa ProductManifold ? _to_array_partition(p0) : p0
end


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


function _armijo_max_decreases(initial_stepsize::Real, contraction::Real, alpha_min::Real)
    initial_stepsize <= alpha_min && return 0
    (contraction <= 0 || contraction >= 1) && return 1000
    n = floor(Int, log(alpha_min / initial_stepsize) / log(contraction))
    return max(n, 0)
end


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


function _safe_cost_function(model_cost)
    return function (M, p)
        _all_finite(p) || return Inf
        c = model_cost(M, p)
        return isfinite(c) ? c : Inf
    end
end


function _layout_adapt_gradient(model_grad)
    return function (M, p)
        g = model_grad(M, p)
        return _align_layout_like_point(p, g)
    end
end

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

@inline function _solver_gradient(state)
    try
        return Manopt.get_gradient(state)
    catch
        return nothing
    end
end

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

@inline _solver_has_converged(state) = Manopt.has_converged(state)


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

@inline function _solver_iteration_source()
    return isdefined(Manopt, :stopped_at) ? :stopped_at : :stop_at_iteration_fallback
end

function _solver_stats(
    model_cost,
    model_grad,
    M,
    p_opt,
    state,
    ::Nothing;
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
    rel_error = sqrt(cost_for_error)
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

function _solver_stats(
    model_cost,
    model_grad,
    M,
    p_opt,
    state,
    normA2::Real;
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
    rel_error = _relative_error_frob_sq(cost_for_error, T(normA2))
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

_solver_debug_callbacks(callbacks...) = Any[cb for cb in callbacks if !isnothing(cb)]

# Allow callers to pass `nothing` (e.g., when verbose/debug is omitted)
_solver_debug_actions(::Nothing, callbacks...) = _solver_debug_callbacks(callbacks...)

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

function _solver_eval_count(problem, sym::Symbol)
    try
        count = get_count(get_objective(problem), sym)
        return count < 0 ? 0 : Int(count)
    catch
        return 0
    end
end

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

function solve_rgd(
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 500,
    stepsize::Real = 1.0,
    tol::Real = 1e-6,
    verbose::Bool = true,
    return_stats::Bool = false,
    normA2 = nothing,
    model_grad = nothing,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback = nothing,
    diagnostics_recorder = nothing,
    iteration_callbacks = (),
    grad_tol = nothing,
    normalized_objective::Bool = false,
)
    p0_local = _solver_point(M, p0)
    T = _scalar_eltype(p0_local)
    model_grad_raw = isnothing(model_grad) ? grad(model_egrad) : model_grad
    model_grad_local = _layout_adapt_gradient(model_grad_raw)
    objective_scale =
        normalized_objective && !isnothing(normA2) && normA2 > 0 ? T(normA2) : one(T)
    solver_cost_base, solver_grad, uses_relative_objective =
        _relative_solver_functions(model_cost, model_grad_local, objective_scale)
    retraction_method = _solver_retraction_method(M, p0_local)
    stepsize_eff_base = T(stepsize) * objective_scale
    armijo_alpha_min = T(1e-8) * objective_scale
    grad_stop_tol = isnothing(grad_tol) ? T(tol) : T(grad_tol)
    tol_g = _dual_stop_grad_tol(T, tol, grad_tol)
    tol_g_raw = uses_relative_objective ? tol_g * objective_scale : tol_g
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)

    stopping = StopWhenAny(
        StopAfterIteration(maxiter),
        StopWhenGradientNormLess(grad_stop_tol),
        StopWhenStepsizeLess(armijo_alpha_min),
        dual_stop,
    )

    use_squaring_armijo = _contains_sqeuclidean_manifold(M)
    use_strict_sqeuclidean = _contains_strict_sqeuclidean_manifold(M)
    initial_stepsize_eff =
        use_squaring_armijo ?
        _adaptive_initial_stepsize(
            M,
            p0_local,
            solver_grad,
            retraction_method,
            stepsize_eff_base;
            alpha_min = armijo_alpha_min,
        ) : stepsize_eff_base
    armijo_contraction = use_squaring_armijo ? T(0.5) : T(0.85)
    armijo_sufficient_decrease = use_squaring_armijo ? T(1e-4) : T(1e-3)
    armijo_stop_decreasing =
        _armijo_max_decreases(initial_stepsize_eff, armijo_contraction, armijo_alpha_min)
    armijo_max_step = use_strict_sqeuclidean ? initial_stepsize_eff : Inf
    armijo_stop_increasing = use_strict_sqeuclidean ? 0 : 100
    armijo_additional_decrease =
        use_strict_sqeuclidean ? ((M, q) -> _all_finite(q)) : ((M, q) -> true)
    solver_cost =
        use_strict_sqeuclidean ? _safe_cost_function(solver_cost_base) : solver_cost_base
    armijo = Manopt.ArmijoLinesearch(
        M;
        retraction_method = retraction_method,
        initial_stepsize = initial_stepsize_eff,
        contraction_factor = armijo_contraction,
        sufficient_decrease = armijo_sufficient_decrease,
        stop_when_stepsize_less = armijo_alpha_min,
        stop_when_stepsize_exceeds = armijo_max_step,
        stop_increasing_at_step = armijo_stop_increasing,
        stop_decreasing_at_step = armijo_stop_decreasing,
        additional_decrease_condition = armijo_additional_decrease,
    )

    progress =
        maxiter > 0 ?
        make_rgd_progress(maxiter; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()
    diagnostics_callback =
        isnothing(diagnostics_recorder) ? nothing :
        _solver_diagnostics_callback(diagnostics_recorder)
    progress_callback = _solver_progress_callback(
        progress,
        solver_cost,
        solver_grad,
        M;
        normA2 = uses_relative_objective ? normA2 : nothing,
        diagnostics_recorder,
    )

    state = gradient_descent(
        M,
        solver_cost,
        solver_grad,
        p0_local;
        retraction_method = retraction_method,
        stepsize = armijo,
        stopping_criterion = stopping,
        debug = _solver_debug_actions(
            verbose,
            post_step_callback,
            diagnostics_callback,
            progress_callback,
            iteration_callbacks...,
        ),
        count = [:Cost, :Gradient],
        return_state = true,
    )

    p_opt = get_solver_result(state)
    iterations_done = _solver_iterations(state, maxiter)
    if verbose
        finish_progress!(
            progress;
            current = iterations_done,
            showvalues = Any[("Status", "Finished"), ("Iterations", iterations_done)],
        )
    end
    solver_info =
        isnothing(diagnostics_recorder) ? (;) :
        _solver_info(diagnostics_recorder, iterations_done)
    solver_info =
        merge(solver_info, (initial_stepsize_eff = Float64(initial_stepsize_eff),))
    if !return_stats
        return p_opt
    end
    return isnothing(normA2) ?
           _solver_stats(
        model_cost,
        model_grad_local,
        M,
        p_opt,
        state,
        nothing;
        tol_T = T(tol),
        maxiter,
        solver = :rgd,
        tiny_grad_tol = tol_g_raw,
        solver_info,
        use_state_gradient = !uses_relative_objective,
    ) :
           _solver_stats(
        model_cost,
        model_grad_local,
        M,
        p_opt,
        state,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rgd,
        tiny_grad_tol = tol_g_raw,
        solver_info,
        use_state_gradient = !uses_relative_objective,
    )
end

function solve_rgd_fixed(
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 2000,
    stepsize::Real = 1.0,
    tol::Real = 1e-6,
    verbose::Bool = true,
    return_stats::Bool = false,
    normA2 = nothing,
    model_grad = nothing,
    post_step_callback = nothing,
    diagnostics_recorder = nothing,
    iteration_callbacks = (),
    grad_tol = nothing,
    normalized_objective::Bool = false,
)
    p0_local = _solver_point(M, p0)
    T = _scalar_eltype(p0_local)
    model_grad_raw = isnothing(model_grad) ? grad(model_egrad) : model_grad
    model_grad_local = _layout_adapt_gradient(model_grad_raw)
    objective_scale =
        normalized_objective && !isnothing(normA2) && normA2 > 0 ? T(normA2) : one(T)
    solver_cost, solver_grad, uses_relative_objective =
        _relative_solver_functions(model_cost, model_grad_local, objective_scale)
    retraction_method = _solver_retraction_method(M, p0_local)
    grad_stop_tol = isnothing(grad_tol) ? T(tol) : T(grad_tol)
    tiny_grad_tol =
        isnothing(grad_tol) ? T(1e-5) :
        (uses_relative_objective ? T(grad_tol) * objective_scale : T(grad_tol))
    stopping =
        StopWhenAny(StopAfterIteration(maxiter), StopWhenGradientNormLess(grad_stop_tol))
    progress =
        maxiter > 0 ?
        make_rgd_fixed_progress(maxiter; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()
    diagnostics_callback =
        isnothing(diagnostics_recorder) ? nothing :
        _solver_diagnostics_callback(diagnostics_recorder)
    progress_callback = _solver_progress_callback(
        progress,
        solver_cost,
        solver_grad,
        M;
        normA2 = uses_relative_objective ? normA2 : nothing,
        diagnostics_recorder,
    )
    state = gradient_descent(
        M,
        solver_cost,
        solver_grad,
        p0_local;
        retraction_method = retraction_method,
        stepsize = Manopt.ConstantStepsize(M, T(stepsize) * objective_scale),
        stopping_criterion = stopping,
        debug = _solver_debug_actions(
            verbose,
            post_step_callback,
            diagnostics_callback,
            progress_callback,
            iteration_callbacks...,
        ),
        count = [:Cost, :Gradient],
        return_state = true,
    )

    p_opt = get_solver_result(state)
    iterations_done = _solver_iterations(state, maxiter)
    if verbose
        finish_progress!(
            progress;
            current = iterations_done,
            showvalues = Any[("Status", "Finished"), ("Iterations", iterations_done)],
        )
    end
    solver_info =
        isnothing(diagnostics_recorder) ? (;) :
        _solver_info(diagnostics_recorder, iterations_done)
    if !return_stats
        return p_opt
    end
    return isnothing(normA2) ?
           _solver_stats(
        model_cost,
        model_grad_local,
        M,
        p_opt,
        state,
        nothing;
        tol_T = T(tol),
        maxiter,
        solver = :rgd_fixed,
        tiny_grad_tol = tiny_grad_tol,
        solver_info,
        use_state_gradient = !uses_relative_objective,
    ) :
           _solver_stats(
        model_cost,
        model_grad_local,
        M,
        p_opt,
        state,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rgd_fixed,
        tiny_grad_tol = tiny_grad_tol,
        solver_info,
        use_state_gradient = !uses_relative_objective,
    )
end

# ========== RGDSolver (AbstractFirstOrderSolver) ==========

"""
    RGDSolver(stepsize=1.0)

Riemannian gradient descent with Armijo backtracking line search. Call via
`solve(RGDSolver(...), model; init=:random, gradient_mode=:riemannian)`.
"""
struct RGDSolver <: AbstractFirstOrderSolver
    stepsize::Float64
end
RGDSolver() = RGDSolver(1.0)

solver_symbol(::RGDSolver) = :rgd
first_order_diagnostics_recorder(::RGDSolver) =
    _SolverDiagnosticsRecorder(line_search_enabled = true)

function run_first_order_solver(
    solver::RGDSolver,
    setup;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
    return_stats::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback,
    diagnostics_recorder,
    iteration_callbacks,
    grad_tol = nothing,
    normalized_objective::Bool = false,
)
    return solve_rgd(
        setup.model_cost,
        setup.model_egrad,
        setup.M,
        setup.p0;
        maxiter,
        stepsize = solver.stepsize,
        tol,
        verbose,
        return_stats,
        normA2 = setup.normA2,
        model_grad = setup.model_grad,
        vector_transport_method,
        post_step_callback,
        diagnostics_recorder,
        iteration_callbacks,
        grad_tol,
        normalized_objective,
    )
end

# ========== RGDFixedSolver (AbstractFirstOrderSolver) ==========

"""
    RGDFixedSolver(stepsize=1.0)

Riemannian gradient descent with a constant stepsize.
Used as the stable fallback for Tucker-product BTD on dependency stacks where
Armijo's rand/allocate_result path is still unreliable.
"""
struct RGDFixedSolver <: AbstractFirstOrderSolver
    stepsize::Float64
end
RGDFixedSolver() = RGDFixedSolver(1.0)

solver_symbol(::RGDFixedSolver) = :rgd_fixed
first_order_diagnostics_recorder(solver::RGDFixedSolver) = _SolverDiagnosticsRecorder(
    line_search_enabled = false,
    fallback_stepsize = solver.stepsize,
)

function run_first_order_solver(
    solver::RGDFixedSolver,
    setup;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
    return_stats::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback,
    diagnostics_recorder,
    iteration_callbacks,
    grad_tol = nothing,
    normalized_objective::Bool = false,
)
    return solve_rgd_fixed(
        setup.model_cost,
        setup.model_egrad,
        setup.M,
        setup.p0;
        maxiter,
        stepsize = solver.stepsize,
        tol,
        verbose,
        return_stats,
        normA2 = setup.normA2,
        model_grad = setup.model_grad,
        post_step_callback,
        diagnostics_recorder,
        iteration_callbacks,
        grad_tol,
        normalized_objective,
    )
end
