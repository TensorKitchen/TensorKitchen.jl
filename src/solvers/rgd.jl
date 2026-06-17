# solvers/rgd.jl — Riemannian Gradient Descent
export RGDSolver, RGDFixedSolver
using Manopt

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
    return _solver_stats(
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
    return _solver_stats(
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
