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
    normalized_objective::Bool = true,
    armijo_alpha_min::Real = 1e-8,
)
    armijo_alpha_min > 0 || throw(
        ArgumentError("armijo_alpha_min must be > 0, got $armijo_alpha_min"),
    )
    setup = _prepare_manopt_solver_functions(
        model_cost,
        model_egrad,
        M,
        p0;
        normA2,
        model_grad,
        tol,
        grad_tol,
        normalized_objective,
    )
    p0_local = setup.p0
    T = setup.T
    retraction_method = _solver_retraction_method(M, p0_local)
    armijo_alpha_min_T = T(armijo_alpha_min)
    tol_g = setup.dual_grad_tol
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)
    stopping = _manopt_stopping(
        maxiter,
        setup.grad_stop_tol,
        dual_stop;
        extra = (StopWhenStepsizeLess(armijo_alpha_min_T),),
    )

    use_squaring_armijo = _contains_sqeuclidean_manifold(M)
    use_strict_sqeuclidean = _contains_strict_sqeuclidean_manifold(M)
    initial_stepsize_eff =
        use_squaring_armijo ?
        _adaptive_initial_stepsize(
            M,
            p0_local,
            setup.solver_grad,
            retraction_method,
            T(stepsize);
            alpha_min = armijo_alpha_min_T,
        ) : T(stepsize)
    armijo_contraction = use_squaring_armijo ? T(0.5) : T(0.85)
    armijo_sufficient_decrease = use_squaring_armijo ? T(1e-4) : T(1e-3)
    armijo_stop_decreasing =
        _armijo_max_decreases(initial_stepsize_eff, armijo_contraction, armijo_alpha_min_T)
    armijo_max_step = use_strict_sqeuclidean ? initial_stepsize_eff : Inf
    armijo_stop_increasing = use_strict_sqeuclidean ? 0 : 100
    armijo_additional_decrease =
        use_strict_sqeuclidean ? ((M, q) -> _all_finite(q)) : ((M, q) -> true)
    solver_cost =
        use_strict_sqeuclidean ? _safe_cost_function(setup.solver_cost) : setup.solver_cost
    armijo = Manopt.ArmijoLinesearch(
        M;
        retraction_method = retraction_method,
        initial_stepsize = initial_stepsize_eff,
        contraction_factor = armijo_contraction,
        sufficient_decrease = armijo_sufficient_decrease,
        stop_when_stepsize_less = armijo_alpha_min_T,
        stop_when_stepsize_exceeds = armijo_max_step,
        stop_increasing_at_step = armijo_stop_increasing,
        stop_decreasing_at_step = armijo_stop_decreasing,
        additional_decrease_condition = armijo_additional_decrease,
    )

    callbacks = _manopt_callbacks(
        n -> make_rgd_progress(n; enabled = verbose, phase = :refinement, dt = 0.2),
        maxiter,
        verbose,
        solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )

    state = gradient_descent(
        M,
        solver_cost,
        setup.solver_grad,
        p0_local;
        retraction_method = retraction_method,
        stepsize = armijo,
        stopping_criterion = stopping,
        debug = callbacks.debug_actions,
        count = [:Cost, :Gradient],
        return_state = true,
    )

    return _manopt_finish_result(
        get_solver_result(state),
        state,
        callbacks.progress,
        diagnostics_recorder,
        solver_cost,
        setup.solver_grad,
        M,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rgd,
        tiny_grad_tol = tol_g,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
        solver_info_extra = (
            initial_stepsize_eff = Float64(initial_stepsize_eff),
            armijo_alpha_min = Float64(armijo_alpha_min_T),
        ),
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
    normalized_objective::Bool = true,
)
    setup = _prepare_manopt_solver_functions(
        model_cost,
        model_egrad,
        M,
        p0;
        normA2,
        model_grad,
        tol,
        grad_tol,
        normalized_objective,
    )
    p0_local = setup.p0
    T = setup.T
    retraction_method = _solver_retraction_method(M, p0_local)
    tiny_grad_tol = isnothing(grad_tol) ? T(1e-5) : T(grad_tol)
    stopping = StopWhenAny(
        StopAfterIteration(maxiter),
        StopWhenGradientNormLess(setup.grad_stop_tol),
    )
    callbacks = _manopt_callbacks(
        n -> make_rgd_fixed_progress(n; enabled = verbose, phase = :refinement, dt = 0.2),
        maxiter,
        verbose,
        setup.solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )
    state = gradient_descent(
        M,
        setup.solver_cost,
        setup.solver_grad,
        p0_local;
        retraction_method = retraction_method,
        stepsize = Manopt.ConstantStepsize(M, T(stepsize)),
        stopping_criterion = stopping,
        debug = callbacks.debug_actions,
        count = [:Cost, :Gradient],
        return_state = true,
    )

    return _manopt_finish_result(
        get_solver_result(state),
        state,
        callbacks.progress,
        diagnostics_recorder,
        setup.solver_cost,
        setup.solver_grad,
        M,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rgd_fixed,
        tiny_grad_tol = tiny_grad_tol,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
    )
end

# ========== RGDSolver (AbstractFirstOrderSolver) ==========

"""
    RGDSolver(stepsize=1.0; armijo_alpha_min=1e-8)

Riemannian gradient descent with Armijo backtracking line search. Call via
`solve(RGDSolver(...), model; init=:random, gradient_mode=:riemannian)`.
"""
struct RGDSolver <: AbstractFirstOrderSolver
    stepsize::Float64
    armijo_alpha_min::Float64
end
function RGDSolver(stepsize::Real = 1.0; armijo_alpha_min::Real = 1e-8)
    stepsize > 0 || throw(ArgumentError("stepsize must be > 0, got $stepsize"))
    armijo_alpha_min > 0 || throw(
        ArgumentError("armijo_alpha_min must be > 0, got $armijo_alpha_min"),
    )
    return RGDSolver(Float64(stepsize), Float64(armijo_alpha_min))
end

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
    normalized_objective::Bool = true,
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
        armijo_alpha_min = solver.armijo_alpha_min,
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
    normalized_objective::Bool = true,
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
