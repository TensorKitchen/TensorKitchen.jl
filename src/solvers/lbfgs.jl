# solvers/lbfgs.jl — Riemannian L-BFGS via Manopt quasi_Newton
export LBFGSSolver

"""
    LBFGSSolver(; memory_size=1, cautious_update=true, initial_scale=1.0,
        nonpositive_curvature_behavior=:ignore,
        nondescent_direction_behavior=:reinitialize_direction_update,
        linesearch=:wolfe, preconditioner=nothing)

Limited-memory Riemannian BFGS wrapper built on `Manopt.quasi_Newton`.
"""
struct LBFGSSolver <: AbstractSecondOrderROSolver
    memory_size::Int
    cautious_update::Bool
    initial_scale::Float64
    nonpositive_curvature_behavior::Symbol
    linesearch::Symbol
    preconditioner::Any
end

function LBFGSSolver(;
    memory_size::Int = 1,
    cautious_update::Bool = true,
    initial_scale::Real = 1.0,
    nonpositive_curvature_behavior::Symbol = :ignore,
    linesearch::Symbol = :wolfe,
    preconditioner = nothing,
)
    memory_size >= 1 || throw(ArgumentError("memory_size must be >= 1, got $memory_size"))
    initial_scale > 0 ||
        throw(ArgumentError("initial_scale must be > 0, got $initial_scale"))
    linesearch in (:wolfe, :hagerzhang) || throw(
        ArgumentError("Unsupported linesearch=$linesearch. Use :wolfe or :hagerzhang."),
    )
    return LBFGSSolver(
        memory_size,
        cautious_update,
        Float64(initial_scale),
        nonpositive_curvature_behavior,
        linesearch,
        preconditioner,
    )
end

solver_symbol(::LBFGSSolver) = :lbfgs

second_order_diagnostics_recorder(::LBFGSSolver) =
    _SolverDiagnosticsRecorder(line_search_enabled = true)

@inline function _lbfgs_linesearch(kind::Symbol)
    kind === :hagerzhang && return Manopt.HagerZhangLinesearch()
    kind === :wolfe && return Manopt.WolfePowellLinesearch()
    throw(ArgumentError("Unsupported linesearch kind $kind."))
end

function solve_lbfgs(
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 1000,
    tol::Real = 1e-6,
    verbose::Bool = true,
    return_stats::Bool = false,
    normA2 = nothing,
    model_grad = nothing,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback = nothing,
    diagnostics_recorder = nothing,
    iteration_callbacks = (),
    memory_size::Int = 1,
    cautious_update::Bool = true,
    initial_scale::Real = 1.0,
    nonpositive_curvature_behavior::Symbol = :ignore,
    linesearch::Symbol = :wolfe,
    preconditioner = nothing,
    grad_tol = nothing,
)
    p0_local = _solver_point(M, p0)
    T = _scalar_eltype(p0_local)
    model_grad_raw = isnothing(model_grad) ? grad(model_egrad) : model_grad
    model_grad_local = _layout_adapt_gradient(model_grad_raw)
    retraction_method = _solver_retraction_method(M, p0_local)
    transport =
        isnothing(vector_transport_method) ?
        _default_vector_transport_method(M, p0_local, retraction_method) :
        vector_transport_method
    tol_g = _dual_stop_grad_tol(T, tol, grad_tol)
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)
    stopping = StopWhenAny(
        StopAfterIteration(maxiter),
        StopWhenGradientNormLess(T(tol)),
        dual_stop,
    )
    progress =
        maxiter > 0 ?
        make_manopt_family_progress(
            maxiter;
            enabled = verbose,
            phase = :refinement,
            method = "L-BFGS",
            dt = 0.2,
        ) : NoMethodProgress()
    diagnostics_callback =
        isnothing(diagnostics_recorder) ? nothing :
        _solver_diagnostics_callback(diagnostics_recorder)
    progress_callback = _solver_progress_callback(
        progress,
        model_cost,
        model_grad_local,
        M;
        diagnostics_recorder,
    )

    state = Manopt.quasi_Newton(
        M,
        model_cost,
        model_grad_local,
        p0_local;
        cautious_update = cautious_update,
        direction_update = Manopt.InverseBFGS(),
        memory_size = memory_size,
        initial_scale = initial_scale,
        preconditioner = preconditioner,
        retraction_method = retraction_method,
        vector_transport_method = transport,
        stepsize = _lbfgs_linesearch(linesearch),
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

    p_opt = _tk_get_solver_result(state)
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
    solver_info = merge(
        solver_info,
        (
            memory_size = memory_size,
            cautious_update = cautious_update,
            initial_scale = initial_scale,
            nonpositive_curvature_behavior = nonpositive_curvature_behavior,
            linesearch = linesearch,
            has_preconditioner = !isnothing(preconditioner),
            uses_nonpositive_curvature_behavior = false,
        ),
    )
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
        solver = :lbfgs,
        tiny_grad_tol = tol_g,
        solver_info,
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
        solver = :lbfgs,
        tiny_grad_tol = tol_g,
        solver_info,
    )
end

function run_second_order_solver(
    solver::LBFGSSolver,
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
)
    return solve_lbfgs(
        setup.model_cost,
        setup.model_egrad,
        setup.M,
        setup.p0;
        maxiter,
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
        memory_size = solver.memory_size,
        cautious_update = solver.cautious_update,
        initial_scale = solver.initial_scale,
        nonpositive_curvature_behavior = solver.nonpositive_curvature_behavior,
        linesearch = solver.linesearch,
        preconditioner = solver.preconditioner,
    )
end
