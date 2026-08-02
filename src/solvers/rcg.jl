# solvers/rcg.jl — Riemannian Conjugate Gradient
export RCGSolver

# RCG coefficient and restart rule selection
function _rcg_coefficient_rule(
    M,
    coefficient::Symbol,
    transport;
    denom_threshold::Real = 1e-10,
    beale_restart::Bool = false,
    restart_threshold::Real = 0.2,
)
    rule =
        coefficient in (:conjugate_descent, :cd) ? Manopt.ConjugateDescentCoefficient() :
        coefficient in (:hager_zhang, :hz) ?
        Manopt.HagerZhangCoefficient(
            M;
            vector_transport_method = transport,
            denom_threshold = denom_threshold,
        ) :
        coefficient in (:polak_ribiere, :pr, :prp) ?
        Manopt.PolakRibiereCoefficient(M; vector_transport_method = transport) :
        coefficient in (:fletcher_reeves, :fr) ? Manopt.FletcherReevesCoefficient() :
        coefficient in (:dai_yuan, :dy) ?
        Manopt.DaiYuanCoefficient(M; vector_transport_method = transport) :
        coefficient in (:hestenes_stiefel, :hs) ?
        Manopt.HestenesStiefelCoefficient(M; vector_transport_method = transport) :
        coefficient in (:liu_storey, :ls) ?
        Manopt.LiuStoreyCoefficient(M; vector_transport_method = transport) :
        coefficient in (:steepest, :steepest_descent, :gd, :gradient_descent) ?
        Manopt.SteepestDescentCoefficient() :
        throw(ArgumentError("Unknown RCG coefficient=$(coefficient)."))

    if beale_restart
        return Manopt.ConjugateGradientBealeRestart(
            M,
            rule;
            threshold = restart_threshold,
            vector_transport_method = transport,
        )
    end

    return rule
end

function _rcg_restart_condition(restart::Symbol; κ::Real = 1e-4)
    return restart in (:never, :none, :no_restart) ? Manopt.NeverRestart() :
           restart in (:non_descent, :nondescent) ? Manopt.RestartOnNonDescent() :
           restart in (:non_sufficient_descent, :sufficient_descent) ?
           Manopt.RestartOnNonSufficientDescent(κ) :
           throw(ArgumentError("Unknown RCG restart=$(restart)."))
end

function solve_rcg(
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 2000,
    tol::Real = 1e-8,
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
    coefficient::Symbol = :hager_zhang,
    restart::Symbol = :non_descent,
    restart_threshold::Real = 0.2,
    sufficient_descent_kappa::Real = 1e-4,
    denom_threshold::Real = 1e-10,
    beale_restart::Bool = false,
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
    transport =
        isnothing(vector_transport_method) ?
        _default_vector_transport_method(M, p0_local, retraction_method) :
        vector_transport_method
    coefficient_rule = _rcg_coefficient_rule(
        M,
        coefficient,
        transport;
        denom_threshold,
        beale_restart,
        restart_threshold,
    )
    restart_rule = _rcg_restart_condition(restart; κ = sufficient_descent_kappa)

    tol_g = setup.dual_grad_tol
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)

    stopping = _manopt_stopping(maxiter, setup.grad_stop_tol, dual_stop)
    callbacks = _manopt_callbacks(
        n -> make_rcg_progress(n; enabled = verbose, phase = :refinement, dt = 0.2),
        maxiter,
        verbose,
        setup.solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )

    state = conjugate_gradient_descent(
        M,
        setup.solver_cost,
        setup.solver_grad,
        p0_local;
        retraction_method = retraction_method,
        vector_transport_method = transport,
        coefficient = coefficient_rule,
        restart_condition = restart_rule,
        stopping_criterion = stopping,
        debug = callbacks.debug_actions,
        count = [:Cost, :Gradient],
        return_state = true,
    )

    return _manopt_finish_result(
        _tk_get_solver_result(state),
        state,
        callbacks.progress,
        diagnostics_recorder,
        setup.solver_cost,
        setup.solver_grad,
        M,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rcg,
        tiny_grad_tol = tol_g,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
        solver_info_extra = (
            rcg_coefficient = coefficient,
            rcg_restart = restart,
            rcg_beale_restart = beale_restart,
            rcg_restart_threshold = Float64(restart_threshold),
            rcg_sufficient_descent_kappa = Float64(sufficient_descent_kappa),
            rcg_denom_threshold = Float64(denom_threshold),
            rcg_transport = string(typeof(transport)),
            rcg_coefficient_rule = string(typeof(coefficient_rule)),
            rcg_restart_rule = string(typeof(restart_rule)),
        ),
    )
end

"""
    RCGSolver(; coefficient=:hager_zhang, restart=:non_descent,
        restart_threshold=0.2, sufficient_descent_kappa=1e-4,
        denom_threshold=1e-10, beale_restart=false)

Riemannian conjugate-gradient solver.

- `coefficient` selects the search-direction update. Accepted long names and
  aliases are `:conjugate_descent`/`:cd`, `:hager_zhang`/`:hz`,
  `:polak_ribiere`/`:pr`/`:prp`, `:fletcher_reeves`/`:fr`,
  `:dai_yuan`/`:dy`, `:hestenes_stiefel`/`:hs`, `:liu_storey`/`:ls`, and
  `:steepest`/`:steepest_descent`/`:gd`/`:gradient_descent`.
- `restart` accepts `:never` (aliases `:none`, `:no_restart`), `:non_descent`
  (alias `:nondescent`), or `:non_sufficient_descent` (alias
  `:sufficient_descent`).
- `restart_threshold` is used by optional Beale restart;
  `sufficient_descent_kappa` is used by the sufficient-descent restart rule;
  and `denom_threshold` safeguards the Hager--Zhang coefficient.
- `beale_restart=true` wraps the selected coefficient rule in Beale restart.
"""
Base.@kwdef struct RCGSolver <: AbstractFirstOrderROSolver
    coefficient::Symbol = :hager_zhang
    restart::Symbol = :non_descent
    restart_threshold::Float64 = 0.2
    sufficient_descent_kappa::Float64 = 1e-4
    denom_threshold::Float64 = 1e-10
    beale_restart::Bool = false
end

solver_symbol(::RCGSolver) = :rcg

first_order_diagnostics_recorder(::RCGSolver) =
    _SolverDiagnosticsRecorder(line_search_enabled = true)

function run_first_order_solver(
    solver::RCGSolver,
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
    return solve_rcg(
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
        normalized_objective,
        coefficient = solver.coefficient,
        restart = solver.restart,
        restart_threshold = solver.restart_threshold,
        sufficient_descent_kappa = solver.sufficient_descent_kappa,
        denom_threshold = solver.denom_threshold,
        beale_restart = solver.beale_restart,
    )
end
