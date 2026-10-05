# solvers/lbfgs.jl — Riemannian L-BFGS via Manopt quasi_Newton
export LBFGSSolver

"""
    LBFGSSolver(; memory_size=1, cautious_update=true, initial_scale=1.0,
        nonpositive_curvature_behavior=:ignore, linesearch=:wolfe, preconditioner=nothing)

Limited-memory Riemannian BFGS wrapper built on `Manopt.quasi_Newton`.

- `memory_size` is the number of curvature pairs retained.
- `cautious_update` lets Manopt reject unsuitable quasi-Newton updates.
- `initial_scale` sets the initial inverse-Hessian scale.
- `linesearch=:wolfe` is always supported; `:hagerzhang` is available when the
  installed Manopt version provides `HagerZhangLinesearch`.
- `preconditioner` is passed to Manopt.
- `nonpositive_curvature_behavior` is retained in configuration and reported
  in `solver_info`, but the current Manopt update path does not consume it;
  `solver_info.uses_nonpositive_curvature_behavior` is therefore `false`.
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
    supported = _lbfgs_supported_linesearches()
    linesearch in supported || throw(
        ArgumentError(
            "Unsupported linesearch=$linesearch. Use one of " * join(supported, ", ") * ".",
        ),
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

function _lbfgs_supported_linesearches()
    base = (:wolfe,)
    return isdefined(Manopt, :HagerZhangLinesearch) ? (base..., :hagerzhang) : base
end

@inline function _lbfgs_linesearch(
    kind::Symbol,
    M,
    p,
    retraction_method,
    transport,
    ::Type{T},
) where {T<:Real}
    kind === :wolfe && return Manopt.WolfePowellLinesearch(
        sufficient_curvature = T(0.9),
        stop_when_stepsize_less = T(1e-8),
        stop_decreasing_at_step = 100,
        retraction_method = retraction_method,
        vector_transport_method = transport,
    )
    if kind === :hagerzhang && isdefined(Manopt, :HagerZhangLinesearch)
        TF = _hagerzhang_workspace_type(T)
        return getproperty(Manopt, :HagerZhangLinesearchStepsize)(
            M;
            initial_guess = getproperty(Manopt, :HagerZhangInitialGuess){TF}(;
                ψ0 = TF(0.01),
                ψ1 = TF(0.01),
                ψ2 = TF(2.0),
                constant_guess = TF(NaN),
                zero_abstol = eps(TF),
                alphamax = TF(Inf),
            ),
            retraction_method = retraction_method,
            vector_transport_method = transport,
            initial_last_stepsize = TF(NaN),
            initial_last_cost = TF(NaN),
            stepsize_limit = TF(Inf),
            candidate_point = copy(M, p),
            candidate_direction = zero_vector(M, p),
            ϵ = TF(1.0e-6),
            δ = TF(0.1),
            σ = TF(0.9),
            ω = TF(1.0e-3),
            θ = TF(0.5),
            γ = TF(0.66),
            ρ = TF(5.0),
            Δ = TF(0.7),
            secant_acceptance_ratio = TF(1.0e-8),
        )
    end
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
    if linesearch === :hagerzhang
        retraction_method = _hagerzhang_retraction_method(M, retraction_method)
    end
    transport =
        isnothing(vector_transport_method) ?
        _default_vector_transport_method(M, p0_local, retraction_method) :
        vector_transport_method
    tol_g = setup.dual_grad_tol
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)
    stopping = _manopt_stopping(maxiter, setup.grad_stop_tol, dual_stop)
    callbacks = _manopt_callbacks(
        n -> make_manopt_family_progress(
            n;
            enabled = verbose,
            phase = :refinement,
            method = "L-BFGS",
            dt = 0.2,
        ),
        maxiter,
        verbose,
        setup.solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )
    preconditioner_kwargs = isnothing(preconditioner) ? NamedTuple() : (; preconditioner)

    state = Manopt.quasi_Newton(
        M,
        setup.solver_cost,
        setup.solver_grad,
        p0_local;
        cautious_update = cautious_update,
        direction_update = Manopt.InverseBFGS(),
        memory_size = memory_size,
        initial_scale = T(initial_scale),
        preconditioner_kwargs...,
        retraction_method = retraction_method,
        vector_transport_method = transport,
        stepsize = _lbfgs_linesearch(
            linesearch,
            M,
            p0_local,
            retraction_method,
            transport,
            T,
        ),
        stopping_criterion = stopping,
        debug = callbacks.debug_actions,
        callbacks = callbacks.solver_callbacks,
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
        solver = :lbfgs,
        tiny_grad_tol = tol_g,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
        solver_info_extra = (
            memory_size = memory_size,
            cautious_update = cautious_update,
            initial_scale = initial_scale,
            nonpositive_curvature_behavior = nonpositive_curvature_behavior,
            linesearch = linesearch,
            has_preconditioner = !isnothing(preconditioner),
            uses_nonpositive_curvature_behavior = false,
        ),
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
    normalized_objective::Bool = true,
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
        normalized_objective,
    )
end
