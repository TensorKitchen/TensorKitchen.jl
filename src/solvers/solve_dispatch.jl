# solvers/solve_dispatch.jl — shared solver normalization and dispatch

function _solver_object(solver, ::Real; kwargs...)
    throw(
        ArgumentError(
            "Unsupported solver specification $(typeof(solver)). Use a solver symbol such as :als, :cls, :rgd, :rgd_fixed, :rcg, :lbfgs, :lm, :gn_cg, :gn_dense, or :btd_tsd, or pass an AbstractSolver object.",
        ),
    )
end

function _solver_object(solver::Symbol, stepsize::Real; kwargs...)
    return _solver_object(Val(solver), stepsize; kwargs...)
end

_solver_object(solver::AbstractSolver, ::Real; kwargs...) = solver

_solver_object(::Val{:als}, ::Real; kwargs...) = ALSSolver()
_solver_object(::Val{:cls}, ::Real; kwargs...) = SymmetricCLS(
    damping = get(kwargs, :cls_damping, 1.0e-10),
    pinv_rtol = get(kwargs, :cls_pinv_rtol, nothing),
    weight_pinv_rtol = get(kwargs, :cls_weight_pinv_rtol, nothing),
    patience = get(kwargs, :cls_patience, 3),
)
_solver_object(::Val{:rgd}, stepsize::Real; kwargs...) =
    RGDSolver(stepsize; armijo_alpha_min = get(kwargs, :armijo_alpha_min, 1e-8))
_solver_object(::Val{:rgd_fixed}, stepsize::Real; kwargs...) = RGDFixedSolver(stepsize)

function _solver_object(::Val{:rcg}, ::Real; kwargs...)
    return RCGSolver(;
        coefficient = get(kwargs, :coefficient, :hager_zhang),
        restart = get(kwargs, :restart, :non_descent),
        restart_threshold = Float64(get(kwargs, :restart_threshold, 0.2)),
        sufficient_descent_kappa = Float64(get(kwargs, :sufficient_descent_kappa, 1e-4)),
        denom_threshold = Float64(get(kwargs, :denom_threshold, 1e-10)),
        beale_restart = Bool(get(kwargs, :beale_restart, false)),
    )
end

function _solver_object(::Val{:lbfgs}, ::Real; kwargs...)
    return LBFGSSolver(;
        memory_size = get(kwargs, :memory_size, 1),
        cautious_update = get(kwargs, :cautious_update, true),
        initial_scale = get(kwargs, :initial_scale, 1.0),
        linesearch = get(kwargs, :linesearch, :wolfe),
        preconditioner = get(kwargs, :preconditioner, nothing),
    )
end

function _solver_object(::Val{:lm}, ::Real; kwargs...)
    return LMSolver(;
        η = get(kwargs, :η, 0.2),
        damping_term_min = get(kwargs, :damping_term_min, 0.1),
        β = get(kwargs, :β, 5.0),
        expect_zero_residual = get(kwargs, :expect_zero_residual, false),
        inner = something(get(kwargs, :inner, nothing), InnerSolveOptions()),
        diagnostics = get(kwargs, :diagnostics, false),
        damping_reduction_threshold = get(kwargs, :damping_reduction_threshold, nothing),
    )
end

const _GN_OPTION_KEYS = (
    :damping,
    :damping_increase,
    :damping_decrease,
    :inner,
    :max_damping_trials,
    :acceptance_ratio,
    :poor_step_ratio,
    :good_step_ratio,
    :damping_policy,
)

function _solver_object(solver::GaussNewtonSolver, ::Real; kwargs...)
    conflicts = [key for key in _GN_OPTION_KEYS if !isnothing(get(kwargs, key, nothing))]
    isempty(conflicts) || throw(
        ArgumentError(
            "Configure $conflicts on GaussNewtonSolver instead of also passing separate keywords.",
        ),
    )
    return solver
end

_gn_option(kwargs, key, default) = something(get(kwargs, key, nothing), default)

function _gauss_newton_solver(linear_solver::Symbol; kwargs...)
    damping_policy = get(kwargs, :damping_policy, nothing)
    if !isnothing(damping_policy)
        conflicts = [
            key for key in _GN_OPTION_KEYS if
            key ∉ (:inner, :damping_policy) && !isnothing(get(kwargs, key, nothing))
        ]
        isempty(conflicts) || throw(
            ArgumentError(
                "damping_policy conflicts with scalar damping options $conflicts.",
            ),
        )
    end
    if isnothing(damping_policy)
        damping_policy = DampingPolicy(
            initial = _gn_option(kwargs, :damping, 1.0e-6),
            increase_factor = _gn_option(kwargs, :damping_increase, 10),
            reduction_factor = _gn_option(kwargs, :damping_decrease, 0.3),
            acceptance_threshold = _gn_option(kwargs, :acceptance_ratio, 1.0e-4),
            increase_threshold = _gn_option(kwargs, :poor_step_ratio, 0.25),
            reduction_threshold = _gn_option(kwargs, :good_step_ratio, 0.75),
            max_trials = _gn_option(kwargs, :max_damping_trials, 8),
        )
    end
    return GaussNewtonSolver(;
        linear_solver,
        inner = _gn_option(kwargs, :inner, _default_symcpd_inner_options()),
        damping = damping_policy,
    )
end

_solver_object(::Val{:gn_cg}, ::Real; kwargs...) = _gauss_newton_solver(:cg; kwargs...)
_solver_object(::Val{:gn_dense}, ::Real; kwargs...) =
    _gauss_newton_solver(:dense; kwargs...)

function _solve_with_solver(solver::GaussNewtonSolver, model; kwargs...)
    # Configuration was consumed by _solver_object. Execution keywords are
    # checked by solve, including unsupported shared RO controls.
    execution =
        (; (key => value for (key, value) in pairs(kwargs) if key ∉ _GN_OPTION_KEYS)...)
    return solve(solver, model; return_stats = true, execution...)
end

function _solver_object(::Val{:btd_tsd}, stepsize::Real; kwargs...)
    return BTDTSDSolver(;
        stepsize,
        schedule = get(kwargs, :schedule, :cyclic),
        block_repeats = get(kwargs, :block_repeats, 1),
        armijo_contraction = get(kwargs, :armijo_contraction, 0.5),
        armijo_sufficient_decrease = get(kwargs, :armijo_sufficient_decrease, 1e-4),
        armijo_alpha_min = get(kwargs, :armijo_alpha_min, 1e-12),
    )
end

function _solver_object(::Val{S}, ::Real; kwargs...) where {S}
    throw(
        ArgumentError(
            "Unknown solver=$S. Use :als, :cls, :rgd, :rgd_fixed, :rcg, :lbfgs, :lm, :gn_cg, :gn_dense, or :btd_tsd.",
        ),
    )
end

function _solve_with_solver(
    solver_obj::SymmetricCLS,
    model;
    init,
    p0 = nothing,
    maxiter::Int,
    tol::Real,
    gradient_mode::Symbol = :riemannian,
    normalization = NoNormalization(),
    verbose::Bool,
    kwargs...,
)
    gradient_mode == :riemannian || throw(
        ArgumentError(
            "SymmetricCLS does not use gradient_mode. Use gradient_mode=:riemannian.",
        ),
    )
    return solve(
        solver_obj,
        model;
        init,
        p0,
        maxiter,
        tol,
        normalization,
        verbose,
        return_stats = true,
        kwargs...,
    )
end

function _solve_with_solver(
    solver_obj::AbstractROSolver,
    model;
    init,
    p0 = nothing,
    maxiter::Int,
    tol::Real,
    gradient_mode::Symbol,
    normalization = NoNormalization(),
    verbose::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    grad_tol = nothing,
    normalized_objective::Bool = true,
    iteration_callbacks = (),
    observation_norm2_cache = nothing,
    kwargs...,
)
    return solve(
        solver_obj,
        model;
        init,
        p0,
        maxiter,
        tol,
        gradient_mode,
        normalization,
        verbose,
        return_stats = true,
        vector_transport_method,
        grad_tol,
        normalized_objective,
        iteration_callbacks,
        observation_norm2_cache,
    )
end

function _solve_with_solver(
    solver_obj::Union{ALSSolver,RALSSolver},
    model;
    init,
    p0 = nothing,
    maxiter::Int,
    tol::Real,
    gradient_mode::Symbol = :riemannian,
    normalization = SeparateLambdaNormalization(),
    verbose::Bool,
    kwargs...,
)
    gradient_mode == :riemannian || throw(
        ArgumentError(
            "ALS solvers do not use gradient_mode. Use gradient_mode=:riemannian.",
        ),
    )

    return solve(
        solver_obj,
        model;
        init,
        p0,
        maxiter,
        tol,
        normalization,
        verbose,
        return_stats = true,
        kwargs...,
    )
end

"""
    _solve_model(model; solver, init, maxiter, stepsize, tol, kwargs...)

Top-level internal solver dispatcher shared by CPD, BTD, NNCPD, and generic
`approx`. Accepts either a public solver symbol or a concrete `AbstractSolver`
object, normalizes it to a solver object, and returns a result-like `NamedTuple`.
"""
function _solve_model(
    model::AbstractDecompositionModel;
    init,
    p0 = nothing,
    solver,
    maxiter::Int,
    stepsize::Real,
    tol::Real,
    gradient_mode::Symbol,
    normalization,
    verbose::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    kwargs...,
)
    return _solve_with_solver(
        _solver_object(solver, stepsize; kwargs...),
        model;
        init,
        p0,
        maxiter,
        tol,
        gradient_mode,
        normalization,
        verbose,
        vector_transport_method,
        kwargs...,
    )
end
