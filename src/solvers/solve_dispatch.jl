# solvers/solve_dispatch.jl — shared solver normalization and dispatch

function _solver_object(solver, ::Real; kwargs...)
    throw(
        ArgumentError(
            "Unsupported solver specification $(typeof(solver)). Use a solver symbol such as :als, :rgd, :rgd_fixed, :rcg, :lbfgs, or :btd_tsd, or pass an AbstractSolver object.",
        ),
    )
end

function _solver_object(solver::Symbol, stepsize::Real; kwargs...)
    return _solver_object(Val(solver), stepsize; kwargs...)
end

_solver_object(solver::AbstractSolver, ::Real; kwargs...) = solver

_solver_object(::Val{:als}, ::Real; kwargs...) = ALSSolver()
_solver_object(::Val{:rgd}, stepsize::Real; kwargs...) = RGDSolver(stepsize)
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
        nonpositive_curvature_behavior = get(
            kwargs,
            :nonpositive_curvature_behavior,
            :ignore,
        ),
        linesearch = get(kwargs, :linesearch, :wolfe),
        preconditioner = get(kwargs, :preconditioner, nothing),
    )
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
            "Unknown solver=$S. Use :als, :rgd, :rgd_fixed, :rcg, :lbfgs, or :btd_tsd.",
        ),
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
