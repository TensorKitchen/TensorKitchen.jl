# solvers/solve_dispatch.jl — shared solver normalization and dispatch

function _solver_object(solver, ::Real; kwargs...)
    throw(
        ArgumentError(
            "Unsupported solver specification $(typeof(solver)). Use a solver symbol such as :als, :rgd, :rgd_fixed, :rcg, :lbfgs, or :btd_tsd, or pass an AbstractSolver object.",
        ),
    )
end

function _solver_object(solver::Symbol, stepsize::Real; kwargs...)
    solvers = (
        als = () -> ALSSolver(),
        rgd = () -> RGDSolver(stepsize),
        rgd_fixed = () -> RGDFixedSolver(stepsize),
        rcg = () -> RCGSolver(),
        lbfgs = () -> LBFGSSolver(;
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
        ),
        btd_tsd = () -> BTDTSDSolver(;
            stepsize,
            schedule = get(kwargs, :schedule, :cyclic),
            block_repeats = get(kwargs, :block_repeats, 1),
            armijo_contraction = get(kwargs, :armijo_contraction, 0.5),
            armijo_sufficient_decrease = get(kwargs, :armijo_sufficient_decrease, 1e-4),
            armijo_alpha_min = get(kwargs, :armijo_alpha_min, 1e-12),
        ),
    )
    f = get(solvers, solver) do
        throw(
            ArgumentError(
                "Unknown solver=$solver. Use :als, :rgd, :rgd_fixed, :rcg, :lbfgs, or :btd_tsd.",
            ),
        )
    end
    return f()
end

_solver_object(solver::AbstractSolver, ::Real; kwargs...) = solver

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
