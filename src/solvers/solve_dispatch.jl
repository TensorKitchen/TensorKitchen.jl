# solvers/solve_dispatch.jl — shared symbol-based solver dispatch

"""
    _solver_object(solver, stepsize; kwargs...) -> AbstractSolver

Construct the concrete solver object selected by a public `solver::Symbol`.
Solver-specific keyword arguments are consumed here before generic dispatch.
"""
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
        tsd = () -> BTDTSDSolver(;
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

"""
    _solve_with_solver(solver_obj::AbstractROSolver, model; kwargs...)

Route Riemannian optimization solvers through their common `solve` interface
with gradient-mode, normalization, and vector-transport options.
"""
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
    )
end

"""
    _solve_with_solver(solver_obj::Union{ALSSolver,RALSSolver}, model; kwargs...)

Route ALS-family solvers through their Euclidean factor-update path, where
gradient-mode and vector transport are not used.
"""
function _solve_with_solver(
    solver_obj::Union{ALSSolver,RALSSolver},
    model;
    init,
    p0 = nothing,
    maxiter::Int,
    tol::Real,
    normalization = SeparateLambdaNormalization(),
    verbose::Bool,
    kwargs...,
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
`approx`. Converts the public solver symbol into a concrete solver and returns
a result-like `NamedTuple`.
"""
function _solve_model(
    model::AbstractDecompositionModel;
    init,
    p0 = nothing,
    solver::Symbol,
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
