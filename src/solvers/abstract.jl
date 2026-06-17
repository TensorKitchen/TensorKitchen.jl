# solvers/abstract.jl — Abstract decomposition solver interface
export AbstractSolverTrace,
    IterationRecord,
    StandardSolverTrace,
    SolverResult,
    record!,
    AbstractSolver,
    AbstractALSSolver,
    AbstractROSolver,
    AbstractFirstOrderROSolver,
    AbstractSecondOrderROSolver,
    AbstractFirstOrderSolver,
    solve

abstract type AbstractSolverTrace end

struct IterationRecord{T<:AbstractFloat}
    iter::Int
    cost::T
    rel_error::T
    grad_norm::Union{T,Nothing}
    extra::NamedTuple
end

mutable struct StandardSolverTrace{T<:AbstractFloat} <: AbstractSolverTrace
    records::Vector{IterationRecord{T}}
    function_evaluations::Int
    gradient_evaluations::Int
    wall_time_ns::Float64
end
StandardSolverTrace{T}() where {T<:AbstractFloat} =
    StandardSolverTrace{T}(IterationRecord{T}[], 0, 0, 0.0)

"""
    _dual_stop_grad_tol(T, tol; grad_tol=nothing)

Gradient tolerance paired with `StopWhenCostRelChangeAndGradientLess`.
Defaults to `sqrt(tol)`; callers may pass an explicit `grad_tol` (for example
`grad_tol = tol` on the nonnegative CPD manifold route).
"""
@inline function _dual_stop_grad_tol(
    ::Type{T},
    tol::Real,
    grad_tol = nothing,
) where {T<:Real}
    return isnothing(grad_tol) ? sqrt(T(tol)) : T(grad_tol)
end

function record!(
    trace::StandardSolverTrace{T},
    iter::Int,
    cost::T,
    rel_error::T,
    grad_norm::Union{T,Nothing},
    extra::NamedTuple,
) where {T<:AbstractFloat}
    push!(trace.records, IterationRecord{T}(iter, cost, rel_error, grad_norm, extra))
end

struct SolverResult{P,T<:AbstractFloat,TR<:AbstractSolverTrace}
    point::P
    cost::T
    rel_error::T
    grad_norm::Union{T,Nothing}
    iterations::Int
    converged::Bool
    solver::Symbol
    solver_info::NamedTuple
    trace::TR
end

point(result::SolverResult) = result.point
cost(result::SolverResult) = result.cost
rel_error(result::SolverResult) = result.rel_error
grad_norm(result::SolverResult) = result.grad_norm
iterations(result::SolverResult) = result.iterations
converged(result::SolverResult) = result.converged
solver(result::SolverResult) = result.solver
solver_info(result::SolverResult) = result.solver_info
trace(result::SolverResult) = result.trace

point(result::NamedTuple) = getproperty(result, :point)
cost(result::NamedTuple) = getproperty(result, :cost)
rel_error(result::NamedTuple) = getproperty(result, :rel_error)
grad_norm(result::NamedTuple) = getproperty(result, :grad_norm)
iterations(result::NamedTuple) = getproperty(result, :iterations)
converged(result::NamedTuple) = getproperty(result, :converged)
solver(result::NamedTuple) = getproperty(result, :solver)
solver_info(result::NamedTuple) = getproperty(result, :solver_info)

"""
    AbstractSolver

Root abstract type for every decomposition solver. Concrete solvers should
descend from one of the domain-specific abstract subtypes
([`AbstractALSSolver`](@ref) or the [`AbstractROSolver`](@ref) tree) rather
than from `AbstractSolver` directly. Implement `solve(solver, model; kwargs...)`
returning a point or stats.
"""
abstract type AbstractSolver end

"""
    AbstractROSolver <: AbstractSolver

Abstract supertype for Riemannian optimization solvers that operate directly
on the product manifold attached to an `AbstractDecompositionModel`.
Subtrees: [`AbstractFirstOrderROSolver`](@ref) and
[`AbstractSecondOrderROSolver`](@ref).
"""
abstract type AbstractROSolver <: AbstractSolver end

"""
    AbstractFirstOrderROSolver <: AbstractROSolver

Abstract supertype for first-order (gradient-based) Riemannian optimizers
such as `RGDSolver`, `RGDFixedSolver`, and `RCGSolver`. Concrete subtypes
must implement [`run_first_order_solver`](@ref) and [`solver_symbol`](@ref)
and may override [`first_order_diagnostics_recorder`](@ref).
"""
abstract type AbstractFirstOrderROSolver <: AbstractROSolver end

"""
    AbstractSecondOrderROSolver <: AbstractROSolver

Abstract supertype reserved for second-order Riemannian optimizers
(Gauss–Newton, trust-region, etc.).
"""
abstract type AbstractSecondOrderROSolver <: AbstractROSolver end

"""
    AbstractFirstOrderSolver <: AbstractFirstOrderROSolver

Alias-style marker for plain first-order (non-conjugate) gradient solvers
(e.g. `RGDSolver`, `RGDFixedSolver`).
"""
abstract type AbstractFirstOrderSolver <: AbstractFirstOrderROSolver end

"""
    AbstractALSSolver <: AbstractSolver

Abstract supertype for alternating-least-squares–style solvers (e.g.
`ALSSolver` for CP-ALS, `RALSSolver` for Randomized CP-ALS). These solvers
operate on Euclidean factor matrices and alternate closed-form (or sampled)
least-squares updates per mode.
"""
abstract type AbstractALSSolver <: AbstractSolver end

"""
    solver_symbol(solver::AbstractSolver) -> Symbol

Canonical symbolic tag of a solver (e.g. `:rgd`, `:rcg`, `:cp_als`). Used for
reporting and by post-step callbacks that specialize on the solver family.
"""
solver_symbol(solver::AbstractSolver) =
    error("solver_symbol not implemented for $(typeof(solver))")

"""
    first_order_diagnostics_recorder(solver::AbstractFirstOrderROSolver)

Diagnostics recorder attached to a first-order Riemannian optimizer. The
default is `nothing` (no recording); solvers with line search / fixed-step
instrumentation override this hook.
"""
first_order_diagnostics_recorder(solver::AbstractFirstOrderROSolver) = nothing

"""
    second_order_diagnostics_recorder(solver::AbstractSecondOrderROSolver)

Diagnostics recorder attached to a second-order Riemannian optimizer. The
default is `nothing` (no recording); concrete solvers with line search
instrumentation may override this hook.
"""
second_order_diagnostics_recorder(solver::AbstractSecondOrderROSolver) = nothing

"""
    run_first_order_solver(solver::AbstractFirstOrderROSolver, setup; kwargs...)

Driver hook for first-order Riemannian solvers. Given a prepared `setup`
(`M`, `p0`, cost/egrad/grad closures, `normA2`), execute the solver and
return either the final point or a stats `NamedTuple` depending on
`return_stats`. Each concrete solver implements this method; the generic
[`solve(::AbstractFirstOrderROSolver, model; ...)`](@ref solve) handles
problem setup, normalization policy validation, and post-step callback
wiring.
"""
function run_first_order_solver(solver::AbstractFirstOrderROSolver, setup; kwargs...)
    error("run_first_order_solver not implemented for $(typeof(solver))")
end

"""
    run_second_order_solver(solver::AbstractSecondOrderROSolver, setup; kwargs...)

Driver hook for second-order / quasi-second-order Riemannian solvers. Given a
prepared `setup` (`M`, `p0`, cost/egrad/grad closures, `normA2`), execute the
solver and return either the final point or a stats `NamedTuple` depending on
`return_stats`.
"""
function run_second_order_solver(solver::AbstractSecondOrderROSolver, setup; kwargs...)
    error("run_second_order_solver not implemented for $(typeof(solver))")
end

"""
    solve(solver::AbstractSolver, model::AbstractDecompositionModel; kwargs...)

Solve a decomposition model on its associated manifold. Returns point (if `return_stats=false`) or NamedTuple with
`(point, cost, rel_error, grad_norm, iterations, converged, solver, solver_info)` (if `return_stats=true`).

Concrete solver types (or the generic `AbstractFirstOrderROSolver` path)
override this method; this fallback only fires when no dispatch matches.
"""
function solve(solver::AbstractSolver, model::AbstractDecompositionModel; kwargs...)
    error("solve not implemented for $(typeof(solver))")
end

function _solve_ro_solver(
    solver::AbstractROSolver,
    model::AbstractDecompositionModel;
    init,
    p0,
    maxiter::Int,
    tol::Real,
    gradient_mode,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing},
    verbose::Bool,
    return_stats::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing},
    grad_tol,
    normalized_objective::Bool,
    iteration_callbacks,
    diagnostics_recorder,
    run_solver::Function,
)
    setup = _prepare_solver_problem(model; init, p0, gradient_mode, verbose)
    normalization_policy = _normalization_policy(normalization)
    supports_normalization_policy(model, normalization_policy) || throw(
        ArgumentError(
            "Normalization policy $(typeof(normalization_policy)) is not supported for model $(typeof(model)).",
        ),
    )
    solver_sym = solver_symbol(solver)
    post_step_callback =
        _solver_post_step_callback(model, setup.M, normalization_policy, solver_sym)
    return run_solver(
        solver,
        setup;
        maxiter,
        tol,
        verbose,
        return_stats,
        vector_transport_method,
        grad_tol,
        normalized_objective,
        post_step_callback,
        diagnostics_recorder,
        iteration_callbacks,
    )
end

function solve(
    solver::AbstractFirstOrderROSolver,
    model::AbstractDecompositionModel{T};
    init = :random,
    p0 = nothing,
    maxiter::Int = 500,
    tol::Real = 1e-8,
    gradient_mode = RiemannianGradientMode(),
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = NoNormalization(),
    verbose::Bool = true,
    return_stats::Bool = false,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    grad_tol = nothing,
    normalized_objective::Bool = true,
    iteration_callbacks = (),
) where {T<:AbstractFloat}
    return _solve_ro_solver(
        solver,
        model;
        init,
        p0,
        maxiter,
        tol,
        gradient_mode,
        normalization,
        verbose,
        return_stats,
        vector_transport_method,
        grad_tol,
        normalized_objective,
        iteration_callbacks,
        diagnostics_recorder = first_order_diagnostics_recorder(solver),
        run_solver = run_first_order_solver,
    )
end

function solve(
    solver::AbstractSecondOrderROSolver,
    model::AbstractDecompositionModel{T};
    init = :random,
    p0 = nothing,
    maxiter::Int = 500,
    tol::Real = 1e-8,
    gradient_mode = RiemannianGradientMode(),
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = NoNormalization(),
    verbose::Bool = true,
    return_stats::Bool = false,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    grad_tol = nothing,
    normalized_objective::Bool = true,
    iteration_callbacks = (),
) where {T<:AbstractFloat}
    return _solve_ro_solver(
        solver,
        model;
        init,
        p0,
        maxiter,
        tol,
        gradient_mode,
        normalization,
        verbose,
        return_stats,
        vector_transport_method,
        grad_tol,
        normalized_objective,
        iteration_callbacks,
        diagnostics_recorder = second_order_diagnostics_recorder(solver),
        run_solver = run_second_order_solver,
    )
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    gradient_mode;
    model_egrad = model_egrad_function(model),
    model_rgrad = supports_rgrad(model) ? model_rgrad_function(model; model_egrad) :
                  nothing,
)
    return _model_gradient_closure(
        model,
        gradient_mode_policy(gradient_mode);
        model_egrad,
        model_rgrad,
    )
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    ::RiemannianGradientMode;
    model_egrad,
    model_rgrad,
)
    return isnothing(model_rgrad) ? ((M, p) -> egrad_to_rgrad(M, p, model_egrad(M, p))) :
           model_rgrad
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    ::ProjectedEGradientMode;
    model_egrad,
    model_rgrad,
)
    supports_egrad_project(model) || throw(
        ArgumentError(
            "gradient_mode=:egrad_project is not supported for model $(typeof(model)). " *
            "Use :riemannian or another model-supported gradient mode.",
        ),
    )
    return (M, p) -> egrad_to_rgrad(M, p, model_egrad(M, p))
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    ::ExactNativeGradientMode;
    model_egrad,
    model_rgrad,
)
    supports_exact_native(model) || throw(
        ArgumentError(
            "gradient_mode=:exact_native is reserved for dedicated native closed-form gradients. " *
            "Use :exact_join (or :riemannian) for generic direct rgrad models.",
        ),
    )
    return model_exact_native_function(model)
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    ::ExactJoinGradientMode;
    model_egrad,
    model_rgrad,
)
    !isnothing(model_rgrad) || throw(
        ArgumentError(
            "gradient_mode=:exact_join requires a model with direct rgrad support.",
        ),
    )
    return model_rgrad
end

function _model_gradient_closure(
    model::AbstractDecompositionModel,
    ::ExactJoinBasisGradientMode;
    model_egrad,
    model_rgrad,
)
    supports_exact_join_basis(model) || throw(
        ArgumentError(
            "gradient_mode=:exact_join_basis requires explicit basis-based direct gradient support.",
        ),
    )
    return model_exact_join_basis_function(model)
end

@inline _unwrap_solver_manifold(M) = hasproperty(M, :M) ? getproperty(M, :M) : M

# The actual methods depend on the registered defaults, e.g. custom manifolds such
# as Segre or SoftplusEuclidean may choose ExponentialRetraction, while sphere-like
# factors may choose their ManifoldsBase default.
@inline function _default_component_retraction_method(Mi, pi)
    return ManifoldsBase.default_retraction_method(Mi, typeof(pi))
end

function _solver_retraction_method(M, p)
    return _solver_retraction_method_unwrapped(_unwrap_solver_manifold(M), p)
end

function _solver_retraction_method_unwrapped(M::ProductManifold, p)
    pparts0 = point_parts(p)
    pparts = pparts0 isa Tuple ? pparts0 : Tuple(pparts0)
    n = length(M.manifolds)
    length(pparts) == n || throw(
        ArgumentError(
            "Cannot derive solver retraction method: ProductManifold has $n factors but point has $(length(pparts)) parts.",
        ),
    )
    methods =
        ntuple(i -> _default_component_retraction_method(M.manifolds[i], pparts[i]), n)
    return ManifoldsBase.ProductRetraction(methods)
end

_solver_retraction_method_unwrapped(M, p) = _default_component_retraction_method(M, p)

"""
    _prepare_solver_problem(model; init, gradient_mode)

Build shared solver inputs for manifold-based optimizers:
- manifold and initial point
- cost/egrad/selected gradient closures
- target norm squared used for relative-error reporting
"""
function _prepare_solver_problem(
    model::AbstractDecompositionModel;
    init = :random,
    p0 = nothing,
    gradient_mode = RiemannianGradientMode(),
    verbose::Bool = false,
)
    M = manifold(model)
    isnothing(M) &&
        throw(ArgumentError("Solver requires a manifold. Model $(typeof(model)) has none."))

    p0_local = isnothing(p0) ? initial_point(model, init; verbose) : p0

    model_cost, model_egrad = model_cost_egrad_functions(model)
    model_rgrad = supports_rgrad(model) ? model_rgrad_function(model; model_egrad) : nothing
    model_grad = _model_gradient_closure(
        model,
        gradient_mode_policy(gradient_mode);
        model_egrad,
        model_rgrad,
    )

    return (
        M = M,
        p0 = p0_local,
        model_cost = model_cost,
        model_egrad = model_egrad,
        model_grad = model_grad,
        normA2 = sum(abs2, tensor(model)),
    )
end
