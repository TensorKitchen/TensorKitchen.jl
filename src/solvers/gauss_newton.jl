# Damped Riemannian Gauss--Newton using the model operator contract.

export GaussNewtonSolver

_default_gauss_newton_inner_options() =
    InnerSolveOptions(tolerance = AdaptiveResidualTolerance(minimum = 1e-10))

_default_gauss_newton_damping() = DampingPolicy(
    initial = 1.0e-6,
    increase_factor = 10,
    reduction_factor = 0.3,
    acceptance_threshold = 1.0e-4,
    increase_threshold = 0.25,
    reduction_threshold = 0.75,
    max_trials = 8,
)

"""
    GaussNewtonSolver(; linear_solver=:cg, inner=..., damping=...)

Damped Riemannian Gauss--Newton for nonlinear least-squares models.
Uses matrix-free tangent CG (`:cg`) or a dense intrinsic normal solve (`:dense`).
Configure iterative solves with `InnerSolveOptions` and step acceptance with
`DampingPolicy`.

Requires consistent least-squares scaling of `cost`, `rgrad`, and
`normal_operator`, and `supports_rgrad(model)=true`. Only the gradient tolerance
indicates convergence. See the advanced optimization guide for defaults,
algorithms, and diagnostics.
"""
struct GaussNewtonSolver{I<:InnerSolveOptions} <: AbstractSecondOrderROSolver
    linear_solver::Symbol
    inner::I
    damping::DampingPolicy
end

function GaussNewtonSolver(;
    linear_solver::Symbol = :cg,
    inner::InnerSolveOptions = _default_gauss_newton_inner_options(),
    damping::DampingPolicy = _default_gauss_newton_damping(),
)
    linear_solver in (:cg, :dense) ||
        throw(ArgumentError("linear_solver must be :cg or :dense, got $linear_solver."))
    damping.initial > 0 ||
        throw(ArgumentError("Gauss--Newton requires positive initial damping."))
    return GaussNewtonSolver(linear_solver, inner, damping)
end

solver_symbol(solver::GaussNewtonSolver) = solver.linear_solver == :cg ? :gn_cg : :gn_dense

@doc raw"""
    _gauss_newton_cg(model, p, b, damping; tol, maxiter)

Solve `(J\*J + damping*I)X = b` in the Riemannian tangent metric by conjugate
gradients. Each Krylov iteration calls [`normal_operator!`](@ref); no Jacobian
or normal matrix is stored. The fourth return value records the initial and
final residual norms, achieved relative residual, requested tolerance, and
termination reason.

"""
function _gauss_newton_cg(
    model::AbstractDecompositionModel{T},
    p,
    b,
    damping::T;
    tol::T,
    maxiter::Int,
    absolute::Bool = false,
) where {T<:AbstractFloat}
    M = manifold(model)
    damped_normal =
        direction ->
            normal_operator(model, p, direction) + _scale_solver_tangent(direction, damping)
    return _tangent_cg(M, p, damped_normal, b; tol, maxiter, absolute)
end

# A common result shape keeps linear-solver details outside the acceptance loop.
function _solve_gn_inner(
    solver::GaussNewtonSolver,
    model,
    p,
    gradient,
    mu;
    tolerance,
    cache,
)
    M = manifold(model)
    if isnothing(cache)
        rhs = _scale_solver_tangent(gradient, -one(mu))
        step, iterations, converged, info = _gauss_newton_cg(
            model,
            p,
            rhs,
            mu;
            tol = tolerance,
            maxiter = _inner_maxiter(solver.inner, manifold_dimension(M)),
            absolute = solver.inner.tolerance isa AbsoluteResidualTolerance,
        )
        return (;
            step,
            iterations,
            converged,
            tolerance,
            initial_residual = info.initial_residual_norm,
            final_residual = info.final_residual_norm,
            reason = info.termination_reason,
        )
    end
    # Never shift the cached matrix in place: rejected trials share this matrix.
    H = copy(cache.normal)
    H[diagind(H)] .+= mu
    coordinates = -(H \ cache.gradient)
    final_residual = norm(H * coordinates + cache.gradient)
    converged = all(isfinite, coordinates) && isfinite(final_residual)
    step = ManifoldsBase.get_vector(M, p, coordinates, cache.basis)
    return (;
        step,
        iterations = nothing,
        converged,
        tolerance = nothing,
        initial_residual = norm(cache.gradient),
        final_residual,
        reason = converged ? :direct_solve : :nonfinite_residual,
    )
end

struct _GNDiagnostics{T,I,L}
    iterations::Vector{I}
    converged::Vector{Bool}
    tolerances::Vector{L}
    initial_residuals::Vector{T}
    final_residuals::Vector{T}
    reasons::Vector{Symbol}
    predicted::Vector{T}
    actual::Vector{T}
    rho::Vector{T}
    damping::Vector{T}
    accepted::Vector{Bool}
end
_GNDiagnostics(::Type{T}, iterative::Bool) where {T} = _GNDiagnostics(
    iterative ? Int[] : Nothing[],
    Bool[],
    iterative ? T[] : Nothing[],
    T[],
    T[],
    Symbol[],
    T[],
    T[],
    T[],
    T[],
    Bool[],
)

function _record_gn_inner!(history, result, mu)
    push!(history.iterations, result.iterations)
    push!(history.converged, result.converged)
    push!(history.tolerances, result.tolerance)
    push!(history.initial_residuals, result.initial_residual)
    push!(history.final_residuals, result.final_residual)
    push!(history.reasons, result.reason)
    push!(history.damping, mu)
end

function _record_gn_trial!(history, predicted, actual, rho, accepted)
    push!(history.predicted, predicted)
    push!(history.actual, actual)
    push!(history.rho, rho)
    push!(history.accepted, accepted)
end

"""
    _solve_gauss_newton(solver, model; ...)

Run damped GN using compatible `cost`, `rgrad`, and `normal_operator` methods.
The cost is half the squared residual norm in the model's tangent metric.
CG uses normal actions; dense solves reuse one normal matrix per outer iteration.
Trials use undamped predicted reduction and the configured damping policy.
Only the gradient tolerance establishes convergence; small steps indicate stagnation.
Returns the point, objective, error, and per-trial diagnostics in `solver_info`.
See the advanced optimization guide for equations, forcing, and references.
"""
function _solve_gauss_newton(
    solver::GaussNewtonSolver,
    model::AbstractDecompositionModel{T};
    init = :auto,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-8,
    grad_tol::Real = tol,
    observation_norm2_cache = nothing,
    verbose::Bool = true,
) where {T<:AbstractFloat}
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative."))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive."))
    isfinite(grad_tol) && grad_tol > 0 ||
        throw(ArgumentError("grad_tol must be finite and positive."))

    M = manifold(model)
    linear_solver = solver.linear_solver
    damping_policy = solver.damping
    inner_options = solver.inner
    prepared = _prepare_model_point(model; init, p0, verbose, observation_norm2_cache)
    requested_init, resolved_init = prepared.requested_init, prepared.resolved_init
    p = _solver_point(M, _independent_solver_point(prepared.p0))
    retraction_method = _solver_retraction_method(M, p)
    current_cost = cost(model, p)
    lower, upper = _damping_bounds(damping_policy, T)
    mu = T(damping_policy.initial)
    isfinite(mu) && mu > 0 ||
        throw(ArgumentError("Initial damping must be positive and finite in $T."))
    mu = clamp(mu, lower, upper)
    tolerance = T(tol)
    gradient_tolerance = T(grad_tol)
    isfinite(tolerance) &&
    tolerance > 0 &&
    isfinite(gradient_tolerance) &&
    gradient_tolerance > 0 ||
        throw(ArgumentError("Stopping tolerances must be positive and finite in $T."))
    history = _GNDiagnostics(T, linear_solver == :cg)
    iterations_done = 0
    converged_flag = false
    termination_reason = :maxiter
    basis = ManifoldsBase.DefaultOrthonormalBasis()

    for iteration = 1:maxiter
        gradient = rgrad(model, p)
        gradient_norm = norm(M, p, gradient)
        if verbose
            println(
                "GN iteration $iteration: cost=$(current_cost), " *
                "grad_norm=$(gradient_norm), damping=$(mu)",
            )
        end
        if gradient_norm <= gradient_tolerance
            converged_flag = true
            termination_reason = :gradient_tolerance
            iterations_done = iteration - 1
            break
        end

        accepted = false
        accepted_step_norm = T(Inf)
        effective_inner_tolerance =
            linear_solver == :cg ?
            _inner_tolerance(inner_options.tolerance, gradient_norm) : nothing
        cache =
            linear_solver == :dense ?
            (
                normal = dense_normal_matrix(model, p),
                gradient = ManifoldsBase.get_coordinates(M, p, gradient, basis),
                basis = basis,
            ) : nothing
        for _ = 1:damping_policy.max_trials
            inner_result = _solve_gn_inner(
                solver,
                model,
                p,
                gradient,
                mu;
                tolerance = effective_inner_tolerance,
                cache,
            )
            _record_gn_inner!(history, inner_result, mu)
            step = inner_result.step
            step_norm = norm(M, p, step)
            if !isfinite(step_norm)
                _record_gn_trial!(history, T(NaN), T(NaN), T(-Inf), false)
                mu = _increase_damping(damping_policy, mu)
                continue
            end
            normal_step = normal_operator(model, p, step)
            predicted_reduction =
                -ManifoldsBase.inner(M, p, gradient, step) -
                T(0.5) * ManifoldsBase.inner(M, p, step, normal_step)
            if !isfinite(predicted_reduction) || predicted_reduction <= zero(T)
                _record_gn_trial!(history, predicted_reduction, T(NaN), T(-Inf), false)
                mu = _increase_damping(damping_policy, mu)
                continue
            end

            candidate = try
                _independent_retract(M, p, step, retraction_method)
            catch error
                error isa DomainError || rethrow()
                nothing
            end
            candidate_cost = isnothing(candidate) ? T(Inf) : cost(model, candidate)
            actual_reduction = current_cost - candidate_cost
            rho = actual_reduction / predicted_reduction
            trial_accepted =
                isfinite(candidate_cost) &&
                isfinite(rho) &&
                rho >= T(damping_policy.acceptance_threshold)
            _record_gn_trial!(
                history,
                predicted_reduction,
                actual_reduction,
                rho,
                trial_accepted,
            )

            if trial_accepted
                p = candidate
                current_cost = candidate_cost
                accepted = true
                accepted_step_norm = step_norm
                mu = _update_accepted_damping(damping_policy, mu, rho)
                break
            end

            mu = _increase_damping(damping_policy, mu)
        end
        iterations_done = iteration
        if !accepted
            termination_reason = :no_acceptable_step
            break
        end
        if accepted_step_norm <= tolerance
            termination_reason = :small_step
            break
        end
    end

    final_gradient = rgrad(model, p)
    final_gradient_norm = norm(M, p, final_gradient)
    if final_gradient_norm <= gradient_tolerance
        converged_flag = true
        termination_reason = :gradient_tolerance
    end
    norm2 = prepared.normA2
    relative_error = _solver_rel_error(current_cost, norm2, false, T)
    solver_symbol = linear_solver == :cg ? :gn_cg : :gn_dense
    return (
        point = p,
        cost = current_cost,
        rel_error = relative_error,
        grad_norm = final_gradient_norm,
        iterations = iterations_done,
        converged = converged_flag,
        solver = solver_symbol,
        solver_info = (
            requested_init = requested_init,
            resolved_init = resolved_init,
            linear_solver = linear_solver,
            matrix_free_normal = linear_solver == :cg,
            materializes_jacobian = false,
            inner = _inner_history_info(
                history.iterations,
                history.tolerances,
                history.initial_residuals,
                history.final_residuals,
                history.reasons;
                solver = linear_solver,
                policy = inner_options.tolerance,
                maxiter = _inner_maxiter(inner_options, manifold_dimension(M)),
                converged = history.converged,
                iterative = linear_solver == :cg,
            ),
            accepted_steps = count(identity, history.accepted),
            rejected_steps = count(!, history.accepted),
            predicted_reduction_history = history.predicted,
            actual_reduction_history = history.actual,
            rho_history = history.rho,
            damping_history = history.damping,
            step_accepted_history = history.accepted,
            acceptance_ratio = T(damping_policy.acceptance_threshold),
            poor_step_ratio = T(damping_policy.increase_threshold),
            good_step_ratio = T(damping_policy.reduction_threshold),
            final_damping = mu,
            damping_policy = damping_policy,
            termination_reason = termination_reason,
            grad_tol = gradient_tolerance,
            normalized_objective = false,
            error_measure = isnothing(norm2) || iszero(norm2) ? :absolute_residual :
                            :relative_residual,
        ),
    )
end

function solve(
    solver::GaussNewtonSolver,
    model::AbstractDecompositionModel{T};
    init = :auto,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-8,
    verbose::Bool = true,
    return_stats::Bool = false,
    gradient_mode = :riemannian,
    normalization = NoNormalization(),
    grad_tol = nothing,
    normalized_objective::Bool = false,
    iteration_callbacks = (),
    vector_transport_method = nothing,
    observation_norm2_cache = nothing,
    kwargs...,
) where {T<:AbstractFloat}
    supports_rgrad(model) || throw(
        ArgumentError(
            "GaussNewtonSolver requires a model implementing rgrad with supports_rgrad(model)=true.",
        ),
    )
    isempty(kwargs) || throw(
        ArgumentError(
            "Unsupported GaussNewtonSolver keywords $(keys(kwargs)). Configure inner and damping on the solver object.",
        ),
    )
    gradient_mode_policy(gradient_mode) isa RiemannianGradientMode ||
        throw(ArgumentError("GaussNewtonSolver requires gradient_mode=:riemannian."))
    _normalization_policy(normalization) isa NoNormalization ||
        throw(ArgumentError("GaussNewtonSolver only supports NoNormalization()."))
    normalized_objective && throw(
        ArgumentError("GaussNewtonSolver currently requires normalized_objective=false."),
    )
    isempty(iteration_callbacks) ||
        throw(ArgumentError("GaussNewtonSolver does not yet support iteration_callbacks."))
    isnothing(vector_transport_method) ||
        throw(ArgumentError("GaussNewtonSolver does not use vector_transport_method."))
    if !isnothing(observation_norm2_cache)
        isfinite(observation_norm2_cache) && observation_norm2_cache >= 0 ||
            throw(ArgumentError("observation_norm2_cache must be finite and nonnegative."))
    end
    result = _solve_gauss_newton(
        solver,
        model;
        init,
        p0,
        maxiter,
        tol,
        grad_tol = isnothing(grad_tol) ? tol : grad_tol,
        observation_norm2_cache,
        verbose,
    )
    return return_stats ? result : result.point
end
