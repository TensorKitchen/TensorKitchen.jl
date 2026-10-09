# Damped Riemannian Gauss--Newton for the structured symmetric model.

export GaussNewtonSolver

_default_symcpd_inner_options() =
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

"""Configure damped Riemannian Gauss--Newton.

`linear_solver=:cg` uses a matrix-free tangent-space Krylov solve, while
`linear_solver=:dense` materializes the intrinsic normal matrix. `inner`
controls the iterative linear solve and `damping` owns step acceptance and
damping updates.
"""
struct GaussNewtonSolver{I<:InnerSolveOptions} <: AbstractSecondOrderROSolver
    linear_solver::Symbol
    inner::I
    damping::DampingPolicy
end

function GaussNewtonSolver(;
    linear_solver::Symbol = :cg,
    inner::InnerSolveOptions = _default_symcpd_inner_options(),
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
    _symcpd_cg(model, p, b, damping; tol, maxiter)

Solve `(J\*J + damping*I)X = b` in the Riemannian tangent metric by conjugate
gradients. Each Krylov iteration calls [`normal_operator!`](@ref); no Jacobian
or normal matrix is stored. The fourth return value records the initial and
final residual norms, achieved relative residual, requested tolerance, and
termination reason.

This is the symmetric intrinsic analogue of the implicit normal products used
by N. Singh, L. Ma, H. Yang, and E. Solomonik, "Comparison of Accuracy and
Scalability of Gauss--Newton and Alternating Least Squares for CANDECOMC/PARAFAC
Decomposition," *SIAM Journal on Scientific Computing* 43(4) (2021),
C290--C311, doi:10.1137/20M1344561. Matrix-free nonlinear least squares for CPD
and BTD was introduced earlier by L. Sorber, M. Van Barel, and L. De Lathauwer,
*SIAM Journal on Optimization* 23(2) (2013), 695--720,
doi:10.1137/120868323.
"""
function _symcpd_cg(
    model::JoinModel{T,B},
    p,
    b,
    damping::T;
    tol::T,
    maxiter::Int,
    absolute::Bool = false,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    M = model.backend.product_manifold
    damped_normal =
        direction ->
            normal_operator(model, p, direction) + _scale_solver_tangent(direction, damping)
    return _tangent_cg(M, p, damped_normal, b; tol, maxiter, absolute)
end

@doc raw"""
    _solve_symcpd_gn(solver, model; ...)

Run damped Riemannian Gauss--Newton on a symmetric [`JoinModel`](@ref). The
step solves

```math
(J(p)^*J(p)+\mu I)\eta=-\operatorname{grad}f(p).
```

`linear_solver=:cg` uses the analytic matrix-free normal action and tangent
conjugate gradients. `linear_solver=:dense` constructs the small intrinsic
normal matrix in an orthonormal tangent basis and solves it directly; it is a
reference/benchmark option, not the scalable path.

For a trial step `eta`, the Levenberg--Marquardt acceptance ratio is

```math
\operatorname{pred}(\eta)
=-\langle g,\eta\rangle
-\tfrac12\langle\eta,J^*J\eta\rangle,
\qquad
\rho=\frac{f(p)-f(R_p(\eta))}{\operatorname{pred}(\eta)}.
```

A trial is accepted only when its predicted reduction is positive and `rho`
exceeds `acceptance_ratio`. Poor trials increase the damping and good trials
decrease it. The predicted reduction intentionally uses the undamped
Gauss--Newton model; damping controls the step rather than redefining the
reported model agreement.

With `inner=InnerSolveOptions(tolerance=AdaptiveResidualTolerance(...))`, the
inner relative residual tolerance is the forcing term

```math
\xi_k=\operatorname{clamp}
\left(c\|\operatorname{grad}f(p_k)\|^\theta,
\xi_{\min},\xi_{\max}\right),
```

where `maximum` is ``\xi_{\max}``, `minimum` is ``\xi_{\min}``, and `scale`
and `power` define the forcing schedule. Thus early linear systems may be
solved approximately while the requested accuracy tightens near a stationary
point. `RelativeResidualTolerance` selects a fixed relative threshold and
`AbsoluteResidualTolerance` selects an absolute residual threshold.

This relative-residual forcing condition follows R. S. Dembo, S. C. Eisenstat,
and T. Steihaug, "Inexact Newton Methods," *SIAM Journal on Numerical
Analysis* 19(2) (1982), 400--408, doi:10.1137/0719025; the power-law schedule
above is TensorKitchen's bounded specialization for the damped GN system.

An inexact CG direction may be used before the inner residual reaches its
tolerance. `solver_info` records convergence, iteration count, requested
tolerance, achieved residual, and termination reason for every CG trial rather
than silently discarding the inner status. A small accepted step stops the
iteration as stagnation but does not by itself mark the solve as converged;
convergence requires the outer gradient tolerance.

The product-of-Veronese Riemannian GN formulation follows Khouja, Khalil, and
Mourrain (2022), doi:10.1016/j.laa.2021.12.008. Their published TensorDec
implementation assembles a dense normal matrix. The `:cg` option instead uses
the operator/Krylov pattern established for tensor GN by Sorber, Van Barel, and
De Lathauwer (2013), doi:10.1137/120868323, and Singh et al. (2021),
doi:10.1137/20M1344561.
"""
function _solve_symcpd_gn(
    solver::GaussNewtonSolver,
    model::JoinModel{T,B};
    init = :auto,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-8,
    verbose::Bool = true,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative."))
    tol > 0 || throw(ArgumentError("tol must be positive."))

    M = model.backend.product_manifold
    linear_solver = solver.linear_solver
    damping_policy = solver.damping
    inner_options = solver.inner
    inner_maxiter = _inner_maxiter(inner_options, manifold_dimension(M))
    requested_init = isnothing(p0) ? _symcpd_init_label(init) : :explicit
    resolved_spec = isnothing(p0) ? _resolve_symcpd_init(model, init) : :explicit
    resolved_init = isnothing(p0) ? _symcpd_init_label(resolved_spec) : :explicit
    p_initial = isnothing(p0) ? initial_point(model, resolved_spec; verbose) : p0
    p = join_solver_point(M, deepcopy(p_initial))
    retraction_method = _solver_retraction_method(M, p)
    current_cost = cost(model, p)
    mu = T(damping_policy.initial)
    tolerance = T(tol)
    inner_iterations_history = Int[]
    inner_converged_history = Bool[]
    inner_tolerance_history = T[]
    inner_initial_residual_history = T[]
    inner_final_residual_history = T[]
    inner_termination_history = Symbol[]
    accepted_steps = 0
    rejected_steps = 0
    predicted_reduction_history = T[]
    actual_reduction_history = T[]
    rho_history = T[]
    damping_history = T[]
    step_accepted_history = Bool[]
    iterations_done = 0
    converged_flag = false
    termination_reason = :maxiter
    final_gradient = rgrad(model, p)
    final_gradient_norm = norm(M, p, final_gradient)
    basis = ManifoldsBase.DefaultOrthonormalBasis()

    for iteration = 1:maxiter
        gradient = rgrad(model, p)
        gradient_norm = norm(M, p, gradient)
        final_gradient = gradient
        final_gradient_norm = gradient_norm
        if verbose
            println(
                "SymCPD GN iteration $iteration: cost=$(current_cost), " *
                "grad_norm=$(gradient_norm), damping=$(mu)",
            )
        end
        if gradient_norm <= tolerance
            converged_flag = true
            termination_reason = :gradient_tolerance
            iterations_done = iteration - 1
            break
        end

        accepted = false
        accepted_step_norm = T(Inf)
        effective_inner_tolerance = _inner_tolerance(inner_options.tolerance, gradient_norm)
        for _ = 1:damping_policy.max_trials
            push!(damping_history, mu)
            step = if linear_solver == :cg
                rhs = _scale_solver_tangent(gradient, -one(T))
                candidate_step, inner_iterations, inner_converged, inner_info =
                    _symcpd_cg(
                        model,
                        p,
                        rhs,
                        mu;
                        tol = effective_inner_tolerance,
                        maxiter = inner_maxiter,
                        absolute = inner_options.tolerance isa AbsoluteResidualTolerance,
                    )
                push!(inner_iterations_history, inner_iterations)
                push!(inner_converged_history, inner_converged)
                push!(inner_tolerance_history, effective_inner_tolerance)
                push!(inner_initial_residual_history, inner_info.initial_residual_norm)
                push!(inner_final_residual_history, inner_info.final_residual_norm)
                push!(inner_termination_history, inner_info.termination_reason)
                candidate_step
            else
                H = dense_normal_matrix(model, p)
                gradient_coordinates =
                    ManifoldsBase.get_coordinates(M, p, gradient, basis)
                H[diagind(H)] .+= mu
                step_coordinates = -(H \ gradient_coordinates)
                ManifoldsBase.get_vector(M, p, step_coordinates, basis)
            end
            step_norm = norm(M, p, step)
            if !isfinite(step_norm)
                push!(predicted_reduction_history, T(NaN))
                push!(actual_reduction_history, T(NaN))
                push!(rho_history, T(-Inf))
                push!(step_accepted_history, false)
                rejected_steps += 1
                mu = _increase_damping(damping_policy, mu)
                continue
            end
            normal_step = normal_operator(model, p, step)
            predicted_reduction =
                -ManifoldsBase.inner(M, p, gradient, step) -
                T(0.5) * ManifoldsBase.inner(M, p, step, normal_step)
            push!(predicted_reduction_history, predicted_reduction)
            if !isfinite(predicted_reduction) || predicted_reduction <= zero(T)
                push!(actual_reduction_history, T(NaN))
                push!(rho_history, T(-Inf))
                push!(step_accepted_history, false)
                rejected_steps += 1
                mu = _increase_damping(damping_policy, mu)
                continue
            end

            candidate = try
                _independent_retract(M, p, step, retraction_method)
            catch
                nothing
            end
            candidate_cost = isnothing(candidate) ? T(Inf) : cost(model, candidate)
            actual_reduction = current_cost - candidate_cost
            rho = actual_reduction / predicted_reduction
            trial_accepted =
                isfinite(candidate_cost) &&
                isfinite(rho) &&
                rho >= T(damping_policy.acceptance_threshold)
            push!(actual_reduction_history, actual_reduction)
            push!(rho_history, rho)
            push!(step_accepted_history, trial_accepted)

            if trial_accepted
                p = candidate
                current_cost = candidate_cost
                accepted = true
                accepted_steps += 1
                accepted_step_norm = step_norm
                mu = _update_accepted_damping(damping_policy, mu, rho)
                break
            end

            rejected_steps += 1
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
    if final_gradient_norm <= tolerance
        converged_flag = true
        termination_reason = :gradient_tolerance
    end
    norm2 = target_norm2(model.backend.target)
    residual_norm2 = max(T(2) * current_cost, zero(T))
    relative_error = norm2 > zero(T) ? sqrt(residual_norm2 / norm2) : sqrt(residual_norm2)
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
                inner_iterations_history,
                inner_tolerance_history,
                inner_initial_residual_history,
                inner_final_residual_history,
                inner_termination_history;
                solver = linear_solver,
                policy = inner_options.tolerance,
                maxiter = inner_maxiter,
                converged = inner_converged_history,
            ),
            accepted_steps = accepted_steps,
            rejected_steps = rejected_steps,
            predicted_reduction_history = predicted_reduction_history,
            actual_reduction_history = actual_reduction_history,
            rho_history = rho_history,
            damping_history = damping_history,
            step_accepted_history = step_accepted_history,
            acceptance_ratio = T(damping_policy.acceptance_threshold),
            poor_step_ratio = T(damping_policy.increase_threshold),
            good_step_ratio = T(damping_policy.reduction_threshold),
            final_damping = mu,
            damping_policy = damping_policy,
            termination_reason = termination_reason,
        ),
    )
end

function solve(
    solver::GaussNewtonSolver,
    model::JoinModel{T,B};
    init = :auto,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-8,
    verbose::Bool = true,
    return_stats::Bool = false,
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    result = _solve_symcpd_gn(solver, model; init, p0, maxiter, tol, verbose)
    return return_stats ? result : result.point
end
