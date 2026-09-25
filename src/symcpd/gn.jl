# Damped Riemannian Gauss--Newton for the structured symmetric model.

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
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    0 < tol < 1 || throw(ArgumentError("CG tol must lie in (0, 1)."))
    M = model.backend.product_manifold
    solution = ManifoldsBase.zero_vector(M, p)
    residual = copy(b)
    direction = copy(residual)
    residual_norm2 = inner(M, p, residual, residual)
    initial_norm = sqrt(max(residual_norm2, zero(T)))
    threshold = tol * initial_norm
    if initial_norm == zero(T)
        info = (
            initial_residual_norm = zero(T),
            final_residual_norm = zero(T),
            relative_residual = zero(T),
            threshold = zero(T),
            tolerance = tol,
            termination_reason = :initial_residual,
        )
        return solution, 0, true, info
    end
    iterations = 0
    converged = false
    termination_reason = :maxiter
    final_norm = initial_norm
    for k = 1:maxiter
        action = normal_operator(model, p, direction)
        action = action + _scale_solver_tangent(direction, damping)
        curvature = inner(M, p, direction, action)
        if !isfinite(curvature) || curvature <= eps(T) * max(residual_norm2, one(T))
            termination_reason =
                isfinite(curvature) ? :nonpositive_curvature : :nonfinite_curvature
            break
        end
        alpha = residual_norm2 / curvature
        solution = solution + _scale_solver_tangent(direction, alpha)
        residual = residual - _scale_solver_tangent(action, alpha)
        next_norm2 = inner(M, p, residual, residual)
        iterations = k
        final_norm = sqrt(max(next_norm2, zero(T)))
        if final_norm <= threshold
            converged = true
            termination_reason = :converged
            break
        end
        beta = next_norm2 / residual_norm2
        direction = residual + _scale_solver_tangent(direction, beta)
        residual_norm2 = next_norm2
    end
    info = (
        initial_residual_norm = initial_norm,
        final_residual_norm = final_norm,
        relative_residual = final_norm / initial_norm,
        threshold = threshold,
        tolerance = tol,
        termination_reason = termination_reason,
    )
    return solution, iterations, converged, info
end

@doc raw"""
    _solve_symcpd_gn(model; linear_solver=:cg, ...)

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

With `adaptive_cg=true`, the inner relative residual tolerance is the forcing
term

```math
\xi_k=\operatorname{clamp}
\left(c\|\operatorname{grad}f(p_k)\|^\theta,
\xi_{\min},\xi_{\max}\right),
```

where `cg_tol` is ``\xi_{\max}``, `cg_min_tol` is ``\xi_{\min}``, and the
scale and exponent are `cg_forcing_scale` and `cg_forcing_power`. Thus early
linear systems may be solved approximately while the requested accuracy
tightens near a stationary point. `adaptive_cg=false` uses the fixed relative
tolerance `cg_tol`.

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
    model::JoinModel{T,B};
    init = :auto,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-8,
    linear_solver::Symbol = :cg,
    damping::Real = 1.0e-6,
    damping_increase::Real = 10,
    damping_decrease::Real = 0.3,
    cg_tol::Real = 1.0e-2,
    adaptive_cg::Bool = true,
    cg_min_tol::Real = 1.0e-10,
    cg_forcing_scale::Real = 1,
    cg_forcing_power::Real = 0.5,
    cg_maxiter::Int = max(20 * manifold_dimension(model.backend.product_manifold), 200),
    max_damping_trials::Int = 8,
    acceptance_ratio::Real = 1.0e-4,
    poor_step_ratio::Real = 0.25,
    good_step_ratio::Real = 0.75,
    verbose::Bool = true,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    linear_solver in (:cg, :dense) ||
        throw(ArgumentError("linear_solver must be :cg or :dense, got $linear_solver."))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative."))
    tol > 0 || throw(ArgumentError("tol must be positive."))
    damping > 0 || throw(ArgumentError("damping must be positive."))
    damping_increase > 1 || throw(ArgumentError("damping_increase must exceed one."))
    0 < damping_decrease <= 1 ||
        throw(ArgumentError("damping_decrease must lie in (0, 1]."))
    0 < cg_tol < 1 || throw(ArgumentError("cg_tol must lie in (0, 1)."))
    if adaptive_cg
        0 < cg_min_tol <= cg_tol ||
            throw(ArgumentError("Require 0 < cg_min_tol <= cg_tol for adaptive CG."))
        cg_forcing_scale > 0 || throw(ArgumentError("cg_forcing_scale must be positive."))
        0 < cg_forcing_power <= 1 ||
            throw(ArgumentError("cg_forcing_power must lie in (0, 1]."))
    end
    cg_maxiter > 0 || throw(ArgumentError("cg_maxiter must be positive."))
    max_damping_trials > 0 || throw(ArgumentError("max_damping_trials must be positive."))
    0 <= acceptance_ratio < poor_step_ratio < good_step_ratio <= 1 || throw(
        ArgumentError(
            "Require 0 <= acceptance_ratio < poor_step_ratio < good_step_ratio <= 1.",
        ),
    )

    M = model.backend.product_manifold
    requested_init = isnothing(p0) ? init : :explicit
    resolved_init = isnothing(p0) ? _resolve_symcpd_init(model, init) : :explicit
    p_initial = isnothing(p0) ? initial_point(model, resolved_init; verbose) : p0
    p = join_solver_point(M, deepcopy(p_initial))
    retraction_method = _solver_retraction_method(M, p)
    current_cost = cost(model, p)
    mu = T(damping)
    tolerance = T(tol)
    total_cg_iterations = 0
    cg_iterations_history = Int[]
    cg_converged_history = Bool[]
    cg_tolerance_history = T[]
    cg_initial_residual_history = T[]
    cg_final_residual_history = T[]
    cg_relative_residual_history = T[]
    cg_termination_history = Symbol[]
    cg_failed_count = 0
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
        effective_cg_tol = if adaptive_cg
            clamp(
                T(cg_forcing_scale) * gradient_norm^T(cg_forcing_power),
                T(cg_min_tol),
                T(cg_tol),
            )
        else
            T(cg_tol)
        end
        for _ = 1:max_damping_trials
            push!(damping_history, mu)
            step = if linear_solver == :cg
                rhs = _scale_solver_tangent(gradient, -one(T))
                candidate_step, cg_iterations, cg_converged, cg_info = _symcpd_cg(
                    model,
                    p,
                    rhs,
                    mu;
                    tol = effective_cg_tol,
                    maxiter = cg_maxiter,
                )
                total_cg_iterations += cg_iterations
                push!(cg_iterations_history, cg_iterations)
                push!(cg_converged_history, cg_converged)
                push!(cg_tolerance_history, effective_cg_tol)
                push!(cg_initial_residual_history, cg_info.initial_residual_norm)
                push!(cg_final_residual_history, cg_info.final_residual_norm)
                push!(cg_relative_residual_history, cg_info.relative_residual)
                push!(cg_termination_history, cg_info.termination_reason)
                cg_failed_count += !cg_converged
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
                mu *= T(damping_increase)
                continue
            end
            normal_step = normal_operator(model, p, step)
            predicted_reduction =
                -inner(M, p, gradient, step) - T(0.5) * inner(M, p, step, normal_step)
            push!(predicted_reduction_history, predicted_reduction)
            if !isfinite(predicted_reduction) || predicted_reduction <= zero(T)
                push!(actual_reduction_history, T(NaN))
                push!(rho_history, T(-Inf))
                push!(step_accepted_history, false)
                rejected_steps += 1
                mu *= T(damping_increase)
                continue
            end

            candidate = try
                retract(M, p, step, retraction_method)
            catch
                nothing
            end
            candidate_cost = isnothing(candidate) ? T(Inf) : cost(model, candidate)
            actual_reduction = current_cost - candidate_cost
            rho = actual_reduction / predicted_reduction
            trial_accepted =
                isfinite(candidate_cost) && isfinite(rho) && rho >= T(acceptance_ratio)
            push!(actual_reduction_history, actual_reduction)
            push!(rho_history, rho)
            push!(step_accepted_history, trial_accepted)

            if trial_accepted
                p = candidate
                current_cost = candidate_cost
                accepted = true
                accepted_steps += 1
                accepted_step_norm = step_norm
                if rho > T(good_step_ratio)
                    mu = max(mu * T(damping_decrease), eps(T))
                elseif rho < T(poor_step_ratio)
                    mu *= T(damping_increase)
                end
                break
            end

            rejected_steps += 1
            mu *= T(damping_increase)
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
            total_cg_iterations = total_cg_iterations,
            cg_iterations_history = cg_iterations_history,
            cg_converged_history = cg_converged_history,
            cg_tolerance_history = cg_tolerance_history,
            cg_initial_residual_history = cg_initial_residual_history,
            cg_final_residual_history = cg_final_residual_history,
            cg_relative_residual_history = cg_relative_residual_history,
            cg_termination_history = cg_termination_history,
            cg_failed_count = cg_failed_count,
            all_cg_converged = isempty(cg_converged_history) ? nothing :
                               all(cg_converged_history),
            adaptive_cg = adaptive_cg,
            cg_min_tol = T(cg_min_tol),
            cg_forcing_scale = T(cg_forcing_scale),
            cg_forcing_power = T(cg_forcing_power),
            accepted_steps = accepted_steps,
            rejected_steps = rejected_steps,
            predicted_reduction_history = predicted_reduction_history,
            actual_reduction_history = actual_reduction_history,
            rho_history = rho_history,
            damping_history = damping_history,
            step_accepted_history = step_accepted_history,
            acceptance_ratio = T(acceptance_ratio),
            poor_step_ratio = T(poor_step_ratio),
            good_step_ratio = T(good_step_ratio),
            final_damping = mu,
            termination_reason = termination_reason,
        ),
    )
end
