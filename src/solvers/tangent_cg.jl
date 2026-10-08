# Conjugate gradients for a self-adjoint positive-definite tangent operator.

"""
    _tangent_cg(M, p, operator, b; tol, maxiter)

Approximately solve `operator(X) = b` in the tangent space at `p` using the
manifold's tangent metric. `operator` supplies the model-specific linear action;
this routine does not construct a matrix or assume a particular decomposition.

Return `(solution, iterations, converged, info)`, where `info` contains the
initial and final residual norms, relative residual, requested tolerance, and
termination reason. The caller is responsible for supplying a self-adjoint,
positive-definite operator in this metric (for example, a damped normal action).
"""
function _tangent_cg(M, p, operator, b; tol::T, maxiter::Int) where {T<:AbstractFloat}
    0 < tol < 1 || throw(ArgumentError("CG tol must lie in (0, 1)."))
    maxiter >= 0 || throw(ArgumentError("CG maxiter must be nonnegative."))
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
    termination_reason = isfinite(initial_norm) ? :maxiter : :nonfinite_residual
    final_norm = initial_norm
    for k = 1:maxiter
        isfinite(residual_norm2) || break
        action = operator(direction)
        curvature = inner(M, p, direction, action)
        if !isfinite(curvature) || curvature <= zero(T)
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
        if !isfinite(next_norm2)
            termination_reason = :nonfinite_residual
            break
        end
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
