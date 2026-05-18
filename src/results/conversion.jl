# results/conversion.jl — optimization result wrappers

_result_solver_symbol(solver::Symbol) = solver
_result_solver_symbol(solver) = :unknown

function _to_approx_result(model::JoinModel{T}, result) where {T<:AbstractFloat}
    comps = extract_components(model, result.point)
    solver_sym = _result_solver_symbol(result.solver)
    solver_info = hasproperty(result, :solver_info) ? result.solver_info : (;)
    return ApproxResult(
        result.point,
        comps,
        result.cost,
        result.rel_error,
        result.grad_norm,
        result.iterations,
        result.converged,
        solver_sym,
        solver_info,
    )
end

function _to_btd_result(model::JoinModel{T}, result) where {T<:AbstractFloat}
    comps = extract_components(model, result.point)
    solver_sym = _result_solver_symbol(result.solver)
    solver_info = hasproperty(result, :solver_info) ? result.solver_info : (;)
    # Reuse solver-reported cost/error instead of reconstructing the full BTD residual again.
    return BTDResult(
        result.point,
        comps,
        result.cost,
        result.rel_error,
        result.grad_norm,
        result.iterations,
        result.converged,
        solver_sym,
        solver_info,
    )
end

_to_cpd_result(model, result, dims, r) =
    throw(ArgumentError("No CPD result converter for model $(typeof(model))."))

_to_cpd_result(model::JoinModel{<:AbstractFloat,<:CPDBackend}, result, dims, r) =
    _cpd_result(model, result, dims, r)
