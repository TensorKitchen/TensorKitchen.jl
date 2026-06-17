# results/conversion.jl — optimization result wrappers

_result_solver_symbol(solver::Symbol) = solver
_result_solver_symbol(solver) = :unknown
_result_solver_info(result) = hasproperty(result, :solver_info) ? solver_info(result) : (;)

function _to_join_result(result_type, model::JoinModel{T}, result) where {T<:AbstractFloat}
    comps = extract_components(model, result.point)
    solver_sym = _result_solver_symbol(result.solver)
    si = _result_solver_info(result)
    return result_type(
        result.point,
        comps,
        result.cost,
        result.rel_error,
        result.grad_norm,
        result.iterations,
        result.converged,
        solver_sym,
        si,
    )
end

_to_approx_result(model::JoinModel{T}, result) where {T<:AbstractFloat} =
    _to_join_result(ApproxResult, model, result)

# Reuse solver-reported cost/error instead of reconstructing the full BTD residual again.
_to_btd_result(model::JoinModel{T}, result) where {T<:AbstractFloat} =
    _to_join_result(BTDResult, model, result)

_to_cpd_result(model, result, dims, r) =
    throw(ArgumentError("No CPD result converter for model $(typeof(model))."))

_to_cpd_result(model::JoinModel{<:AbstractFloat,<:CPDBackend}, result, dims, r) =
    _cpd_result(model, result, dims, r)
