# results/rel_error.jl — explicit reconstruction error helpers
export rel_error, reconstruction_error, relative_error

"""
    rel_error(A, x)

Relative Frobenius reconstruction error `‖A - Â‖_F / ‖A‖_F` (absolute `‖A-Â‖_F` if `‖A‖_F = 0`),
computed from an explicit approximation `Â`.

Supported second arguments:
- `Â::AbstractArray`
- [`CPDResult`](@ref), [`TuckerResult`](@ref), [`ApproxResult`](@ref), [`BTDResult`](@ref)
- `(core, factors)` for Tucker via `rel_error(A, core, factors)`

# Solver field `rel_error` vs. this function

`cpd` / `btd` / join solvers set `result.rel_error` (a **field**) from fast in-loop statistics. This
function** recomputes the error from the explicit reconstruction.
"""
function rel_error(A::AbstractArray, Ahat::AbstractArray)
    return relative_frobenius_error(A, Ahat)
end

function rel_error(A::AbstractArray, r::CPDResult)
    return relative_frobenius_error(A, reconstruct(r))
end

function rel_error(A::AbstractArray, td::TuckerResult)
    return relative_frobenius_error(A, reconstruct(td))
end

function rel_error(A::AbstractArray, r::ApproxResult)
    return relative_frobenius_error(A, reconstruct(r))
end

function rel_error(A::AbstractArray, r::BTDResult)
    return relative_frobenius_error(A, reconstruct(r))
end

function rel_error(A::AbstractArray, core, factors)
    return relative_frobenius_error(A, reconstruct_tucker(core, factors))
end

reconstruction_error(A, core, factors) = rel_error(A, core, factors)
relative_error(A::AbstractArray, td::TuckerResult) = rel_error(A, td)
