# results/rel_error.jl — explicit reconstruction error helpers
export rel_error, reconstruction_error, relative_error

"""
    rel_error(A, x)

Relative Frobenius reconstruction error `‖A - Â‖_F / ‖A‖_F` (absolute `‖A-Â‖_F` if `‖A‖_F = 0`),
computed from an explicit approximation `Â`.

Supported second arguments:
- `Â::AbstractArray`
- [`CPDResult`](@ref) / [`ApproxResult`](@ref) / [`BTDResult`](@ref) as `res` (as in `res = cpd(A, r)`)
- [`TuckerResult`](@ref) as `tucker_res` (as in `tucker_res = tucker(A, mlrank)`)
- `(core, factors)` for Tucker via `rel_error(A, core, factors)`

"""
function rel_error(A::AbstractArray, Ahat::AbstractArray)
    return relative_frobenius_error(A, Ahat)
end

function rel_error(A::AbstractArray, res::Union{CPDResult,TuckerResult,ApproxResult,BTDResult})
    return relative_frobenius_error(A, reconstruct(res))
end

function rel_error(A::AbstractArray, core, factors)
    return relative_frobenius_error(A, reconstruct_tucker(core, factors))
end

reconstruction_error(A, core, factors) = rel_error(A, core, factors)
relative_error(A::AbstractArray, tucker_res::TuckerResult) = rel_error(A, tucker_res)
