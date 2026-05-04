export tucker

"""
    tucker(A, ranks; method=:sthosvd, kwargs...) -> TuckerResult

Tucker decomposition. `method`: `:sthosvd` (default), `:thosvd`/`:hosvd`, or `:hooi`.
"""
function tucker(A, ranks; method::Symbol = :sthosvd, kwargs...)
    fn = (sthosvd = sthosvd, thosvd = thosvd, hosvd = thosvd, hooi = hooi)
    f = get(fn, method) do
        throw(ArgumentError("Unknown method=$method. Use :sthosvd, :thosvd, :hosvd, or :hooi."))
    end
    return f(A, ranks; kwargs...)
end
