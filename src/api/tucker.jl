export tucker

"""
    tucker(A, ranks; method=:sthosvd, kwargs...)
    Computes Tucker decomposition. `method`: `:sthosvd` (default) or `:hooi`.
"""
function tucker(A, ranks; method::Symbol = :sthosvd, kwargs...)
    fn = (sthosvd = sthosvd, hooi = hooi)
    f = get(fn, method) do
        throw(ArgumentError("Unknown method=$method. Use :sthosvd or :hooi."))
    end
    return f(A, ranks; kwargs...)
end
