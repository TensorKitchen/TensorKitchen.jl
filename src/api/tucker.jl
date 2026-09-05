export tucker

"""
    tucker(A, ranks; method=:sthosvd, kwargs...)

Compress `A` into a Tucker core and one factor matrix per tensor mode.

# Inputs

- `A`: numerical input tensor.
- `ranks`: tuple with one retained rank per mode of `A`.

# Output

Returns a [`TuckerResult`](@ref). Use `core` and `factors` to access the compact
representation, `reconstruct` to rebuild the
approximation, and `rel_error(A, result)` to measure reconstruction error.

# Common options

- `method=:sthosvd` performs a direct decomposition.
- `method=:hooi` iteratively refines a Tucker decomposition.
- `svd_backend=:randomized` can reduce memory use for large inputs when used
  with `method=:sthosvd`.

# Example

```julia
A = randn(20, 15, 10)
result = tucker(A, (5, 4, 3))
compressed = core(result)
A_approx = reconstruct(result)
```
"""
function tucker(A, ranks; method::Symbol = :sthosvd, kwargs...)
    return _tucker(Val(method), A, ranks; kwargs...)
end

_tucker(::Val{:sthosvd}, A, ranks; kwargs...) = sthosvd(A, ranks; kwargs...)
_tucker(::Val{:hooi}, A, ranks; kwargs...) = hooi(A, ranks; kwargs...)

function _tucker(::Val{M}, A, ranks; kwargs...) where {M}
    throw(ArgumentError("Unknown method=$M. Use :sthosvd or :hooi."))
end
