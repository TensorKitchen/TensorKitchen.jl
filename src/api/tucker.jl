export tucker

"""
    tucker(A, ranks; method=:sthosvd, compute_type=nothing,
        materialize=false, conversion_block_length=65_536, kwargs...)

Compress `A` into a Tucker core and one factor matrix per tensor mode.

# Inputs

- `A`: real-valued input tensor. Storage conversion can remain lazy when the
  randomized ST-HOSVD path is selected.
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
- `compute_type=nothing` selects the arithmetic precision from the input
  storage type.
- `materialize=false` avoids a full compute-precision input copy. A lazy
  converted input currently requires `method=:sthosvd` and
  `svd_backend=:randomized`; set `materialize=true` for exact ST-HOSVD or HOOI.
- `conversion_block_length=65_536` controls the bounded conversion buffer.

# Example

```julia
A = randn(20, 15, 10)
result = tucker(A, (5, 4, 3))
compressed = core(result)
A_approx = reconstruct(result)
```
"""
function tucker(
    A::AbstractArray{<:Real},
    ranks;
    method::Symbol = :sthosvd,
    compute_type = nothing,
    materialize::Bool = false,
    conversion_block_length::Int = 65_536,
    kwargs...,
)
    A_prepared =
        prepare_tensor(A; compute_type, materialize, block_length = conversion_block_length)
    if A_prepared isa ComputeArray
        method === :sthosvd || throw(
            ArgumentError(
                "A lazy ComputeArray currently supports Tucker decomposition only with " *
                "method=:sthosvd and svd_backend=:randomized. Set materialize=true " *
                "to use HOOI.",
            ),
        )
        get(kwargs, :svd_backend, :exact) === :randomized || throw(
            ArgumentError(
                "A lazy ComputeArray requires svd_backend=:randomized for ST-HOSVD. " *
                "Set svd_backend=:randomized, or set materialize=true for the exact backend.",
            ),
        )
    end
    return _tucker(Val(method), A_prepared, ranks; kwargs...)
end

_tucker(::Val{:sthosvd}, A, ranks; kwargs...) = sthosvd(A, ranks; kwargs...)
_tucker(::Val{:hooi}, A, ranks; kwargs...) = hooi(A, ranks; kwargs...)

function _tucker(::Val{M}, A, ranks; kwargs...) where {M}
    throw(ArgumentError("Unknown method=$M. Use :sthosvd or :hooi."))
end
