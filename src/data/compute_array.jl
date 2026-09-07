# data/compute_array.jl — storage/compute precision separation

export ComputeArray,
    prepare_tensor, materialize_tensor, storage_type, compute_type, is_materialized

"""
    ComputeArray{T}(storage)

A read-only `AbstractArray` view that keeps `storage` in its native element type and converts each value to floating-point type `T` only when it is read.

`ComputeArray` never allocates a full converted copy. It is intended for exact blockwise or streaming kernels where, for example, `Int16` observations should remain stored as `Int16` while arithmetic is performed in `Float32`.
"""
struct ComputeArray{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    storage::A

    function ComputeArray{T}(
        storage::A,
    ) where {T<:AbstractFloat,N,A<:AbstractArray{<:Real,N}}
        return new{T,N,A}(storage)
    end
end

Base.parent(A::ComputeArray) = A.storage
Base.size(A::ComputeArray) = size(parent(A))
Base.axes(A::ComputeArray) = axes(parent(A))
Base.length(A::ComputeArray) = length(parent(A))
Base.IndexStyle(::Type{<:ComputeArray{T,N,A}}) where {T,N,A} = Base.IndexStyle(A)

@inline function Base.getindex(A::ComputeArray{T}, I...) where {T}
    return T(@inbounds getindex(parent(A), I...))
end

function Base.similar(A::ComputeArray, ::Type{T}, dims::Dims) where {T}
    return similar(parent(A), T, dims)
end

Base.copy(A::ComputeArray{T}) where {T} = materialize_tensor(A, T)

"""Return the element type used by the underlying stored data."""
storage_type(A::ComputeArray) = eltype(parent(A))
storage_type(A::AbstractArray) = eltype(A)

"""Return the floating-point type exposed to numerical kernels."""
compute_type(A::AbstractArray) = eltype(A)

"""Return whether `A` already stores values in its exposed compute type."""
is_materialized(::ComputeArray) = false
is_materialized(::AbstractArray) = true

@inline _default_compute_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _default_compute_type(::Type{T}) where {T<:Real} = float(T)

function _resolve_compute_type(A::AbstractArray{<:Real}, requested)
    T = isnothing(requested) ? _default_compute_type(eltype(A)) : requested
    T isa Type && T <: AbstractFloat || throw(
        ArgumentError("compute_type must be a floating-point type; received $requested"),
    )
    return T
end

"""
    materialize_tensor(A, T=compute_type(A); block_length=65_536)

Allocate a dense array with element type `T` and copy/convert `A` into it in linear blocks. This is the explicit opt-in path for algorithms that require a materialized compute-precision tensor.
"""
function materialize_tensor(
    A::AbstractArray{<:Real,N},
    ::Type{T} = compute_type(A);
    block_length::Int = 65_536,
) where {T<:AbstractFloat,N}
    block_length >= 1 || throw(ArgumentError("block_length must be positive"))
    out = similar(A isa ComputeArray ? parent(A) : A, T, size(A))
    source = A isa ComputeArray{T} ? A : ComputeArray{T}(A)
    first_index = firstindex(source)
    last_index = lastindex(source)
    for start = first_index:block_length:last_index
        stop = min(start + block_length - 1, last_index)
        @inbounds for index = start:stop
            out[index] = source[index]
        end
    end
    return out
end

"""
    prepare_tensor(A; compute_type=nothing, materialize=false,
                   block_length=65_536)

Prepare real-valued raw data for TensorKitchen numerical kernels.

- `materialize=false` (default) returns `A` unchanged when it already has the
  requested floating-point type, otherwise returns a lazy [`ComputeArray`](@ref).
- `materialize=true` explicitly allocates a full compute-precision copy using
  [`materialize_tensor`](@ref).

This function is mathematically separate from randomized sketching: it changes
storage and arithmetic representation only and never discards observations.
"""
function prepare_tensor(
    A::AbstractArray{<:Real};
    compute_type = nothing,
    materialize::Bool = false,
    block_length::Int = 65_536,
)
    T = _resolve_compute_type(A, compute_type)
    if materialize
        return materialize_tensor(A, T; block_length)
    elseif eltype(A) === T
        return A
    elseif A isa ComputeArray{T}
        return A
    end
    return ComputeArray{T}(A)
end
