# data/compute_array.jl — storage/compute precision separation

export ComputeArray,
    prepare_tensor,
    materialize_tensor,
    raw_storage,
    convert_block!,
    foreach_compute_block,
    storage_type,
    compute_type,
    is_materialized

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

function Base.copy(::ComputeArray)
    throw(
        ArgumentError(
            "copy(::ComputeArray) would materialize the full compute tensor; " *
            "call materialize_tensor(A) explicitly if that is intended",
        ),
    )
end

"""Return the ultimate physical storage, unwrapping any compute views."""
raw_storage(A::ComputeArray) = raw_storage(parent(A))
raw_storage(A::AbstractArray) = A

ComputeArray{T}(storage::ComputeArray) where {T<:AbstractFloat} =
    ComputeArray{T}(raw_storage(storage))

"""Return the element type used by the underlying stored data."""
storage_type(A::AbstractArray) = eltype(raw_storage(A))

"""Return the floating-point type exposed to numerical kernels."""
compute_type(A::AbstractArray) = eltype(A)

"""Return whether `A` already stores values in its exposed compute type."""
is_materialized(::ComputeArray) = false
is_materialized(::AbstractArray) = true

@inline _default_compute_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _default_compute_type(::Type{<:Union{Int8,UInt8,Int16,UInt16}}) = Float32
@inline _default_compute_type(::Type{T}) where {T<:Integer} = Float64
@inline _default_compute_type(::Type{T}) where {T<:Real} = float(T)

function _resolve_compute_type(A::AbstractArray{<:Real}, requested)
    T = isnothing(requested) ? _default_compute_type(eltype(A)) : requested
    T isa Type && T <: AbstractFloat || throw(
        ArgumentError("compute_type must be a floating-point type; received $requested"),
    )
    return T
end

"""
    convert_block!(dest, source, indices)

Convert the linear `indices` from `source` into the bounded compute buffer
`dest`. `length(dest)` must equal `length(indices)`.
"""
function convert_block!(
    dest::AbstractVector{TC},
    source::AbstractArray{<:Real},
    indices::AbstractUnitRange{<:Integer},
) where {TC<:AbstractFloat}
    length(dest) == length(indices) || throw(
        DimensionMismatch(
            "destination length $(length(dest)) must equal block length $(length(indices))",
        ),
    )
    raw = raw_storage(source)
    @inbounds for (output_index, source_index) in enumerate(indices)
        dest[output_index] = TC(raw[source_index])
    end
    return dest
end

"""
    foreach_compute_block(f, A; compute_type=nothing, block_length=65_536)

Call `f(block, indices)` for consecutive linear blocks of `A`. `block` is a
compute-precision view into a bounded scratch buffer that is reused on the next
iteration; callers must copy it if they need to retain it.
"""
function foreach_compute_block(
    f,
    A::AbstractArray{<:Real};
    compute_type = nothing,
    block_length::Int = 65_536,
)
    block_length >= 1 || throw(ArgumentError("block_length must be positive"))
    raw = raw_storage(A)
    requested = isnothing(compute_type) && A isa ComputeArray ? eltype(A) : compute_type
    T = _resolve_compute_type(raw, requested)
    buffer = Vector{T}(undef, min(block_length, length(raw)))
    first_index = firstindex(raw)
    last_index = lastindex(raw)
    for start = first_index:block_length:last_index
        stop = min(start + block_length - 1, last_index)
        indices = start:stop
        block = @view buffer[1:length(indices)]
        convert_block!(block, raw, indices)
        f(block, indices)
    end
    return nothing
end

"""
    materialize_tensor(A, T=compute_type(A); block_length=65_536)

Allocate a dense array with element type `T` and copy/convert `A` into it in linear blocks. This is the explicit opt-in path for algorithms that require a materialized compute-precision tensor.
"""
function materialize_tensor(
    A::AbstractArray{<:Real,N},
    ::Type{T};
    block_length::Int = 65_536,
) where {T<:AbstractFloat,N}
    block_length >= 1 || throw(ArgumentError("block_length must be positive"))
    raw = raw_storage(A)
    out = similar(raw, T, size(A))
    foreach_compute_block(raw; compute_type = T, block_length) do block, indices
        @inbounds for (block_index, output_index) in enumerate(indices)
            out[output_index] = block[block_index]
        end
    end
    return out
end


function materialize_tensor(A::AbstractArray{<:Real}; block_length::Int = 65_536)
    requested = A isa ComputeArray ? eltype(A) : nothing
    T = _resolve_compute_type(raw_storage(A), requested)
    return materialize_tensor(A, T; block_length)
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
    raw = raw_storage(A)
    requested = isnothing(compute_type) && A isa ComputeArray ? eltype(A) : compute_type
    T = _resolve_compute_type(raw, requested)
    if materialize
        return materialize_tensor(raw, T; block_length)
    elseif eltype(raw) === T
        return raw
    elseif A isa ComputeArray{T} && parent(A) === raw
        return A
    end
    return ComputeArray{T}(raw)
end
