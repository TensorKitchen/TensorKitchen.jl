# kernels/implicit_norm.jl — observation-preserving reductions

export observation_norm2, observation_stats

@inline _observation_accumulator_type(::Type{T}) where {T<:AbstractFloat} = T
@inline _observation_accumulator_type(::Type{T}) where {T<:Union{Float16,Float32}} = Float64

"""
    observation_norm2(A; compute_type=nothing, block_length=65_536)

Compute the squared Frobenius norm from every observation. Native
floating-point arrays use their backend reduction; lazy or converting inputs
use one bounded compute-precision buffer. No randomized approximation or full
converted tensor is created. Streaming Float16 and Float32 reductions accumulate
in Float64 and round once to the selected compute type.
"""
function observation_norm2(
    A::AbstractArray{<:Real};
    compute_type = nothing,
    block_length::Int = 65_536,
)
    requested = isnothing(compute_type) && A isa ComputeArray ? eltype(A) : compute_type
    T = _resolve_compute_type(raw_storage(A), requested)
    if !(A isa ComputeArray) && eltype(A) === T
        return sum(abs2, A)
    end
    Tacc = _observation_accumulator_type(T)
    total = zero(Tacc)
    foreach_compute_block(A; compute_type = T, block_length) do block, _
        @inbounds for value in block
            total += abs2(Tacc(value))
        end
    end
    return T(total)
end

"""
    observation_stats(A; compute_type=nothing, block_length=65_536)

Inspect every observation in one bounded-memory pass. The returned named tuple
contains:

- `norm2`: squared Frobenius norm in the selected compute type,
- `has_nonfinite`: whether any value is `NaN` or infinite,
- `minimum`: the smallest converted value, or `nothing` for an empty tensor,
- `has_negative`: whether any value is less than zero.

Like [`observation_norm2`](@ref), this operation neither sketches the data nor
creates a full compute-precision copy. Its Float16 and Float32 squared-norm
reduction uses a Float64 accumulator before returning `norm2` in the selected
compute type.
"""
function observation_stats(
    A::AbstractArray{<:Real};
    compute_type = nothing,
    block_length::Int = 65_536,
)
    requested = isnothing(compute_type) && A isa ComputeArray ? eltype(A) : compute_type
    T = _resolve_compute_type(raw_storage(A), requested)
    Tacc = _observation_accumulator_type(T)
    norm2_acc = zero(Tacc)
    minimum_value = nothing
    has_nonfinite = false
    has_negative = false

    if !(A isa ComputeArray) && eltype(A) === T
        @inbounds for value in A
            norm2_acc += abs2(Tacc(value))
            has_nonfinite |= !isfinite(value)
            has_negative |= value < zero(T)
            minimum_value = isnothing(minimum_value) ? value : min(minimum_value, value)
        end
        norm2 = T(norm2_acc)
        return (; norm2, has_nonfinite, minimum = minimum_value, has_negative)
    end

    foreach_compute_block(A; compute_type = T, block_length) do block, _
        @inbounds for value in block
            norm2_acc += abs2(Tacc(value))
            has_nonfinite |= !isfinite(value)
            has_negative |= value < zero(T)
            minimum_value = isnothing(minimum_value) ? value : min(minimum_value, value)
        end
    end

    norm2 = T(norm2_acc)
    return (; norm2, has_nonfinite, minimum = minimum_value, has_negative)
end
