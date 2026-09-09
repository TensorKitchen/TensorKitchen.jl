# kernels/implicit_norm.jl — observation-preserving reductions

export observation_norm2

"""
    observation_norm2(A; compute_type=nothing, block_length=65_536)

Compute the squared Frobenius norm from every observation using one bounded
compute-precision buffer. No randomized approximation or full converted tensor
is created.
"""
function observation_norm2(
    A::AbstractArray{<:Real};
    compute_type = nothing,
    block_length::Int = 65_536,
)
    requested = isnothing(compute_type) && A isa ComputeArray ? eltype(A) : compute_type
    T = _resolve_compute_type(raw_storage(A), requested)
    total = zero(T)
    foreach_compute_block(A; compute_type = T, block_length) do block, _
        @inbounds for value in block
            total += abs2(value)
        end
    end
    return total
end
