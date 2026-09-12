# kernels/implicit_contractions.jl — observation-preserving tensor products

@inline function _axis_chunks(axis::AbstractUnitRange, chunk_length::Int)
    chunk_length >= 1 || throw(ArgumentError("chunk length must be positive"))
    first_index = first(axis)
    last_index = last(axis)
    return [
        start:min(start+chunk_length-1, last_index) for
        start = first_index:chunk_length:last_index
    ]
end

function _mode_block_lengths(dims::NTuple{N,Int}, mode::Int, block_columns::Int) where {N}
    block_columns >= 1 || throw(ArgumentError("block_columns must be positive"))
    lengths = ones(Int, N)
    lengths[mode] = dims[mode]
    remaining = block_columns
    @inbounds for m = 1:N
        m == mode && continue
        lengths[m] = min(dims[m], max(remaining, 1))
        remaining = max(div(remaining, lengths[m]), 1)
    end
    return Tuple(lengths)
end

"""Apply a mode matrix blockwise without materializing a mode unfolding."""
function _implicit_mode_product(
    A::AbstractArray,
    U::AbstractMatrix,
    mode::Int;
    block_columns::Int = 65_536,
)
    N = ndims(A)
    1 <= mode <= N || throw(ArgumentError("mode must be in 1:$N; received $mode"))
    block_columns >= 1 || throw(ArgumentError("block_columns must be positive"))
    size(U, 2) == size(A, mode) || throw(
        DimensionMismatch(
            "size(U, 2)=$(size(U, 2)) must match size(A, mode)=$(size(A, mode))",
        ),
    )

    tensor_labels = ntuple(identity, N)
    new_label = N + 1
    matrix_labels = (new_label, mode)
    output_labels = ntuple(m -> m == mode ? new_label : m, N)
    output_dims = ntuple(m -> m == mode ? size(U, 1) : size(A, m), N)
    output = Array{promote_type(eltype(A), eltype(U))}(undef, output_dims)
    block_lengths = _mode_block_lengths(size(A), mode, block_columns)
    block_axes = ntuple(m -> _axis_chunks(axes(A, m), block_lengths[m]), N)

    for block in Iterators.product(block_axes...)
        block_indices = Tuple(block)
        A_block = @view A[block_indices...]
        partial = TensorOperations.tensorcontract(
            output_labels,
            U,
            matrix_labels,
            A_block,
            tensor_labels,
        )
        output_indices = ntuple(m -> m == mode ? axes(output, m) : block_indices[m], N)
        @views output[output_indices...] .= partial
    end
    return output
end

"""Compute a mode-wise tensor cross product blockwise without unfoldings."""
function _implicit_mode_cross(
    A::AbstractArray,
    B::AbstractArray,
    mode::Int;
    block_columns::Int = 65_536,
)
    N = ndims(A)
    ndims(B) == N || throw(DimensionMismatch("A and B must have the same number of modes"))
    1 <= mode <= N || throw(ArgumentError("mode must be in 1:$N; received $mode"))
    block_columns >= 1 || throw(ArgumentError("block_columns must be positive"))
    @inbounds for m = 1:N
        m == mode && continue
        size(A, m) == size(B, m) || throw(
            DimensionMismatch(
                "A and B must agree outside mode $mode; " *
                "size(A, $m)=$(size(A, m)), size(B, $m)=$(size(B, m))",
            ),
        )
    end

    labels_A = ntuple(identity, N)
    new_label = N + 1
    labels_B = ntuple(m -> m == mode ? new_label : m, N)
    output = zeros(promote_type(eltype(A), eltype(B)), size(A, mode), size(B, mode))
    block_lengths = _mode_block_lengths(size(A), mode, block_columns)
    block_axes = ntuple(m -> _axis_chunks(axes(A, m), block_lengths[m]), N)

    for block in Iterators.product(block_axes...)
        block_indices_A = Tuple(block)
        block_indices_B = ntuple(m -> m == mode ? axes(B, m) : block_indices_A[m], N)
        A_block = @view A[block_indices_A...]
        B_block = @view B[block_indices_B...]
        partial = TensorOperations.tensorcontract(
            (mode, new_label),
            A_block,
            labels_A,
            B_block,
            labels_B,
        )
        output .+= partial
    end
    return output
end
