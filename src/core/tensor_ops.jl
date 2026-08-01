# core/tensor_ops.jl — Generic tensor contractions, unfold, and mode products
export unfold_mode,
    mode_n_product,
    relative_frobenius_error,
    cross_component,
    build_cross_matrix,
    build_cross_matrix_unit,
    factors_from_components,
    components_from_factors,
    grad_lambda_cp,
    cp_rankr_cost_value,
    cross_component_except_mode,
    cross_term_gradU,
    gradU_column_cp

using LinearAlgebra
using TensorOperations

@inline function _mode_first_perm(N::Int, mode::Int)
    perm = Vector{Int}(undef, N)
    perm[1] = mode
    j = 2
    @inbounds for m = 1:N
        m == mode && continue
        perm[j] = m
        j += 1
    end
    return perm
end

@inline function _other_mode_matrices_reverse(U::Vector, mode::Int)
    N = length(U)
    mats = Vector{eltype(U)}(undef, N - 1)
    j = 1
    @inbounds for m = N:-1:1
        m == mode && continue
        mats[j] = U[m]
        j += 1
    end
    return mats
end

@inline function _relative_error_frob_sq(
    n_residual_sq::T,
    n_target_sq::T,
) where {T<:AbstractFloat}
    r = max(n_residual_sq, zero(T))
    return n_target_sq > zero(T) ? sqrt(r / n_target_sq) : sqrt(r)
end

function relative_frobenius_error(A::AbstractArray, B::AbstractArray)
    size(A) == size(B) || throw(
        DimensionMismatch(
            "relative_frobenius_error: size(A)=$(size(A)), size(B)=$(size(B))",
        ),
    )
    Ta, Tb = eltype(A), eltype(B)
    Tacc = promote_type(float(Ta), float(Tb))
    sA = zero(Tacc)
    sR = zero(Tacc)
    @inbounds for i in eachindex(A)
        a = A[i]
        b = B[i]
        sA += abs2(a)
        d = a - b
        sR += abs2(d)
    end
    return _relative_error_frob_sq(sR, sA)
end

function unfold_mode(A::AbstractArray, mode::Int)
    dims = size(A)
    N = length(dims)
    mode == 1 && return reshape(A, dims[1], :)
    perm = _mode_first_perm(N, mode)
    Aperm = permutedims(A, perm)
    return reshape(Aperm, dims[mode], :)
end

function mode_n_product(A::AbstractArray, U::AbstractMatrix, mode::Int)
    dims = size(A)
    N = length(dims)
    n, newn = dims[mode], size(U, 1)
    perm = _mode_first_perm(N, mode)
    Aperm = mode == 1 ? A : permutedims(A, perm)
    A2 = reshape(Aperm, n, :)
    B2 = U * A2
    newdims = (newn, dims[perm[2:end]]...)
    Bperm = reshape(B2, newdims)
    return permutedims(Bperm, invperm(perm))
end

# These helpers perform mode contractions from tensor index labels instead of
# first constructing a mode unfolding. They are intentionally internal: the
# public API remains `mode_n_product`, while scalable Tucker algorithms can use
# these operations without materializing a tensor-sized `permutedims` copy.
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

"""
    _implicit_mode_sketch(A, mode, sketch_rank, rng; block_columns)

Compute `unfold_mode(A, mode) * Omega` for an implicit Gaussian matrix `Omega`
without constructing either the mode unfolding or the complete random matrix.
The non-mode columns are visited in tensor blocks containing at most roughly
`block_columns` entries, and each partial random projection is evaluated as a
tensor contraction.
"""
function _implicit_mode_sketch(
    A::AbstractArray{T,N},
    mode::Int,
    sketch_rank::Int,
    rng::AbstractRNG;
    block_columns::Int,
) where {T<:AbstractFloat,N}
    1 <= mode <= N || throw(ArgumentError("mode must be in 1:$N; received $mode"))
    sketch_rank >= 1 || throw(ArgumentError("sketch_rank must be positive"))

    other_modes = [m for m = 1:N if m != mode]
    block_lengths = _mode_block_lengths(size(A), mode, block_columns)
    block_axes = ntuple(m -> _axis_chunks(axes(A, m), block_lengths[m]), N)
    tensor_labels = ntuple(identity, N)
    sketch_label = N + 1
    omega_labels = Tuple(vcat(other_modes, sketch_label))
    sketch = zeros(T, size(A, mode), sketch_rank)

    for block in Iterators.product(block_axes...)
        block_indices = Tuple(block)
        A_block = @view A[block_indices...]
        omega_dims = Tuple(vcat([size(A_block, m) for m in other_modes], sketch_rank))
        omega = randn(rng, T, omega_dims)
        partial = TensorOperations.tensorcontract(
            (mode, sketch_label),
            A_block,
            tensor_labels,
            omega,
            omega_labels,
        )
        sketch .+= partial
    end
    return sketch
end

@inline function _rank1_entry_product(
    I::CartesianIndex{N},
    U::Vector{<:AbstractVector{T}},
) where {T<:AbstractFloat,N}
    prod_val = one(T)
    @inbounds for m = 1:N
        prod_val *= U[m][I[m]]
    end
    return prod_val
end

@inline function _rank1_entry_product_parts(I::CartesianIndex{N}, parts) where {N}
    T = eltype(parts[2])
    prod_val = one(T)
    @inbounds for m = 1:N
        prod_val *= parts[m+1][I[m]]
    end
    return prod_val
end

@inline function _rank1_entry_product_except(
    I::CartesianIndex{N},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    prod_val = one(T)
    @inbounds for m = 1:N
        m == mode && continue
        prod_val *= U[m][I[m]]
    end
    return prod_val
end

@inline function _rank1_entry_product_except_parts(
    I::CartesianIndex{N},
    parts,
    mode::Int,
) where {N}
    T = eltype(parts[2])
    prod_val = one(T)
    @inbounds for m = 1:N
        m == mode && continue
        prod_val *= parts[m+1][I[m]]
    end
    return prod_val
end

@inline function _rank1_entry_product_except_column(
    I::CartesianIndex{N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    k::Int,
) where {T<:AbstractFloat,N}
    prod_val = one(T)
    @inbounds for m = 1:N
        m == mode && continue
        prod_val *= U[m][I[m], k]
    end
    return prod_val
end

function rank1_inner(
    A::AbstractArray{T,N},
    U::Vector{<:AbstractVector{T}},
) where {T<:AbstractFloat,N}
    s = zero(T)
    @inbounds for I in CartesianIndices(A)
        s += A[I] * _rank1_entry_product(I, U)
    end
    return s
end

function rank1_inner(
    A::AbstractArray{T,3},
    U::Vector{<:AbstractVector{T}},
) where {T<:AbstractFloat}
    s = zero(T)
    u1, u2, u3 = U
    @tensor s = A[i, j, k] * u1[i] * u2[j] * u3[k]
    return s
end

function rank1_inner(
    A::AbstractArray{T,4},
    U::Vector{<:AbstractVector{T}},
) where {T<:AbstractFloat}
    s = zero(T)
    u1, u2, u3, u4 = U
    @tensor s = A[i, j, k, l] * u1[i] * u2[j] * u3[k] * u4[l]
    return s
end

function rank1_inner_parts(A::AbstractArray{T,N}, parts) where {T<:AbstractFloat,N}
    s = zero(T)
    @inbounds for I in CartesianIndices(A)
        s += A[I] * _rank1_entry_product_parts(I, parts)
    end
    return s
end

function rank1_mode_contract(
    A::AbstractArray{T,N},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    out = Vector{T}(undef, size(A, mode))
    return rank1_mode_contract!(out, A, U, mode)
end

function rank1_mode_contract(
    A::AbstractArray{T,3},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat}
    out = Vector{T}(undef, size(A, mode))
    return rank1_mode_contract!(out, A, U, mode)
end

function rank1_mode_contract(
    A::AbstractArray{T,4},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat}
    out = Vector{T}(undef, size(A, mode))
    return rank1_mode_contract!(out, A, U, mode)
end

function rank1_mode_contract_parts(
    A::AbstractArray{T,N},
    parts,
    mode::Int,
) where {T<:AbstractFloat,N}
    out = Vector{T}(undef, size(A, mode))
    return rank1_mode_contract_parts!(out, A, parts, mode)
end

function rank1_mode_contract!(
    out::AbstractVector{T},
    A::AbstractArray{T,N},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    # In-place fallback used by CP gradient code to avoid one fresh vector per contraction.
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(A)
        out[I[mode]] += A[I] * _rank1_entry_product_except(I, U, mode)
    end
    return out
end

function rank1_mode_contract!(
    out::AbstractVector{T},
    A::AbstractArray{T,3},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat}
    u1, u2, u3 = U
    if mode == 1
        @tensor out[i] = A[i, j, k] * u2[j] * u3[k]
    elseif mode == 2
        @tensor out[j] = A[i, j, k] * u1[i] * u3[k]
    elseif mode == 3
        @tensor out[k] = A[i, j, k] * u1[i] * u2[j]
    else
        throw(ArgumentError("mode must be between 1 and 3"))
    end
    return out
end

function rank1_mode_contract!(
    out::AbstractVector{T},
    A::AbstractArray{T,4},
    U::Vector{<:AbstractVector{T}},
    mode::Int,
) where {T<:AbstractFloat}
    u1, u2, u3, u4 = U
    if mode == 1
        @tensor out[i] = A[i, j, k, l] * u2[j] * u3[k] * u4[l]
    elseif mode == 2
        @tensor out[j] = A[i, j, k, l] * u1[i] * u3[k] * u4[l]
    elseif mode == 3
        @tensor out[k] = A[i, j, k, l] * u1[i] * u2[j] * u4[l]
    elseif mode == 4
        @tensor out[l] = A[i, j, k, l] * u1[i] * u2[j] * u3[k]
    else
        throw(ArgumentError("mode must be between 1 and 4"))
    end
    return out
end

function rank1_mode_contract_parts!(
    out::AbstractVector{T},
    A::AbstractArray{T,N},
    parts,
    mode::Int,
) where {T<:AbstractFloat,N}
    # Native-Segre variant mirrors `rank1_mode_contract!` but reads factors directly from point parts.
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(A)
        out[I[mode]] += A[I] * _rank1_entry_product_except_parts(I, parts, mode)
    end
    return out
end

function rank1_mode_contract_column(
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    k::Int,
) where {T<:AbstractFloat,N}
    out = similar(vec(U[1]), T, size(A, mode))
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(A)
        out[I[mode]] += A[I] * _rank1_entry_product_except_column(I, U, mode, k)
    end
    return out
end

function rank1_mode_contract_column!(
    out::AbstractVector{T},
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    k::Int,
) where {T<:AbstractFloat,N}
    fill!(out, zero(T))
    @inbounds for m = 1:N
        size(U[m], 1) == size(A, m) ||
            throw(DimensionMismatch("mode $m factor row count mismatch"))
    end
    @inbounds for I in CartesianIndices(A)
        out[I[mode]] += A[I] * _rank1_entry_product_except_column(I, U, mode, k)
    end
    return out
end
# 3D version
function rank1_mode_contract_column(
    A::AbstractArray{T,3},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    k::Int,
) where {T<:AbstractFloat}
    out = similar(vec(U[1]), T, size(A, mode))
    u1 = @view U[1][:, k]
    u2 = @view U[2][:, k]
    u3 = @view U[3][:, k]
    if mode == 1
        @tensor out[i] = A[i, j, l] * u2[j] * u3[l]
    elseif mode == 2
        @tensor out[j] = A[i, j, l] * u1[i] * u3[l]
    elseif mode == 3
        @tensor out[l] = A[i, j, l] * u1[i] * u2[j]
    else
        throw(ArgumentError("mode must be between 1 and 3"))
    end
    return out
end
# 4D version
function rank1_mode_contract_column(
    A::AbstractArray{T,4},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    k::Int,
) where {T<:AbstractFloat}
    out = similar(vec(U[1]), T, size(A, mode))
    u1 = @view U[1][:, k]
    u2 = @view U[2][:, k]
    u3 = @view U[3][:, k]
    u4 = @view U[4][:, k]
    if mode == 1
        @tensor out[i] = A[i, j, l, m] * u2[j] * u3[l] * u4[m]
    elseif mode == 2
        @tensor out[j] = A[i, j, l, m] * u1[i] * u3[l] * u4[m]
    elseif mode == 3
        @tensor out[l] = A[i, j, l, m] * u1[i] * u2[j] * u4[m]
    elseif mode == 4
        @tensor out[m] = A[i, j, l, m] * u1[i] * u2[j] * u3[l]
    else
        throw(ArgumentError("mode must be between 1 and 4"))
    end
    return out
end

function rank1_inner_component(
    A::AbstractArray{T,N},
    components::Vector{RankOneTensor{T}},
    k::Int,
) where {T<:AbstractFloat,N}
    return rank1_inner(A, components[k].vectors)
end

function rank1_mode_contract_component(
    A::AbstractArray{T,N},
    components::Vector{RankOneTensor{T}},
    k::Int,
    m::Int,
) where {T<:AbstractFloat,N}
    return rank1_mode_contract(A, components[k].vectors, m)
end

function cross_component(a::RankOneTensor{T}, b::RankOneTensor{T}) where {T<:AbstractFloat}
    length(a.vectors) == length(b.vectors) ||
        throw(DimensionMismatch("RankOneTensor mode count"))
    return a.λ * b.λ * prod(dot(a.vectors[m], b.vectors[m]) for m in eachindex(a.vectors))
end

function build_cross_matrix(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat}
    r = length(components)
    cross_mat = zeros(T, r, r)
    for k = 1:r
        for l = 1:r
            cross_mat[k, l] = cross_component(components[k], components[l])
        end
    end
    return cross_mat
end

function build_cross_matrix_unit(
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    r = length(components)
    cross_mat = zeros(T, r, r)
    for k = 1:r
        for l = 1:r
            cross_mat[k, l] = prod(
                dot(components[k].vectors[m], components[l].vectors[m]) for
                m in eachindex(components[1].vectors)
            )
        end
    end
    return cross_mat
end

function grad_lambda_cp(
    λ::Vector{T},
    inner::Vector{T},
    cross_mat::Matrix{T},
) where {T<:AbstractFloat}
    return cross_mat * λ - inner
end

function cp_rankr_cost_value(
    normA2::T,
    λ::Vector{T},
    inner::Vector{T},
    cross_mat::Matrix{T},
) where {T<:AbstractFloat}
    return T(0.5) * normA2 + T(0.5) * dot(λ, cross_mat * λ) - dot(λ, inner)
end

cross_component_except_mode(
    a::RankOneTensor{T},
    b::RankOneTensor{T},
    exclude_m::Int,
) where {T<:AbstractFloat} =
    prod(dot(a.vectors[j], b.vectors[j]) for j in setdiff(eachindex(a.vectors), exclude_m))

function cross_term_gradU(
    components::Vector{RankOneTensor{T}},
    k::Int,
    m::Int,
) where {T<:AbstractFloat}
    out = zeros(T, length(components[k].vectors[m]))
    for l in setdiff(eachindex(components), k)
        coef =
            components[l].λ * cross_component_except_mode(components[k], components[l], m)
        out .+= coef .* components[l].vectors[m]
    end
    return out
end

gradU_column_cp(
    λ_k::T,
    U_mk::Vector{T},
    contract_km::Vector{T},
    cross_term::Vector{T},
) where {T<:AbstractFloat} = -λ_k .* contract_km .+ λ_k .* cross_term

cp_reconstruction_norm2(components::Vector{RankOneTensor{T}}) where {T<:AbstractFloat} =
    sum(
        cross_component(components[i], components[j]) for
        i in eachindex(components), j in eachindex(components)
    )

function cp_inner_AX(
    A::AbstractArray{T,N},
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat,N}
    return sum(
        components[k].λ * rank1_inner(A, components[k].vectors) for
        k in eachindex(components)
    )
end

@inline function _cp_residual_sq_from_G_unreliable(
    n2::T,
    normA2::T,
    normX2::T,
    innerAX::T,
) where {T<:AbstractFloat}
    (!isfinite(n2) || n2 < zero(T)) && return true
    scale = max(normA2, normX2, 2 * abs(innerAX), one(T))
    # When ‖A-X‖² is tiny vs. ‖A‖²,‖X‖², the difference ‖A‖²+‖X‖²-2⟨A,X⟩ cancels badly; use explicit residual.
    return n2 < sqrt(eps(T)) * scale
end

function cp_residual_stats(
    A::AbstractArray{T,N},
    normA2::T,
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat,N}
    normX2 = cp_reconstruction_norm2(components)
    innerAX = cp_inner_AX(A, components)
    n2 = normA2 + normX2 - 2 * innerAX
    if _cp_residual_sq_from_G_unreliable(n2, normA2, normX2, innerAX)
        return cp_residual_stats_explicit(A, normA2, components)
    end
    return (n2, T(0.5) * n2, _relative_error_frob_sq(n2, normA2))
end

function cp_residual_stats_explicit(
    A::AbstractArray{T,N},
    normA2::T,
    λ::Vector{T},
    U::Vector{Matrix{T}},
) where {T<:AbstractFloat,N}
    X = reconstruct_cpd_rankr(λ, U)
    residual = X .- A
    n2 = sum(abs2, residual)
    return (n2, T(0.5) * n2, _relative_error_frob_sq(n2, normA2))
end

function cp_residual_stats_explicit(
    A::AbstractArray{T,N},
    normA2::T,
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat,N}
    λ = [c.λ for c in components]
    U = factors_from_components(components)
    return cp_residual_stats_explicit(A, normA2, λ, U)
end

function factors_from_components(
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat}
    isempty(components) && return Vector{Matrix{T}}()
    r = length(components)
    N = length(components[1].vectors)
    return [hcat((components[k].vectors[m] for k = 1:r)...) for m = 1:N]
end

function components_from_factors(
    λ::Vector{T},
    U::Vector{Matrix{T}},
) where {T<:AbstractFloat}
    r = length(λ)
    N = length(U)
    return [RankOneTensor(λ[k], [Vector(@view U[m][:, k]) for m = 1:N]) for k = 1:r]
end
