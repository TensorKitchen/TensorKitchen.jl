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
    return a.λ * b.λ * prod(dot(a.vectors[m], b.vectors[m]) for m = 1:length(a.vectors))
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
                m = 1:length(components[1].vectors)
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
    prod(dot(a.vectors[j], b.vectors[j]) for j in setdiff(1:length(a.vectors), exclude_m))

function cross_term_gradU(
    components::Vector{RankOneTensor{T}},
    k::Int,
    m::Int,
) where {T<:AbstractFloat}
    out = zeros(T, length(components[k].vectors[m]))
    for l in setdiff(1:length(components), k)
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
        cross_component(components[i], components[j]) for i = 1:length(components),
        j = 1:length(components)
    )

function cp_inner_AX(
    A::AbstractArray{T,N},
    components::Vector{RankOneTensor{T}},
) where {T<:AbstractFloat,N}
    return sum(
        components[k].λ * rank1_inner(A, components[k].vectors) for k = 1:length(components)
    )
end

@inline function _cp_residual_sq_from_gram_unreliable(
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
    if _cp_residual_sq_from_gram_unreliable(n2, normA2, normX2, innerAX)
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
    return [RankOneTensor(λ[k], [Vector(U[m][:, k]) for m = 1:N]) for k = 1:r]
end
