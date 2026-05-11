# cpd/core/mttkrp.jl — CPD-specific MTTKRP kernels and dispatch
# Improvement of resolving mttkrp bottleneck still in progress 
export mttkrp, khatri_rao

@inline _mttkrp_needs_kr_workspace(method::Symbol) = method == :khatri_rao
@inline _mttkrp_needs_tmp_workspace(method::Symbol) = method in (:direct3, :direct4)

@inline function _mttkrp_auto_method_3way(kr_rows::Int, r::Int, mode::Int)
    # Benchmark-guided 3-way table:
    # - mode 3 consistently favored explicit KR
    # - modes 1/2 favored direct3 on moderate/large KR shapes
    # - mode 1 at very large KR + higher rank tilted back toward KR
    if mode == 3
        return :khatri_rao
    elseif kr_rows <= 400
        return :khatri_rao
    elseif mode == 1 && kr_rows >= 1800 && r >= 16
        return :khatri_rao
    else
        return :direct3
    end
end

@inline function _mttkrp_auto_method(dims::NTuple{N,Int}, r::Int, mode::Int) where {N}
    kr_rows = div(prod(dims), dims[mode])
    if N == 3
        return _mttkrp_auto_method_3way(kr_rows, r, mode)
    elseif N == 4
        # 4-way dense sweeps still favored the BLAS/KR path at moderate KR sizes.
        return kr_rows * r > 400_000 ? :contract : :khatri_rao
    else
        return kr_rows * r > 200_000 ? :contract : :khatri_rao
    end
end

@inline function _mttkrp_resolve_method(
    method::Symbol,
    dims::NTuple{N,Int},
    r::Int,
    mode::Int,
) where {N}
    method == :auto && return _mttkrp_auto_method(dims, r, mode)
    method == :khatri_rao && return :khatri_rao
    if method == :direct
        N == 3 && return :direct3
        N == 4 && return :direct4
        return :contract
    else
        throw(
            ArgumentError(
                "Unknown mttkrp method=$method. Use :auto, :khatri_rao, or :direct.",
            ),
        )
    end
end

# forming Khatri-Rao product helper, the loop is costly.
function khatri_rao(mats::AbstractVector{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    if isempty(mats)
        throw(ArgumentError("khatri_rao: empty matrix list"))
    end
    r = size(mats[1], 2)
    for m in mats
        size(m, 2) == r || throw(ArgumentError("khatri_rao: column counts must match"))
    end
    out = copy(mats[1])
    for i = 2:length(mats)
        A = mats[i]
        new = similar(out, T, size(out, 1) * size(A, 1), r)
        rows_out = size(out, 1)
        rows_A = size(A, 1)
        @inbounds for k = 1:r
            idx = 1
            for io = 1:rows_out
                scale = out[io, k]
                for ia = 1:rows_A
                    new[idx, k] = scale * A[ia, k]
                    idx += 1
                end
            end
        end
        out = new
    end
    return out
end

function khatri_rao!(
    out::AbstractMatrix{T},
    mats::AbstractVector{<:AbstractMatrix{T}},
    work::AbstractMatrix{T},
) where {T<:AbstractFloat}
    if isempty(mats)
        throw(ArgumentError("khatri_rao!: empty matrix list"))
    end
    r = size(mats[1], 2)
    for m in mats
        size(m, 2) == r || throw(ArgumentError("khatri_rao!: column counts must match"))
    end
    final_rows = prod(size(m, 1) for m in mats)
    size(out) == (final_rows, r) || throw(
        DimensionMismatch(
            "khatri_rao!: out has size $(size(out)), expected ($(final_rows), $r)",
        ),
    )
    size(work) == size(out) || throw(
        DimensionMismatch(
            "khatri_rao!: work has size $(size(work)), expected $(size(out))",
        ),
    )

    rows_cur = size(mats[1], 1)
    copyto!(view(out, 1:rows_cur, :), mats[1])
    src = out
    dst = work
    for i = 2:length(mats)
        A = mats[i]
        rows_A = size(A, 1)
        new_rows = rows_cur * rows_A
        @inbounds for k = 1:r
            idx = 1
            for io = 1:rows_cur
                scale = src[io, k]
                for ia = 1:rows_A
                    dst[idx, k] = scale * A[ia, k]
                    idx += 1
                end
            end
        end
        rows_cur = new_rows
        src, dst = dst, src
    end
    if src !== out
        copyto!(view(out, 1:rows_cur, :), view(src, 1:rows_cur, :))
    end
    return out
end

function _mttkrp_khatri_rao(
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    mats = _other_mode_matrices_reverse(U, mode)
    kr = khatri_rao(mats)
    A_mode = unfold_mode(A, mode)
    return A_mode * kr
end

function _mttkrp_khatri_rao!(
    out::AbstractMatrix{T},
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
    kr_buf::AbstractMatrix{T},
    kr_work::AbstractMatrix{T},
) where {T<:AbstractFloat,N}
    mats = _other_mode_matrices_reverse(U, mode)
    kr = khatri_rao!(kr_buf, mats, kr_work)
    A_mode = unfold_mode(A, mode)
    mul!(out, A_mode, kr)
    return out
end

# Direct and contraction kernels
@inline function _accumulate_scaled_columns!(
    out::AbstractMatrix{T},
    tmp::AbstractMatrix{T},
    w::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds for q in axes(out, 2)
        α = w[q]
        for i in axes(out, 1)
            out[i, q] += α * tmp[i, q]
        end
    end
    return out
end

@inline function _accumulate_scaled_columns!(
    out::AbstractMatrix{T},
    tmp::AbstractMatrix{T},
    w1::AbstractVector{T},
    w2::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds for q in axes(out, 2)
        α = w1[q] * w2[q]
        for i in axes(out, 1)
            out[i, q] += α * tmp[i, q]
        end
    end
    return out
end

function _mttkrp_direct3(
    A::AbstractArray{T,3},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    out = similar(U[1], T, size(A, mode), size(U[1], 2))
    fill!(out, zero(T))
    tmp = similar(out)
    return _mttkrp_direct3!(out, tmp, A, U, mode)
end

function _mttkrp_direct3!(
    out::AbstractMatrix{T},
    tmp::AbstractMatrix{T},
    A::AbstractArray{T,3},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    I, J, K = size(A)
    fill!(out, zero(T))
    if mode == 1
        @inbounds for k = 1:K
            mul!(tmp, @view(A[:, :, k]), U[2])
            _accumulate_scaled_columns!(out, tmp, @view(U[3][k, :]))
        end
    elseif mode == 2
        @inbounds for k = 1:K
            mul!(tmp, transpose(@view(A[:, :, k])), U[1])
            _accumulate_scaled_columns!(out, tmp, @view(U[3][k, :]))
        end
    elseif mode == 3
        @inbounds for j = 1:J
            mul!(tmp, transpose(reshape(@view(A[:, j, :]), I, K)), U[1])
            _accumulate_scaled_columns!(out, tmp, @view(U[2][j, :]))
        end
    else
        throw(ArgumentError("mode must be between 1 and 3"))
    end
    return out
end

function _mttkrp_direct4(
    A::AbstractArray{T,4},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    out = similar(U[1], T, size(A, mode), size(U[1], 2))
    fill!(out, zero(T))
    tmp = similar(out)
    return _mttkrp_direct4!(out, tmp, A, U, mode)
end

function _mttkrp_direct4!(
    out::AbstractMatrix{T},
    tmp::AbstractMatrix{T},
    A::AbstractArray{T,4},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    I, J, K, L = size(A)
    fill!(out, zero(T))
    if mode == 1
        @inbounds for l = 1:L
            for k = 1:K
                mul!(tmp, @view(A[:, :, k, l]), U[2])
                _accumulate_scaled_columns!(out, tmp, @view(U[3][k, :]), @view(U[4][l, :]))
            end
        end
    elseif mode == 2
        @inbounds for l = 1:L
            for k = 1:K
                mul!(tmp, transpose(@view(A[:, :, k, l])), U[1])
                _accumulate_scaled_columns!(out, tmp, @view(U[3][k, :]), @view(U[4][l, :]))
            end
        end
    elseif mode == 3
        @inbounds for l = 1:L
            for j = 1:J
                mul!(tmp, transpose(reshape(@view(A[:, j, :, l]), I, K)), U[1])
                _accumulate_scaled_columns!(out, tmp, @view(U[2][j, :]), @view(U[4][l, :]))
            end
        end
    elseif mode == 4
        @inbounds for k = 1:K
            for j = 1:J
                mul!(tmp, transpose(reshape(@view(A[:, j, k, :]), I, L)), U[1])
                _accumulate_scaled_columns!(out, tmp, @view(U[2][j, :]), @view(U[3][k, :]))
            end
        end
    else
        throw(ArgumentError("mode must be between 1 and 4"))
    end
    return out
end

function _mttkrp_contract(
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    dims = size(A)
    r = size(U[1], 2)
    out = similar(U[1], T, dims[mode], r)
    fill!(out, zero(T))
    for k = 1:r
        out[:, k] = rank1_mode_contract_column(A, U, mode, k)
    end
    return out
end

function _mttkrp_contract!(
    out::AbstractMatrix{T},
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat,N}
    r = size(U[1], 2)
    @inbounds for q = 1:r
        out[:, q] = rank1_mode_contract_column(A, U, mode, q)
    end
    return out
end

function _mttkrp_contract(
    A::AbstractArray{T,3},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    r = size(U[1], 2)
    out = similar(U[1], T, size(A, mode), r)
    @inbounds for q = 1:r
        out[:, q] = rank1_mode_contract_column(A, U, mode, q)
    end
    return out
end

function _mttkrp_contract(
    A::AbstractArray{T,4},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int,
) where {T<:AbstractFloat}
    r = size(U[1], 2)
    out = similar(U[1], T, size(A, mode), r)
    @inbounds for q = 1:r
        out[:, q] = rank1_mode_contract_column(A, U, mode, q)
    end
    return out
end

#Public API
function mttkrp(
    A::AbstractArray{T,N},
    components::Vector{RankOneTensor{T}},
    mode::Int;
    method::Symbol = :auto,
) where {T<:AbstractFloat,N}
    return mttkrp(A, factors_from_components(components), mode; method)
end

function mttkrp(
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int;
    method::Symbol = :auto,
) where {T<:AbstractFloat,N}
    dims = size(A)
    mode < 1 && throw(ArgumentError("mode must be >= 1"))
    mode > N && throw(ArgumentError("mode must be <= ndims(A)"))
    isempty(U) && throw(ArgumentError("mttkrp: factor list is empty"))
    length(U) == N ||
        throw(DimensionMismatch("mttkrp: expected $N factor matrices, got $(length(U))"))

    r = size(U[1], 2)
    for m = 1:N
        size(U[m], 1) == dims[m] || throw(
            DimensionMismatch(
                "mttkrp: U[$m] has $(size(U[m], 1)) rows, expected $(dims[m])",
            ),
        )
        size(U[m], 2) == r ||
            throw(DimensionMismatch("mttkrp: all factors must have same column count"))
    end

    method_eff = _mttkrp_resolve_method(method, dims, r, mode)

    if method_eff == :khatri_rao
        return _mttkrp_khatri_rao(A, U, mode)
    elseif method_eff == :direct3
        N == 3 ||
            throw(ArgumentError("method=:direct3 is only supported for 3-way tensors"))
        return _mttkrp_direct3(A, U, mode)
    elseif method_eff == :direct4
        N == 4 ||
            throw(ArgumentError("method=:direct4 is only supported for 4-way tensors"))
        return _mttkrp_direct4(A, U, mode)
    elseif method_eff == :contract
        return _mttkrp_contract(A, U, mode)
    else
        throw(
            ArgumentError(
                "Unknown mttkrp method=$method. Use :auto, :khatri_rao, or :direct.",
            ),
        )
    end
end

function mttkrp!(
    out::AbstractMatrix{T},
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    mode::Int;
    method::Symbol = :auto,
    work = nothing,
    kr_buf = nothing,
    kr_work = nothing,
) where {T<:AbstractFloat,N}
    dims = size(A)
    size(out, 1) == dims[mode] || throw(
        DimensionMismatch("mttkrp!: out has $(size(out,1)) rows, expected $(dims[mode])"),
    )
    r = size(U[1], 2)
    size(out, 2) == r ||
        throw(DimensionMismatch("mttkrp!: out has $(size(out,2)) columns, expected $r"))

    method_eff = _mttkrp_resolve_method(method, dims, r, mode)
    if method_eff == :khatri_rao
        isnothing(kr_buf) && throw(
            ArgumentError("mttkrp!: method=:khatri_rao requires a KR workspace buffer"),
        )
        isnothing(kr_work) && throw(
            ArgumentError(
                "mttkrp!: method=:khatri_rao requires a secondary KR workspace buffer",
            ),
        )
        _mttkrp_khatri_rao!(out, A, U, mode, kr_buf, kr_work)
    elseif method_eff == :direct3
        N == 3 ||
            throw(ArgumentError("method=:direct3 is only supported for 3-way tensors"))
        isnothing(work) &&
            throw(ArgumentError("mttkrp!: method=:direct3 requires a work buffer"))
        size(work) == size(out) || throw(
            DimensionMismatch(
                "mttkrp!: work size $(size(work)) must match out size $(size(out))",
            ),
        )
        _mttkrp_direct3!(out, work, A, U, mode)
    elseif method_eff == :direct4
        N == 4 ||
            throw(ArgumentError("method=:direct4 is only supported for 4-way tensors"))
        isnothing(work) &&
            throw(ArgumentError("mttkrp!: method=:direct4 requires a work buffer"))
        size(work) == size(out) || throw(
            DimensionMismatch(
                "mttkrp!: work size $(size(work)) must match out size $(size(out))",
            ),
        )
        _mttkrp_direct4!(out, work, A, U, mode)
    elseif method_eff == :contract
        _mttkrp_contract!(out, A, U, mode)
    else
        throw(
            ArgumentError(
                "Unknown mttkrp method=$method. Use :auto, :khatri_rao, or :direct.",
            ),
        )
    end
    return out
end
