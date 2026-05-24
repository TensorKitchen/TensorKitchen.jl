# btd/core/inner_prod.jl — Tucker-block inner products for BTD

@inline _tucker_data(p::Manifolds.TuckerPoint) = (p.hosvd.core, p.hosvd.U)

function _workspace_tensor!(
    cache::_WorkspaceTensorCache{T,N},
    ::Type{T},
    dims::NTuple{N,Int},
) where {T,N}
    idx = findfirst(==(dims), cache.dims)
    if isnothing(idx)
        push!(cache.dims, dims)
        push!(cache.bufs, Array{T}(undef, dims))
        idx = length(cache.bufs)
    end
    return cache.bufs[idx]::Array{T,N}
end

@inline function _workspace_copy_tensor!(
    ws::BTDContractionWorkspace{T,N},
    A::AbstractArray{T,N},
) where {T,N}
    buf = _workspace_tensor!(ws.persist, T, size(A))
    copyto!(buf, A)
    return buf
end

@inline _perm_tuple(N::Int, mode::Int) = (mode, ntuple(i -> i < mode ? i : i + 1, N - 1)...)

@inline _invperm_tuple(perm::NTuple{N,Int}) where {N} =
    ntuple(i -> findfirst(isequal(i), perm), N)

function _mode_n_product!(
    ws::BTDContractionWorkspace{T,N},
    out::AbstractArray{T,N},
    A::AbstractArray{T,N},
    U::AbstractMatrix{T},
    mode::Int,
) where {T,N}
    dims = size(A)
    n = dims[mode]
    size(U, 2) == n || throw(
        DimensionMismatch(
            "mode_n_product!: size(U,2)=$(size(U,2)) must match tensor mode size $n for mode $mode.",
        ),
    )

    # Fast path for mode-1 contraction: avoid permutedims! entirely.
    if mode == 1
        A2 = reshape(A, n, :)
        B2 = reshape(out, size(U, 1), :)
        mul!(B2, U, A2)
        return out
    end

    # N=3 specialized paths for mode-2/3 also avoid permutedims!.
    if N == 3 && mode == 2
        r = size(U, 1)
        Ut = transpose(U)
        @inbounds for k in axes(A, 3)
            Ak = reshape(view(A, :, :, k), size(A, 1), size(A, 2))
            Bk = reshape(view(out, :, :, k), size(out, 1), r)
            mul!(Bk, Ak, Ut)
        end
        return out
    end
    if N == 3 && mode == 3
        r = size(U, 1)
        A2 = reshape(A, :, size(A, 3))
        B2 = reshape(out, :, r)
        mul!(B2, A2, transpose(U))
        return out
    end

    perm = _perm_tuple(N, mode)
    invperm = _invperm_tuple(perm)
    perm_in_dims = ntuple(i -> dims[perm[i]], N)
    perm_out_dims = (size(U, 1), ntuple(i -> dims[perm[i+1]], N - 1)...)

    tmp_in = _workspace_tensor!(ws.perm_in, T, perm_in_dims)
    tmp_out = _workspace_tensor!(ws.perm_out, T, perm_out_dims)

    permutedims!(tmp_in, A, perm)
    A2 = reshape(tmp_in, n, :)
    B2 = reshape(tmp_out, size(U, 1), :)
    mul!(B2, U, A2)
    permutedims!(out, tmp_out, invperm)
    return out
end

function _apply_mode_products(
    ws::BTDContractionWorkspace{T,N},
    A::AbstractArray{T,N},
    mode_mats::Tuple,
) where {T,N}
    current = A
    use_slot1 = true
    for (mode, U) in mode_mats
        out_dims = ntuple(i -> i == mode ? size(U, 1) : size(current, i), N)
        out = _workspace_tensor!(use_slot1 ? ws.tensor_slot1 : ws.tensor_slot2, T, out_dims)
        _mode_n_product!(ws, out, current, U, mode)
        current = out
        use_slot1 = !use_slot1
    end
    return current
end

function _apply_mode_products(
    backend::BTDBackend{T},
    A::AbstractArray{T,N},
    mode_mats::Tuple,
) where {T,N}
    return _apply_mode_products(backend.workspace, A, mode_mats)
end

function _all_except_mode_products(
    backend::BTDBackend{T},
    A::AbstractArray{T,3},
    mats::NTuple{3,<:AbstractMatrix{T}},
) where {T}
    # N=3 fast path: directly form the three "all except mode m" projections.
    x1 = _apply_mode_products(backend, A, ((2, mats[2]), (3, mats[3])))
    x2 = _apply_mode_products(backend, A, ((1, mats[1]), (3, mats[3])))
    x3 = _apply_mode_products(backend, A, ((1, mats[1]), (2, mats[2])))
    return Array{T,3}[
        _workspace_copy_tensor!(backend.workspace, x1),
        _workspace_copy_tensor!(backend.workspace, x2),
        _workspace_copy_tensor!(backend.workspace, x3),
    ]
end

function _all_except_mode_products(
    backend::BTDBackend{T},
    A::AbstractArray{T,N},
    mats::NTuple{N,<:AbstractMatrix{T}},
) where {T,N}
    N == 1 && return Array{T,N}[_workspace_copy_tensor!(backend.workspace, A)]

    prefix = Vector{Array{T,N}}(undef, N)
    suffix = Vector{Array{T,N}}(undef, N + 1)
    prefix[1] = _workspace_copy_tensor!(backend.workspace, A)
    for m = 1:(N-1)
        prefix[m+1] = _workspace_copy_tensor!(
            backend.workspace,
            _apply_mode_products(backend, prefix[m], ((m, mats[m]),)),
        )
    end

    suffix[N+1] = _workspace_copy_tensor!(backend.workspace, A)
    for m = N:-1:2
        suffix[m] = _workspace_copy_tensor!(
            backend.workspace,
            _apply_mode_products(backend, suffix[m+1], ((m, mats[m]),)),
        )
    end

    excepts = Vector{Array{T,N}}(undef, N)
    excepts[1] = _workspace_copy_tensor!(backend.workspace, suffix[2])
    excepts[N] = _workspace_copy_tensor!(backend.workspace, prefix[N])
    for m = 2:(N-1)
        if (m - 1) <= (N - m)
            current = prefix[m]
            for k = (m+1):N
                current = _apply_mode_products(backend, current, ((k, mats[k]),))
            end
            excepts[m] = _workspace_copy_tensor!(backend.workspace, current)
        else
            current = suffix[m+1]
            for k = 1:(m-1)
                current = _apply_mode_products(backend, current, ((k, mats[k]),))
            end
            excepts[m] = _workspace_copy_tensor!(backend.workspace, current)
        end
    end
    return excepts
end

function _tucker_tucker_inner(p::Manifolds.TuckerPoint, q::Manifolds.TuckerPoint)
    Gp, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)

    N = length(Up)
    ndims(Gp) == N ||
        throw(DimensionMismatch("Tucker core/factor count mismatch for first point."))
    ndims(Gq) == N ||
        throw(DimensionMismatch("Tucker core/factor count mismatch for second point."))
    N == length(Uq) ||
        throw(DimensionMismatch("Tucker mode count mismatch between points."))
    @inbounds for k = 1:N
        size(Up[k], 2) == size(Gp, k) ||
            throw(DimensionMismatch("Tucker factor dimension mismatch for mode $k."))
        size(Uq[k], 2) == size(Gq, k) ||
            throw(DimensionMismatch("Tucker factor dimension mismatch for mode $k."))
        size(Up[k], 1) == size(Uq[k], 1) ||
            throw(DimensionMismatch("Tucker factor dimension mismatch for mode $k."))
    end

    Ht = Gq
    @inbounds for k = 1:N
        Ht = mode_n_product(Ht, Up[k]' * Uq[k], k)
    end
    return sum(Gp .* Ht)
end

function _tucker_tucker_inner(
    backend::BTDBackend{T},
    p::Manifolds.TuckerPoint,
    q::Manifolds.TuckerPoint,
) where {T}
    Gp, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)

    N = length(Up)
    ndims(Gp) == N ||
        throw(DimensionMismatch("Tucker core/factor count mismatch for first point."))
    ndims(Gq) == N ||
        throw(DimensionMismatch("Tucker core/factor count mismatch for second point."))
    N == length(Uq) ||
        throw(DimensionMismatch("Tucker mode count mismatch between points."))
    mode_mats = ntuple(k -> (k, Matrix{T}(Up[k]' * Uq[k])), N)
    Ht = _apply_mode_products(backend, Gq, mode_mats)
    return sum(Gp .* Ht)
end

function _target_tucker_inner(A::AbstractArray, p::Manifolds.TuckerPoint)
    G, U = _tucker_data(p)
    N = length(U)
    ndims(G) == N || throw(DimensionMismatch("Tucker core/factor count mismatch."))
    @inbounds for k = 1:N
        size(U[k], 2) == size(G, k) ||
            throw(DimensionMismatch("Tucker factor dimension mismatch for mode $k."))
        size(U[k], 1) == size(A, k) ||
            throw(DimensionMismatch("Tucker factor dimension mismatch for mode $k."))
    end

    At = A
    @inbounds for k in eachindex(U)
        At = mode_n_product(At, U[k]', k)
    end
    return sum(At .* G)
end

function _target_tucker_inner(
    backend::BTDBackend{T},
    A::AbstractArray{T,N},
    p::Manifolds.TuckerPoint,
) where {T,N}
    G, U = _tucker_data(p)
    mode_mats = ntuple(k -> (k, transpose(U[k])), length(U))
    At = _apply_mode_products(backend, A, mode_mats)
    return sum(At .* G)
end

function _tucker_project_target(p::Manifolds.TuckerPoint, A::AbstractArray)
    G, U = _tucker_data(p)
    At = A
    @inbounds for k in eachindex(U)
        At = mode_n_product(At, U[k]', k)
    end
    size(At) == size(G) || throw(DimensionMismatch("Projected target/core size mismatch."))
    return At
end

function _tucker_project_target(
    backend::BTDBackend{T},
    p::Manifolds.TuckerPoint,
    A::AbstractArray{T,N},
) where {T,N}
    G, U = _tucker_data(p)
    mode_mats = ntuple(k -> (k, transpose(U[k])), length(U))
    At = _apply_mode_products(backend, A, mode_mats)
    size(At) == size(G) || throw(DimensionMismatch("Projected target/core size mismatch."))
    return _workspace_copy_tensor!(backend.workspace, At)
end

function _tucker_project_target_except_mode(
    p::Manifolds.TuckerPoint,
    A::AbstractArray,
    m::Int,
)
    _, U = _tucker_data(p)
    At = A
    @inbounds for k in eachindex(U)
        k == m && continue
        At = mode_n_product(At, U[k]', k)
    end
    return At
end

function _tucker_project_target_except_mode(
    backend::BTDBackend{T},
    p::Manifolds.TuckerPoint,
    A::AbstractArray{T,N},
    m::Int,
) where {T,N}
    _, U = _tucker_data(p)
    mode_mats = Tuple((k, transpose(U[k])) for k in eachindex(U) if k != m)
    At = _apply_mode_products(backend, A, mode_mats)
    return _workspace_copy_tensor!(backend.workspace, At)
end

function _tucker_cross_core(p::Manifolds.TuckerPoint, q::Manifolds.TuckerPoint)
    _, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)
    Ht = Gq
    @inbounds for k in eachindex(Up)
        Ht = mode_n_product(Ht, Up[k]' * Uq[k], k)
    end
    return Ht
end

function _tucker_cross_core(
    backend::BTDBackend{T},
    p::Manifolds.TuckerPoint,
    q::Manifolds.TuckerPoint,
) where {T}
    _, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)
    mode_mats = ntuple(k -> (k, Matrix{T}(Up[k]' * Uq[k])), length(Up))
    return _workspace_copy_tensor!(
        backend.workspace,
        _apply_mode_products(backend, Gq, mode_mats),
    )
end

function _tucker_cross_except_mode(
    p::Manifolds.TuckerPoint,
    q::Manifolds.TuckerPoint,
    m::Int,
)
    _, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)
    Ht = Gq
    @inbounds for k in eachindex(Up)
        k == m && continue
        Ht = mode_n_product(Ht, Up[k]' * Uq[k], k)
    end
    return Ht
end

function _tucker_cross_except_mode(
    backend::BTDBackend{T},
    p::Manifolds.TuckerPoint,
    q::Manifolds.TuckerPoint,
    m::Int,
) where {T}
    _, Up = _tucker_data(p)
    Gq, Uq = _tucker_data(q)
    mode_mats = Tuple((k, Matrix{T}(Up[k]' * Uq[k])) for k in eachindex(Up) if k != m)
    return _workspace_copy_tensor!(
        backend.workspace,
        _apply_mode_products(backend, Gq, mode_mats),
    )
end
