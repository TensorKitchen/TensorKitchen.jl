# join/join_init.jl — Init helpers for JoinBackend / BTDBackend manifold-type dispatch

function _reshape_like_tucker(M::Manifolds.Tucker, X, what::AbstractString)
    dims = factor_dims(M)
    size(X) == dims && return X
    length(X) == prod(dims) || throw(
        DimensionMismatch(
            "$what length $(length(X)) does not match Tucker ambient size $(prod(dims)) for dims $dims.",
        ),
    )
    return reshape(vec(X), dims)
end

function _sphere_init(M, target, init::Symbol)
    n = ManifoldsBase.representation_size(M)
    n = n isa Tuple ? prod(n) : n
    T = eltype(target)
    if init == :deterministic
        x = zeros(T, n)
        x[1] = one(T)
        return x
    end
    if init == :target
        # Reuse the target's flattened storage/backend instead of forcing Array(target).
        x = vec(target)
        nx = norm(x)
        nx > eps(T) ||
            throw(ArgumentError("init=:target failed because target has near-zero norm."))
        return x ./ nx
    end
    x = randn(T, n)
    return x ./ norm(x)
end

function _segre_init(M::Manifolds.Segre, target, init::Symbol)
    dims = factor_dims(M)
    T = eltype(target)

    if init == :deterministic
        factors = [
            begin
                u = zeros(T, d)
                u[1] = one(T)
                u
            end for d in dims
        ]
        return pack_point_rank1_segre(one(T), factors)
    end

    if init == :random
        factors = [random_unit_vector(d, T) for d in dims]
        return pack_point_rank1_segre(one(T), factors)
    end

    throw(
        ArgumentError(
            "No default init=$init for Manifolds.Segre. Use init=:random or :deterministic.",
        ),
    )
end

function _tucker_init(M::Manifolds.Tucker, target, init::Symbol)
    d = factor_dims(M)
    r = multilinear_rank(M)
    T = eltype(target)
    A = _reshape_like_tucker(M, target, "Tucker init target")
    if init == :random
        core = randn(T, r...)
        factors = [_rand_orthonormal_tucker(d[m], r[m], T) for m in eachindex(d)]
        return Manifolds.TuckerPoint(core, factors...)
    end
    if init in (:tucker, :tucker_diag)
        core, factors = tucker_hosvd(A, r)
        return Manifolds.TuckerPoint(core, factors...)
    end
    if init == :sthosvd
        td = sthosvd(A, r)
        return Manifolds.TuckerPoint(td.core, td.factors...)
    end
    if init in (:hosvd, :thosvd)
        throw(
            ArgumentError(
                "Tucker init=$init is no longer supported in the pipeline. Use :sthosvd, :tucker, or :tucker_diag.",
            ),
        )
    end

    # fallback to random
    core = randn(T, r...)
    factors = [_rand_orthonormal_tucker(d[m], r[m], T) for m in eachindex(d)]
    return Manifolds.TuckerPoint(core, factors...)
end

function _rand_orthonormal_tucker(n::Int, r::Int, ::Type{T}) where {T<:AbstractFloat}
    Q, _ = qr(randn(T, n, r))
    return Matrix(Q[:, 1:r])
end

function _tucker_all_except_mode_products(
    core::AbstractArray{T,N},
    factors,
) where {T<:AbstractFloat,N}
    N == 1 && return Array{T,N}[Array(core)]
    products = Vector{Array{T,N}}(undef, N)
    @inbounds for m = 1:N
        # Precompute all "except mode m" contractions once so factor gradients can reuse them.
        B = core
        for j = 1:N
            j == m && continue
            B = mode_n_product(B, factors[j], j)
        end
        products[m] = Array(B)
    end
    return products
end

function _tucker_egrad(M::Manifolds.Tucker, p, R)
    p isa Manifolds.TuckerPoint || throw(
        ArgumentError(
            "Expected native TuckerPoint for Manifolds.Tucker, got $(typeof(p)).",
        ),
    )
    R = _reshape_like_tucker(M, R, "Tucker residual")
    core = p.hosvd.core
    factors = p.hosvd.U
    N = length(factors)

    grad_core = copy(R)
    for m = 1:N
        grad_core = mode_n_product(grad_core, factors[m]', m)
    end

    residual_unfolds = [unfold_mode(R, m) for m = 1:N]
    # Reuse these intermediates across all factor-gradient blocks.
    all_except_mode = _tucker_all_except_mode_products(core, factors)
    grad_factors = Vector{Matrix{eltype(core)}}(undef, N)
    for m = 1:N
        Bm = unfold_mode(all_except_mode[m], m)
        grad_factors[m] = residual_unfolds[m] * transpose(Bm)
    end

    return Manifolds.TuckerTangentVector(grad_core, Tuple(grad_factors))
end
