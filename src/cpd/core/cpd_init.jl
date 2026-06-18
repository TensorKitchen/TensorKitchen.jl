# cpd/core/cpd_init.jl — CP init: init_cpd_factors, init_cp_rank1, cp_init_tucker
export cp_init_tucker

function _cp_tucker_decomposition(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int};
    method::Symbol = :hosvd,
) where {T<:AbstractFloat,N}
    return _cp_tucker_decomposition(Val(method), A, ranks)
end

function _cp_tucker_decomposition(::Val{:hosvd}, A, ranks)
    return tucker_hosvd(A, ranks)
end

function _cp_tucker_decomposition(::Val{:sthosvd}, A, ranks)
    t = sthosvd(A, ranks)
    return t.core, t.factors
end

function _cp_tucker_decomposition(::Val{:thosvd}, A, ranks)
    t = thosvd(A, ranks)
    return t.core, t.factors
end

function _cp_tucker_decomposition(::Val{:hooi}, A, ranks)
    t = hooi(A, ranks; init = :sthosvd, maxiter = 20, verbose = false)
    return t.core, t.factors
end

function _cp_tucker_decomposition(::Val{M}, A, ranks) where {M}
    throw(ArgumentError("Unknown method=$M. Use :hosvd, :sthosvd, :thosvd, or :hooi."))
end

function _cp_core_diag_init(core::AbstractArray{T,N}, r::Int) where {T<:AbstractFloat,N}
    core_dims = size(core)
    λ0 = _tucker_diag(core, r)
    U0 = Vector{Matrix{T}}(undef, N)
    for m = 1:N
        rm = core_dims[m]
        Um = fill!(similar(core, rm, r), zero(T))
        n_eye = min(rm, r)
        for k = 1:n_eye
            Um[k, k] = one(T)
        end
        if rm > 0 && r > n_eye
            Um[:, (n_eye+1):r] .= random_unit_matrix(rm, r - n_eye, T)
        end
        U0[m] = Um
    end
    return λ0, U0
end

function _compose_tucker_cp(
    factors::Vector{<:AbstractMatrix{T}},
    λcore::AbstractVector{T},
    Ucore::Vector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    U = [
        begin
            Um = similar(factors[m], size(factors[m], 1), size(Ucore[m], 2))
            mul!(Um, factors[m], Ucore[m])
            Um
        end for m in eachindex(factors)
    ]
    λ = Vector{T}(λcore)
    normalize_factors!(U, λ)
    return λ, U
end


"""Optimal CP weights for fixed factor columns (Frobenius), via `Z \\ vec(A)` on rank-1 basis columns."""
function _cp_least_squares_lambda(
    A::AbstractArray{T,N},
    U0::Vector{Matrix{T}},
    r::Int,
) where {T<:AbstractFloat,N}
    dlen = length(A)
    Z = similar(A, T, dlen, r)
    @inbounds for k = 1:r
        vecs = [collect(@view U0[m][:, k]) for m in eachindex(U0)]
        Z[:, k] .= vec(reconstruct_cp_rank1(one(T), vecs))
    end
    return Vector{T}(Z \ vec(A))
end

function init_cpd_factors(
    A::AbstractArray{T,N},
    r::Int;
    init::Symbol = :random,
) where {T<:AbstractFloat,N}
    dims = size(A)
    init == :random && return (ones(T, r), [random_unit_matrix(dims[m], r, T) for m = 1:N])
    init in (:hosvd, :tucker_diag, :tucker) || throw(
        ArgumentError("Unknown init=$init. Use :random, :hosvd, :tucker_diag, or :tucker."),
    )
    ranks = ntuple(_ -> r, N)
    core, factors = tucker_hosvd(A, ranks)
    U0 = [
        size(factors[m], 2) < r ?
        hcat(
            factors[m],
            random_unit_matrix(size(factors[m], 1), r - size(factors[m], 2), T),
        ) : factors[m] for m = 1:N
    ]
    λ0 = if init == :tucker_diag
        _tucker_diag(core, r)
    elseif init == :tucker
        _cp_least_squares_lambda(A, U0, r)
    else
        ones(T, r)
    end
    return λ0, U0
end

function init_cp_rank1(
    A::AbstractArray{T,N};
    init::Symbol = :random,
) where {T<:AbstractFloat,N}
    dims = size(A)
    init == :random && return [random_unit_vector(dims[m], T) for m = 1:N]
    init in (:hosvd, :tucker_diag, :tucker) || throw(
        ArgumentError("Unknown init=$init. Use :random, :hosvd, :tucker_diag, or :tucker."),
    )
    _, factors = tucker_hosvd(A, ntuple(_ -> 1, N))
    return [vec(factors[m][:, 1]) for m = 1:N]
end

function cp_init_tucker(
    A::AbstractArray{T,N},
    r::Int;
    method::Symbol = :sthosvd,
) where {T<:AbstractFloat,N}
    r >= 1 || throw(ArgumentError("rank r must be ≥ 1, got r=$r"))
    dims = size(A)
    ranks = ntuple(i -> min(r, dims[i]), N)
    core, factors = _cp_tucker_decomposition(A, ranks; method)
    λcore0, Ucore0 = _cp_core_diag_init(core, r)
    λcore, Ucore = fit_cp_als(
        core,
        r;
        maxiter = 10,
        tol = zero(T),
        init_factors = (λcore0, Ucore0),
        verbose = false,
        return_stats = false,
    )
    return _compose_tucker_cp(factors, λcore, Ucore)
end
