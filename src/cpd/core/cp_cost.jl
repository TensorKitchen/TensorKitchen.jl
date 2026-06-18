# cpd/core/cp_cost.jl — Cost and Euclidean gradients for Segre (rank-1) and Secant (rank-r) CP
export cost_segre, egrad_segre, cost_secant_rankr, egrad_secant_rankr

function cost_segre(A::AbstractArray{T,N}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}
    normA2 = sum(abs2, A)
    return function (M, p)
        λ, U = unpack_point_rank1(p, dims)
        inner = rank1_inner(A, U)
        return 0.5 * (normA2 + λ^2 - 2 * λ * inner)
    end
end

"""
    egrad_segre(A, dims)

Return an ambient gradient representative `(M, p)` for the rank-1 Segre cost.
This function returns the Euclidean gradient of the cost function.
"""
function egrad_segre(A::AbstractArray{T,N}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}
    # Reuse one contraction buffer per mode inside the closure to cut temporary vectors.
    grad_buf = [Vector{T}(undef, dims[m]) for m in eachindex(dims)]
    return function (M, p)
        if M isa Manifolds.Segre
            parts = point_parts(p)
            λ, inner = parts[1][1], rank1_inner_parts(A, parts)
            grad_U = Vector{Vector{T}}(undef, length(dims))
            for m in eachindex(dims)
                g = grad_buf[m]
                rank1_mode_contract_parts!(g, A, parts, m)
                rmul!(g, -λ)
                grad_U[m] = copy(g)
            end
            return pack_tangent_rank1_segre(λ - inner, grad_U)
        else
            λ, U = unpack_point_rank1(p, dims)
            inner = rank1_inner(A, U)
            grad_U = Vector{Vector{T}}(undef, length(dims))
            for m in eachindex(dims)
                g = grad_buf[m]
                rank1_mode_contract!(g, A, U, m)
                rmul!(g, -λ)
                grad_U[m] = copy(g)
            end
            return pack_tangent_rank1(λ - inner, grad_U)
        end
    end
end

@inline function _require_vector_for_squaring_metric(M, p)
    if (M isa SqEuclidean || M isa SoftplusEuclidean) && !(p isa AbstractVector)
        throw(
            ArgumentError(
                "Pullback metric manifolds expect flat vector points. " *
                "Got p::$(typeof(p)); use pack_point_rank1_to_vector/pack_point_rankr_to_vector.",
            ),
        )
    end
    return nothing
end

@inline _uses_softplus_pullback(M) = M isa SoftplusEuclidean
@inline _uses_softplus_pullback(M::ProductManifold) =
    any(_uses_softplus_pullback, M.manifolds)
@inline _uses_softplus_pullback(M::MetricManifold) = _uses_softplus_pullback(M.manifold)
@inline _uses_softplus_pullback(M::PowerManifold) = _uses_softplus_pullback(M.manifold)

@inline function _softplus_value(x::Real)
    x > 20 ? x : log1p(exp(x))
end

@inline function _softplus_derivative(x::Real)
    x >= 0 ? one(x) / (one(x) + exp(-x)) : begin
        ex = exp(x)
        ex / (one(x) + ex)
    end
end

@inline function _invsoftplus(x::Real)
    x > 20 ? x : log(expm1(max(x, eps(x))))
end

function cost_segre_nn(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
) where {T<:AbstractFloat,N}
    return function (M, p)
        _require_vector_for_squaring_metric(M, p)
        λ̃, Ũ = unpack_point_rank1(p, dims)
        if _uses_softplus_pullback(M)
            λ = _softplus_value(λ̃)
            U = [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
            X = reconstruct_cp_rank1(λ, U)
        else
            X = embed_point_rank1_nn(p, dims)
        end
        return 0.5 * sum(abs2, A .- X)
    end
end

function egrad_segre_nn(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
) where {T<:AbstractFloat,N}
    # Nonnegative rank-1 gradients use the same contraction reuse pattern as
    # the unconstrained case, but the returned egrad is with respect to the
    # latent coordinates p after the chart chain rule, not with respect to the
    # transformed nonnegative variable.
    grad_buf = [Vector{T}(undef, dims[m]) for m in eachindex(dims)]
    return function (M, p)
        _require_vector_for_squaring_metric(M, p)
        λ̃, Ũ = unpack_point_rank1(p, dims)
        use_softplus = _uses_softplus_pullback(M)
        λ = use_softplus ? _softplus_value(λ̃) : λ̃^2
        U =
            use_softplus ? [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)] :
            [Ũ[m] .^ 2 for m in eachindex(Ũ)]
        inner = rank1_inner(A, U)
        normsq = one(T)
        mode_normsq = Vector{T}(undef, length(U))
        @inbounds for m in eachindex(U)
            nm = sum(abs2, U[m])
            mode_normsq[m] = nm
            normsq *= nm
        end
        grad_λ = λ * normsq - inner
        grad_U = Vector{Vector{T}}(undef, length(dims))
        for m in eachindex(dims)
            g = grad_buf[m]
            rank1_mode_contract!(g, A, U, m)
            rmul!(g, -λ)
            g .+= (λ^2 * (normsq / mode_normsq[m])) .* U[m]
            grad_U[m] = copy(g)
        end
        if use_softplus
            grad_λ̃ = grad_λ * _softplus_derivative(λ̃)
            grad_Ũ = [grad_U[m] .* _softplus_derivative.(Ũ[m]) for m in eachindex(Ũ)]
        else
            grad_λ̃ = grad_λ * 2 * λ̃
            grad_Ũ = [grad_U[m] .* 2 .* Ũ[m] for m in eachindex(Ũ)]
        end
        return p isa Vector ? pack_point_rank1_to_vector(grad_λ̃, grad_Ũ) :
               pack_point_rank1(grad_λ̃, grad_Ũ)
    end
end

_gram_matrices(U::Vector{Matrix{T}}) where {T<:AbstractFloat} =
    [U[m]' * U[m] for m in eachindex(U)]

function _cross_unit_from_grams(grams::Vector{Matrix{T}}) where {T<:AbstractFloat}
    r = size(grams[1], 1)
    C = ones(T, r, r)
    for G in grams
        C .*= G
    end
    return C
end

function _inner_from_mttkrp_first_mode(
    U::Vector{Matrix{T}},
    M1::Matrix{T},
) where {T<:AbstractFloat}
    r = size(M1, 2)
    inner = Vector{T}(undef, r)
    @inbounds for k = 1:r
        inner[k] = dot(@view(U[1][:, k]), @view(M1[:, k]))
    end
    return inner
end

mutable struct _CPRankrEvalCache{T<:AbstractFloat}
    snapshot::Any
    valid::Bool
    contracts::Vector{Matrix{T}}
    grams::Vector{Matrix{T}}
    cross_mat::Matrix{T}
    inner::Vector{T}
end

function _CPRankrEvalCache(
    ::Type{T},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    contracts = [Matrix{T}(undef, dims[m], r) for m = 1:N]
    grams = [Matrix{T}(undef, r, r) for _ = 1:N]
    return _CPRankrEvalCache{T}(
        nothing,
        false,
        contracts,
        grams,
        Matrix{T}(undef, r, r),
        Vector{T}(undef, r),
    )
end

@inline _cp_rankr_point_snapshot(x::Number) = x
@inline _cp_rankr_point_snapshot(x::AbstractArray) = copy(x)
@inline _cp_rankr_point_snapshot(x::Tuple) = map(_cp_rankr_point_snapshot, x)
@inline _cp_rankr_point_snapshot(x) =
    hasproperty(x, :x) ? map(_cp_rankr_point_snapshot, getproperty(x, :x)) : deepcopy(x)

@inline _cp_rankr_same_point_snapshot(s::Number, p) = s == p
@inline _cp_rankr_same_point_snapshot(s::AbstractArray, p::AbstractArray) = s == p
@inline function _cp_rankr_same_point_snapshot(s::Tuple, p::Tuple)
    length(s) == length(p) || return false
    @inbounds for i in eachindex(s)
        _cp_rankr_same_point_snapshot(s[i], p[i]) || return false
    end
    return true
end
@inline function _cp_rankr_same_point_snapshot(s::Tuple, p)
    return hasproperty(p, :x) && _cp_rankr_same_point_snapshot(s, Tuple(getproperty(p, :x)))
end
@inline function _cp_rankr_same_point(cache::_CPRankrEvalCache, p)
    return cache.valid && _cp_rankr_same_point_snapshot(cache.snapshot, p)
end

function _cp_rankr_refresh_cache!(
    cache::_CPRankrEvalCache{T},
    A::AbstractArray{T,N},
    U::Vector{Matrix{T}},
    p;
    method::Symbol = :auto,
) where {T<:AbstractFloat,N}
    if _cp_rankr_same_point(cache, p)
        return cache
    end

    @inbounds for m = 1:N
        copyto!(cache.contracts[m], mttkrp(A, U, m; method))
        mul!(cache.grams[m], transpose(U[m]), U[m])
    end

    fill!(cache.cross_mat, one(T))
    @inbounds for m = 1:N
        cache.cross_mat .*= cache.grams[m]
    end

    @inbounds for k in eachindex(cache.inner)
        cache.inner[k] = dot(@view(U[1][:, k]), @view(cache.contracts[1][:, k]))
    end

    cache.snapshot = _cp_rankr_point_snapshot(p)
    cache.valid = true
    return cache
end

@inline function _scale_columns_by_lambda!(
    G::AbstractMatrix{T},
    λ::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds for k in eachindex(λ)
        @views G[:, k] .*= λ[k]
    end
    return G
end

function _rankr_gradU_from_terms(
    U,
    λ::AbstractVector{T},
    contracts,
    grams,
) where {T<:AbstractFloat}
    Nmodes = length(U)
    r = length(λ)
    gradU = Vector{Matrix{T}}(undef, Nmodes)
    λouter = λ * transpose(λ)
    @inbounds for m = 1:Nmodes
        prod_except = ones(T, r, r)
        for j = 1:Nmodes
            j == m && continue
            prod_except .*= grams[j]
        end
        C = prod_except .* λouter
        Gm = U[m] * C
        Gm .-= contracts[m] .* transpose(λ)
        gradU[m] = Gm
    end
    return gradU
end

"""
    cost_secant_rankr(A, dims, r)

Return a cost function (M, p) -> Float64 for the rank-r secant CP objective
‖A - Σₖ λₖ u₁⁽ᵏ⁾ ⊗ ... ⊗ uₙ⁽ᵏ⁾‖²/2. Built from Segre costs per component plus cross terms.
`p` follows the join-parameter layout from `pack_point_rankr(λ, U, r)`.
"""
function cost_secant_rankr(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    normA2 = sum(abs2, A)
    return function (M, p)
        λ, U = unpack_rankr_join(p, dims, r)
        M1 = mttkrp(A, U, 1; method = :auto)
        inner = _inner_from_mttkrp_first_mode(U, M1)
        grams = _gram_matrices(U)
        cross_mat = _cross_unit_from_grams(grams)
        return cp_rankr_cost_value(normA2, λ, inner, cross_mat)
    end
end

"""
    egrad_secant_rankr(A, dims, r; scale_by_lambda, lambda_eps)

Return the Euclidean gradient for the rank-r secant CP cost.
Uses Segre gradient per component plus cross-term contributions.
"""
function egrad_secant_rankr(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int;
    scale_by_lambda::Bool = true,
    lambda_eps::Float64 = 1e-10,
) where {T<:AbstractFloat,N}
    lambda_eps_T = T(lambda_eps)
    return function (M, p)
        λ, U = unpack_rankr_join(p, dims, r)
        Nmodes = length(U)

        contracts = Vector{Matrix{T}}(undef, Nmodes)
        for m = 1:Nmodes
            contracts[m] = mttkrp(A, U, m; method = :auto)
        end

        inner = _inner_from_mttkrp_first_mode(U, contracts[1])
        grams = _gram_matrices(U)
        cross_mat = _cross_unit_from_grams(grams)
        grad_λ = grad_lambda_cp(λ, inner, cross_mat)

        gradU = _rankr_gradU_from_terms(U, λ, contracts, grams)

        if scale_by_lambda
            for k = 1:r
                λ_abs = max(abs(λ[k]), lambda_eps_T)
                for m = 1:Nmodes
                    gradU[m][:, k] ./= λ_abs
                end
            end
        end

        return pack_rankr_join_tuple(grad_λ, gradU, r)
    end
end

# ---- Rank-r via canonical Product(Euclidean(r), Product(Sphere,r)^d) ----

@inline function _canonical_rankr_fill_factors!(
    Ubuf,
    parts,
    dims::NTuple{N,Int},
    r::Int,
) where {N}
    length(parts) == N + 1 ||
        throw(DimensionMismatch("expected $(N+1) tuple parts, got $(length(parts))"))
    λ = parts[1]
    length(λ) == r || throw(DimensionMismatch("expected λ length $r, got $(length(λ))"))
    @inbounds for m = 1:N
        mode_m = parts[m+1]
        length(mode_m) == r ||
            throw(DimensionMismatch("mode $m has $(length(mode_m)) vectors, expected $r"))
        if !isassigned(Ubuf, m) || size(Ubuf[m], 1) != dims[m] || size(Ubuf[m], 2) != r
            Um = Matrix{eltype(λ)}(undef, dims[m], r)
            Ubuf[m] = Um
        else
            Um = Ubuf[m]
        end
        for k = 1:r
            uk = mode_m[k]
            length(uk) == dims[m] || throw(
                DimensionMismatch(
                    "mode $m, component $k has length $(length(uk)), expected $(dims[m])",
                ),
            )
            Um[:, k] .= uk
        end
    end
    return λ
end

"""
    cost_rankr_canonical(A, dims, r)

Rank-r CP cost on canonical product manifold `ℝ^r × ∏_m (S^(n_m-1))^r`.
Point layout is `(λ, mode₁, ..., mode_d)` with each `mode_m = (u_{m,1}, ..., u_{m,r})`
stored in the product-manifold-compatible container.
"""
function cost_rankr_canonical(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    normA2 = sum(abs2, A)
    Nmodes = length(dims)
    Ubuf = Vector{Matrix{T}}(undef, Nmodes)
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        parts = normalize_rankr_canonical_point(p, dims, r)
        λ = _canonical_rankr_fill_factors!(Ubuf, parts, dims, r)
        _cp_rankr_refresh_cache!(cache, A, Ubuf, p)
        return cp_rankr_cost_value(normA2, λ, cache.inner, cache.cross_mat)
    end
end

"""
    egrad_rankr_canonical(A, dims, r; scale_by_lambda, lambda_eps)

Euclidean gradient on canonical point layout `(λ, mode₁, ..., mode_d)`.
Returns tangent in the same container layout.
"""
function egrad_rankr_canonical(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int;
    scale_by_lambda::Bool = true,
    lambda_eps::Float64 = 1e-10,
) where {T<:AbstractFloat,N}
    lambda_eps_T = T(lambda_eps)
    Nmodes = length(dims)
    Ubuf = Vector{Matrix{T}}(undef, Nmodes)
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        parts = normalize_rankr_canonical_point(p, dims, r)
        λ = _canonical_rankr_fill_factors!(Ubuf, parts, dims, r)
        _cp_rankr_refresh_cache!(cache, A, Ubuf, p)
        grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
        gradU = _rankr_gradU_from_terms(Ubuf, λ, cache.contracts, cache.grams)

        if scale_by_lambda
            for k = 1:r
                λ_abs = max(abs(λ[k]), lambda_eps_T)
                for m = 1:Nmodes
                    gradU[m][:, k] ./= λ_abs
                end
            end
        end

        return wrap_rankr_canonical_tangent_like(p, grad_λ, gradU, r)
    end
end

# ---- Rank-r via Product(Manifolds.Segre, ...) ----

"""
    cost_rankr_native(A, dims, r)

Rank-r CP cost on `ProductManifold(Manifolds.Segre(...), ...)` with one
`Manifolds.Segre` point per component.
For `p = (p₁, …, pᵣ)`, unpacks native factors `(λ, U)` and evaluates
the same CP objective as canonical/join geometry via MTTKRP/Gram contractions.
"""
function cost_rankr_native(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    normA2 = sum(abs2, A)
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        λ, U = unpack_rankr_native(p, dims, r)
        _cp_rankr_refresh_cache!(cache, A, U, p)
        return cp_rankr_cost_value(normA2, λ, cache.inner, cache.cross_mat)
    end
end

"""
    egrad_rankr_native(A, dims, r; scale_by_lambda, lambda_eps)

Euclidean gradient on `ProductManifold(Manifolds.Segre(...), ...)` where each
component uses the `Manifolds.Segre` point layout.
Unpacks native factors `(λ, U)`, computes Euclidean CP gradients in matrix form,
then repacks each rank-1 component as `[[νₖ], u̇₁ₖ, ..., u̇_dₖ]`.
Returns ProductManifold-compatible tuple `(g₁, …, gᵣ)`.
"""
function egrad_rankr_native(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int;
    scale_by_lambda::Bool = true,
    lambda_eps::Real = 1e-10,
) where {T<:AbstractFloat,N}
    lambda_eps_T = T(lambda_eps)
    Nmodes = length(dims)
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        λ, U = unpack_rankr_native(p, dims, r)
        _cp_rankr_refresh_cache!(cache, A, U, p)
        grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
        gradU = _rankr_gradU_from_terms(U, λ, cache.contracts, cache.grams)

        if scale_by_lambda
            for k = 1:r
                λ_abs = max(abs(λ[k]), lambda_eps_T)
                for m = 1:Nmodes
                    gradU[m][:, k] ./= λ_abs
                end
            end
        end

        grad_parts = Vector{Vector{Vector{T}}}(undef, r)
        @inbounds for k = 1:r
            grad_Uk = [Vector(@view gradU[m][:, k]) for m = 1:Nmodes]
            grad_parts[k] = pack_tangent_rank1_segre(grad_λ[k], grad_Uk)
        end

        return hasproperty(p, :x) ? ArrayPartition(grad_parts...) : (grad_parts...,)
    end
end

"""
    cost_secant_rankr_nn(A, dims, r)

Nonnegative rank-r cost via squaring: p stores (λ̃, Ũ); cost uses λ=λ̃², U=Ũ² (elementwise).
"""
function cost_secant_rankr_nn(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    normA2 = sum(abs2, A)
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        _require_vector_for_squaring_metric(M, p)
        λ̃, Ũ = unpack_point_rankr(p, dims, r)
        use_softplus = _uses_softplus_pullback(M)
        λ = use_softplus ? _softplus_value.(λ̃) : λ̃ .^ 2
        U =
            use_softplus ? [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)] :
            [Ũ[m] .^ 2 for m in eachindex(Ũ)]
        _cp_rankr_refresh_cache!(cache, A, U, p)
        return cp_rankr_cost_value(normA2, λ, cache.inner, cache.cross_mat)
    end
end

"""
    egrad_secant_rankr_nn(A, dims, r)

Nonnegative rank-r gradient via squaring: chain rule grad_λ̃ = grad_λ .* 2λ̃, grad_Ũ = grad_U .* 2Ũ.
"""
function egrad_secant_rankr_nn(
    A::AbstractArray{T,N},
    dims::NTuple{N,Int},
    r::Int,
) where {T<:AbstractFloat,N}
    cache = _CPRankrEvalCache(T, dims, r)
    return function (M, p)
        _require_vector_for_squaring_metric(M, p)
        λ̃, Ũ = unpack_point_rankr(p, dims, r)
        use_softplus = _uses_softplus_pullback(M)
        λ = use_softplus ? _softplus_value.(λ̃) : λ̃ .^ 2
        U =
            use_softplus ? [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)] :
            [Ũ[m] .^ 2 for m in eachindex(Ũ)]
        _cp_rankr_refresh_cache!(cache, A, U, p)
        grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
        gradU = _rankr_gradU_from_terms(U, λ, cache.contracts, cache.grams)
        if use_softplus
            grad_λ̃ = grad_λ .* _softplus_derivative.(λ̃)
            grad_Ũ = gradU
            for m in eachindex(U)
                grad_Ũ[m] .*= _softplus_derivative.(Ũ[m])
            end
        else
            grad_λ̃ = grad_λ .* 2 .* λ̃
            grad_Ũ = gradU
            for m in eachindex(U)
                grad_Ũ[m] .*= 2 .* Ũ[m]
            end
        end
        return p isa Vector ? pack_point_rankr_to_vector(grad_λ̃, grad_Ũ, r) :
               pack_point_rankr(grad_λ̃, grad_Ũ, r)
    end
end
