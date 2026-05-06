# cpd/model/rankr.jl — Rank-r CPD model
export embed_point

"""
    RankRCPDModel{T, N}

Rank-r CPD model: approximates tensor A as ∑_k λ_k · u₁ₖ ⊗ u₂ₖ ⊗ ... ⊗ u_dₖ.

Fields:
- `A`: target tensor
- `dims`: dimensions tuple
- `r`: rank
- `M`: parameter manifold (native/canonical/nonnegative Euclidean)
- `scale_by_lambda`: whether to scale gradients by lambda
- `lambda_eps`: epsilon for lambda scaling
"""
struct RankRCPDModel{T<:AbstractFloat,N,A<:AbstractArray{T,N},M<:AbstractManifold} <:
       AbstractDecompositionModel{T}
    # Keep the target/manifold concrete so hot CP rank-r code can specialize fully.
    A::A
    dims::NTuple{N,Int}
    r::Int
    M::M
    geometry::Symbol
    scale_by_lambda::Bool
    lambda_eps::T
    nonnegative::Bool
end

@inline _is_native_rankr_geometry(geometry::Symbol) = geometry == :native

function RankRCPDModel(
    A::AbstractArray{T,N},
    r::Int;
    geometry::Symbol = :canonical,
    scale_by_lambda::Bool = true,
    lambda_eps::Real = 1e-10,
    nonnegative::Bool = false,
    use_pullback_metric::Bool = false,
    pullback_eps::Real = 1e-8,
) where {T<:AbstractFloat,N}
    dims = size(A)
    geometry in (:native, :canonical, :squaring_metric, :softplus_metric) || throw(
        ArgumentError(
            "Unknown geometry=$geometry. Use :native, :canonical, :squaring_metric, or :softplus_metric (regularized pullback-style geometries require nonnegative=true).",
        ),
    )
    (geometry ∉ (:squaring_metric, :softplus_metric) || nonnegative) ||
        throw(ArgumentError("geometry=$geometry requires nonnegative=true."))
    (!nonnegative || !_is_native_rankr_geometry(geometry)) || throw(
        ArgumentError(
            "nonnegative=true with geometry=:native not supported. Use :canonical, :squaring_metric, or :softplus_metric.",
        ),
    )
    geometry_eff = _is_native_rankr_geometry(geometry) ? :native : geometry
    M = if nonnegative
        # Nonnegative rank-r: structured join layout with product of (Sq)Euclidean factors
        use_pullback_sq = use_pullback_metric || geometry == :squaring_metric
        use_pullback_softplus = geometry == :softplus_metric
        Nd = length(dims)
        n_factors = r * (Nd + 1)
        manifolds = Vector{AbstractManifold}(undef, n_factors)
        idx = 1
        @inbounds for k = 1:r
            manifolds[idx] =
                use_pullback_softplus ? SoftplusEuclidean(1; ε = pullback_eps) :
                (use_pullback_sq ? SqEuclidean(1; ε = pullback_eps) : Euclidean(1))
            idx += 1
            for m = 1:Nd
                manifolds[idx] =
                    use_pullback_softplus ?
                    SoftplusEuclidean(dims[m]; ε = pullback_eps) :
                    (
                        use_pullback_sq ? SqEuclidean(dims[m]; ε = pullback_eps) :
                        Euclidean(dims[m])
                    )
                idx += 1
            end
        end
        ProductManifold(manifolds...)
    elseif geometry_eff == :native
        SegreProduct(dims, r)
    else
        CanonicalCP(dims, r)
    end
    scale_by_lambda_eff = nonnegative ? false : scale_by_lambda
    lambda_eps_T = T(lambda_eps)
    return RankRCPDModel(
        A,
        dims,
        r,
        M,
        geometry_eff,
        scale_by_lambda_eff,
        lambda_eps_T,
        nonnegative,
    )
end

tensor(model::RankRCPDModel) = model.A
manifold(model::RankRCPDModel) = model.M

function embed_point(model::RankRCPDModel{T,N}, p) where {T,N}
    if model.nonnegative
        λ̃, Ũ = unpack_point_rankr(p, model.dims, model.r)
        if model.geometry == :softplus_metric
            λ = _softplus_value.(λ̃)
            U = [_softplus_value.(Ũ[m]) for m = 1:length(Ũ)]
            return reconstruct_cpd_rankr(λ, U)
        end
        return embed_point_rankr_nn(p, model.dims, model.r)
    end
    λ, U =
        model.geometry == :native ? unpack_rankr_native(p, model.dims, model.r) :
        unpack_rankr_canonical(p, model.dims, model.r)
    return reconstruct_cpd_rankr(λ, U)
end

function cost(model::RankRCPDModel{T,N}, p) where {T,N}
    return model_cost_function(model)(model.M, p)
end

function egrad(model::RankRCPDModel{T,N}, p) where {T,N}
    return model_egrad_function(model)(model.M, p)
end

function model_cost_function(model::RankRCPDModel{T,N}) where {T,N}
    return model.nonnegative ? cost_secant_rankr_nn(model.A, model.dims, model.r) :
           (
        model.geometry == :native ? cost_rankr_native(model.A, model.dims, model.r) :
        cost_rankr_canonical(model.A, model.dims, model.r)
    )
end

function model_cost_egrad_functions(model::RankRCPDModel{T,N}) where {T,N}
    if model.nonnegative
        cache = _CPRankrEvalCache(T, model.dims, model.r)
        cost_fn = function (M, p)
            _require_vector_for_squaring_metric(M, p)
            λ̃, Ũ = unpack_point_rankr(p, model.dims, model.r)
            use_softplus = model.geometry == :softplus_metric
            λ = use_softplus ? _softplus_value.(λ̃) : λ̃ .^ 2
            U =
                use_softplus ? [_softplus_value.(Ũ[m]) for m = 1:length(Ũ)] :
                [Ũ[m] .^ 2 for m = 1:length(Ũ)]
            _cp_rankr_refresh_cache!(cache, model.A, U, p)
            return cp_rankr_cost_value(sum(abs2, model.A), λ, cache.inner, cache.cross_mat)
        end
        egrad_fn = function (M, p)
            _require_vector_for_squaring_metric(M, p)
            λ̃, Ũ = unpack_point_rankr(p, model.dims, model.r)
            use_softplus = model.geometry == :softplus_metric
            λ = use_softplus ? _softplus_value.(λ̃) : λ̃ .^ 2
            U =
                use_softplus ? [_softplus_value.(Ũ[m]) for m = 1:length(Ũ)] :
                [Ũ[m] .^ 2 for m = 1:length(Ũ)]
            _cp_rankr_refresh_cache!(cache, model.A, U, p)
            grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
            gradU = _rankr_gradU_from_terms(U, λ, cache.contracts, cache.grams)
            # These are Euclidean gradients with respect to the latent
            # coordinates p, i.e. ∇_p F̃(p) after applying the chart chain rule.
            # egrad_to_rgrad then applies the regularized metric inverse.
            if use_softplus
                grad_λ̃ = grad_λ .* _softplus_derivative.(λ̃)
                grad_Ũ = [gradU[m] .* _softplus_derivative.(Ũ[m]) for m = 1:length(U)]
            else
                grad_λ̃ = grad_λ .* 2 .* λ̃
                grad_Ũ = [gradU[m] .* 2 .* Ũ[m] for m = 1:length(U)]
            end
            return p isa Vector ? pack_point_rankr_to_vector(grad_λ̃, grad_Ũ, model.r) :
                   pack_point_rankr(grad_λ̃, grad_Ũ, model.r)
        end
        return cost_fn, egrad_fn
    elseif model.geometry == :native
        cache = _CPRankrEvalCache(T, model.dims, model.r)
        normA2 = sum(abs2, model.A)
        cost_fn = function (M, p)
            λ, U = unpack_rankr_native(p, model.dims, model.r)
            _cp_rankr_refresh_cache!(cache, model.A, U, p)
            return cp_rankr_cost_value(normA2, λ, cache.inner, cache.cross_mat)
        end
        egrad_fn = function (M, p)
            λ, U = unpack_rankr_native(p, model.dims, model.r)
            _cp_rankr_refresh_cache!(cache, model.A, U, p)
            grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
            gradU = _rankr_gradU_from_terms(U, λ, cache.contracts, cache.grams)
            if model.scale_by_lambda
                for k = 1:model.r
                    λ_abs = max(abs(λ[k]), model.lambda_eps)
                    for m = 1:length(U)
                        gradU[m][:, k] ./= λ_abs
                    end
                end
            end
            grad_parts = Vector{Vector{Vector{T}}}(undef, model.r)
            @inbounds for k = 1:model.r
                grad_Uk = [Vector(@view gradU[m][:, k]) for m = 1:length(U)]
                grad_parts[k] = pack_tangent_rank1_segre(grad_λ[k], grad_Uk)
            end
            return hasproperty(p, :x) ? ArrayPartition(grad_parts...) : (grad_parts...,)
        end
        return cost_fn, egrad_fn
    else
        cache = _CPRankrEvalCache(T, model.dims, model.r)
        normA2 = sum(abs2, model.A)
        Nmodes = length(model.dims)
        Ubuf = Vector{Matrix{T}}(undef, Nmodes)
        cost_fn = function (M, p)
            parts = normalize_rankr_canonical_point(p, model.dims, model.r)
            λ = _canonical_rankr_fill_factors!(Ubuf, parts, model.dims, model.r)
            _cp_rankr_refresh_cache!(cache, model.A, Ubuf, p)
            return cp_rankr_cost_value(normA2, λ, cache.inner, cache.cross_mat)
        end
        egrad_fn = function (M, p)
            parts = normalize_rankr_canonical_point(p, model.dims, model.r)
            λ = _canonical_rankr_fill_factors!(Ubuf, parts, model.dims, model.r)
            _cp_rankr_refresh_cache!(cache, model.A, Ubuf, p)
            grad_λ = grad_lambda_cp(λ, cache.inner, cache.cross_mat)
            gradU = _rankr_gradU_from_terms(Ubuf, λ, cache.contracts, cache.grams)
            if model.scale_by_lambda
                for k = 1:model.r
                    λ_abs = max(abs(λ[k]), model.lambda_eps)
                    for m = 1:Nmodes
                        gradU[m][:, k] ./= λ_abs
                    end
                end
            end
            return wrap_rankr_canonical_tangent_like(p, grad_λ, gradU, model.r)
        end
        return cost_fn, egrad_fn
    end
end

function model_egrad_function(model::RankRCPDModel{T,N}) where {T,N}
    return model.nonnegative ? egrad_secant_rankr_nn(model.A, model.dims, model.r) :
           (
        model.geometry == :native ?
        egrad_rankr_native(
            model.A,
            model.dims,
            model.r;
            scale_by_lambda = model.scale_by_lambda,
            lambda_eps = model.lambda_eps,
        ) :
        egrad_rankr_canonical(
            model.A,
            model.dims,
            model.r;
            scale_by_lambda = model.scale_by_lambda,
            lambda_eps = model.lambda_eps,
        )
    )
end

function model_rgrad_function(model::RankRCPDModel{T,N}; model_egrad = nothing) where {T,N} # Riemannian gradient function for canonical geometry
    if model.nonnegative # nonnegative case
        egrad_fn =
            isnothing(model_egrad) ? # get the Euclidean gradient if the model_egrad is nothing  
            egrad_secant_rankr_nn(model.A, model.dims, model.r) : # get the Euclidean gradient for the nonnegative case
            model_egrad # get the Euclidean gradient
        return (M, p) -> egrad_to_rgrad(M, p, egrad_fn(M, p)) # convert Euclidean to Riemannian gradient on (product) (Sq)Euclidean
    end
    if model.geometry == :native # native case (Segre geometry)
        egrad_fn =
            isnothing(model_egrad) ?
            egrad_rankr_native(
                model.A,
                model.dims,
                model.r;
                scale_by_lambda = model.scale_by_lambda,
                lambda_eps = model.lambda_eps,
            ) : model_egrad
        return (M, p) -> egrad_to_rgrad(M, p, egrad_fn(M, p))
    end
    return (M, p) -> rgrad(model, p) # Riemannian gradient for canonical geometry (canonical geometry)   
end

supports_rgrad(model::RankRCPDModel) = true # Riemannian gradient is supported for canonical geometry
supports_egrad_project(model::RankRCPDModel) = true
supports_exact_native(model::RankRCPDModel) =
    (model.geometry == :native) && !model.nonnegative
function model_exact_native_function(model::RankRCPDModel)
    egrad_fn = egrad_rankr_native(
        model.A,
        model.dims,
        model.r;
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
    )
    return (M, p) -> egrad_to_rgrad(M, p, egrad_fn(M, p))
end
cp_als_data(model::RankRCPDModel) = (model.A, model.r)

@inline function _segre_component_tensorvec(comp)
    parts = parts_tuple(comp)
    λ = parts[1][1]
    U = [parts[m+1] for m = 1:(length(parts)-1)]
    return vec(reconstruct_cp_rank1(λ, U))
end

function _segre_tangent_tensorvec(comp, Xcomp)
    pparts = parts_tuple(comp)
    xparts = parts_tuple(Xcomp)
    d = length(pparts) - 1
    length(xparts) == d + 1 ||
        throw(DimensionMismatch("Segre tangent part count mismatch."))

    λ = pparts[1][1]
    ν = xparts[1][1]
    U = [pparts[m+1] for m = 1:d]
    Udot = [xparts[m+1] for m = 1:d]

    v = vec(reconstruct_cp_rank1(ν, U))
    @inbounds for m = 1:d
        Um = [j == m ? Udot[j] : U[j] for j = 1:d]
        v .+= vec(reconstruct_cp_rank1(λ, Um))
    end
    return v
end

"""
    rgrad_exact(model, p)

Fast closed-form Riemannian gradient for CPD on
`ProductManifold(Manifolds.Segre(...), ...)`.
Uses the Euclidean gradient in the `Manifolds.Segre` representation plus the package-local
`egrad_to_rgrad(::Manifolds.Segre, ...)` / product-manifold projection path.
"""
function rgrad_exact(model::RankRCPDModel{T,N}, p) where {T,N} # exact gradient for geometry=:native
    model.geometry == :native ||
        throw(ArgumentError("rgrad_exact requires geometry=:native."))
    model.nonnegative && throw(ArgumentError("rgrad_exact requires nonnegative=false."))
    egrad_fn = egrad_rankr_native(
        model.A,
        model.dims,
        model.r;
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
    )
    return egrad_to_rgrad(model.M, p, egrad_fn(model.M, p))
end

# Reference basis-sweep implementation retained for comparison/history:
#
# function _rgrad_exact_basis_reference(model::RankRCPDModel{T, N}, p) where {T, N}
#     model.geometry == :native || throw(ArgumentError("rgrad_exact requires geometry=:native."))
#     model.nonnegative && throw(ArgumentError("rgrad_exact requires nonnegative=false."))
#     M = model.M
#     p_work = hasproperty(p, :x) ? p : ArrayPartition(p...)
#     comps = parts_tuple(p_work)
#     length(comps) == model.r || throw(DimensionMismatch("point must contain $(model.r) Manifolds.Segre components, got $(length(comps))"))
#     basis = ManifoldsBase.DefaultOrthonormalBasis()
#     ambient_dim = prod(model.dims)
#
#     residual = zeros(T, ambient_dim)
#     @inbounds for k in 1:model.r
#         residual .+= _segre_component_tensorvec(comps[k])
#     end
#     residual .-= vec(model.A)
#
#     dM = manifold_dimension(M)
#     coeff = zeros(T, dM)
#     e_j = zeros(T, dM)
#     Xj_buf = Vector{T}(undef, ambient_dim)
#     @inbounds for j in 1:dM
#         fill!(e_j, zero(T))
#         e_j[j] = one(T)
#         Xj = ManifoldsBase.get_vector(M, p_work, e_j, basis)
#         Xparts = parts_tuple(Xj)
#         fill!(Xj_buf, zero(T))
#         for k in 1:model.r
#             Xj_buf .+= _segre_tangent_tensorvec(comps[k], Xparts[k])
#         end
#         coeff[j] = dot(Xj_buf, residual)
#     end
#     g = ManifoldsBase.get_vector(M, p_work, coeff, basis)
#     return hasproperty(p, :x) ? g : Tuple(getproperty(g, :x))
# end

function rgrad(model::RankRCPDModel{T,N}, p) where {T,N} # Riemannian gradient for canonical geometry
    if model.nonnegative
        eg = egrad(model, p) # get the Euclidean gradient
        return egrad_to_rgrad(manifold(model), p, eg) # convert Euclidean to Riemannian gradient on (product) (Sq)Euclidean
    end
    if model.geometry == :native
        return rgrad_exact(model, p) # fast native closed-form via local Segre projection
    end

    λ, U = unpack_rankr_canonical(p, model.dims, model.r) 

    Nmodes = length(U) 
    r = model.r 

    contracts = Vector{Matrix{T}}(undef, Nmodes) 
    for m = 1:Nmodes
        contracts[m] = mttkrp(model.A, U, m; method = :auto) 
    end

    inner = _inner_from_mttkrp_first_mode(U, contracts[1]) 
    grams = _gram_matrices(U)
    cross_mat = _cross_unit_from_grams(grams) 
    grad_λ = grad_lambda_cp(λ, inner, cross_mat) 

    gradU = _rankr_gradU_from_terms(U, λ, contracts, grams)
    for m = 1:Nmodes
        Gm = gradU[m]
        if model.scale_by_lambda
            for k = 1:r
                Gm[:, k] ./= max(abs(λ[k]), model.lambda_eps) 
            end
        end

        # Tangent projection on each sphere factor (u_{m,k}^T g_{m,k} = 0).
        for k = 1:r
            uk = @view U[m][:, k] # view the k-th column of the m-th mode
            gk = @view Gm[:, k] # view the k-th column of the gradient for the m-th mode
            Gm[:, k] .-= dot(uk, gk) .* uk
        end
        gradU[m] = Gm # assign the gradient for the m-th mode
    end

    return wrap_rankr_canonical_tangent_like(p, grad_λ, gradU, r)
end

function initial_point(
    model::RankRCPDModel{T,N},
    init::Union{Symbol,BuiltinInitializer};
    verbose::Bool = false,
) where {T,N}
    init == :alswarm && return initial_point(model, ALSWarmStartInit(); verbose)
    init_sym = _builtin_initializer_symbol(init)
    λ0, U0 = init_cpd_factors(model.A, model.r; init = init_sym) # initialize the factors
    if model.nonnegative
        if model.geometry == :softplus_metric
            λ̃0 = _invsoftplus.(max.(abs.(λ0), eps(T)))
            Ũ0 = [_invsoftplus.(max.(abs.(U0[m]), eps(T))) for m = 1:length(U0)]
        else
            λ̃0 = sqrt.(max.(abs.(λ0), eps(T)))
            Ũ0 = [sqrt.(max.(abs.(U0[m]), eps(T))) for m = 1:length(U0)]
        end
        return pack_point_rankr(λ̃0, Ũ0, model.r) # structured join layout
    end
    return model.geometry == :native ? pack_rankr_native(λ0, U0, model.r) :
           pack_rankr_canonical(λ0, U0, model.r) # native: ArrayPartition, canonical: tuple            
end

initial_point(model::RankRCPDModel, init::PointInit; kwargs...) = init.point
initial_point(model::RankRCPDModel, init::FunctionInit; kwargs...) = init.f(model)

supports_normalization_policy(model::RankRCPDModel, policy::AbstractNormalizationPolicy) =
    model.nonnegative || policy isa Union{NoNormalization,SeparateLambdaNormalization}

"""
    cpd_point(model::RankRCPDModel, p)

Interpret a rank-`r` solver point `p` as a canonical [`CPDPoint`](@ref).

This erases the distinction between canonical, `ProductManifold(Manifolds.Segre(...), ...)`, and
nonnegative internal layouts so that backend postprocessing can operate on a
single CP representation.
"""
function cpd_point(model::RankRCPDModel{T,N}, p) where {T<:AbstractFloat,N}
    if model.nonnegative
        λ̃, Ũ = unpack_point_rankr(p, model.dims, model.r)
        if model.geometry == :softplus_metric
            return CPDPoint(
                _softplus_value.(λ̃),
                [_softplus_value.(Ũ[m]) for m = 1:length(Ũ)],
            )
        end
        return CPDPoint(λ̃ .^ 2, [Ũ[m] .^ 2 for m = 1:length(Ũ)])
    end
    λ, U =
        model.geometry == :native ? unpack_rankr_native(p, model.dims, model.r) :
        unpack_rankr_canonical(p, model.dims, model.r)
    return CPDPoint(λ, U)
end

"""
    pack_cpd_point(model::RankRCPDModel, point)

Pack a canonical [`CPDPoint`](@ref) back into the layout required by `model`.

Depending on the CPD geometry this rebuilds either a
`ProductManifold(Manifolds.Segre(...), ...)` point, a canonical CP tuple, or
the internal nonnegative parameterization.
"""
function pack_cpd_point(
    model::RankRCPDModel{T,N},
    point::CPDPoint{T},
) where {T<:AbstractFloat,N}
    length(lambda(point)) == model.r || throw(
        DimensionMismatch(
            "CPDPoint has $(length(lambda(point))) weights, expected rank $(model.r)",
        ),
    )
    if model.nonnegative
        if model.geometry == :softplus_metric
            λ̃ = _invsoftplus.(max.(lambda(point), zero(T)))
            Ũ = [_invsoftplus.(max.(F, zero(T))) for F in factors(point)]
        else
            λ̃ = sqrt.(max.(lambda(point), zero(T)))
            Ũ = [sqrt.(max.(F, zero(T))) for F in factors(point)]
        end
        return pack_point_rankr(λ̃, Ũ, model.r)
    end
    return model.geometry == :native ?
           pack_rankr_native(lambda(point), factors(point), model.r) :
           pack_rankr_canonical(lambda(point), factors(point), model.r)
end

"""
    post_step!(model::RankRCPDModel, p; normalization=...)

Rank-`r` CPD post-step hook.

This is the backend normalization adapter used by manifold solvers: convert the
current iterate to [`CPDPoint`](@ref), apply the requested normalization in CP
coordinates, then pack back into the model-specific layout.
"""
function post_step!(
    model::RankRCPDModel,
    p;
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = nothing,
    kwargs...,
)
    policy = _normalization_policy(normalization)
    supports_normalization_policy(model, policy) || throw(
        ArgumentError(
            "Normalization policy $(typeof(policy)) is incompatible with RankRCPDModel geometry=$(model.geometry), nonnegative=$(model.nonnegative).",
        ),
    )
    if policy isa NoNormalization ||
       (!model.nonnegative && policy isa SeparateLambdaNormalization)
        return p
    end
    q = cpd_point(model, p)
    normalize_components!(q, policy)
    return pack_cpd_point(model, q)
end
