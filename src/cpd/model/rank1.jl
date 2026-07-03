# cpd/model/rank1.jl — Rank-1 CPD model
export embed_point

"""
    Rank1CPDModel{T, N}

Rank-1 CPD: A ≈ λ · u₁ ⊗ u₂ ⊗ ... ⊗ u_d. Fields: A, dims, M, scale_by_lambda, lambda_eps, nonnegative.
"""
struct Rank1CPDModel{T<:AbstractFloat,N,A<:AbstractArray{T,N},M<:AbstractManifold} <:
       AbstractDecompositionModel{T}
    # Keep the target/manifold concrete so model fields stay fully inferable.
    A::A
    dims::NTuple{N,Int}
    M::M
    scale_by_lambda::Bool
    lambda_eps::T
    nonnegative::Bool
end

@inline _rank1_uses_softplus_metric(M) = false
@inline _rank1_uses_softplus_metric(M::SoftplusEuclidean) = true
@inline _rank1_uses_softplus_metric(M::ProductManifold) =
    any(mi -> mi isa SoftplusEuclidean, M.manifolds)

function Rank1CPDModel(
    A::AbstractArray{T,N};
    scale_by_lambda::Bool = true,
    lambda_eps::Real = 1e-10,
    nonnegative::Bool = false,
    use_pullback_metric::Bool = false,
    use_softplus_metric::Bool = false,
    pullback_eps::Real = 1e-8,
) where {T<:AbstractFloat,N}
    dims = size(A)
    M = if nonnegative
        # Structured nonnegative layout: [[λ̃], ũ₁, …, ũ_d] on a product manifold
        factors = Vector{AbstractManifold}(undef, N + 1)
        factors[1] =
            use_softplus_metric ? SoftplusEuclidean(1; ε = pullback_eps) :
            (use_pullback_metric ? SqEuclidean(1; ε = pullback_eps) : Euclidean(1))
        @inbounds for m = 1:N
            factors[m+1] =
                use_softplus_metric ? SoftplusEuclidean(dims[m]; ε = pullback_eps) :
                (
                    use_pullback_metric ? SqEuclidean(dims[m]; ε = pullback_eps) :
                    Euclidean(dims[m])
                )
        end
        ProductManifold(factors...)
    else
        Manifolds.Segre(dims...)
    end
    scale_by_lambda_eff = nonnegative ? false : scale_by_lambda
    lambda_eps_T = T(lambda_eps)
    return Rank1CPDModel(A, dims, M, scale_by_lambda_eff, lambda_eps_T, nonnegative)
end

tensor(model::Rank1CPDModel) = model.A
manifold(model::Rank1CPDModel) = model.M

_cp_parameterization(model::Rank1CPDModel) =
    model.nonnegative ?
    (_rank1_uses_softplus_metric(model.M) ? SoftplusNNCPParam() : SquaredNNCPParam()) :
    NativeCPParam()

function embed_point(model::Rank1CPDModel{T,N}, p) where {T,N}
    return _cp_rank1_embed_tensor(_cp_parameterization(model), model.dims, p)
end

function residual(model::Rank1CPDModel{T,N}, p) where {T<:AbstractFloat,N}
    return vec(embed_point(model, p)) .- vec(model.A)
end

function differential_action!(
    out::AbstractVector{T},
    model::Rank1CPDModel{T,N},
    p,
    X,
) where {T<:AbstractFloat,N}
    length(out) == length(model.A) || throw(
        DimensionMismatch(
            "differential_action! output length $(length(out)) != ambient length $(length(model.A)).",
        ),
    )
    λ, U, λ̇, U̇ = _cp_rank1_decode_tangent_factors(_cp_parameterization(model), model.dims, p, X)
    return _cp_rank1_tangent_tensorvec!(out, λ, U, λ̇, U̇)
end

function adjoint_action(model::Rank1CPDModel{T,N}, p, a::AbstractVector; kwargs...) where {T<:AbstractFloat,N}
    length(a) == length(model.A) || throw(
        DimensionMismatch(
            "adjoint_action expected ambient vector of length $(length(model.A)) for $(typeof(model)), got $(length(a)).",
        ),
    )
    Aadj = reshape(a, model.dims)
    eg = _cp_rank1_linear_egrad(_cp_parameterization(model), model.dims, p, Aadj)
    return egrad_to_rgrad(model.M, p, eg)
end

function cost(model::Rank1CPDModel{T,N}, p) where {T,N}
    return model_cost_function(model)(model.M, p)
end

function egrad(model::Rank1CPDModel{T,N}, p) where {T,N}
    return model_egrad_function(model)(model.M, p)
end

function model_cost_function(model::Rank1CPDModel{T,N}) where {T,N}
    return model.nonnegative ? cost_segre_nn(model.A, model.dims) :
           cost_segre(model.A, model.dims)
end

function model_egrad_function(model::Rank1CPDModel{T,N}) where {T,N}
    base =
        model.nonnegative ? egrad_segre_nn(model.A, model.dims) :
        egrad_segre(model.A, model.dims)
    model.nonnegative && return base

    if model.scale_by_lambda
        return function (M, p)
            grad = base(M, p)
            λ_abs = max(abs(unpack_point_rank1(p, model.dims)[1]), model.lambda_eps)
            grad_λ, grad_U = unpack_point_rank1(grad, model.dims)
            return pack_tangent_rank1_segre(grad_λ, [g / λ_abs for g in grad_U])
        end
    end

    return function (M, p)
        grad = base(M, p)
        return M isa Manifolds.Segre ?
               pack_tangent_rank1_segre(unpack_point_rank1(grad, model.dims)...) : grad
    end
end

function model_rgrad_function(model::Rank1CPDModel{T,N}; model_egrad = nothing) where {T,N}
    if model.nonnegative
        egrad_fn =
            isnothing(model_egrad) ? egrad_segre_nn(model.A, model.dims) : model_egrad
        return (M, p) -> egrad_to_rgrad(M, p, egrad_fn(M, p))
    end
    return (M, p) -> rgrad(model, p)
end

supports_rgrad(model::Rank1CPDModel) = true
cp_als_data(model::Rank1CPDModel) = throw(
    ArgumentError("ALSSolver/RALSSolver require RankRCPDModel (r>=2), got Rank1CPDModel."),
)

function rgrad(model::Rank1CPDModel{T,N}, p) where {T,N}
    if model.nonnegative
        eg = egrad_segre_nn(model.A, model.dims)(model.M, p)
        return egrad_to_rgrad(manifold(model), p, eg)
    end

    parts = point_parts(p)
    λ = parts[1][1]
    inner = rank1_inner_parts(model.A, parts)
    grad_λ = λ - inner
    grad_U = Vector{Vector{T}}(undef, length(model.dims))
    for m in eachindex(model.dims)
        # Reuse the in-place contraction helper so only the final tangent vectors are allocated.
        g = Vector{T}(undef, model.dims[m])
        rank1_mode_contract_parts!(g, model.A, parts, m)
        rmul!(g, -λ)
        grad_U[m] = g
    end

    if model.scale_by_lambda
        λ_abs = max(abs(λ), model.lambda_eps)
        for m in eachindex(grad_U)
            grad_U[m] ./= λ_abs
        end
    end
    for m in eachindex(grad_U)
        um = parts[m+1]
        grad_U[m] .-= dot(um, grad_U[m]) .* um
    end

    return pack_tangent_rank1_segre(grad_λ, grad_U)
end

function initial_point(
    model::Rank1CPDModel{T,N},
    init::Union{Symbol,BuiltinInitializer};
    verbose::Bool = false,
) where {T,N}
    init == :alswarm && return initial_point(model, ALSWarmStartInit(); verbose)
    init_sym = _builtin_initializer_symbol(init)
    U0 = init_cp_rank1(model.A; init = init_sym)
    return _cp_rank1_seed_point(_cp_parameterization(model), one(T), U0)
end

initial_point(model::Rank1CPDModel, init::PointInit; kwargs...) = init.point
initial_point(model::Rank1CPDModel, init::FunctionInit; kwargs...) = init.f(model)

supports_normalization_policy(model::Rank1CPDModel, policy::AbstractNormalizationPolicy) =
    model.nonnegative ?
    policy isa Union{NoNormalization,NonnegativeSeparateLambdaNormalization} :
    policy isa Union{NoNormalization,SeparateLambdaNormalization}

"""
    cpd_point(model::Rank1CPDModel, p)

Interpret a rank-1 solver point `p` as a canonical [`CPDPoint`](@ref).

This removes layout-specific details such as `Manifolds.Segre` packing or
nonnegative squared parameterization so backend postprocessing can work with a
uniform `lambda + factors` representation.
"""
function cpd_point(model::Rank1CPDModel{T,N}, p) where {T<:AbstractFloat,N}
    λ, U = _cp_rank1_decode_factors(_cp_parameterization(model), model.dims, p)
    return CPDPoint(T[λ], [reshape(U[m], :, 1) for m in eachindex(U)])
end

function pack_cpd_point(
    model::Rank1CPDModel{T,N},
    point::CPDPoint{T},
) where {T<:AbstractFloat,N}
    length(lambda(point)) == 1 || throw(
        DimensionMismatch(
            "Rank-1 CPDPoint must have exactly one weight, got $(length(lambda(point)))",
        ),
    )
    λ = lambda(point)[1]
    U = [Vector(@view F[:, 1]) for F in factors(point)]
    return _cp_rank1_encode_point(_cp_parameterization(model), λ, U)
end

function post_step!(
    model::Rank1CPDModel,
    p;
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = nothing,
    kwargs...,
)
    policy = _normalization_policy(normalization)
    supports_normalization_policy(model, policy) || throw(
        ArgumentError(
            "Normalization policy $(typeof(policy)) is incompatible with Rank1CPDModel nonnegative=$(model.nonnegative).",
        ),
    )
    policy isa NoNormalization && return p
    if model.nonnegative
        policy isa NonnegativeSeparateLambdaNormalization || throw(
            ArgumentError(
                "Nonnegative Rank1CPDModel supports NoNormalization() and " *
                "NonnegativeSeparateLambdaNormalization() only.",
            ),
        )
    elseif !(policy isa SeparateLambdaNormalization)
        throw(
            ArgumentError(
                "Signed Rank1CPDModel supports NoNormalization() and " *
                "SeparateLambdaNormalization() only.",
            ),
        )
    end
    q = cpd_point(model, p)
    normalize_components!(q, policy)
    return pack_cpd_point(model, q)
end
