# Euclidean → Riemannian gradient conversion (egrad_to_rgrad, grad)
export egrad_to_rgrad, egrad_to_rgrad!, grad, norm

"""
    egrad_to_rgrad(M, p, egrad)

Convert a Euclidean coordinate gradient `egrad` at `p` to a Riemannian tangent
vector on `M`.

For embedded manifolds, this often coincides with converting an ambient
gradient by projection. For `SqEuclidean` and `SoftplusEuclidean`, however,
`egrad` must already be the Euclidean gradient with respect to the latent
coordinates `p`, not the ambient gradient with respect to the transformed
nonnegative variable.
- Embedded manifolds: `project(M, p, egrad)`.
- SqEuclidean / SoftplusEuclidean: inverse of the regularized diagonal metric
  applied to the latent-coordinate Euclidean gradient.
- Generic manifolds without a specialized projection path are expected to
  implement `project(M, p, egrad)` explicitly. An internal embedded-Euclidean
  basis fallback exists for debugging, but production solver paths should not
  rely on it.
"""
egrad_to_rgrad(M::SqEuclidean, p::AbstractVector, egrad::AbstractVector) =
    pullback_metric_inverse(M, p, egrad)

egrad_to_rgrad(M::SoftplusEuclidean, p::AbstractVector, egrad::AbstractVector) =
    pullback_metric_inverse(M, p, egrad)

function egrad_to_rgrad(M::SqEuclidean, p, egrad)
    throw(
        ArgumentError(
            "SqEuclidean expects vector point/gradient. Got p::$(typeof(p)), egrad::$(typeof(egrad)). " *
            "Use flat layout (pack_point_rankr_to_vector / pack_point_rank1_to_vector). " *
            "The gradient must be with respect to the latent coordinates p, not the transformed nonnegative variable.",
        ),
    )
end

function egrad_to_rgrad(M::SoftplusEuclidean, p, egrad)
    throw(
        ArgumentError(
            "SoftplusEuclidean expects vector point/gradient. Got p::$(typeof(p)), egrad::$(typeof(egrad)). " *
            "Use flat layout (pack_point_rankr_to_vector / pack_point_rank1_to_vector). " *
            "The gradient must be with respect to the latent coordinates p, not the transformed nonnegative variable.",
        ),
    )
end

function egrad_to_rgrad(M::Manifolds.Segre, p, egrad)
    parts = _grad_parts(p)
    gparts = _grad_parts(egrad)
    if !_is_structured(gparts)
        throw(
            ArgumentError(
                "Manifolds.Segre expects a structured Euclidean gradient in the native (λ, u₁, …, u_d) representation. " *
                "Ambient tensor gradients must be converted by a model-specific path before calling egrad_to_rgrad.",
            ),
        )
    end

    length(parts) == length(gparts) || throw(
        DimensionMismatch(
            "Segre egrad parts mismatch: point $(length(parts)), egrad $(length(gparts)).",
        ),
    )
    d = length(parts) - 1
    d >= 1 || throw(DimensionMismatch("Segre point must have [λ] and ≥1 mode factor."))

    T = promote_type(eltype(_grad_vec(parts[1])), eltype(_grad_vec(gparts[1])))
    out = Vector{Vector{T}}(undef, d + 1)
    out[1] = _grad_vec(gparts[1], T)
    length(out[1]) == 1 || throw(DimensionMismatch("Segre λ-gradient must have length 1."))
    @inbounds for m in eachindex(Base.OneTo(d))
        u = _grad_vec(parts[m+1], T)
        gm = _grad_vec(gparts[m+1], T)
        length(u) == length(gm) ||
            throw(DimensionMismatch("Mode $m gradient length mismatch."))
        gm .-= dot(u, gm) .* u
        out[m+1] = gm
    end
    return out
end

egrad_to_rgrad(
    M::Manifolds.Tucker,
    p::Manifolds.TuckerPoint,
    egrad::Manifolds.TuckerTangentVector,
) = ManifoldsBase.project(M, p, egrad)

function _egrad_to_rgrad_product(M::ProductManifold, p, eparts_in)
    factors = M.manifolds
    n = length(factors)
    eparts = eparts_in isa Tuple ? eparts_in : Tuple(eparts_in)
    length(eparts) == n ||
        throw(DimensionMismatch("Product egrad must have $n parts, got $(length(eparts))"))
    pparts = _grad_parts(p)
    pparts = pparts isa Tuple ? pparts : (pparts...,)
    length(pparts) == n ||
        throw(DimensionMismatch("Product point must have $n parts, got $(length(pparts))"))
    result = ntuple(i -> egrad_to_rgrad(factors[i], pparts[i], eparts[i]), n)
    return hasproperty(p, :x) ? ArrayPartition(result...) : (result...,)
end

egrad_to_rgrad(M::ProductManifold, p, egrad::Tuple) = _egrad_to_rgrad_product(M, p, egrad)
egrad_to_rgrad(M::ProductManifold, p, egrad::ArrayPartition) =
    _egrad_to_rgrad_product(M, p, egrad.x)

egrad_to_rgrad(M, p, egrad) = _project_or_error(M, p, egrad)

function _project_or_error(M, p, egrad)
    if !applicable(ManifoldsBase.project, M, p, egrad)
        throw(
            ArgumentError(
                "No specialized egrad_to_rgrad/project(M,p,⋅) path for $(typeof(M)). " *
                "Production solver paths do not use the embedded Euclidean basis fallback.",
            ),
        )
    end
    try
        return ManifoldsBase.project(M, p, egrad)
    catch err
        if err isa MethodError && err.f === ManifoldsBase.project!
            throw(
                ArgumentError(
                    "project(M,p,⋅) is declared for $(typeof(M)) but project! is missing. " *
                    "Implement a specialized projection path instead of relying on the embedded Euclidean basis fallback.",
                ),
            )
        end
        rethrow(err)
    end
end

function _embedded_euclidean_basis_fallback(
    M,
    p,
    egrad::AbstractVector{T},
) where {T<:Number}
    d = manifold_dimension(M)
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    coeff = zeros(T, d)
    e_j = zeros(T, d)
    u = similar(egrad, T, length(egrad))
    @inbounds for j in eachindex(Base.OneTo(d))
        fill!(e_j, zero(T))
        e_j[j] = one(T)
        ξj = ManifoldsBase.get_vector(M, p, e_j, basis)
        u = ManifoldsBase.embed!(M, u, p, ξj)
        coeff[j] = dot(u, egrad)
    end
    return ManifoldsBase.get_vector(M, p, coeff, basis)
end

function _embedded_euclidean_basis_fallback(M, p, egrad)
    throw(
        ArgumentError(
            "No project(M,p,⋅) for $(typeof(M)); the embedded Euclidean basis fallback requires a numeric ambient vector and assumes an embedded Euclidean metric. Got egrad::$(typeof(egrad)).",
        ),
    )
end

function egrad_to_rgrad!(M, X, p, egrad)
    if applicable(ManifoldsBase.project!, M, X, p, egrad)
        return ManifoldsBase.project!(M, X, p, egrad)
    end
    copyto!(X, egrad_to_rgrad(M, p, egrad))
    return X
end

function egrad_to_rgrad!(
    M::SqEuclidean,
    X::AbstractVector,
    p::AbstractVector,
    egrad::AbstractVector,
)
    X .= pullback_metric_inverse(M, p, egrad)
    return X
end

function egrad_to_rgrad!(
    M::SoftplusEuclidean,
    X::AbstractVector,
    p::AbstractVector,
    egrad::AbstractVector,
)
    X .= pullback_metric_inverse(M, p, egrad)
    return X
end

function egrad_to_rgrad!(M::SqEuclidean, X, p, egrad)
    throw(
        ArgumentError(
            "SqEuclidean expects vector X/p/egrad. Got X::$(typeof(X)), p::$(typeof(p)), egrad::$(typeof(egrad)).",
        ),
    )
end

function egrad_to_rgrad!(M::SoftplusEuclidean, X, p, egrad)
    throw(
        ArgumentError(
            "SoftplusEuclidean expects vector X/p/egrad. Got X::$(typeof(X)), p::$(typeof(p)), egrad::$(typeof(egrad)).",
        ),
    )
end

grad(M, p, egrad) = egrad_to_rgrad(M, p, egrad)

grad(egrad_fn::Function) = (M, p) -> egrad_to_rgrad(M, p, egrad_fn(M, p))

# ---- Helpers ----
@inline _grad_parts(x) = hasproperty(x, :x) ? x.x : x
@inline _grad_vec(x, ::Type{T}) where {T} = Vector{T}(_grad_parts(x))
@inline _grad_vec(x) = _grad_vec(x, eltype(_grad_parts(x)))
@inline _is_structured(parts) =
    (parts isa Tuple || parts isa AbstractVector) &&
    !isempty(parts) &&
    _grad_parts(first(parts)) isa AbstractVector
