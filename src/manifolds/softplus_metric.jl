# manifolds/softplus_metric.jl — Regularized pullback-inspired geometry for
# nonnegative CP (softplus parameterization).
#
# With x = softplus(p), the Euclidean pullback gives
# G_ii = σ(p_i)^2. We use G_ii = σ(p_i)^2 + ε for the same regularized
# positive-definite metric interface as SqEuclidean.

export SoftplusEuclidean, softplus_metric_inverse

@inline _sp_sigmoid(x::Real) = x >= 0 ? one(x) / (one(x) + exp(-x)) : begin
    ex = exp(x)
    ex / (one(x) + ex)
end

@inline _sp_softplus(x::Real) = x >= 0 ? x + log1p(exp(-x)) : log1p(exp(x))

struct SoftplusEuclidean{T<:Real} <: AbstractManifold{ManifoldsBase.ℝ}
    n::Int
    ε::T
end

SoftplusEuclidean(n::Int; ε::Real = 1e-8) = SoftplusEuclidean{typeof(ε)}(n, ε)

ManifoldsBase.manifold_dimension(M::SoftplusEuclidean) = M.n

const _euclidean_sp(M::SoftplusEuclidean) = Manifolds.Euclidean(M.n)

ManifoldsBase.allocate_result(M::SoftplusEuclidean, ::typeof(rand)) =
    ManifoldsBase.allocate_result(_euclidean_sp(M), rand)
ManifoldsBase.allocate_result(M::SoftplusEuclidean, ::typeof(rand), p) =
    ManifoldsBase.allocate_result(_euclidean_sp(M), rand, p)
Random.rand!(rng::AbstractRNG, M::SoftplusEuclidean, pX; kwargs...) =
    Random.rand!(rng, _euclidean_sp(M), pX; kwargs...)
Random.rand!(M::SoftplusEuclidean, pX; kwargs...) =
    Random.rand!(Random.default_rng(), _euclidean_sp(M), pX; kwargs...)

ManifoldsBase.retract!(
    M::SoftplusEuclidean,
    q,
    p,
    X,
    ::ManifoldsBase.ExponentialRetraction,
) = ManifoldsBase.retract!(_euclidean_sp(M), q, p, X, ManifoldsBase.ExponentialRetraction())

ManifoldsBase.project!(M::SoftplusEuclidean, Y, p, X) =
    ManifoldsBase.project!(_euclidean_sp(M), Y, p, X)

ManifoldsBase.exp!(M::SoftplusEuclidean, q, p, X) =
    ManifoldsBase.exp!(_euclidean_sp(M), q, p, X)

ManifoldsBase.log!(M::SoftplusEuclidean, X, p, q) =
    ManifoldsBase.log!(_euclidean_sp(M), X, p, q)

function softplus_metric_diag(M::SoftplusEuclidean, p::AbstractVector)
    σ = _sp_sigmoid.(p)
    return σ .* σ .+ M.ε
end

function softplus_metric_inverse(M::SoftplusEuclidean, p::AbstractVector, X::AbstractVector)
    return X ./ softplus_metric_diag(M, p)
end

pullback_metric_inverse(M::SoftplusEuclidean, p::AbstractVector, X::AbstractVector) =
    softplus_metric_inverse(M, p, X)

function ManifoldsBase.inner(M::SoftplusEuclidean, p, X::AbstractVector, Y::AbstractVector)
    g = softplus_metric_diag(M, p)
    return dot(X, g .* Y)
end

function ManifoldsBase.get_coordinates_orthonormal(
    M::SoftplusEuclidean,
    p::AbstractVector,
    X::AbstractVector,
    ::ManifoldsBase.RealNumbers,
)
    return sqrt.(softplus_metric_diag(M, p)) .* X
end

function ManifoldsBase.get_coordinates_orthonormal!(
    M::SoftplusEuclidean,
    c,
    p::AbstractVector,
    X::AbstractVector,
    ::ManifoldsBase.RealNumbers,
)
    c .= sqrt.(softplus_metric_diag(M, p)) .* X
    return c
end

function ManifoldsBase.get_vector_orthonormal(
    M::SoftplusEuclidean,
    p::AbstractVector,
    c::AbstractVector,
    ::ManifoldsBase.RealNumbers,
)
    return c ./ sqrt.(softplus_metric_diag(M, p))
end

function ManifoldsBase.get_vector_orthonormal!(
    M::SoftplusEuclidean,
    X,
    p::AbstractVector,
    c::AbstractVector,
    ::ManifoldsBase.RealNumbers,
)
    X .= c ./ sqrt.(softplus_metric_diag(M, p))
    return X
end
