# manifolds/squaring_metric.jl — Regularized pullback-inspired geometry for
# nonnegative CP (squaring parameterization).
#
# With x = x̃², the exact Euclidean pullback gives G_ii = 4*p_i², which is
# degenerate at p_i = 0. We therefore use the regularized form
# G_ii = 4*p_i² + ε to keep G positive definite everywhere.
# Riemannian gradient: grad = G^{-1} * egrad.

export SqEuclidean,
    pullback_metric_inverse,
    sm_cost_nn_quadratic,
    sm_egrad_nn_quadratic,
    sm_cost_nn_rank1,
    sm_egrad_nn_rank1,
    sm_cost_repelling_test,
    sm_egrad_repelling_test,
    sm_cost_double_circle,
    sm_egrad_double_circle,
    sm_double_circle_at_zero

"""
    SqEuclidean(n; ε=1e-8)

Euclidean space `ℝⁿ` induced by the squared map `p ↦ p.^2`.
- `inner(M, p, X, Y)` = X' G(p) Y with G(p) = diag(4*p.² + ε)
- Riemannian gradient: grad = G(p)^{-1} * egrad

This is not the exact pullback metric globally: the exact pullback
`diag(4*p.^2)` is singular at `p = 0`, so `ε > 0` regularizes it into a true
Riemannian metric.

Compatible with Manopt: uses `inner` for norm and Euclidean retractions in the
latent coordinates.
"""
struct SqEuclidean{T<:Real} <: AbstractManifold{ManifoldsBase.ℝ}
    n::Int
    ε::T
end

SqEuclidean(n::Int; ε::Real = 1e-8) = SqEuclidean{typeof(ε)}(n, ε)

ManifoldsBase.manifold_dimension(M::SqEuclidean) = M.n

# Tangent spaces are Euclidean vector spaces. We use Euclidean projection
# and addition retraction in latent coordinates. This is a retraction-based
# optimization geometry, not necessarily the exact geodesic geometry of G.
const _euclidean(M::SqEuclidean) = Manifolds.Euclidean(M.n)

# allocate_result/rand! required by Manopt (e.g. ArmijoLinesearchStepsize)
ManifoldsBase.allocate_result(M::SqEuclidean, ::typeof(rand)) =
    ManifoldsBase.allocate_result(_euclidean(M), rand)
ManifoldsBase.allocate_result(M::SqEuclidean, ::typeof(rand), p) =
    ManifoldsBase.allocate_result(_euclidean(M), rand, p)
Random.rand!(rng::AbstractRNG, M::SqEuclidean, pX; kwargs...) =
    Random.rand!(rng, _euclidean(M), pX; kwargs...)
Random.rand!(M::SqEuclidean, pX; kwargs...) =
    Random.rand!(Random.default_rng(), _euclidean(M), pX; kwargs...)

ManifoldsBase.retract!(M::SqEuclidean, q, p, X, ::ManifoldsBase.ExponentialRetraction) =
    ManifoldsBase.retract!(_euclidean(M), q, p, X, ManifoldsBase.ExponentialRetraction())

ManifoldsBase.project!(M::SqEuclidean, Y, p, X) =
    ManifoldsBase.project!(_euclidean(M), Y, p, X)

ManifoldsBase.exp!(M::SqEuclidean, q, p, X) = ManifoldsBase.exp!(_euclidean(M), q, p, X)

ManifoldsBase.log!(M::SqEuclidean, X, p, q) = ManifoldsBase.log!(_euclidean(M), X, p, q)


function pullback_metric_diag(M::SqEuclidean, p::AbstractVector)
    return 4 .* (p .^ 2) .+ M.ε
end

function pullback_metric_inverse(M::SqEuclidean, p::AbstractVector, X::AbstractVector)
    return X ./ pullback_metric_diag(M, p)
end

function ManifoldsBase.inner(M::SqEuclidean, p, X::AbstractVector, Y::AbstractVector)
    g = pullback_metric_diag(M, p)
    return dot(X, g .* Y)
end

# -----------------------------
# 2D benchmark objectives for squaring/nonnegative experiments
# p = [x, y]
# -----------------------------

@inline function _sm_xy(p::AbstractVector)
    length(p) == 2 ||
        throw(DimensionMismatch("expected p of length 2, got length=$(length(p))"))
    return p[1], p[2]
end

"""
    sm_cost_nn_quadratic(p; a=1, b=1)

f(x,y) = 0.5*((x^2-a)^2 + (y^2-b)^2)
"""
function sm_cost_nn_quadratic(p::AbstractVector; a::Real = 1.0, b::Real = 1.0)
    x, y = _sm_xy(p)
    return 0.5 * ((x^2 - a)^2 + (y^2 - b)^2)
end

function sm_egrad_nn_quadratic(p::AbstractVector; a::Real = 1.0, b::Real = 1.0)
    x, y = _sm_xy(p)
    return [2 * x * (x^2 - a), 2 * y * (y^2 - b)]
end

function sm_cost_nn_rank1(p::AbstractVector; A::Real = 1.0)
    x, y = _sm_xy(p)
    r = A - x^2 * y^2
    return 0.5 * r^2
end

function sm_egrad_nn_rank1(p::AbstractVector; A::Real = 1.0)
    x, y = _sm_xy(p)
    r = A - x^2 * y^2
    return [-2 * x * y^2 * r, -2 * y * x^2 * r]
end

sm_cost_repelling_test(p::AbstractVector) = sm_cost_nn_quadratic(p; a = 1.0, b = 1.0)

sm_egrad_repelling_test(p::AbstractVector) = sm_egrad_nn_quadratic(p; a = 1.0, b = 1.0)

@inline function _sm_circle_terms(
    x::Real,
    y::Real;
    c1::NTuple{2,<:Real} = (5.0, 5.0),
    c2::NTuple{2,<:Real} = (-1.5, 5.0),
    r1::Real = 1.0,
    r2::Real = 1.0,
)
    C1 = (x - c1[1])^2 + (y - c1[2])^2 - r1^2
    C2 = (x - c2[1])^2 + (y - c2[2])^2 - r2^2
    return C1, C2
end

function sm_cost_double_circle(
    p::AbstractVector;
    c1::NTuple{2,<:Real} = (5.0, 5.0),
    c2::NTuple{2,<:Real} = (-1.5, 5.0),
    r1::Real = 1.0,
    r2::Real = 1.0,
)
    x, y = _sm_xy(p)
    C1, C2 = _sm_circle_terms(x, y; c1, c2, r1, r2)
    return C1 * C2
end

function sm_egrad_double_circle(
    p::AbstractVector;
    c1::NTuple{2,<:Real} = (5.0, 5.0),
    c2::NTuple{2,<:Real} = (-1.5, 5.0),
    r1::Real = 1.0,
    r2::Real = 1.0,
)
    x, y = _sm_xy(p)
    C1, C2 = _sm_circle_terms(x, y; c1, c2, r1, r2)
    dfdx = 2 * (x - c1[1]) * C2 + 2 * (x - c2[1]) * C1
    dfdy = 2 * (y - c1[2]) * C2 + 2 * (y - c2[2]) * C1
    return [dfdx, dfdy]
end

function sm_double_circle_at_zero(;
    c1::NTuple{2,<:Real} = (5.0, 5.0),
    c2::NTuple{2,<:Real} = (-1.5, 5.0),
    r1::Real = 1.0,
    r2::Real = 1.0,
)
    p0 = [0.0, 0.0]
    C1, C2 = _sm_circle_terms(0.0, 0.0; c1, c2, r1, r2)
    f0 = sm_cost_double_circle(p0; c1, c2, r1, r2)
    g0 = sm_egrad_double_circle(p0; c1, c2, r1, r2)
    return (point = p0, C1 = C1, C2 = C2, value = f0, gradient = g0)
end
