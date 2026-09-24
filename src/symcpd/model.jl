# Intrinsic Veronese join model. Target and model algebra are deliberately
# separate: target terms are contractions, while component interactions are
# evaluated through the polynomial kernel.

export SymCPDModel, component_inner, data_inner, compressed_coordinates

@doc raw"""
    SymCPDModel(target, rank)

Matrix-free model for the symmetric rank-`rank` approximation

```math
\widehat A = \sum_{r=1}^R \lambda_r x_r^{\otimes D},
\qquad \|x_r\|_2=1.
```

Each point component is stored intrinsically as `([lambda], x)` on
`Manifolds.Veronese(n, D)`. The objective is evaluated as

```math
f(p)=\tfrac12\|A\|_F^2
-\sum_r\lambda_r\langle A,x_r^{\otimes D}\rangle
+\tfrac12\sum_{r,s}\lambda_r\lambda_s(x_r^\top x_s)^D.
```

Consequently, neither the predicted tensor nor a compressed Veronese vector is
formed by `cost` or `rgrad`. `target` may be a [`DenseSymmetricTarget`](@ref)
or [`CompressedSymmetricTarget`](@ref).

The product-of-Veronese least-squares formulation and its Riemannian
Gauss--Newton structure are described by R. Khouja, H. Khalil, and B. Mourrain,
"Riemannian Newton optimization methods for the symmetric tensor approximation
problem," *Linear Algebra and its Applications* 637 (2022), 175--211,
doi:10.1016/j.laa.2021.12.008. This implementation differs computationally by
using the kernel as an operator and not assembling their dense normal matrix.
"""
struct SymCPDModel{
    T<:AbstractFloat,
    B<:AbstractSymmetricTarget{T},
    V<:Manifolds.Veronese,
    P<:ProductManifold,
} <: AbstractDecompositionModel{T}
    target::B
    rank::Int
    n::Int
    order::Int
    component_manifold::V
    product_manifold::P
end

function SymCPDModel(
    target::B,
    rank::Int,
) where {T<:AbstractFloat,B<:DenseSymmetricTarget{T}}
    rank >= 1 || throw(ArgumentError("rank must be positive, got $rank."))
    A = target.data
    d = ndims(A)
    n = size(A, 1)
    d >= 1 || throw(ArgumentError("The target order must be positive."))
    all(==(n), size(A)) || throw(
        DimensionMismatch("A symmetric target must have equal mode sizes, got $(size(A))."),
    )
    V = Manifolds.Veronese(n, d)
    P = ProductManifold(ntuple(_ -> V, rank)...)
    return SymCPDModel{T,B,typeof(V),typeof(P)}(target, rank, n, d, V, P)
end

function SymCPDModel(
    target::B,
    rank::Int,
) where {T<:AbstractFloat,B<:CompressedSymmetricTarget{T}}
    rank >= 1 || throw(ArgumentError("rank must be positive, got $rank."))
    V = Manifolds.Veronese(target.n, target.order)
    P = ProductManifold(ntuple(_ -> V, rank)...)
    return SymCPDModel{T,B,typeof(V),typeof(P)}(target, rank, target.n, target.order, V, P)
end

manifold(model::SymCPDModel) = model.product_manifold
tensor(model::SymCPDModel) = _symmetric_target_storage(model.target)
supports_rgrad(::SymCPDModel) = true
supports_egrad_project(::SymCPDModel) = false

# Preserve Veronese support in the generic materialized JoinModel as a
# reference path. SymCPDModel itself does not use this ambient length.
ambient_length(M::Manifolds.Veronese) = manifold_dimension(get_embedding(M))

function egrad(model::SymCPDModel, p)
    throw(
        ArgumentError(
            "SymCPDModel provides a direct intrinsic Riemannian gradient. " *
            "Use gradient_mode=:riemannian.",
        ),
    )
end

function initial_point(model::SymCPDModel{T}, init::Symbol; kwargs...) where {T}
    init == :random || throw(
        ArgumentError("SymCPDModel supports init=:random or an explicit p0, got $init."),
    )
    parts = ntuple(_ -> begin
        p = rand(model.component_manifold)
        ([T(p[1][1])], T.(p[2]))
    end, model.rank)
    return join_point(model.product_manifold, parts)
end

@doc raw"""
    component_inner(model, p, q)

Return the Frobenius inner product of two symmetric rank-one components,

```math
\langle \lambda x^{\otimes D},\mu y^{\otimes D}\rangle
=\lambda\mu(x^\top y)^D.
```

This is the homogeneous polynomial (Veronese) kernel. Differentiating this
identity gives the Gauss--Newton blocks in Proposition 4.9 of Khouja, Khalil,
and Mourrain (2022), doi:10.1016/j.laa.2021.12.008.
"""
function component_inner(model::SymCPDModel, p, q)
    pp = point_parts(p)
    qp = point_parts(q)
    return pp[1][1] * qp[1][1] * dot(pp[2], qp[2])^model.order
end

@doc raw"""
    data_inner(model, p)

Return `\langle A, lambda*x^(tensor D)\rangle` using the selected target
backend. This is the target-side polynomial evaluation; unlike model-model
interactions it cannot, for an arbitrary target, be reduced to pairwise
rank-one kernels. See the gradient/polynomial-evaluation construction in
Khouja, Khalil, and Mourrain (2022), Proposition 4.9,
doi:10.1016/j.laa.2021.12.008.
"""
function data_inner(model::SymCPDModel, p)
    pp = point_parts(p)
    return pp[1][1] * _symmetric_target_inner(model.target, pp[2])
end

function cost(model::SymCPDModel{T}, p) where {T}
    parts = join_parts(model.product_manifold, p)
    length(parts) == model.rank ||
        throw(DimensionMismatch("Expected $(model.rank) components."))
    value = T(0.5) * symmetric_target_norm2(model.target)
    @inbounds for r = 1:model.rank
        value -= data_inner(model, parts[r])
        value += T(0.5) * component_inner(model, parts[r], parts[r])
        for s = 1:(r-1)
            value += component_inner(model, parts[r], parts[s])
        end
    end
    return value
end

@doc raw"""
    rgrad(model::SymCPDModel, p)

Evaluate the intrinsic Riemannian gradient without constructing `Ahat`, a
residual tensor, or Veronese coordinate vectors. With
`a_r = <A,x_r^(tensor D)>`, `b_r = A(x_r,...,x_r,.)`, and
`c_rs = x_r' * x_s`, the coordinate derivatives are

```math
\partial_{\lambda_r}f=-a_r+\sum_s\lambda_s c_{rs}^D,
```

```math
\nabla_{x_r}f=-D\lambda_r b_r
+D\lambda_r\sum_s\lambda_s c_{rs}^{D-1}x_s.
```

The factor derivative is projected to `x_r^perp` and divided by
`D*lambda_r^2`, the spherical block of the induced Veronese metric
`d_lambda^2 + D*lambda^2*g_S`.

The target-evaluation and Veronese Gauss--Newton formulas follow Khouja,
Khalil, and Mourrain (2022), doi:10.1016/j.laa.2021.12.008. The induced metric
is the full-symmetry case of S. Jacobsson, L. Swijsen, J. Van der Veken, and
N. Vannieuwenhoven, "Warped Geometries of Segre--Veronese Manifolds,"
*SIAM Journal on Matrix Analysis and Applications* 47(3) (2026), 1551--1577,
doi:10.1137/25M1790099.
"""
function rgrad(model::SymCPDModel{T}, p) where {T}
    M = model.product_manifold
    parts = join_parts(M, p)
    length(parts) == model.rank ||
        throw(DimensionMismatch("Expected $(model.rank) components."))
    d = model.order
    values = ntuple(
        r -> begin
            pr = point_parts(parts[r])
            lambda_r = pr[1][1]
            x_r = pr[2]
            a_r, b_r = _symmetric_target_inner_and_contraction(model.target, x_r)
            radial = -a_r
            factor_covector = (-T(d) * lambda_r) .* b_r
            @inbounds for s = 1:model.rank
                ps = point_parts(parts[s])
                lambda_s = ps[1][1]
                x_s = ps[2]
                c = dot(x_r, x_s)
                radial += lambda_s * c^d
                factor_covector .+= (T(d) * lambda_r * lambda_s * c^(d - 1)) .* x_s
            end
            factor_covector .-= dot(x_r, factor_covector) .* x_r
            factor_gradient = factor_covector ./ (T(d) * lambda_r^2)
            ([T(radial)], factor_gradient)
        end,
        model.rank,
    )
    return join_tangent_like(M, p, values)
end

"""
    compressed_coordinates(result)

Return the Bombieri--Weyl orthonormal symmetric coordinates represented by a
`SymCPDResult`. This is an explicit output conversion; optimization does not
use the vector on the model side. The normalization follows the homogeneous
polynomial inner product used by Khouja, Khalil, and Mourrain (2022),
doi:10.1016/j.laa.2021.12.008.
"""
function compressed_coordinates(res::SymCPDResult)
    n = size(res.factors, 1)
    M = Manifolds.Veronese(n, res.order)
    T = eltype(res.weights)
    out = zeros(T, binomial(n + res.order - 1, res.order))
    work = similar(out)
    for c in res.components
        ManifoldsBase.embed!(M, work, c.point)
        out .+= work
    end
    return out
end

function _symcpd_result(model::SymCPDModel{T}, result, n::Int, d::Int) where {T}
    parts = join_parts(model.product_manifold, result.point)
    r = length(parts)
    weights_out = Vector{T}(undef, r)
    factors_out = Matrix{T}(undef, n, r)
    components_out = Vector{SymCPDComponent}(undef, r)
    for k = 1:r
        pk = parts[k]
        pk_parts = point_parts(pk)
        weight = T(pk_parts[1][1])
        factor = Vector{T}(pk_parts[2])
        weights_out[k] = weight
        factors_out[:, k] = factor
        components_out[k] = SymCPDComponent(pk, weight, factor, d)
    end
    return SymCPDResult(
        result.point,
        components_out,
        weights_out,
        factors_out,
        d,
        T(result.cost),
        T(result.rel_error),
        T(result.grad_norm),
        result.iterations,
        result.converged,
        _result_solver_symbol(result.solver),
        _result_solver_info(result),
    )
end
