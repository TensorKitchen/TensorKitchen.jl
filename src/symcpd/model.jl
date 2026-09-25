# Intrinsic symmetric rank-one component and its kernelized JoinModel backend.
# Target terms are contractions, while component interactions are evaluated
# through the homogeneous polynomial kernel.

export SymmetricRankOne,
    SymmetricCPDBackend, SymCPDModel, component_inner, data_inner, compressed_coordinates

@doc raw"""
    SymmetricRankOne(n, order)
    SymmetricRankOne(; dimension, order)

Describe one nonzero symmetric rank-one tensor

```math
\phi(\lambda,x)=\lambda x^{\otimes D},
\qquad \|x\|_2=1.
```

This is the symmetric analogue of an ordinary CP rank-one component: the
Segre geometry of independent mode factors is replaced by the Veronese
geometry of one factor repeated in every mode. A rank-`R` symmetric CP model
is therefore constructed as `JoinModel(SymmetricRankOne(n, D), R, target)`.

The Veronese formulation follows R. Khouja, H. Khalil, and B. Mourrain,
"Riemannian Newton optimization methods for the symmetric tensor approximation
problem," *Linear Algebra and its Applications* 637 (2022), 175--211,
doi:10.1016/j.laa.2021.12.008.
"""
struct SymmetricRankOne{V<:AbstractManifold} <: AbstractJoinComponent
    manifold::V

    function SymmetricRankOne(manifold::V) where {V<:AbstractManifold}
        _is_symcpd_manifold(manifold) || throw(
            ArgumentError("SymmetricRankOne requires the configured Veronese manifold."),
        )
        return new{V}(manifold)
    end
end

SymmetricRankOne(n::Int, order::Int) = SymmetricRankOne(_symcpd_manifold(n, order))
SymmetricRankOne(; dimension::Int, order::Int) = SymmetricRankOne(dimension, order)

component_manifold(component::SymmetricRankOne) = component.manifold
component_embedding(::SymmetricRankOne) = DefaultJoinEmbedding()
kind(::SymmetricRankOne) = :Veronese

function _symmetric_component_size(component::SymmetricRankOne)
    return _symcpd_manifold_size(component.manifold)
end

function ambient_length(component::SymmetricRankOne)
    n, order = _symmetric_component_size(component)
    return _check_symmetric_coordinate_size(n, order)
end

@doc raw"""
    SymmetricCPDBackend

Kernelized backend used when a [`JoinModel`](@ref) repeats a
[`SymmetricRankOne`](@ref) component. It represents the symmetric rank-`rank`
approximation

```math
\widehat A = \sum_{r=1}^R \lambda_r x_r^{\otimes D},
\qquad \|x_r\|_2=1.
```

Each component point is stored intrinsically as `([lambda], x)` on
`Manifolds.Veronese(n, D)`. The objective is evaluated as

```math
f(p)=\tfrac12\|A\|_F^2
-\sum_r\lambda_r\langle A,x_r^{\otimes D}\rangle
+\tfrac12\sum_{r,s}\lambda_r\lambda_s(x_r^\top x_s)^D.
```

Consequently, neither the predicted tensor nor a compressed Veronese vector is
formed by `cost` or `rgrad`. `target` may be a [`DenseSymmetricTarget`](@ref),
[`CompressedSymmetricTarget`](@ref), or [`FunctionalSymmetricTarget`](@ref).

The product-of-Veronese least-squares formulation and its Riemannian
Gauss--Newton structure are described by R. Khouja, H. Khalil, and B. Mourrain,
"Riemannian Newton optimization methods for the symmetric tensor approximation
problem," *Linear Algebra and its Applications* 637 (2022), 175--211,
doi:10.1016/j.laa.2021.12.008. This implementation differs computationally by
using the kernel as an operator and not assembling their dense normal matrix.
"""
struct SymmetricCPDBackend{
    T<:AbstractFloat,
    B<:AbstractSymmetricTarget{T},
    C<:SymmetricRankOne,
    CS<:Tuple,
    P<:ProductManifold,
} <: AbstractJoinBackend
    target::B
    component::C
    components::CS
    rank::Int
    n::Int
    order::Int
    product_manifold::P
end

function JoinModel(
    component::SymmetricRankOne,
    rank::Int,
    target::B,
) where {T<:AbstractFloat,B<:AbstractSymmetricTarget{T}}
    rank >= 1 || throw(ArgumentError("rank must be positive, got $rank."))
    n, order = _symmetric_component_size(component)
    target_n, target_order = _symmetric_target_size(target)
    (n, order) == (target_n, target_order) || throw(
        DimensionMismatch(
            "SymmetricRankOne has dimension/order ($n, $order), but target has " *
            "($target_n, $target_order).",
        ),
    )
    components = ntuple(_ -> component, rank)
    product = ProductManifold(ntuple(_ -> component.manifold, rank)...)
    backend = SymmetricCPDBackend{T,B,typeof(component),typeof(components),typeof(product)}(
        target,
        component,
        components,
        rank,
        n,
        order,
        product,
    )
    return JoinModel{T,typeof(backend)}(backend)
end

function JoinModel(
    component::SymmetricRankOne,
    rank::Int,
    target::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    n, order = _symmetric_component_size(component)
    N == order || throw(
        DimensionMismatch("Expected an order-$order dense symmetric target, got order $N."),
    )
    size(target) == ntuple(_ -> n, order) || throw(
        DimensionMismatch(
            "Expected dense symmetric target size $(ntuple(_ -> n, order)), got $(size(target)).",
        ),
    )
    return JoinModel(component, rank, DenseSymmetricTarget(target))
end

function JoinModel(
    component::SymmetricRankOne,
    rank::Int,
    target::AbstractVector{T},
) where {T<:AbstractFloat}
    n, order = _symmetric_component_size(component)
    return JoinModel(component, rank, CompressedSymmetricTarget(target, n, order))
end

JoinModel(component::SymmetricRankOne, target::AbstractSymmetricTarget) =
    JoinModel(component, 1, target)
JoinModel(
    component::SymmetricRankOne,
    target::AbstractArray{T,N},
) where {T<:AbstractFloat,N} = JoinModel(component, 1, target)

@doc raw"""
    SymCPDModel(target, rank)

Compatibility constructor for a symmetric rank-`rank` decomposition. It
creates `SymmetricRankOne(n, D)` from `target` and returns
`JoinModel(component, rank, target)`; `SymCPDModel` is not a separate model
type.

The component/join formulation follows the product-of-Veronese approximation
in Khouja, Khalil, and Mourrain (2022),
doi:10.1016/j.laa.2021.12.008.
"""
function SymCPDModel(target::AbstractSymmetricTarget, rank::Int)
    n, order = _symmetric_target_size(target)
    return JoinModel(SymmetricRankOne(n, order), rank, target)
end

_backend_components(backend::SymmetricCPDBackend) = backend.components
manifold(model::JoinModel{T,B}) where {T<:AbstractFloat,B<:SymmetricCPDBackend} =
    model.backend.product_manifold
tensor(model::JoinModel{T,B}) where {T<:AbstractFloat,B<:SymmetricCPDBackend} =
    _symmetric_target_storage(model.backend.target)
supports_rgrad(::JoinModel{T,B}) where {T<:AbstractFloat,B<:SymmetricCPDBackend} = true
supports_egrad_project(::JoinModel{T,B}) where {T<:AbstractFloat,B<:SymmetricCPDBackend} =
    false

function egrad(model::JoinModel{T,B}, p) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    throw(
        ArgumentError(
            "The symmetric JoinModel provides a direct intrinsic Riemannian gradient. " *
            "Use gradient_mode=:riemannian.",
        ),
    )
end

function initial_point(
    model::JoinModel{T,B},
    init::Symbol;
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    init == :random || throw(
        ArgumentError(
            "The symmetric JoinModel supports init=:random or an explicit p0, got $init.",
        ),
    )
    backend = model.backend
    parts = ntuple(_ -> begin
        _symcpd_random_point(backend.component.manifold, T)
    end, backend.rank)
    return join_point(backend.product_manifold, parts)
end

@doc raw"""
    component_inner(component::SymmetricRankOne, p, q)

Return the Frobenius inner product of two symmetric rank-one components,

```math
\langle \lambda x^{\otimes D},\mu y^{\otimes D}\rangle
=\lambda\mu(x^\top y)^D.
```

This is the homogeneous polynomial (Veronese) kernel. Differentiating this
identity gives the Gauss--Newton blocks in Proposition 4.9 of Khouja, Khalil,
and Mourrain (2022), doi:10.1016/j.laa.2021.12.008.
"""
function component_inner(component::SymmetricRankOne, p, q)
    pp = point_parts(p)
    qp = point_parts(q)
    _, order = _symmetric_component_size(component)
    return pp[1][1] * qp[1][1] * dot(pp[2], qp[2])^order
end

component_inner(
    model::JoinModel{T,B},
    p,
    q,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend} =
    component_inner(model.backend.component, p, q)

@doc raw"""
    data_inner(model, p)

Return `\langle A, lambda*x^(tensor D)\rangle` using the selected target
backend. This is the target-side polynomial evaluation; unlike model-model
interactions it cannot, for an arbitrary target, be reduced to pairwise
rank-one kernels. See the gradient/polynomial-evaluation construction in
Khouja, Khalil, and Mourrain (2022), Proposition 4.9,
doi:10.1016/j.laa.2021.12.008.
"""
function data_inner(
    model::JoinModel{T,B},
    p,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    pp = point_parts(p)
    return pp[1][1] * evaluate(model.backend.target, pp[2])
end

function cost(model::JoinModel{T,B}, p) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    parts = join_parts(backend.product_manifold, p)
    length(parts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) components."))
    value = T(0.5) * target_norm2(backend.target)
    @inbounds for r = 1:backend.rank
        value -= data_inner(model, parts[r])
        value += T(0.5) * component_inner(model, parts[r], parts[r])
        for s = 1:(r-1)
            value += component_inner(model, parts[r], parts[s])
        end
    end
    return value
end

@doc raw"""
    rgrad(model, p)

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
function rgrad(model::JoinModel{T,B}, p) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    M = backend.product_manifold
    parts = join_parts(M, p)
    length(parts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) components."))
    d = backend.order
    values = ntuple(
        r -> begin
            pr = point_parts(parts[r])
            lambda_r = pr[1][1]
            x_r = pr[2]
            a_r, b_r = _symmetric_target_inner_and_contraction(backend.target, x_r)
            radial = -a_r
            factor_covector = (-T(d) * lambda_r) .* b_r
            @inbounds for s = 1:backend.rank
                ps = point_parts(parts[s])
                lambda_s = ps[1][1]
                x_s = ps[2]
                c = dot(x_r, x_s)
                radial += lambda_s * c^d
                factor_covector .+= (T(d) * lambda_r * lambda_s * c^(d - 1)) .* x_s
            end
            factor_covector .-= dot(x_r, factor_covector) .* x_r
            factor_gradient = factor_covector ./ (T(d) * lambda_r^2)
            _symcpd_tangent(T(radial), factor_gradient)
        end,
        backend.rank,
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
    M = _symcpd_manifold(n, res.order)
    T = eltype(res.weights)
    out = zeros(T, binomial(n + res.order - 1, res.order))
    work = similar(out)
    for c in res.components
        _symcpd_embed_coordinates!(work, M, c.point)
        out .+= work
    end
    return out
end

function _symcpd_result(
    model::JoinModel{T,B},
    result,
    n::Int,
    d::Int,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    parts = join_parts(model.backend.product_manifold, result.point)
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

function extract_components(
    model::JoinModel{T,B},
    p,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    parts = join_parts(backend.product_manifold, p)
    length(parts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) components."))
    components_out = Vector{SymCPDComponent}(undef, backend.rank)
    for k = 1:backend.rank
        pk = parts[k]
        pk_parts = point_parts(pk)
        components_out[k] =
            SymCPDComponent(pk, T(pk_parts[1][1]), Vector{T}(pk_parts[2]), backend.order)
    end
    return components_out
end
