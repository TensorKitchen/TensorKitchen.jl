# Target-side storage and contractions for symmetric CPD.

export AbstractSymmetricTarget,
    DenseSymmetricTarget, CompressedSymmetricTarget, symmetric_target_norm2

abstract type AbstractSymmetricTarget{T<:AbstractFloat} end

@doc raw"""
    DenseSymmetricTarget(A)

Store a dense order-`D` symmetric tensor `A` for matrix-free symmetric CPD.
The model never forms its predicted tensor.  Target evaluations use only

```math
a(x)=\langle A,x^{\otimes D}\rangle,
\qquad
b(x)=A(x,\ldots,x,\cdot),
```

where `b(x)` is the one-mode contraction satisfying
`\nabla a(x) = D*b(x)` for symmetric `A`.

This target/model separation follows the polynomial-evaluation formulation in
R. Khouja, H. Khalil, and B. Mourrain, "Riemannian Newton optimization methods
for the symmetric tensor approximation problem," *Linear Algebra and its
Applications* 637 (2022), 175--211, Proposition 4.9,
doi:10.1016/j.laa.2021.12.008.
"""
struct DenseSymmetricTarget{T<:AbstractFloat,N,A<:AbstractArray{T,N}} <:
       AbstractSymmetricTarget{T}
    data::A
    norm2::T
end

function DenseSymmetricTarget(A::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    N >= 1 || throw(ArgumentError("The target order must be positive."))
    n = size(A, 1)
    all(==(n), size(A)) || throw(
        DimensionMismatch("A symmetric target must have equal mode sizes, got $(size(A))."),
    )
    return DenseSymmetricTarget{T,N,typeof(A)}(A, T(sum(abs2, A)))
end

"""
    CompressedSymmetricTarget(coefficients, n, order)

Store a symmetric tensor in Bombieri--Weyl orthonormal coordinates.  The
coordinate vector has length `binomial(n + order - 1, order)`, but model-model
terms still use the kernel `(x' * y)^order` and never construct a compressed
Veronese vector.  Target contractions visit the coordinates one at a time.

The normalization is the isometric symmetric-tensor/polynomial convention used
for the Veronese geometry in Khouja, Khalil, and Mourrain (2022),
doi:10.1016/j.laa.2021.12.008.  In TensorKitchen, compressed storage is a
target backend rather than the intrinsic point representation.
"""
struct CompressedSymmetricTarget{T<:AbstractFloat,V<:AbstractVector{T}} <:
       AbstractSymmetricTarget{T}
    coefficients::V
    n::Int
    order::Int
    norm2::T
end

function CompressedSymmetricTarget(
    coefficients::AbstractVector{T},
    n::Int,
    order::Int,
) where {T<:AbstractFloat}
    n > 0 || throw(ArgumentError("n must be positive, got $n."))
    order > 0 || throw(ArgumentError("order must be positive, got $order."))
    expected = binomial(n + order - 1, order)
    length(coefficients) == expected || throw(
        DimensionMismatch(
            "Expected $expected compressed coordinates for n=$n and order=$order, " *
            "got $(length(coefficients)).",
        ),
    )
    return CompressedSymmetricTarget{T,typeof(coefficients)}(
        coefficients,
        n,
        order,
        T(sum(abs2, coefficients)),
    )
end

@doc raw"""Return `\|A\|_F^2` without expanding a compressed symmetric target."""
symmetric_target_norm2(target::AbstractSymmetricTarget) = target.norm2

_symmetric_target_storage(target::DenseSymmetricTarget) = target.data
_symmetric_target_storage(target::CompressedSymmetricTarget) = target.coefficients
_symmetric_target_size(target::DenseSymmetricTarget) =
    (size(target.data, 1), ndims(target.data))
_symmetric_target_size(target::CompressedSymmetricTarget) = (target.n, target.order)

function _symmetric_target_inner(target::DenseSymmetricTarget, x::AbstractVector)
    A = target.data
    d = ndims(A)
    T = promote_type(eltype(A), eltype(x))
    value = zero(T)
    @inbounds for I in CartesianIndices(A)
        monomial = one(T)
        for mode = 1:d
            monomial *= x[I[mode]]
        end
        value += A[I] * monomial
    end
    return value
end

function _symmetric_target_inner_and_contraction(
    target::DenseSymmetricTarget,
    x::AbstractVector,
)
    A = target.data
    d = ndims(A)
    n = size(A, 1)
    length(x) == n || throw(DimensionMismatch("Expected a factor of length $n."))
    T = promote_type(eltype(A), eltype(x))
    value = zero(T)
    contraction = zeros(T, n)
    @inbounds for I in CartesianIndices(A)
        trailing = one(T)
        for mode = 2:d
            trailing *= x[I[mode]]
        end
        contraction[I[1]] += A[I] * trailing
        value += A[I] * x[I[1]] * trailing
    end
    return value, contraction
end

function _symmetric_target_inner(target::CompressedSymmetricTarget, x::AbstractVector)
    length(x) == target.n ||
        throw(DimensionMismatch("Expected a factor of length $(target.n)."))
    M = Manifolds.Veronese(target.n, target.order)
    T = promote_type(eltype(target.coefficients), eltype(x))
    value = zero(T)
    Manifolds._veronese_coordinates(M, x) do k, monomial, _
        value += target.coefficients[k] * monomial
    end
    return value
end

function _symmetric_target_inner_and_contraction(
    target::CompressedSymmetricTarget,
    x::AbstractVector,
)
    length(x) == target.n ||
        throw(DimensionMismatch("Expected a factor of length $(target.n)."))
    M = Manifolds.Veronese(target.n, target.order)
    T = promote_type(eltype(target.coefficients), eltype(x))
    value = zero(T)
    contraction = zeros(T, target.n)
    inv_order = inv(T(target.order))
    Manifolds._veronese_coordinates(M, x; differential = true) do k, monomial, gradient
        coefficient = target.coefficients[k]
        value += coefficient * monomial
        contraction .+= (coefficient * inv_order) .* gradient
    end
    return value, contraction
end
