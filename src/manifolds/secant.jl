# manifolds/secant.jl — ProductManifold join constructors for Segre and Tucker families.
#
# Pipeline: Manopt uses manifold M for retraction (exp map) and project (egrad→Riemannian grad).
# - ProductManifold(Euclidean(1), Sphere×...) is the placeholder for Segre (CPD rank-1).
# - CanonicalCP: Euclidean(r) × Π_m ProductManifold(Sphere(n_m-1), ..., Sphere(n_m-1)) for rank-r CPD.
# - `join_product(Manifolds.Segre(...), r)` gives flattened CP layout.
# - `join_product(Manifolds.Tucker(...), b)` gives Tucker block join manifold for BTD.


export CPJoin, TuckerJoin, SegreProduct, CanonicalCP

"""
Named tensor-decomposition parameter manifolds built from join_product.

These are semantic constructors for CPD/BTD-style models.
"""



"""
    SegreProduct(dims, r)

Product manifold `Manifolds.Segre(dims) × ... × Manifolds.Segre(dims)` (r factors).
Each component point uses the `Manifolds.Segre` layout `[[λ], x₁, …, x_d]`.
"""
function SegreProduct(dims::NTuple{N,Int}, r::Int) where {N}
    r >= 2 || throw(
        ArgumentError("Rank-r ProductManifold(Manifolds.Segre, ...) needs r>=2, got r=$r"),
    )
    return ProductManifold(ntuple(_ -> Manifolds.Segre(dims...), r)...)
end

SegreProduct(dims::Vector{T}, r::Int) where {T<:Int} = SegreProduct(Tuple(dims), r)

# Canonical rank-r CP: ℝ^r × ∏_m (S^(n_m-1))^r
function CanonicalCP(dims::NTuple{N,Int}, r::Int) where {N}
    r >= 2 || throw(ArgumentError("Canonical rank-r CP manifold needs r>=2, got r=$r"))
    mode_factors = ntuple(m -> ProductManifold(ntuple(_ -> Sphere(dims[m] - 1), r)...), N)
    return ProductManifold(Euclidean(r), mode_factors...)
end

CanonicalCP(dims::Vector{T}, r::Int) where {T<:Int} = CanonicalCP(Tuple(dims), r)

CPJoin(M::Manifolds.Segre, r::Int) = join_product(M, r)
CPJoin(dims::NTuple{N,T}, r::Int) where {N,T<:Int} = CPJoin(Manifolds.Segre(dims), r)
CPJoin(dims::Vector{T}, r::Int) where {T<:Int} = CPJoin(Tuple(dims), r)
CPJoin(M::Manifolds.Tucker, b::Int) = join_product(M, b)

TuckerJoin(M::Manifolds.Tucker, b::Int) = join_product(M, b)
TuckerJoin(dims::NTuple{N,T1}, ranks::NTuple{N,T2}, b::Int) where {N,T1<:Int,T2<:Int} =
    TuckerJoin(Manifolds.Tucker(dims, ranks), b)
TuckerJoin(dims::Vector{T1}, ranks::Vector{T2}, b::Int) where {T1<:Int,T2<:Int} =
    TuckerJoin(Tuple(dims), Tuple(ranks), b)
