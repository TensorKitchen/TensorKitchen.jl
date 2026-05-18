# manifolds/join.jl — ProductManifold-first join constructors.
#
# Why join_product instead of "just ProductManifold"? ProductManifold is always the
# result type; join_product(base, r) decides how to expand base into r components:
# - Manifolds.Segre → flattened (Euclidean(1), Sphere, ...) × r (one λ + spheres per rank-1).
# - ProductManifold → repeat each existing product factor r times.
# - Manifolds.Tucker → repeat the Tucker manifold r times.
# - Generic manifold → ProductManifold(base, base, ..., base).
# So we use join_product (via CPJoin/TuckerJoin) wherever we want this expansion;
# the pipeline uses it for BTD (btd → TuckerJoin) and for the CPJoin/TuckerJoin APIs.

export join_product, dim, product



"""
Generic product parameterization of join models.

join_product(base, r) constructs the product-domain manifold used to
parameterize sums of r structured components. It does not construct the
image join/secant variety itself.
"""
function _segre_flat_factors(M::Manifolds.Segre)
    dims = factor_dims(M)
    return (Euclidean(1), (Sphere(n - 1) for n in dims)...)
end

"""
    join_product(base, r) 

Decides how to expand base into r components:
* Manifolds.Segre → flattened (Euclidean(1), Sphere, ...) × r (one λ + spheres per rank-1).
* ProductManifold → repeat each existing product factor r times.
* Manifolds.Tucker → repeat the Tucker manifold r times.
* Generic manifold → ProductManifold(base, base, ..., base).
"""
function join_product(base::AbstractManifold, r::Int)
    r >= 1 || throw(ArgumentError("Join rank must be at least 1, got r=$r."))
    return _join_product(base, r)
end

function _join_product(base::Manifolds.Segre, r::Int)
    factors = _segre_flat_factors(base)
    return ProductManifold([deepcopy(f) for _ = 1:r for f in factors]...)
end

function _join_product(base::ProductManifold, r::Int)
    factors = base.manifolds
    return ProductManifold([deepcopy(f) for _ = 1:r for f in factors]...)
end

function _join_product(base::AbstractManifold, r::Int)
    return ProductManifold(ntuple(_ -> deepcopy(base), r)...)
end

product(M::ProductManifold) = M
dim(M::ProductManifold) = manifold_dimension(M)
