# manifolds/join.jl — ProductManifold-first join constructors.
#
# Why join_product instead of "just ProductManifold"? ProductManifold is always the
# result type; join_product(base, r) decides how to expand base into r components:
# - Manifolds.Segre → flattened (Euclidean(1), Sphere, ...) × r (one λ + spheres per rank-1).
# - Manifolds.Tucker / ProductManifold → repeat each factor r times.
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
    join_product(base, r) -> ProductManifold

Construct rank-`r` join parameter manifold as a plain `ProductManifold`.
For `Manifolds.Segre`, uses flattened `(Euclidean(1), Sphere, ..., Sphere)` factors
per component to preserve current CP parameter layout.
"""
function join_product(base::T, r::Int) where {T<:AbstractManifold}
    r >= 2 || throw(ArgumentError("Join rank must be at least 2, got r=$r."))

    M = if base isa Manifolds.Segre
        factors = _segre_flat_factors(base)
        ProductManifold([deepcopy(f) for _ = 1:r for f in factors]...)
    elseif base isa ProductManifold
        factors = base.manifolds
        ProductManifold([deepcopy(f) for _ = 1:r for f in factors]...)
    else
        ProductManifold(ntuple(_ -> deepcopy(base), r)...)
    end

    return M
end

product(M::ProductManifold) = M
dim(M::ProductManifold) = manifold_dimension(M)
