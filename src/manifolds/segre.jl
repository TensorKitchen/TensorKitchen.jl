export dim, factor_dims

"""
    Manifolds.Segre(dims)

`Manifolds.Segre` manifold used for rank-1 CP.
Point format for rank-1 is:
`[[λ], x₁, x₂, …, x_d]` with λ > 0 and xᵢ ∈ S^{nᵢ-1}.
"""
function Manifolds.Segre(dims::NTuple{N,T}) where {N,T<:Int}
    N >= 2 || throw(ArgumentError("Segre needs order N >= 2, got N=$N."))
    for (i, n) in enumerate(dims)
        n > 0 || throw(ArgumentError("dims[$i] must be positive, got $n."))
    end
    return Manifolds.Segre(Int.(dims)...)
end

Manifolds.Segre(dims::Vector{T}) where {T<:Int} = Manifolds.Segre(Tuple(dims))
dim(M::Manifolds.Segre) = manifold_dimension(M)

function factor_dims(M::Manifolds.Segre)
    dims = typeof(M).parameters[2]
    return ntuple(i -> Int(dims[i]), length(dims))
end
