# manifolds/tucker.jl — TensorDecomposition extensions for Manifolds.Tucker

export multilinear_rank, factor_dims, dim

Manifolds.Tucker(dims::Vector{T1}, ranks::Vector{T2}) where {T1<:Int,T2<:Int} =
    Manifolds.Tucker(Tuple(Int.(dims)), Tuple(Int.(ranks)), ManifoldsBase.ℝ)

ManifoldsBase.default_retraction_method(
    ::Manifolds.Tucker,
    ::Type{<:Manifolds.TuckerPoint},
) = ManifoldsBase.PolarRetraction()
ManifoldsBase.default_retraction_method(::Manifolds.Tucker, ::Manifolds.TuckerPoint) =
    ManifoldsBase.PolarRetraction()

@inline _tucker_size_data(M::Manifolds.Tucker) = ManifoldsBase.get_parameter(M.size)

factor_dims(M::Manifolds.Tucker) = _tucker_size_data(M)[1]
multilinear_rank(M::Manifolds.Tucker) = _tucker_size_data(M)[2]
dim(M::Manifolds.Tucker) = manifold_dimension(M)
