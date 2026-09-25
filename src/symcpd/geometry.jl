# Thin adapter around the provisional Manifolds.jl Veronese API. Keep all
# constructor and representation assumptions here so an upstream API change
# does not leak into the SymCPD computational layer.

@inline _symcpd_manifold(n::Int, order::Int) = Manifolds.Veronese(n, order)

@inline _is_symcpd_manifold(::Manifolds.Veronese) = true
@inline _is_symcpd_manifold(::AbstractManifold) = false

@inline function _symcpd_manifold_size(M)
    return Manifolds.get_parameter(M.size)
end

@inline _symcpd_point(lambda, factor) = ([lambda], factor)
@inline _symcpd_tangent(radial, factor) = ([radial], factor)

function _symcpd_random_point(M, ::Type{T}) where {T<:AbstractFloat}
    p = rand(M)
    return _symcpd_point(T(p[1][1]), T.(p[2]))
end

function _manifold_init(M::Manifolds.Veronese, target, init::Symbol)
    init == :random ||
        throw(ArgumentError("Veronese supports init=:random or an explicit p0."))
    return _symcpd_random_point(M, eltype(target))
end
