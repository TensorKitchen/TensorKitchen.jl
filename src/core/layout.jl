# core/layout.jl — Container boundaries.
# A component manifold owns its native point representation. The join and
# solver may wrap or unwrap only the outer collection of components.

@inline point_parts(p) = hasproperty(p, :x) ? p.x : p
@inline parts_tuple(p) = (_p = point_parts(p); _p isa Tuple ? _p : Tuple(_p))
@inline _partition_parts(parts::Tuple) = ArrayPartition(parts)

# Manopt applies ordinary vector-space arithmetic to its outer ArrayPartition.
# RecursiveArrayTools delegates that arithmetic to each component, but a
# Veronese tangent is a native tuple of arrays. Preserve that tuple while
# lifting the operations over its leaves at the solver boundary.
const _TupleComponentPartition =
    ArrayPartition{T,P} where {T,P<:Tuple{Vararg{<:Tuple}}}
@inline _native_neg(x::Tuple) = map(_native_neg, x)
@inline _native_neg(x) = -x
@inline _native_add(x::Tuple, y::Tuple) = map(_native_add, x, y)
@inline _native_add(x, y) = x + y
@inline _native_sub(x::Tuple, y::Tuple) = map(_native_sub, x, y)
@inline _native_sub(x, y) = x - y
@inline _native_scale(a, x::Tuple) = map(y -> _native_scale(a, y), x)
@inline _native_scale(a, x) = a * x
@inline _native_copy(x::Tuple) = map(_native_copy, x)
@inline _native_copy(x) = copy(x)
@inline _native_similar(x::Tuple) = map(_native_similar, x)
@inline _native_similar(x) = similar(x)
@inline _native_similar(x::Tuple, ::Type{T}) where {T} =
    map(y -> _native_similar(y, T), x)
@inline _native_similar(x, ::Type{T}) where {T} = similar(x, T)
@inline _native_zero(x::Tuple) = map(_native_zero, x)
@inline _native_zero(x) = zero(x)
Base.:-(x::_TupleComponentPartition) = _partition_parts(map(_native_neg, x.x))
Base.:+(x::_TupleComponentPartition, y::_TupleComponentPartition) =
    _partition_parts(map(_native_add, x.x, y.x))
Base.:-(x::_TupleComponentPartition, y::_TupleComponentPartition) =
    _partition_parts(map(_native_sub, x.x, y.x))
Base.:*(a::Number, x::_TupleComponentPartition) =
    _partition_parts(map(y -> _native_scale(a, y), x.x))
Base.:*(x::_TupleComponentPartition, a::Number) = a * x
Base.:/(x::_TupleComponentPartition, a::Number) = inv(a) * x
Base.copy(x::_TupleComponentPartition) = _partition_parts(map(_native_copy, x.x))
Base.similar(x::_TupleComponentPartition) = _partition_parts(map(_native_similar, x.x))
Base.similar(x::_TupleComponentPartition, ::Type{T}) where {T} =
    _partition_parts(map(y -> _native_similar(y, T), x.x))
Base.zero(x::_TupleComponentPartition) = _partition_parts(map(_native_zero, x.x))
ManifoldsBase.allocate(x::_TupleComponentPartition) = similar(x)
ManifoldsBase.allocate(x::_TupleComponentPartition, ::Type{T}) where {T} = similar(x, T)
@inline _native_copyto!(dest::Tuple, src::Tuple) =
    map(_native_copyto!, dest, src)
@inline _native_copyto!(dest, src) = copyto!(dest, src)
function Base.copyto!(
    dest::_TupleComponentPartition,
    bc::Base.Broadcast.Broadcasted{<:RecursiveArrayTools.ArrayPartitionStyle},
)
    for i in eachindex(dest.x)
        _native_copyto!(dest.x[i], copy(RecursiveArrayTools.unpack(bc, i)))
    end
    return dest
end
@inline wrap_like_point(p, vals::Tuple) =
    hasproperty(p, :x) ? _partition_parts(vals) : vals
@inline _unwrap_part(x) = hasproperty(x, :x) ? x.x : x

@inline join_parts(::ProductManifold, p::Tuple) = p
@inline join_parts(M::ProductManifold, p::ArrayPartition) =
    ManifoldsBase.submanifold_components(M, p)
@inline join_point(::ProductManifold, parts::Tuple) = _partition_parts(parts)
@inline join_tangent_like(::ProductManifold, p, parts::Tuple) = wrap_like_point(p, parts)

# Traverse the manifold tree, not the Julia value tree: a nested ProductManifold
# owns another product boundary, whereas a Veronese/Segre native tuple is opaque.
join_solver_point(::AbstractManifold, p) = p
function join_solver_point(M::ProductManifold, p)
    parts = join_parts(M, p)
    length(parts) == length(M.manifolds) || throw(DimensionMismatch(
        "ProductManifold expects $(length(M.manifolds)) components, got $(length(parts)).",
    ))
    return join_point(
        M,
        ntuple(i -> join_solver_point(M.manifolds[i], parts[i]), length(parts)),
    )
end

# Adapt one container level. In particular, a native tuple such as a Veronese
# ([λ], x) is never recursively converted into an ArrayPartition.
@inline outer_container_like(p, x) =
    hasproperty(p, :x) ?
    (hasproperty(x, :x) ? x : (x isa Tuple ? _partition_parts(x) : x)) :
    (hasproperty(x, :x) ? Tuple(getproperty(x, :x)) : x)

# Pack canonical rank-r CP tangents in one place so the hot gradient paths do
# not each rebuild the same nested ArrayPartition layout by hand.
function wrap_rankr_canonical_tangent_like(p, grad_λ, gradU, r::Int)
    grad_modes = ntuple(m -> ntuple(k -> Vector(@view gradU[m][:, k]), r), length(gradU))
    if hasproperty(p, :x)
        mode_parts = ntuple(m -> ArrayPartition(grad_modes[m]...), length(grad_modes))
        return ArrayPartition(grad_λ, mode_parts...)
    end
    return (grad_λ, grad_modes...)
end
