# Narrow compatibility adapters for Manopt/Manifolds integration.
#
# Keep dependency workarounds in this file so solver implementations can retain
# TensorKitchen's scalar and storage invariants without extending methods on
# external point or manifold types.

"""
    _LMSubproblemManifold(manifold)

Internal decorator used only for Levenberg--Marquardt tangent-space
subproblems. Manopt currently allocates a coordinate scratch vector while
applying its matrix-free LM surrogate, even when a `FunctionVectorialType`
Jacobian ignores that scratch space. Manifolds' `ProductManifold` allocator
delegates this allocation to its first point component; this fails when that
component is a `TuckerPoint` or a native Veronese tuple rather than an array.

The decorator forwards manifold operations unchanged and only supplies the
flat coordinate scratch allocation and manifold-aware tangent-space copying.
It can be removed when the upstream allocation path supports structured
product components directly.
"""
struct _LMSubproblemManifold{F,M<:ManifoldsBase.AbstractManifold{F}} <:
       ManifoldsBase.AbstractDecoratorManifold{F}
    manifold::M
end

ManifoldsBase.decorated_manifold(M::_LMSubproblemManifold) = M.manifold
ManifoldsBase.get_forwarding_type(::_LMSubproblemManifold, _) =
    ManifoldsBase.SimpleForwardingType()
ManifoldsBase.get_forwarding_type(::_LMSubproblemManifold, _, ::Type) =
    ManifoldsBase.SimpleForwardingType()

@inline _lm_coordinate_storage(p::Manifolds.TuckerPoint) = p.hosvd.core
@inline _lm_coordinate_storage(p::Manifolds.TuckerTangentVector) = p.Ċ
@inline _lm_coordinate_storage(p::RecursiveArrayTools.ArrayPartition) =
    _lm_coordinate_storage(first(p.x))
@inline _lm_coordinate_storage(p::Tuple) = _lm_coordinate_storage(first(p))
@inline _lm_coordinate_storage(p::AbstractArray) = p

function ManifoldsBase.allocate_result(
    M::_LMSubproblemManifold,
    ::typeof(ManifoldsBase.get_coordinates),
    p,
    X,
    basis::ManifoldsBase.AbstractBasis,
)
    storage = _lm_coordinate_storage(p)
    T = _scalar_eltype(p)
    n = ManifoldsBase.number_of_coordinates(M.manifold, basis)
    return similar(storage, T, n)
end

@inline function _lm_subproblem_manifold(M)
    return _lm_needs_storage_adapter(M) ? _LMSubproblemManifold(M) : M
end

_lm_needs_storage_adapter(M) = _contains_tucker_manifold(M) || M isa Manifolds.Veronese
_lm_needs_storage_adapter(M::ProductManifold) = any(_lm_needs_storage_adapter, M.manifolds)

# CR copies points of a tangent space. The generic Fiber fallback uses array
# copying, which cannot handle native Veronese tuples inside ArrayPartition.
# Scope this correction to our decorator and delegate to the base geometry.
function ManifoldsBase.copyto!(
    TpM::ManifoldsBase.TangentSpace{F,<:_LMSubproblemManifold},
    dest,
    src,
) where {F}
    M = ManifoldsBase.base_manifold(TpM).manifold
    return ManifoldsBase.copyto!(M, dest, ManifoldsBase.base_point(TpM), src)
end

# Hager--Zhang currently seeds its evaluation history with
# `UnivariateTriple(0.0, ...)`, which selects Float64 even for Float32 states.
# Keep its internal scalar workspace at Float64 until that literal is made
# type-generic upstream; TensorKitchen still converts retraction steps at the
# line-search boundary and retains Float32 model points and tangents.
@inline _hagerzhang_workspace_type(::Type{T}) where {T<:Real} = T === Float32 ? Float64 : T
