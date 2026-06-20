# join/join_backend.jl — Generic join and BTD backend implementation

_is_manifold_like(::AbstractManifold) = true
_is_manifold_like(_) = false
_is_join_component_like(::JoinComponent) = true
_is_join_component_like(x) = _is_manifold_like(x)

_wrap_join_component(component::JoinComponent) = component
_wrap_join_component(manifold::AbstractManifold) = JoinComponent(manifold)

_component_manifold(component::JoinComponent) = component.manifold
_component_manifold(manifold::AbstractManifold) = manifold
_backend_components(backend::JoinBackend) = backend.components
_backend_components(backend::BTDBackend) = backend.manifolds

function _as_join_manifold_tuple(manifolds::Tuple)
    all(_is_manifold_like, manifolds) || throw(
        ArgumentError(
            "All entries in manifolds tuple must be AbstractManifold. Got types: $(map(typeof, manifolds)).",
        ),
    )
    return manifolds
end

function _as_join_manifold_tuple(manifolds::AbstractVector)
    all(_is_manifold_like, manifolds) || throw(
        ArgumentError(
            "All entries in manifolds vector must be AbstractManifold. Got types: $(map(typeof, manifolds)).",
        ),
    )
    return Tuple(manifolds)
end

_as_join_manifold_tuple(M::ProductManifold) = Tuple(M.manifolds)

function _as_join_component_tuple(components::Tuple)
    all(_is_join_component_like, components) || throw(
        ArgumentError(
            "All join components must be AbstractManifold or JoinComponent. Got types: $(map(typeof, components)).",
        ),
    )
    return ntuple(k -> _wrap_join_component(components[k]), length(components))
end

function _as_join_component_tuple(components::AbstractVector)
    all(_is_join_component_like, components) || throw(
        ArgumentError(
            "All join components must be AbstractManifold or JoinComponent. Got types: $(map(typeof, components)).",
        ),
    )
    return Tuple(_wrap_join_component(c) for c in components)
end

_as_join_component_tuple(M::ProductManifold) = _as_join_component_tuple(Tuple(M.manifolds))

function _uniform_segre_dims(components::Tuple)
    isempty(components) && return nothing
    first_manifold = _component_manifold(first(components))
    first_manifold isa Manifolds.Segre || return nothing
    dims = factor_dims(first_manifold)
    all(c -> begin
        M = _component_manifold(c)
        M isa Manifolds.Segre && factor_dims(M) == dims
    end, components) || return nothing
    return dims
end

@inline function _check_parts_len(parts, expected::Int, where_fn::AbstractString)
    length(parts) == expected || throw(
        DimensionMismatch(
            "$where_fn expected $expected components, got $(length(parts)). " *
            "Ensure point layout matches ProductManifold component count.",
        ),
    )
end

_manifold_init(M, target, init) =
    _manifold_init(M, target, _builtin_initializer_symbol(init))

_manifold_init(M::Manifolds.Sphere, target, init::Symbol) = _sphere_init(M, target, init)
_manifold_init(M::Manifolds.Segre, target, init::Symbol) = _segre_init(M, target, init)
_manifold_init(M::Manifolds.Tucker, target, init::Symbol) = _tucker_init(M, target, init)

_component_init(component, target, init) =
    _manifold_init(_component_manifold(component), target, init)

function _manifold_init(M, target, init_sym::Symbol)
    init_sym == :random && return rand(M)
    throw(
        ArgumentError(
            "No default init for manifold $(typeof(M)). Use init=:random or provide init_point.",
        ),
    )
end

"""
    _component_egrad(component, p, residual)

Compute the component Euclidean gradient induced by a join residual. Tucker
components use their native tensor gradient; other components copy the residual.
"""
_component_egrad(::DefaultJoinEmbedding, M, p, residual) = copy(residual)
_component_egrad(::DefaultJoinEmbedding, M::Manifolds.Tucker, p, residual) =
    _tucker_egrad(M, p, residual)

function _component_egrad(::DefaultJoinEmbedding, M::Manifolds.Segre, p, residual)
    dims = factor_dims(M)
    R = reshape(residual, dims)
    parts = point_parts(p)
    λ = parts[1][1]
    grad_λ = rank1_inner_parts(R, parts)
    grad_U = Vector{Vector{eltype(R)}}(undef, length(dims))
    @inbounds for m in eachindex(dims)
        g = rank1_mode_contract_parts(R, parts, m)
        rmul!(g, λ)
        grad_U[m] = g
    end
    return pack_tangent_rank1_segre(grad_λ, grad_U)
end

_component_egrad(component::JoinComponent, p, residual) =
    _component_egrad(component.embedding, component.manifold, p, residual)
_component_egrad(M, p, residual) = _component_egrad(DefaultJoinEmbedding(), M, p, residual)

_manifold_egrad(M, p, residual) = _component_egrad(M, p, residual)

"""
    _ambient_vector(M, p, target_len) returns AbstractVector

Embed a component point into the flattened ambient tensor space, checking that
its length matches the target.
"""
function _ambient_vector(M, p, target_len::Int)
    emb = ManifoldsBase.embed(M, p)
    length(emb) == target_len || throw(
        DimensionMismatch(
            "Embedding length $(length(emb)) does not match target length $target_len. " *
            "Check manifold embedding dimension and point layout.",
        ),
    )
    return vec(emb)
end

function _ambient_vector(M::Manifolds.Tucker, p::Manifolds.TuckerPoint, target_len::Int)
    core = p.hosvd.core
    factors = p.hosvd.U
    X = reconstruct_tucker(core, factors)
    length(X) == target_len || throw(
        DimensionMismatch(
            "Tucker reconstructed tensor length $(length(X)) != target length $target_len.",
        ),
    )
    return vec(X)
end

function _ambient_vector(M::Manifolds.Tucker, p, target_len::Int)
    throw(
        ArgumentError(
            "Expected native TuckerPoint for Manifolds.Tucker, got $(typeof(p)).",
        ),
    )
end

"""
    _ambient_tensor(M, p, target_shape) returns AbstractArray

Embed a component point into ambient space and reshape it to the target tensor
shape.
"""
function _ambient_tensor(M, p, target_shape::Tuple)
    return reshape(_ambient_vector(M, p, prod(target_shape)), target_shape)
end

"""
    ambient_length(M) returns Int

Return the flattened ambient tensor length represented by a component manifold.
"""
function ambient_length(M::AbstractManifold)
    rs = ManifoldsBase.representation_size(M)
    rs === nothing && throw(
        ArgumentError(
            "Cannot infer ambient length for $(typeof(M)). Define ambient_length(::$(typeof(M))) " *
            "or use a manifold with a concrete representation_size.",
        ),
    )
    return rs isa Tuple ? prod(rs) : Int(rs)
end

ambient_length(M::Manifolds.Segre) = prod(factor_dims(M))
ambient_length(M::Manifolds.Tucker) = prod(factor_dims(M))
ambient_length(component::JoinComponent) = ambient_length(component.manifold)

"""
    _join_vector_workspace_like(target, n) returns AbstractVector

Allocate a flattened work vector with the same storage style and scalar type as
the target tensor.
"""
@inline function _join_vector_workspace_like(
    target::AbstractArray{T},
    n::Int,
) where {T<:AbstractFloat}
    # Match the target backend for all flattened join work buffers.
    return similar(vec(target), T, n)
end

"""
    _validate_join_ambient_compatibility(manifolds, target) validates the join ambient compatibility

Ensure every join component embeds into the same flattened ambient space as the
target tensor.
"""
function _validate_join_ambient_compatibility(components::Tuple, target::AbstractArray)
    target_len = length(target)
    failures = String[]
    @inbounds for k in eachindex(components)
        mk = ambient_length(components[k])
        if mk != target_len
            push!(
                failures,
                "[$k] $(typeof(components[k])) has ambient length $mk but target has length $target_len",
            )
        end
    end
    isempty(failures) || throw(
        DimensionMismatch(
            "All manifolds in a join must share the same flattened ambient length as the target.\n" *
            Base.join(failures, "\n"),
        ),
    )
end

"""
    _sum_backend_instance(B, manifolds, target; init_point=nothing) returns AbstractJoinBackend

Construct either a generic `JoinBackend` or `BTDBackend` with shared target,
product manifold, and reusable reconstruction/residual buffers.
"""
function _sum_backend_instance(
    B::Type,
    manifolds,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    throw(ArgumentError("Unsupported join backend type $B. Use JoinBackend or BTDBackend."))
end

function _sum_backend_parts(
    components,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    r = length(components)
    # Keep the original target representation instead of eagerly materializing Array.
    tgt = target
    _validate_join_ambient_compatibility(components, tgt)

    tflat = vec(tgt)
    tgt_len = length(tgt)
    # One ambient buffer per component lets reconstruction reuse storage across iterations.
    component_bufs = [_join_vector_workspace_like(tgt, tgt_len) for _ = 1:r]
    work_rec = _join_vector_workspace_like(tgt, tgt_len)
    work_residual = _join_vector_workspace_like(tgt, tgt_len)
    manifolds = ntuple(k -> _component_manifold(components[k]), r)

    return (;
        components,
        manifolds,
        r,
        target = tgt,
        target_size = size(tgt),
        target_flat = tflat,
        target_len = tgt_len,
        product = ProductManifold(manifolds...),
        init_point,
        work_rec,
        work_residual,
        component_bufs,
    )
end

function _sum_backend_instance(
    ::Type{JoinBackend},
    components,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    parts = _sum_backend_parts(components, target; init_point)
    return JoinBackend(
        parts.components,
        parts.r,
        parts.target,
        parts.target_size,
        parts.target_flat,
        parts.product,
        parts.init_point,
        parts.work_rec,
        parts.work_residual,
        _JoinResidualWORO(_join_vector_workspace_like(parts.target, parts.target_len)),
        parts.component_bufs,
    )
end

function _sum_backend_instance(
    ::Type{BTDBackend},
    components,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    parts = _sum_backend_parts(components, target; init_point)
    return BTDBackend(
        parts.manifolds,
        parts.r,
        parts.target,
        parts.target_size,
        parts.target_flat,
        parts.product,
        parts.init_point,
        parts.work_rec,
        parts.work_residual,
        BTDContractionWorkspace{T,N}(),
        sum(abs2, parts.target),
        parts.component_bufs,
    )
end

"""
    JoinModel(manifolds, target; init_point=nothing)

Construct a generic sum-of-manifolds approximation model from explicit
component manifolds and a target tensor.
"""
function JoinModel(
    components::Tuple,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    components_tuple = _as_join_component_tuple(components)
    r = length(components_tuple)
    r >= 1 || throw(ArgumentError("JoinModel(manifolds, target) needs r >= 1, got r=$r"))
    b = _sum_backend_instance(JoinBackend, components_tuple, target; init_point)
    return JoinModel{T,typeof(b)}(b)
end

"""
    JoinModel(base, r, target; init_point=nothing)

Construct a join model by repeating a base manifold `r` times.
"""
function JoinModel(
    components::AbstractVector,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(_as_join_component_tuple(components), target; init_point)
end

function JoinModel(
    M::ProductManifold,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(_as_join_component_tuple(M), target; init_point)
end

function JoinModel(
    base::AbstractManifold,
    r::Int,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(ntuple(_ -> base, r), target; init_point)
end

function JoinModel(
    base::JoinComponent,
    r::Int,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(ntuple(_ -> base, r), target; init_point)
end

function JoinModel(
    base::AbstractManifold,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel((base,), target; init_point)
end

function JoinModel(
    base::JoinComponent,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel((base,), target; init_point)
end

manifold(model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}}) =
    model.backend.M_product
tensor(model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}}) =
    model.backend.target

function initial_point(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
    init::Union{Symbol,BuiltinInitializer};
    kwargs...,
)
    backend = model.backend
    M = backend.M_product
    if !isnothing(backend.init_point)
        return backend.init_point(M, init)
    end
    parts =
        ntuple(k -> _component_init(backend.components[k], backend.target, init), backend.r)
    return ArrayPartition(parts...)
end

function initial_point(
    model::JoinModel{<:AbstractFloat,<:JoinBackend},
    init::ALSWarmStartInit;
    verbose::Bool = false,
    kwargs...,
)
    backend = model.backend
    dims = _uniform_segre_dims(backend.components)
    isnothing(dims) && throw(
        ArgumentError(
            "ALSWarmStartInit for a generic JoinModel requires all component manifolds to be Manifolds.Segre with identical factor_dims.",
        ),
    )
    dims == backend.target_shape || throw(
        DimensionMismatch(
            "Uniform Segre factor_dims $dims must match target size $(backend.target_shape) for ALS warm start.",
        ),
    )
    warm_model = JoinModel(backend.target, backend.r; geometry = :canonical)
    p_canonical = initial_point(warm_model, init; verbose, kwargs...)
    return canonical_to_joinpoint(p_canonical, backend.target_shape, backend.r)
end

# Gradient path: always recomputes the ambient reconstruction and marks the
# WORO cache fresh so that the immediately following cost evaluation can reuse it.
function _join_residual_grad!(backend::JoinBackend, p)
    _join_reconstruct!(backend.work_rec, backend, p)
    backend.work_residual .= backend.work_rec .- backend.target_flat
    copyto!(backend.woro.residual, backend.work_residual)
    backend.woro.fresh = true
    return backend.work_residual
end

function _join_residual_cost!(backend::JoinBackend, p)
    if backend.woro.fresh
        # Cost often follows gradient at the same iterate, so reuse the fresh residual once.
        backend.woro.fresh = false
        return backend.woro.residual
    end
    _join_reconstruct!(backend.work_rec, backend, p)
    backend.work_residual .= backend.work_rec .- backend.target_flat
    return backend.work_residual
end

function cost(model::JoinModel{<:AbstractFloat,<:JoinBackend}, p)
    residual = _join_residual_cost!(model.backend, p)
    return 0.5 * sum(abs2, residual)
end

function egrad(model::JoinModel{<:AbstractFloat,<:JoinBackend}, p)
    backend = model.backend
    residual = _join_residual_grad!(backend, p)
    parts = point_parts(p)
    vals =
        ntuple(k -> _component_egrad(backend.components[k], parts[k], residual), backend.r)
    return wrap_like_point(p, vals)
end

function _join_basis_project(components::Tuple, p, residual)
    parts = point_parts(p)
    _check_parts_len(parts, length(components), "_join_basis_project")
    vals = ntuple(k -> begin
        ck = components[k]
        Mk = _component_manifold(ck)
        pk = parts[k]
        eg = _component_egrad(ck, pk, residual)
        egrad_to_rgrad(Mk, pk, eg)
    end, length(components))
    return wrap_like_point(p, vals)
end

supports_rgrad(::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}}) = true
supports_exact_join_basis(::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}}) =
    true
model_exact_native_function(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
) = throw(
    ArgumentError(
        "exact_native is only defined for CPD native geometry models, not $(typeof(model.backend)).",
    ),
)
model_exact_join_basis_function(model::JoinModel{<:AbstractFloat,<:JoinBackend}) =
    (M, p) -> begin
        backend = model.backend
        residual = _join_residual!(backend, p)
        _join_basis_project(backend.components, p, residual)
    end

function extract_components(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
    p,
)
    backend = model.backend
    components = _backend_components(backend)
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "extract_components")
    T = eltype(backend.target)
    N = length(backend.target_shape)
    point_type = Union{}
    manifold_type = Union{}
    @inbounds for k = 1:backend.r
        point_type = typejoin(point_type, typeof(parts[k]))
        manifold_type = typejoin(manifold_type, typeof(_component_manifold(components[k])))
    end
    comps = Vector{DecompositionComponent{T,N,point_type,manifold_type}}(undef, backend.r)
    @inbounds for k = 1:backend.r
        # Components keep only point/manifold metadata and reconstruct derived tensors on demand.
        comps[k] = DecompositionComponent{T,N,point_type,manifold_type}(
            parts[k],
            _component_manifold(components[k]),
            backend.target_shape,
        )
    end
    return comps
end

function rgrad(model::JoinModel{<:AbstractFloat,<:JoinBackend}, p)
    backend = model.backend
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "rgrad")
    residual = _join_residual_grad!(backend, p)
    vals = ntuple(k -> begin
        ck = backend.components[k]
        Mk = _component_manifold(ck)
        pk = parts[k]
        eg = _component_egrad(ck, pk, residual)
        egrad_to_rgrad(Mk, pk, eg)
    end, backend.r)
    return wrap_like_point(p, vals)
end

"""
    _component_ambient_embedding!(out, component, p)

Write the ambient embedding of one join component into `out`.
Component manifolds may specialize this hook when their native point structure
admits a more direct embedding than the generic `embed!` path.
"""
function _component_ambient_embedding!(out::AbstractVector, M, p)
    return _component_ambient_embedding!(out, DefaultJoinEmbedding(), M, p)
end

function _component_ambient_embedding!(out::AbstractVector, ::DefaultJoinEmbedding, M, p)
    ManifoldsBase.embed!(M, out, p)
    return out
end

function _component_ambient_embedding!(
    out::AbstractVector,
    ::DefaultJoinEmbedding,
    M::Manifolds.Tucker,
    p::Manifolds.TuckerPoint,
)
    core = p.hosvd.core
    factors = p.hosvd.U
    reconstruct_tucker!(reshape(out, factor_dims(M)), core, factors)
    return out
end

function _component_ambient_embedding!(out::AbstractVector, M::Manifolds.Segre, p)
    return _component_ambient_embedding!(out, DefaultJoinEmbedding(), M, p)
end

function _component_ambient_embedding!(
    out::AbstractVector,
    ::DefaultJoinEmbedding,
    M::Manifolds.Segre,
    p,
)
    copyto!(out, _segre_component_tensorvec(p))
    return out
end

function _component_ambient_embedding!(out::AbstractVector, M::Manifolds.Tucker, p)
    return _component_ambient_embedding!(out, DefaultJoinEmbedding(), M, p)
end

function _component_ambient_embedding!(
    out::AbstractVector,
    ::DefaultJoinEmbedding,
    M::Manifolds.Tucker,
    p,
)
    throw(
        ArgumentError(
            "Expected native TuckerPoint for Manifolds.Tucker, got $(typeof(p)).",
        ),
    )
end

function _component_ambient_embedding!(out::AbstractVector, component::JoinComponent, p)
    return _component_ambient_embedding!(out, component.embedding, component.manifold, p)
end

"""
    _component_ambient_pushforward!(out, component, p, X)

Write the ambient pushforward `DΦ(p)[X]` of one join component into `out`.
This is the component-level differential used by LM Jacobian assembly on
generic `JoinModel`s.
"""
function _component_ambient_pushforward!(out::AbstractVector, M, p, X)
    return _component_ambient_pushforward!(out, DefaultJoinEmbedding(), M, p, X)
end

function _component_ambient_pushforward!(
    out::AbstractVector,
    ::DefaultJoinEmbedding,
    M,
    p,
    X,
)
    emb = ManifoldsBase.embed(M, p, X)
    length(emb) == length(out) || throw(
        DimensionMismatch(
            "Tangent embedding length $(length(emb)) does not match output length $(length(out)).",
        ),
    )
    copyto!(out, vec(emb))
    return out
end

function _component_ambient_pushforward!(out::AbstractVector, M::Manifolds.Segre, p, X)
    return _component_ambient_pushforward!(out, DefaultJoinEmbedding(), M, p, X)
end

function _component_ambient_pushforward!(
    out::AbstractVector,
    ::DefaultJoinEmbedding,
    M::Manifolds.Segre,
    p,
    X,
)
    copyto!(out, _segre_tangent_tensorvec(p, X))
    return out
end

function _component_ambient_pushforward!(
    out::AbstractVector,
    component::JoinComponent,
    p,
    X,
)
    return _component_ambient_pushforward!(
        out,
        component.embedding,
        component.manifold,
        p,
        X,
    )
end

function _subtract_ambient_tensor!(
    residual::AbstractArray{T,N},
    component,
    p,
    work_vec::AbstractVector{T},
) where {T<:AbstractFloat,N}
    length(work_vec) == length(residual) || throw(
        DimensionMismatch(
            "_subtract_ambient_tensor!: work length $(length(work_vec)) != residual length $(length(residual))",
        ),
    )
    _component_ambient_embedding!(work_vec, component, p)
    residual_vec = vec(residual)
    @inbounds for i in eachindex(residual_vec, work_vec)
        residual_vec[i] -= work_vec[i]
    end
    return residual
end

"""
    _join_reconstruct!(out, backend, p)

Reconstruct the ambient join approximation represented by `p` into `out`.

For component manifolds `M_k` with embeddings
`Phi_k : M_k -> R^n`, the generic join model optimizes

```math
f(p_1, ..., p_r) =
    \\frac{1}{2}\\left\\|\\sum_{k=1}^r \\Phi_k(p_k) - A\\right\\|^2.
```

This backend implements the mathematical core of that model:

- `_join_reconstruct!` computes `sum_k Phi_k(p_k)`.
- `_join_residual!` computes `sum_k Phi_k(p_k) - A`.
- `cost(model, p)` computes `1/2 * ||residual||^2`.
- `_manifold_egrad` applies the adjoint embedding derivative
  `DPhi_k(p_k)'` to the residual for each component.
- `rgrad(model, p)` projects those component gradients to tangent spaces.
- `extract_components(model, p)` converts the optimized component points
  `p_k` into result components.

The method writes each component embedding into preallocated component buffers,
then accumulates those buffers into `out`. This avoids allocating one dense
ambient tensor per component during solver iterations.
"""
function _join_reconstruct!(out::AbstractArray, backend::Union{JoinBackend,BTDBackend}, p)
    components = _backend_components(backend)
    r = backend.r
    bufs = backend.component_bufs

    parts = point_parts(p)
    _check_parts_len(parts, r, "_join_reconstruct")

    fill!(out, zero(eltype(out)))

    @inbounds for k = 1:r
        # Reconstruct each component into its preallocated workspace.
        _component_ambient_embedding!(bufs[k], components[k], parts[k])

        # Accumulate into the output tensor without allocating a Khatri-Rao-sized object.
        out .+= bufs[k]
    end

    return out
end

function _join_residual!(backend::Union{JoinBackend,BTDBackend}, p)
    _join_reconstruct!(backend.work_rec, backend, p)
    backend.work_residual .= backend.work_rec
    backend.work_residual .-= backend.target_flat
    return backend.work_residual
end
