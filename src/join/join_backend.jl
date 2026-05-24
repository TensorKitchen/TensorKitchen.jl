# join/join_backend.jl — Generic join and BTD backend implementation

_is_manifold_like(::AbstractManifold) = true
_is_manifold_like(_) = false

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

function _manifold_init(M, target, init_sym::Symbol)
    init_sym == :random && return rand(M)
    throw(
        ArgumentError(
            "No default init for manifold $(typeof(M)). Use init=:random or provide init_point.",
        ),
    )
end

"""
    _manifold_egrad(M, p, residual)

Compute the component Euclidean gradient induced by a join residual. Tucker
components use their native tensor gradient; other components copy the residual.
"""
_manifold_egrad(M, p, residual) = copy(residual)
_manifold_egrad(M::Manifolds.Tucker, p, residual) = _tucker_egrad(M, p, residual)

function _manifold_egrad(M::Manifolds.Segre, p, residual)
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
function _validate_join_ambient_compatibility(manifolds::Tuple, target::AbstractArray)
    target_len = length(target)
    failures = String[]
    @inbounds for k in eachindex(manifolds)
        mk = ambient_length(manifolds[k])
        if mk != target_len
            push!(
                failures,
                "[$k] $(typeof(manifolds[k])) has ambient length $mk but target has length $target_len",
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
    manifolds,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    r = length(manifolds)
    # Keep the original target representation instead of eagerly materializing Array.
    tgt = target
    _validate_join_ambient_compatibility(manifolds, tgt)

    tflat = vec(tgt)
    tgt_len = length(tgt)
    # One ambient buffer per component lets reconstruction reuse storage across iterations.
    component_bufs = [_join_vector_workspace_like(tgt, tgt_len) for _ in eachindex(manifolds)]
    work_rec = _join_vector_workspace_like(tgt, tgt_len)
    work_residual = _join_vector_workspace_like(tgt, tgt_len)

    return (;
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
    manifolds,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    parts = _sum_backend_parts(manifolds, target; init_point)
    return JoinBackend(
        parts.manifolds,
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
    manifolds,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    parts = _sum_backend_parts(manifolds, target; init_point)
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
    manifolds::Tuple{Vararg{AbstractManifold}},
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    r = length(manifolds)
    r >= 1 || throw(ArgumentError("JoinModel(manifolds, target) needs r >= 1, got r=$r"))
    b = _sum_backend_instance(JoinBackend, manifolds, target; init_point)
    return JoinModel{T,typeof(b)}(b)
end

"""
    JoinModel(base, r, target; init_point=nothing)

Construct a join model by repeating a base manifold `r` times.
"""
function JoinModel(
    manifolds::AbstractVector,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(_as_join_manifold_tuple(manifolds), target; init_point)
end

function JoinModel(
    M::ProductManifold,
    target::AbstractArray{T,N};
    init_point = nothing,
) where {T<:AbstractFloat,N}
    return JoinModel(_as_join_manifold_tuple(M), target; init_point)
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
    base::AbstractManifold,
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
        ntuple(k -> _manifold_init(backend.manifolds[k], backend.target, init), backend.r)
    return ArrayPartition(parts...)
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
    vals = ntuple(k -> _manifold_egrad(backend.manifolds[k], parts[k], residual), backend.r)
    return wrap_like_point(p, vals)
end

function _join_basis_project(manifolds::Tuple, p, residual)
    parts = point_parts(p)
    _check_parts_len(parts, length(manifolds), "_join_basis_project")
    vals = ntuple(k -> begin
        Mk = manifolds[k]
        pk = parts[k]
        eg = _manifold_egrad(Mk, pk, residual)
        egrad_to_rgrad(Mk, pk, eg)
    end, length(manifolds))
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
        _join_basis_project(backend.manifolds, p, residual)
    end

function extract_components(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
    p,
)
    backend = model.backend
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "extract_components")
    T = eltype(backend.target)
    N = length(backend.target_shape)
    point_type = Union{}
    manifold_type = Union{}
    @inbounds for k in eachindex(parts)
        point_type = typejoin(point_type, typeof(parts[k]))
        manifold_type = typejoin(manifold_type, typeof(backend.manifolds[k]))
    end
    comps = Vector{DecompositionComponent{T,N,point_type,manifold_type}}(undef, backend.r)
    @inbounds for k in eachindex(comps)
        # Components keep only point/manifold metadata and reconstruct derived tensors on demand.
        comps[k] = DecompositionComponent{T,N,point_type,manifold_type}(
            parts[k],
            backend.manifolds[k],
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
        Mk = backend.manifolds[k]
        pk = parts[k]
        eg = _manifold_egrad(Mk, pk, residual)
        egrad_to_rgrad(Mk, pk, eg)
    end, backend.r)
    return wrap_like_point(p, vals)
end

function _ambient_vector!(out::AbstractVector, M, p)
    ManifoldsBase.embed!(M, out, p)
    return out
end

function _ambient_vector!(
    out::AbstractVector,
    M::Manifolds.Tucker,
    p::Manifolds.TuckerPoint,
)
    core = p.hosvd.core
    factors = p.hosvd.U
    reconstruct_tucker!(reshape(out, factor_dims(M)), core, factors)
    return out
end

function _ambient_vector!(out::AbstractVector, M::Manifolds.Tucker, p)
    throw(
        ArgumentError(
            "Expected native TuckerPoint for Manifolds.Tucker, got $(typeof(p)).",
        ),
    )
end

function _subtract_ambient_tensor!(
    residual::AbstractArray{T,N},
    M,
    p,
    work_vec::AbstractVector{T},
) where {T<:AbstractFloat,N}
    length(work_vec) == length(residual) || throw(
        DimensionMismatch(
            "_subtract_ambient_tensor!: work length $(length(work_vec)) != residual length $(length(residual))",
        ),
    )
    _ambient_vector!(work_vec, M, p)
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
    manifolds = backend.manifolds
    r = backend.r
    bufs = backend.component_bufs

    parts = point_parts(p)
    _check_parts_len(parts, r, "_join_reconstruct")

    fill!(out, zero(eltype(out)))

    @inbounds for k in eachindex(parts)
        # Reconstruct each component into its preallocated workspace.
        _ambient_vector!(bufs[k], manifolds[k], parts[k])

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
