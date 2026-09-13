# btd/model.jl — BTD backend and model hooks

mutable struct _WorkspaceTensorCache{T,N}
    dims::Vector{NTuple{N,Int}}
    bufs::Vector{Array{T,N}}
end

_WorkspaceTensorCache{T,N}() where {T,N} =
    _WorkspaceTensorCache{T,N}(NTuple{N,Int}[], Array{T,N}[])

mutable struct BTDContractionWorkspace{T,N}
    tensor_slot1::_WorkspaceTensorCache{T,N}
    tensor_slot2::_WorkspaceTensorCache{T,N}
    perm_in::_WorkspaceTensorCache{T,N}
    perm_out::_WorkspaceTensorCache{T,N}
    persist::_WorkspaceTensorCache{T,N}
end

BTDContractionWorkspace{T,N}() where {T,N} = BTDContractionWorkspace{T,N}(
    _WorkspaceTensorCache{T,N}(),
    _WorkspaceTensorCache{T,N}(),
    _WorkspaceTensorCache{T,N}(),
    _WorkspaceTensorCache{T,N}(),
    _WorkspaceTensorCache{T,N}(),
)

"""
    BTDBackend

Backend state for block-term decomposition as a sum of Tucker blocks. Stores
the target tensor, product manifold, contraction workspace, and target norm.
Ambient reconstruction buffers are allocated only by legacy operations that
explicitly request an ambient tensor.
"""
struct BTDBackend{
    T,
    N,
    CT<:Tuple,
    MT<:Tuple,
    A<:AbstractArray{T,N},
    MP<:ProductManifold,
    I,
    S<:Real,
} <: AbstractJoinBackend
    components::CT
    manifolds::MT
    r::Int
    # Preserve the target array/backend without conversion or copying.
    target::A
    target_shape::NTuple{N,Int}
    M_product::MP
    init_point::I
    workspace::BTDContractionWorkspace{T,N}
    target_normsq::S
end

egrad(model::JoinModel{<:AbstractFloat,<:BTDBackend}, p) = _btd_egrad(model.backend, p)
supports_rgrad(::JoinModel{<:AbstractFloat,<:BTDBackend}) = true
supports_exact_join_basis(::JoinModel{<:AbstractFloat,<:BTDBackend}) = true
model_exact_native_function(model::JoinModel{<:AbstractFloat,<:BTDBackend}) = throw(
    ArgumentError(
        "exact_native is only defined for CPD native geometry models, not $(typeof(model.backend)).",
    ),
)
model_exact_join_basis_function(model::JoinModel{<:AbstractFloat,<:BTDBackend}) =
    (_, p) -> rgrad(model, p)

function _require_materialized_btd_ambient_target(
    backend::BTDBackend,
    operation::AbstractString,
)
    is_materialized(backend.target) && return nothing
    throw(
        ArgumentError(
            "$operation requires an explicit materialized tensor for BTD. " *
            "Call materialize_tensor(A) before constructing the model, or use the " *
            "observation-preserving BTD cost, gradient, and projected ALS paths.",
        ),
    )
end

"""Copy a materialized BTD target for an explicitly ambient operation."""
function _btd_ambient_target_copy(backend::BTDBackend)
    _require_materialized_btd_ambient_target(backend, "BTD structured initialization")
    target_flat = vec(backend.target)
    residual_flat = _join_vector_workspace_like(backend.target, length(target_flat))
    copyto!(residual_flat, target_flat)
    return reshape(residual_flat, backend.target_shape)
end

"""Allocate one operation-local ambient vector for a legacy BTD operation."""
function _btd_ambient_work_vector(backend::BTDBackend)
    _require_materialized_btd_ambient_target(backend, "BTD ambient operation")
    return _join_vector_workspace_like(backend.target, length(backend.target))
end

function _btd_sequential_tucker_init(model::JoinModel{<:AbstractFloat,<:BTDBackend}, init)
    backend = model.backend
    # Initialize Tucker blocks on the current residual rather than cloning the
    # same full-tensor fit into every block.
    residual = _btd_ambient_target_copy(backend)
    component_buf = _btd_ambient_work_vector(backend)
    parts = Vector{Manifolds.TuckerPoint{eltype(backend.target)}}(undef, backend.r)
    for k = 1:backend.r
        component = _backend_component(backend, k)
        Mk = _backend_manifold(backend, k)
        pk = _component_init(component, residual, init)
        parts[k] = pk
        _subtract_ambient_tensor!(residual, Mk, pk, component_buf)
    end
    return ArrayPartition(parts...)
end

function _btd_block_ranks_by_mode(backend::BTDBackend{T,N}) where {T,N}
    ranks = Vector{NTuple{N,Int}}(undef, backend.r)
    for b = 1:backend.r
        M = _backend_manifold(backend, b)
        M isa Manifolds.Tucker || throw(
            ArgumentError(
                "BTD HOSVD multistart expects Tucker manifolds, got $(typeof(M)) at block $b.",
            ),
        )
        ranks[b] = multilinear_rank(M)
    end
    return ranks
end

function _btd_hosvd_subspaces(backend::BTDBackend{T,N}, ranks_by_block) where {T,N}
    A = backend.target
    return ntuple(N) do mode
        total_rank = sum(r[mode] for r in ranks_by_block)
        kept_rank = min(size(A, mode), total_rank)
        U, _, _ = svd(unfold_mode(A, mode))
        Matrix(@view U[:, 1:kept_rank])
    end
end

function _btd_split_columns(
    rng::AbstractRNG,
    total::Int,
    ranks::Vector{Int},
    candidate::Int,
)
    total >= maximum(ranks) || throw(
        DimensionMismatch(
            "BTD HOSVD split has only $total basis vectors for block rank $(maximum(ranks)).",
        ),
    )
    order = candidate == 1 ? collect(1:total) : randperm(rng, total)
    cols = Vector{Vector{Int}}(undef, length(ranks))
    offset = 0
    for b in eachindex(ranks)
        rb = ranks[b]
        cols[b] = [order[mod1(offset + j, total)] for j = 1:rb]
        offset += rb
    end
    return cols
end

function _btd_project_core(A::AbstractArray{T,N}, factors) where {T<:AbstractFloat,N}
    core = A
    for mode = 1:N
        core = mode_n_product(core, factors[mode]', mode)
    end
    return core
end

function _btd_hosvd_split_candidate(
    rng::AbstractRNG,
    backend::BTDBackend{T,N},
    ranks_by_block,
    subspaces,
    candidate::Int,
) where {T<:AbstractFloat,N}
    columns_by_mode = ntuple(N) do mode
        ranks_m = [ranks_by_block[b][mode] for b = 1:backend.r]
        _btd_split_columns(rng, size(subspaces[mode], 2), ranks_m, candidate)
    end

    residual = _btd_ambient_target_copy(backend)
    component_buf = _btd_ambient_work_vector(backend)
    # Candidate blocks are always Tucker points here
    parts = Vector{Manifolds.TuckerPoint{T}}(undef, backend.r)
    for b = 1:backend.r
        factors =
            ntuple(mode -> Matrix(@view subspaces[mode][:, columns_by_mode[mode][b]]), N)
        core = _btd_project_core(residual, factors)
        pk = Manifolds.TuckerPoint(core, factors...)
        parts[b] = pk
        _subtract_ambient_tensor!(
            residual,
            _backend_manifold(backend, b),
            pk,
            component_buf,
        )
    end
    return ArrayPartition(parts...)
end

"""
    initial_point(model::JoinModel{<:BTDBackend}, init; verbose=false)

Create a BTD initial point from symbols or builtin initializer objects,
including sequential Tucker initialization and multistart variants.
"""
function initial_point(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    init::Union{Symbol,BuiltinInitializer};
    verbose::Bool = false,
)
    init == :alswarm && return initial_point(model, BTDALSWarmStartInit(); verbose)
    init == :hosvd_multistart &&
        return initial_point(model, BTDHOSVDMultistartInit(); verbose)
    backend = model.backend
    M = backend.M_product
    if !isnothing(backend.init_point)
        return backend.init_point(M, init)
    end
    init_sym = _builtin_initializer_symbol(init)
    if init_sym == :random
        parts = ntuple(
            k -> _component_init(_backend_component(backend, k), backend.target, init),
            backend.r,
        )
        return ArrayPartition(parts...)
    end

    return _btd_sequential_tucker_init(model, init)
end

function initial_point(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    init::BTDALSWarmStartInit;
    verbose::Bool = false,
)
    backend = model.backend
    p_base = initial_point(model, init.base_init; verbose)
    warm = fit_btd_als(
        backend.target,
        backend;
        p0 = p_base,
        maxiter = init.nsteps,
        tol = 0.0,
        block_method = init.block_method,
        block_maxiter = init.block_maxiter,
        verbose,
        return_stats = true,
        progress_phase = :initialization,
    )
    return warm.point
end

function initial_point(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    init::BTDHOSVDMultistartInit;
    verbose::Bool = false,
)
    backend = model.backend
    ranks_by_block = _btd_block_ranks_by_mode(backend)
    subspaces = _btd_hosvd_subspaces(backend, ranks_by_block)
    rng = isnothing(init.seed) ? Random.default_rng() : MersenneTwister(init.seed)
    best_point = nothing
    best_cost = Inf
    split_candidate = 1

    for c = 1:init.candidates
        p_candidate = if init.include_sequential && c == 1
            _btd_sequential_tucker_init(model, :sthosvd)
        else
            p = _btd_hosvd_split_candidate(
                rng,
                backend,
                ranks_by_block,
                subspaces,
                split_candidate,
            )
            split_candidate += 1
            p
        end

        p_screened = if init.screening_steps > 0
            fit_btd_als(
                backend.target,
                backend;
                p0 = p_candidate,
                maxiter = init.screening_steps,
                tol = 0.0,
                block_method = init.block_method,
                block_maxiter = init.block_maxiter,
                verbose,
                return_stats = true,
                progress_phase = :initialization,
            ).point
        else
            p_candidate
        end

        cst = cost(model, p_screened)
        if cst < best_cost
            best_cost = cst
            best_point = p_screened
        end
    end

    return best_point
end


function rgrad(model::JoinModel{<:AbstractFloat,<:BTDBackend}, p)
    backend = model.backend
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "BTD rgrad")
    eg = point_parts(_btd_egrad(backend, p))
    vals = ntuple(
        k -> egrad_to_rgrad(_backend_manifold(backend, k), parts[k], eg[k]),
        backend.r,
    )
    return wrap_like_point(p, vals)
end
