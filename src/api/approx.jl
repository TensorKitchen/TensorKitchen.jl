# api/approx.jl — user-facing generic approximation entry points
export approx

"""
    _join_manifold_init_sym(M, init_sym) -> Bool

Return whether a built-in initializer symbol is supported by a single join
component manifold. Used to validate generic `approx(...; init=:alswarm)`.
"""
@inline function _join_manifold_init_sym(M, init_sym::Symbol)
    if M isa Manifolds.Sphere
        return init_sym in (:random, :deterministic, :target)
    elseif M isa Manifolds.Segre
        return init_sym in (:random, :deterministic)
    elseif M isa Manifolds.Tucker
        return init_sym in (:random, :tucker, :tucker_diag, :sthosvd)
    end
    return init_sym == :random
end

"""
    _validate_warm_init(model, warm_init)

Validate that an ALS warm-start initializer is compatible with every component
of a generic `JoinModel`; throws an `ArgumentError` with per-component guidance.
"""
function _validate_warm_init(model::JoinModel, warm_init)
    backend = model.backend
    backend isa JoinBackend || return nothing
    init_sym = _builtin_initializer_symbol(warm_init)
    for (k, M) in enumerate(backend.manifolds)
        if !_join_manifold_init_sym(M, init_sym)
            throw(
                ArgumentError(
                    "approx(...; init=:alswarm, warm_init=$init_sym) is incompatible with " *
                    "component $k ($(typeof(M))). " *
                    "Choose a warm_init supported by all manifold components (for mixed joins, :random is safest). " *
                    "Supported warm_init for this component: " *
                    (
                        M isa Manifolds.Sphere ? "(:random, :deterministic, :target)" :
                        M isa Manifolds.Segre ? "(:random, :deterministic)" :
                        M isa Manifolds.Tucker ?
                        "(:random, :tucker, :tucker_diag, :sthosvd)" :
                        "(:random)"
                    ),
                ),
            )
        end
    end
    return nothing
end

"""
    approx(model; init=:alswarm, solver=:rgd, ...) -> ApproxResult

Core approximation entry-point. Solves any model that implements the common manifold hooks.
"""
function approx(
    model::JoinModel{T};
    init = :alswarm,
    solver = :rgd,
    maxiter = 500,
    stepsize = 1.0,
    tol = 1e-6,
    gradient_mode = :riemannian,
    verbose = true,
    vector_transport_method = nothing,
    warm_steps = 500,
    warm_init = :random,
    kwargs...,
) where {T<:AbstractFloat}
    init == :alswarm && _validate_warm_init(model, warm_init)
    init_eff = init == :alswarm ? ALSWarmStartInit(warm_steps; base_init = warm_init) : init
    result = _solve_model(
        model;
        init = init_eff,
        solver = solver,
        maxiter = maxiter,
        stepsize = stepsize,
        tol = tol,
        gradient_mode = gradient_mode,
        normalization = NoNormalization(),
        verbose = verbose,
        vector_transport_method = vector_transport_method,
        kwargs...,
    )
    return _to_approx_result(model, result)
end

"""
    approx(manifolds, target; dispatch=:auto, kwargs...) returns a Union{ApproxResult,CPDResult,BTDResult}
    approx(base, r, target; dispatch=:auto, kwargs...) returns a Union{ApproxResult,CPDResult,BTDResult}
    approx(base, target; kwargs...) returns a ApproxResult

Convenience overloads for generic join approximation from manifold specifications:
- `Tuple`/`Vector` of manifolds
- `ProductManifold` (its factors become join components)
- single `base::AbstractManifold` (one component)

By default, `approx` auto-routes by manifold family:
- uniform `Manifolds.Segre` summands calls `cpd(...)`
- uniform `Manifolds.Tucker` summands calls `btd(...)`
- otherwise calls `JoinModel(...)` and returns a `ApproxResult`

For generic mixed joins, components only need to agree on the flattened ambient
length of `target`; each manifold may still use its own native ambient shape.

Use `dispatch=:generic` to force the generic join path, `dispatch=:cpd` to
require CPD routing, or `dispatch=:btd` to require BTD routing.
"""
function approx(
    manifolds::Tuple{Vararg{AbstractManifold}},
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    if dispatch == :cpd || (dispatch == :auto && _all_segre_uniform(manifolds))
        _all_segre_uniform(manifolds) || throw(
            ArgumentError(
                "approx(...; dispatch=:cpd) requires all manifolds to be Manifolds.Segre with identical factor_dims.",
            ),
        )
        return cpd(target, length(manifolds); kwargs...)
    end
    if dispatch == :btd ||
       (dispatch == :auto && _all_tucker_uniform(manifolds, size(target)))
        _all_tucker_uniform(manifolds, size(target)) || throw(
            ArgumentError(
                "approx(...; dispatch=:btd) requires all manifolds to be Manifolds.Tucker with identical factor_dims/multilinear_rank matching the target.",
            ),
        )
        return btd(target, length(manifolds), multilinear_rank(first(manifolds)); kwargs...)
    end
    return approx(JoinModel(manifolds, target); kwargs...)
end

"""
    approx(manifolds::AbstractVector, target; dispatch=:auto, kwargs...)

Vector-manifold overload; preserves the same routing rules as the tuple overload
while accepting dynamically assembled component lists.
"""
function approx(
    manifolds::AbstractVector,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    if dispatch == :cpd || (dispatch == :auto && _all_segre_uniform(manifolds))
        _all_segre_uniform(manifolds) || throw(
            ArgumentError(
                "approx(...; dispatch=:cpd) requires all manifolds to be Manifolds.Segre with identical factor_dims.",
            ),
        )
        return cpd(target, length(manifolds); kwargs...)
    end
    if dispatch == :btd ||
       (dispatch == :auto && _all_tucker_uniform(manifolds, size(target)))
        _all_tucker_uniform(manifolds, size(target)) || throw(
            ArgumentError(
                "approx(...; dispatch=:btd) requires all manifolds to be Manifolds.Tucker with identical factor_dims/multilinear_rank matching the target.",
            ),
        )
        return btd(target, length(manifolds), multilinear_rank(first(manifolds)); kwargs...)
    end
    return approx(JoinModel(manifolds, target); kwargs...)
end

"""
    approx(M::ProductManifold, target; dispatch=:auto, kwargs...)

Use the factors of a product manifold as join components and route to CPD, BTD,
or the generic join solver according to `dispatch`.
"""
function approx(
    M::ProductManifold,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    mfs = Tuple(M.manifolds)
    if dispatch == :cpd || (dispatch == :auto && _all_segre_uniform(mfs))
        _all_segre_uniform(mfs) || throw(
            ArgumentError(
                "approx(...; dispatch=:cpd) requires all manifolds to be Manifolds.Segre with identical factor_dims.",
            ),
        )
        return cpd(target, length(mfs); kwargs...)
    end
    if dispatch == :btd || (dispatch == :auto && _all_tucker_uniform(mfs, size(target)))
        _all_tucker_uniform(mfs, size(target)) || throw(
            ArgumentError(
                "approx(...; dispatch=:btd) requires all manifolds to be Manifolds.Tucker with identical factor_dims/multilinear_rank matching the target.",
            ),
        )
        return btd(target, length(mfs), multilinear_rank(first(mfs)); kwargs...)
    end
    return approx(JoinModel(M, target); kwargs...)
end

"""
    approx(base::Manifolds.Segre, r, target; dispatch=:auto, kwargs...)

Build a rank-`r` Segre join and route to the CPD pipeline unless generic
dispatch is explicitly requested.
"""
function approx(
    base::Manifolds.Segre,
    r::Int,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    return dispatch in (:auto, :cpd) ? cpd(target, r; kwargs...) :
           approx(JoinModel(base, r, target); kwargs...)
end

"""
    approx(base::Manifolds.Tucker, r, target; dispatch=:auto, kwargs...)

Build a `r`-block Tucker join and route to the BTD pipeline unless generic
dispatch is explicitly requested.
"""
function approx(
    base::Manifolds.Tucker,
    r::Int,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    return dispatch in (:auto, :btd) ? btd(target, r, multilinear_rank(base); kwargs...) :
           approx(JoinModel(base, r, target); kwargs...)
end

"""
    approx(base::AbstractManifold, r, target; dispatch=:auto, kwargs...)

Fallback rank-`r` join constructor for non-specialized manifolds. Forced CPD or
BTD dispatch is rejected because the base manifold family is not known.
"""
function approx(
    base::AbstractManifold,
    r::Int,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    dispatch == :cpd &&
        throw(ArgumentError("approx(...; dispatch=:cpd) requires Manifolds.Segre inputs."))
    dispatch == :btd &&
        throw(ArgumentError("approx(...; dispatch=:btd) requires Manifolds.Tucker inputs."))
    return approx(JoinModel(base, r, target); kwargs...)
end

"""
    approx(base::AbstractManifold, target; dispatch=:auto, kwargs...)

Single-component generic approximation fallback. Use this when no CPD/BTD
family-specific routing is intended.
"""
function approx(
    base::AbstractManifold,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    dispatch = _normalize_approx_dispatch(dispatch)
    dispatch == :cpd &&
        throw(ArgumentError("approx(...; dispatch=:cpd) requires Manifolds.Segre inputs."))
    dispatch == :btd &&
        throw(ArgumentError("approx(...; dispatch=:btd) requires Manifolds.Tucker inputs."))
    return approx(JoinModel(base, target); kwargs...)
end

"""
    approx(manifolds, target; kwargs...)

Error-reporting fallback for unsupported manifold specifications.
"""
function approx(manifolds, target::AbstractArray{T,N}; kwargs...) where {T<:AbstractFloat,N}
    throw(
        ArgumentError(
            "Unsupported manifolds specification $(typeof(manifolds)) for approx(). " *
            "Use one of: Tuple{Vararg{AbstractManifold}}, AbstractVector of manifolds, " *
            "ProductManifold, AbstractManifold (single component), or " *
            "(base::AbstractManifold, r::Int, target).",
        ),
    )
end
