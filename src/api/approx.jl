# api/approx.jl — user-facing generic approximation entry points
export approx

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
                        "(:random, :tucker, :tucker_diag, :sthosvd)" : "(:random)"
                    ),
                ),
            )
        end
    end
    return nothing
end

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
## Generic Join Approximation

`approx(...)` is the main frontend for join decomposition, which works in two stages:

1. build an initial point
2. refine it with the selected solver

### Supported Forms
* `approx(model; kwargs...)` : Builds a generic join model and routes to the generic join solver. Model can be any type that can be converted to a JoinModel, such as a tuple of manifolds, a ProductManifold, or a single manifold.
   - Example:
    ```julia
    approx(JoinModel(Manifolds.Sphere(2), , target); kwargs...) 
    ```
* `approx(manifolds, target; kwargs...)` : Builds a generic join model from a tuple of existing manifolds and routes to the generic join solver. 
   - Example:
    ```julia
    approx((Manifolds.Segre(2, 3), Manifolds.Segre(2, 3)), target; kwargs...)
    ```
* `approx(M::ProductManifold, target; kwargs...)` : Uses the factors of a product manifold as join components and route to CPD, BTD,
or the generic join solver according to `dispatch`. 
   - Example:
    ```julia
    approx(ProductManifold(Manifolds.Segre(2, 3), Manifolds.Segre(2, 3)), target; kwargs...)
    approx(ProductManifold(Manifolds.Tucker(2, 3), Manifolds.Tucker(2, 3)), target; kwargs...)
    ```
* `approx(base, r, target; kwargs...)` : Builds a rank-`r` Segre join and route to the CPD pipeline unless generic
dispatch is explicitly requested. 
   - Example:
    ```julia
    approx(Manifolds.Segre(2, 3), 3, target; kwargs...)
    ```
* `approx(base, target; kwargs...)` : Builds a single-component generic join and route to the generic join solver.
   - Example:
    ```julia
    approx(Manifolds.Sphere(2), target; kwargs...)
    ```
* By default, `approx` auto-routes by manifold family:
    - uniform `Manifolds.Segre` summands calls `cpd(...)`
    - uniform `Manifolds.Tucker` summands calls `btd(...)`
    - otherwise calls `JoinModel(...)` and returns a `ApproxResult`

### Return Types

* Depending on the manifold family, `approx(...)` may return:
    - `ApproxResult` for the generic join path
    - `CPDResult` when auto-routed to `cpd(...)`
    - `BTDResult` when auto-routed to `btd(...)`

### Main Options

For the generic join path:
* `init = :alswarm`: Sets the algorithm to find the initial point. Runs ALS first and uses the result as the initial point for refinement.
    - Important: When init = :alswarm, warm_init must be supported by every component manifold
    - For mixed joins, `warm_init = :random` is usually the safest choice
* `solver = :rgd`: Sets the algorithm for refinement. Possible options are:
    - `rgd` (default): Riemannian gradient descent
    - `rgd_fixed`: Riemannian gradient descent with fixed step size
    - `rcg`: Riemannian conjugate gradient
    - `lbfgs`: Limited-memory quasi-Newton
    - `als`: Alternating Least Squares

## Extended Options

* `p0 = nothing`: Explicit initial point. If provided, it overrides the default initial point.
* `:alswarm`: ALS warm start option.
    - `warm_init = TuckerInit()`: Before finding the warm start initial point, this sets the good starting point for ALS.
    - `warm_steps = 500`: Once finding the best initial point from warm_init, it runs this many ALS iterations to refine the initial point.
* `dispatch=:generic` to force the generic join path
* `dispatch=:cpd` to require CPD routing
* `dispatch=:btd` to require BTD routing

##Notes##
* `:als` is not a solver option for `approx(...)`. However, if `approx(...)` auto-routes to `cpd(...)` or `btd(...)`, then those specialized pipelines may support ALS separately.
* For generic mixed joins, use manifold solvers such as `:rgd`, `:rcg`, or `:lbfgs` instead of `:als`.
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
