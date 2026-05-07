# api/approx.jl — user-facing generic approximation entry points
export approx

function approx(
    model::JoinModel{T};
    init = :random,
    solver = :rgd,
    maxiter = 500,
    stepsize = 1.0,
    tol = 1e-6,
    gradient_mode = :riemannian,
    verbose = true,
    vector_transport_method = nothing,
    kwargs...,
) where {T<:AbstractFloat}
    result = _solve_model(
        model;
        init = init,
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

* `approx(model; kwargs...)` : It is for an already constructed join model (`JoinModel(...)`) and routes to the generic join solver. Model can be a `JoinModel` of a tuple of manifolds, a `ProductManifold`, or a single manifold.
    - model means an already constructed JoinModel.
    - It fully fixes the decomposition structure and target.
    - `approx(model; ...)` just solves that model.
    Example: If you want to approximate a point on the sphere, you can build a `JoinModel` and then use `approx` to refine it.

```julia
target = [1.2, 0.4, -0.3]
model = JoinModel(Manifolds.Sphere(2), target)
approx(model; maxiter = 100, verbose = false)
# returns an ApproxResult
```

* `approx(manifolds, target; kwargs...)` : Builds a generic join model from a tuple of existing manifolds and routes to according to dispatch.
    Example:

```julia
target = randn(2, 3)
approx((Manifolds.Segre((2, 3)), Manifolds.Segre((2, 3))), target; verbose = false)
# This builds a join with 2 copies of Manifolds.Segre((2, 3))".
```  
    
* `approx(M::ProductManifold, target; kwargs...)` : Uses the factors of a product manifold and route to CPD, BTD, or the generic join solver according to `dispatch`. 
    - M::ProductManifold means that you already have a product manifold whose factors are the join components.
    - approx(M, target; ...) uses those factors directly.
    - It is more explicit than `base`, because the components are already listed.
    
   Example:

```julia
# Example 1
target = randn(2, 3)
M = ProductManifold(Manifolds.Segre((2, 3)), Manifolds.Segre((2, 3)))
approx(M, target; verbose = false)
# returns an CPDResult

# Example 2
target = randn(4, 3, 2)
M = ProductManifold(
    Manifolds.Tucker((4, 3, 2), (2, 2, 2)),
    Manifolds.Tucker((4, 3, 2), (2, 2, 2)),
)
approx(M, target; verbose = false)
# returns an BTDResult
```

* `approx(base, r, target; kwargs...)` : builds a rank-r Segre join and routes to CPD when base isa Manifolds.Segre and BTD when base isa Manifolds.Tucker unless generic
dispatch is explicitly requested. 
    - base means one manifold template, not yet a full join.
    - `approx(base, r, target; ...)` repeats that same manifold r times to build a join.
    - `approx(base, target; ...)` builds a one-component join.

```julia
target = randn(2, 3)
approx(Manifolds.Segre((2, 3)), 2, target; verbose = false)
# returns an CPDResult
```

* `approx(base, target; kwargs...)` : Builds a single-component generic join and route to the generic join solver.
    Example:

```julia
target = [1.2, 0.4, -0.3]   
approx(Manifolds.Sphere(2), target; verbose = false)
# returns an ApproxResult
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
* `init = :random`: Sets the algorithm to find the initial point.
* `solver = :rgd`: Sets the algorithm for refinement. Possible options are:
    - `rgd` (default): Riemannian gradient descent
    - `rgd_fixed`: Riemannian gradient descent with fixed step size
    - `rcg`: Riemannian conjugate gradient
    - `lbfgs`: Limited-memory quasi-Newton

##Notes##
* `:als` is not a solver option for `approx(...)`. However, if `approx(...)` auto-routes to `cpd(...)` or `btd(...)`, then those specialized pipelines may support ALS separately.
* `warm_steps` and `warm_init` are not part of the generic `approx(...)` path. Generic joins start from random initial point and then use manifold solvers for refinement.
* For generic mixed joins, use manifold solvers such as `:rgd`, `:rcg`, or `:lbfgs`.
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
