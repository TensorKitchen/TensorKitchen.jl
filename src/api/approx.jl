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

@doc """
    approx(model; kwargs...)
    approx(manifolds, target; dispatch=:auto, kwargs...)
    approx(product_manifold, target; dispatch=:auto, kwargs...)
    approx(base, component_count, target; dispatch=:auto, kwargs...)
    approx(base, target; dispatch=:auto, kwargs...)

Approximate `target` with components described by one or more manifolds, or
solve an existing `JoinModel`.

# Inputs

- `manifolds`: one manifold or a collection of allowed component manifolds.
- `product_manifold`: a `ProductManifold` whose factors are the components.
- `base`: one component manifold used once or repeated `component_count` times.
- `component_count`: positive number of repeated components.
- `target`: vector or tensor to approximate.
- `model`: an existing `JoinModel` containing both the structure and target.

# Output

Returns an [`ApproxResult`](@ref) for a general join. A compatible collection
of CP or Tucker components may return a specialized `CPDResult` or `BTDResult`.
Use `components`, `reconstruct`, and `rel_error(target, result)`
to inspect the fit.

# Common options

- `dispatch=:auto` routes uniform Segre components to `cpd`, uniform compatible
  Tucker components to `btd`, and other component families to the general Join
  solver. Use `:generic`, `:cpd`, or `:btd` to request a compatible route.
- For the general Join path, `init=:random`, `solver=:rgd`, `maxiter=500`,
  `stepsize=1.0`, `tol=1e-6`, `p0=nothing`, and `verbose=true` are the current
  defaults. Use `p0` to supply an explicit packed starting point.
- `gradient_mode=:riemannian` selects the intrinsic gradient supplied to the
  manifold optimizer. `vector_transport_method=nothing` uses the default
  transport associated with the selected retraction.
- The general Join path supports `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, and
  `:lm`; generic `:als` is not available. A specialized CPD or BTD route
  accepts the options of that pipeline instead.
- Supported generic initializers depend on the component family: Sphere
  components accept `:random`, `:deterministic`, or `:target`; Segre components
  accept `:random` or `:deterministic`; Tucker components accept `:random`,
  `:tucker`, `:tucker_diag`, or `:sthosvd`.
- `target` must have a floating-point element type.

# Example

```julia
using TensorKitchen, Manifolds

target = [1.2, 0.4]
circle = Sphere(1)
result = approx((circle, circle), target; verbose = false)
target_approx = reconstruct(result)
```
""" approx
function _approx_manifold_collection(
    dispatch::AutoApproxDispatch,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    _all_segre_uniform(manifolds) && return _approx_manifold_collection(
        CPDApproxDispatch(),
        manifolds,
        target;
        kwargs...,
    )
    _all_tucker_uniform(manifolds, size(target)) && return _approx_manifold_collection(
        BTDApproxDispatch(),
        manifolds,
        target;
        kwargs...,
    )
    return _approx_manifold_collection(
        GenericApproxDispatch(),
        manifolds,
        target;
        kwargs...,
    )
end

function _approx_manifold_collection(
    ::CPDApproxDispatch,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    _all_segre_uniform(manifolds) || throw(
        ArgumentError(
            "approx(...; dispatch=:cpd) requires all manifolds to be Manifolds.Segre with identical factor_dims.",
        ),
    )
    return cpd(target, length(manifolds); kwargs...)
end

function _approx_manifold_collection(
    ::BTDApproxDispatch,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    _all_tucker_uniform(manifolds, size(target)) || throw(
        ArgumentError(
            "approx(...; dispatch=:btd) requires all manifolds to be Manifolds.Tucker with identical factor_dims/multilinear_rank matching the target.",
        ),
    )
    return btd(target, length(manifolds), multilinear_rank(first(manifolds)); kwargs...)
end

function _approx_manifold_collection(
    ::GenericApproxDispatch,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    return approx(JoinModel(manifolds, target); kwargs...)
end

function approx(
    manifolds::Tuple{Vararg{AbstractManifold}},
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    return _approx_manifold_collection(
        approx_dispatch(dispatch),
        manifolds,
        target;
        kwargs...,
    )
end

function approx(
    manifolds::AbstractVector,
    target::AbstractArray{T,N};
    dispatch::Symbol = :auto,
    kwargs...,
) where {T<:AbstractFloat,N}
    return _approx_manifold_collection(
        approx_dispatch(dispatch),
        manifolds,
        target;
        kwargs...,
    )
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
    mfs = Tuple(M.manifolds)
    return _approx_product_manifold(approx_dispatch(dispatch), M, mfs, target; kwargs...)
end

function _approx_product_manifold(
    dispatch::Union{AutoApproxDispatch,CPDApproxDispatch,BTDApproxDispatch},
    M::ProductManifold,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    return _approx_manifold_collection(dispatch, manifolds, target; kwargs...)
end

function _approx_product_manifold(
    ::GenericApproxDispatch,
    M::ProductManifold,
    manifolds,
    target::AbstractArray;
    kwargs...,
)
    return approx(JoinModel(M, target); kwargs...)
end

_approx_segre_rank(
    ::Union{AutoApproxDispatch,CPDApproxDispatch},
    base::Manifolds.Segre,
    r::Int,
    target::AbstractArray;
    kwargs...,
) = cpd(target, r; kwargs...)

_approx_segre_rank(
    ::AbstractApproxDispatch,
    base::Manifolds.Segre,
    r::Int,
    target::AbstractArray;
    kwargs...,
) = approx(JoinModel(base, r, target); kwargs...)

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
    return _approx_segre_rank(approx_dispatch(dispatch), base, r, target; kwargs...)
end

_approx_tucker_rank(
    ::Union{AutoApproxDispatch,BTDApproxDispatch},
    base::Manifolds.Tucker,
    r::Int,
    target::AbstractArray;
    kwargs...,
) = btd(target, r, multilinear_rank(base); kwargs...)

_approx_tucker_rank(
    ::AbstractApproxDispatch,
    base::Manifolds.Tucker,
    r::Int,
    target::AbstractArray;
    kwargs...,
) = approx(JoinModel(base, r, target); kwargs...)

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
    return _approx_tucker_rank(approx_dispatch(dispatch), base, r, target; kwargs...)
end

_reject_generic_rank_dispatch(::Union{AutoApproxDispatch,GenericApproxDispatch}) = nothing

function _reject_generic_rank_dispatch(::CPDApproxDispatch)
    throw(ArgumentError("approx(...; dispatch=:cpd) requires Manifolds.Segre inputs."))
end

function _reject_generic_rank_dispatch(::BTDApproxDispatch)
    throw(ArgumentError("approx(...; dispatch=:btd) requires Manifolds.Tucker inputs."))
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
    _reject_generic_rank_dispatch(approx_dispatch(dispatch))
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
    _reject_generic_rank_dispatch(approx_dispatch(dispatch))
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
