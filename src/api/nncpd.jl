# api/nncpd.jl — user-facing nonnegative CP decomposition entry points
export nncpd

"""
    nncpd(A; r=nothing, kwargs...) -> CPDResult

Nonnegative CP decomposition. If `r` is omitted, uses the same rank heuristic as
`cpd(A; r=nothing, ...)`.
"""
function nncpd(
    A::AbstractArray{T,N};
    r::Union{Int,Nothing} = nothing,
    kwargs...,
) where {T<:AbstractFloat,N}
    dims = size(A)
    r_eff = r === nothing ? max(1, minimum(dims)) : r
    if r === nothing && get(kwargs, :verbose, true)
        println(
            "Rank not specified. Using heuristic r=$r_eff. Pass r explicitly to control model complexity.",
        )
    end
    return nncpd(A, r_eff; kwargs...)
end

"""
    nncpd(A, r; kwargs...) -> CPDResult

Nonnegative CP decomposition with its own frontend defaults.

Key options:
- `solver=:rgd`, `geometry=:softplus_metric` is the default manifold route.
- `solver=:als` uses the canonical nonnegative ALS route and ignores manifold
  geometry.
- `warm_steps=500` enables an ALS warm start before manifold refinement.

Geometry guide:
- `geometry=:softplus_metric`
  Default and usually the safest choice. Uses a regularized pullback-inspired
  geometry induced by the softplus chart. Good general-purpose option for dense
  strictly positive data.
- `geometry=:squaring_metric`
  Uses a regularized pullback-inspired geometry induced by the squaring chart.
  This can be useful on some sparse or near-zero exact nonnegative tensors, but
  it is more chart-sensitive and should be treated as a specialized option.
- `geometry=:canonical`
  Plain nonnegative CP coordinates without the pullback-style manifold geometry.
  This is the natural choice with `solver=:als`.

Notes:
- `pullback_eps` regularizes the diagonal metric in `:softplus_metric` and
  `:squaring_metric`.
- `cpd(A, r; nonnegative=true, ...)` routes here and adopts the same effective
  defaults.
"""
function nncpd(
    A::AbstractArray{T,N},
    r::Int;
    init = :auto,
    p0 = nothing,
    warm_steps = 500,
    warm_init = TuckerInit(),
    solver = :rgd,
    geometry = (solver == :als ? :canonical : :softplus_metric),
    maxiter = 500,
    stepsize = 0.01,
    tol = 1e-6,
    gradient_mode = :riemannian,
    normalization = :auto,
    scale_by_lambda = true,
    lambda_eps = 1e-10,
    verbose = true,
    vector_transport_method = nothing,
    pullback_eps = 1e-8,
    als_polish_max_steps = nothing,
    als_polish_chunk::Int = 10,
    als_polish_rel_improve = 1e-10,
    kwargs...,
) where {T<:AbstractFloat,N}
    return _cpd_impl(
        A,
        r;
        init = init,
        p0 = p0,
        warm_steps = warm_steps,
        warm_init = warm_init,
        solver = solver,
        geometry = geometry,
        maxiter = maxiter,
        stepsize = stepsize,
        tol = tol,
        gradient_mode = gradient_mode,
        normalization = normalization,
        scale_by_lambda = scale_by_lambda,
        lambda_eps = lambda_eps,
        nonnegative = true,
        verbose = verbose,
        vector_transport_method = vector_transport_method,
        pullback_eps = pullback_eps,
        als_polish_max_steps = als_polish_max_steps,
        als_polish_chunk = als_polish_chunk,
        als_polish_rel_improve = als_polish_rel_improve,
        kwargs...,
    )
end
