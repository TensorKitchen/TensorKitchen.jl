# api/nncpd.jl — user-facing nonnegative CP decomposition entry points
export nncpd

"""
     nncpd(A, r; kwargs...)

Computes a nonnegative rank-`r` CP approximation of `A` in two steps: (1) the first step finds an initial point; (2) the second step refines the initial point. Returns a [`CPDResult`](@ref). 
If `r` is omitted, uses the smallest tensor mode as a conservative heuristic rank.
`cpd(A, r; nonnegative=true, ...)` routes here and adopts the same effective defaults.

## Main Options 
* `init = :auto`: Sets the algorithm to find the initial point.
* `solver = :rgd`, `geometry=:softplus_metric`: is the default manifold route.

Geometry guide:
- `geometry=:softplus_metric`
  Default and usually the safest choice. 
- `geometry=:squaring_metric`
  Uses a regularized pullback-inspired geometry induced by the squaring chart.
- `geometry=:canonical`
  Plain nonnegative CP coordinates without the pullback-style manifold geometry.
  This is the natural choice with `solver=:als`.

## Extended Options
* `p0 = nothing`: 
* `warm_steps = 500`: 
* `warm_init = TuckerInit()`:
* `maxiter = 500`:
* `stepsize = 1.0`:
* `tol = 1e-6`:
* `gradient_mode = :riemannian`:
* `normalization = :auto`: 
* `scale_by_lambda = true`:
* `lambda_eps = 1e-10`:
* `nonnegative::Bool = false`:
* `verbose = true`:
* `vector_transport_method = nothing`:
* `pullback_eps = 1e-8`:
* `als_polish_max_steps = nothing`:
* `als_polish_chunk::Int = 10`:
* `als_polish_rel_improve = 1e-10`:

## Example 
```julia-repl
julia> A = randn(20, 15, 10); r = 35;
julia> B = abs.(A)
julia> nncpd(B, r)
CPDResult{Float64}
  Order:        3
  Dimensions:   (20, 15, 10)
  Rank:         35
  Rel. error:   0.3765605093526155
``` 





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
