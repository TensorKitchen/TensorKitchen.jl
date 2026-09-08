# api/nncpd.jl — user-facing nonnegative CP decomposition entry points
export nncpd

_nncpd_default_geometry(::ALSSolver) = :canonical
_nncpd_default_geometry(::AbstractSolver) = :softplus_metric

function _nncpd_effective_geometry(geometry, solver::AbstractSolver)
    return isnothing(geometry) ? _nncpd_default_geometry(solver) : geometry
end

"""
    nncpd(A, rank; init=:auto, p0=nothing, warm_steps=500,
        warm_init=TuckerInit(), solver=:rgd, geometry=nothing,
        maxiter=500, stepsize=0.01, tol=1e-6,
        gradient_mode=:riemannian, normalization=:auto,
        scale_by_lambda=true, lambda_eps=1e-10, verbose=true,
        vector_transport_method=nothing, pullback_eps=1e-8,
        component_trace=false, kwargs...)
    nncpd(A; r=nothing, kwargs...)

Approximate `A` with a nonnegative CP decomposition containing `rank`
components. This is intended for nonnegative data such as counts,
intensities, and concentrations.

# Inputs

- `A`: numerical input tensor.
- `rank`: number of nonnegative rank-one components.

# Output

Returns a [`CPDResult`](@ref) with nonnegative weights and factors. Use
`weights`, `factors`, `reconstruct`, and
`rel_error(A, result)` to inspect the fit.

# Options

- `solver=:rgd`: supports `:als`, `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, and
  `:lm`.
- `init=:auto`, `warm_steps=500`, and `warm_init=TuckerInit()` configure
  initialization in the same way as [`cpd`](@ref).
- `geometry=nothing`: selects `:canonical` for ALS and `:softplus_metric` for
  manifold solvers. Explicit choices are `:canonical`, `:softplus_metric`, and
  `:squaring_metric`.
- `maxiter=500`, `stepsize=0.01`, `tol=1e-6`, and `verbose=true` control the
  solve.
- `gradient_mode`, `normalization`, `scale_by_lambda`, `lambda_eps`,
  `vector_transport_method`, and `pullback_eps` have the meanings described for
  [`cpd`](@ref).
- `component_trace=false` disables tracing by default. Set it to `true` to
  record per-component diagnostics for manifold solvers. With `solver=:als`,
  `miniter`, `projected_grad_tol`, `nn_update`, and `mttkrp_method` are
  documented by [`fit_cp_als`](@ref).

`A` must have floating-point element type. If `rank`/`r` is omitted, the
smallest tensor dimension is used as a heuristic; pass it explicitly for
reproducible model selection.

# Example

```julia
A = abs.(randn(20, 15, 10))
result = nncpd(A, 5; verbose=false)
A_approx = reconstruct(result)
```

Unlike unconstrained CPD, every returned weight and factor is nonnegative up to
floating-point roundoff.
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
    geometry = nothing,
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
    kwargs...,
) where {T<:AbstractFloat,N}
    solver_obj = _solver_object(solver, stepsize; kwargs...)
    geometry_eff = _nncpd_effective_geometry(geometry, solver_obj)
    return _cpd_impl(
        A,
        r;
        init = init,
        p0 = p0,
        warm_steps = warm_steps,
        warm_init = warm_init,
        solver = solver_obj,
        geometry = geometry_eff,
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
        kwargs...,
    )
end
