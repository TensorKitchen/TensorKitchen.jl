# api/cpd.jl — user-facing CP decomposition entry points
export cpd

"""
    CPDOpts

Keyword-struct form of the CPD frontend options. Used by the legacy
`cpd(A, r, opts)` path and for keyword validation in the compact API.
"""
Base.@kwdef struct CPDOpts{T<:Real}
    solver::Symbol = :rgd
    geometry::Symbol = :canonical
    init::Any = :auto
    maxiter::Int = (solver in (:rgd, :rcg)) ? 2000 : 500
    tol::T = 1e-6
    gradient_mode::Symbol = :riemannian
    nonnegative::Bool = false
    stepsize::T = 1.0
    normalization::Symbol = :separate
    pullback_eps::T = 1e-8
    verbose::Bool = true
end

function _validate_opts(opts::CPDOpts)
    if opts.solver == :als && opts.geometry != :canonical
        throw(
            ArgumentError("ALS solver requires geometry=:canonical. Got: $(opts.geometry)"),
        )
    end
    if opts.nonnegative &&
       opts.solver in (:rgd, :rcg) &&
       opts.geometry ∉ (:squaring_metric, :softplus_metric)
        @warn "Nonnegative RGD/RCG usually requires geometry=:softplus_metric or :squaring_metric."
    end
    return opts
end

"""
    _resolve_cpd_init(init, solver) -> initializer

Resolve `:auto` to the default CPD initializer for the selected solver family.
"""
@inline function _resolve_cpd_init(init, solver::Symbol)
    init == :auto || return init
    return solver == :als ? :tucker : :alswarm
end

function _pullback_eps_value(::Type{T}, pullback_eps) where {T<:AbstractFloat}
    ε = T(pullback_eps)
    isfinite(ε) && ε > zero(T) ||
        throw(ArgumentError("pullback_eps must be finite and positive, got $pullback_eps."))
    return ε
end

function _merge_res_solver_info(res, patch::NamedTuple)
    si0 = hasproperty(res, :solver_info) ? solver_info(res) : (;)
    return (
        point = point(res),
        cost = cost(res),
        rel_error = rel_error(res),
        grad_norm = grad_norm(res),
        iterations = iterations(res),
        converged = converged(res),
        solver = solver(res),
        solver_info = merge(si0, patch),
    )
end

function _pack_cpd_explicit_p0(model, p0)
    p0 isa CPDPoint && return pack_cpd_point(model, p0)
    p0 isa CPDResult && return pack_cpd_point(model, cpd_point(p0))
    return p0
end

"""
    _cpd_als_warm_then_pack(target, init; tol, normalization, ...)

Run the same CP-ALS solve as `solver=:als` on a canonical `JoinModel` (matching
`target`'s nonnegative mode and factor scaling), then pack the resulting
[`CPDResult`](@ref) into `target`'s manifold coordinates for manifold refinement.
"""
function _cpd_als_warm_then_pack(
    target::JoinModel{<:AbstractFloat,<:CPDBackend},
    init::ALSWarmStartInit;
    tol::Real,
    normalization,
    verbose::Bool,
    pullback_eps::Real,
    kwargs...,
)
    inner = cpd_model(target)
    A = tensor(target)
    r = inner isa RankRCPDModel ? inner.r : 1
    T = eltype(A)
    warm_init = _resolve_cpd_init(init.base_init, :als)
    if r == 1
        warm_out = fit_cp_als(
            A,
            1;
            init = warm_init,
            maxiter = init.nsteps,
            tol = tol,
            normalization = normalization,
            mttkrp_method = get(kwargs, :mttkrp_method, :auto),
            nonnegative = inner.nonnegative,
            verbose = verbose,
            return_stats = true,
        )
        return pack_cpd_point(target, CPDPoint(warm_out.weights, warm_out.factors))
    end
    warm_model = JoinModel(
        A,
        r;
        geometry = :canonical,
        scale_by_lambda = inner.scale_by_lambda,
        lambda_eps = inner.lambda_eps,
        nonnegative = inner.nonnegative,
        use_pullback_metric = false,
        pullback_eps = pullback_eps,
    )
    warm_result = _solve_model(
        warm_model;
        init = warm_init,
        solver = :als,
        maxiter = init.nsteps,
        stepsize = one(T),
        tol = tol,
        gradient_mode = :riemannian,
        normalization = normalization,
        verbose = verbose,
        vector_transport_method = nothing,
        nonnegative = inner.nonnegative,
        kwargs...,
    )
    warm_cpd = _to_cpd_result(warm_model, warm_result, size(A), r)
    return pack_cpd_point(target, cpd_point(warm_cpd))
end

function initial_point(
    model::RankRCPDModel{T,N},
    init::ALSWarmStartInit;
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    ε = _pullback_eps_value(T, 1e-8)
    target = JoinModel(
        model.A,
        model.r;
        geometry = model.geometry,
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
        nonnegative = model.nonnegative,
        use_pullback_metric = (model.geometry == :squaring_metric),
        pullback_eps = ε,
    )
    norm_als = model.nonnegative ? NoNormalization() : SeparateLambdaNormalization()
    return _cpd_als_warm_then_pack(
        target,
        init;
        tol = T(1e-6),
        normalization = norm_als,
        verbose,
        pullback_eps = ε,
    )
end

function initial_point(
    model::Rank1CPDModel{T,N},
    init::ALSWarmStartInit;
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    ε = _pullback_eps_value(T, 1e-8)
    geom =
        model.nonnegative ?
        (_rank1_uses_softplus_metric(model.M) ? :softplus_metric : :squaring_metric) :
        :native
    target = JoinModel(
        model.A,
        1;
        geometry = geom,
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
        nonnegative = model.nonnegative,
        use_pullback_metric = (geom == :squaring_metric),
        pullback_eps = ε,
    )
    norm_als = model.nonnegative ? NoNormalization() : SeparateLambdaNormalization()
    return _cpd_als_warm_then_pack(
        target,
        init;
        tol = T(1e-6),
        normalization = norm_als,
        verbose,
        pullback_eps = ε,
    )
end

function _cpd_impl(
    A::AbstractArray{T,N},
    r::Int;
    init,
    p0 = nothing,
    warm_steps,
    warm_init,
    solver,
    geometry,
    maxiter,
    stepsize,
    tol,
    gradient_mode,
    normalization,
    scale_by_lambda,
    lambda_eps,
    nonnegative::Bool,
    verbose,
    vector_transport_method,
    pullback_eps = 1e-8,
    kwargs...,
) where {T<:AbstractFloat,N}
    haskey(kwargs, :softplus_beta) && throw(
        ArgumentError(
            "softplus_beta has been removed. Use pullback_eps to tune softplus pullback regularization.",
        ),
    )
    dims = size(A)
    init_resolved = _resolve_cpd_init(init, solver)
    init_eff =
        init_resolved == :alswarm ? ALSWarmStartInit(warm_steps; base_init = warm_init) :
        init_resolved
    geometry_eff = _is_native_rankr_geometry(geometry) ? :native : geometry
    manifold_solvers = (:rgd, :rgd_fixed, :rcg)
    als_family = (:als,)
    normalization_eff =
        normalization == :auto ?
        (
            solver ∈ als_family ?
            (nonnegative ? NoNormalization() : SeparateLambdaNormalization()) :
            NoNormalization()
        ) : _normalization_policy(normalization)
    als_normalization_eff =
        normalization == :auto ?
        (nonnegative ? NoNormalization() : SeparateLambdaNormalization()) :
        _normalization_policy(normalization)
    pullback_eps_eff = _pullback_eps_value(T, pullback_eps)

    r >= 1 || throw(ArgumentError("rank r must be >= 1, got r=$r"))
    nonnegative &&
        solver ∉ (:als, :rgd, :rgd_fixed, :rcg) &&
        throw(
            ArgumentError(
                "nonnegative=true requires solver=:als, :rgd, :rgd_fixed, or :rcg. Got solver=$solver.",
            ),
        )
    geometry_eff ∈ (:native, :canonical, :squaring_metric, :softplus_metric) || throw(
        ArgumentError(
            "Unknown geometry=$geometry. Use :native, :canonical, :squaring_metric, or :softplus_metric (regularized pullback-style geometries require nonnegative=true).",
        ),
    )
    (geometry_eff ∉ (:squaring_metric, :softplus_metric) || nonnegative) ||
        throw(ArgumentError("geometry=$geometry_eff requires nonnegative=true."))
    solver ∈ manifold_solvers ||
        solver ∈ als_family ||
        throw(
            ArgumentError(
                "Unknown solver=$solver. Use a manifold solver (:rgd, :rgd_fixed, :rcg) or CP-ALS (:als).",
            ),
        )
    if solver ∈ als_family
        geometry_eff == :canonical || throw(
            ArgumentError(
                "solver=$solver does not use manifold geometry. Pass geometry=:canonical (default) or switch to a manifold solver (:rgd, :rgd_fixed, :rcg) for geometry=:native/:squaring_metric/:softplus_metric.",
            ),
        )
        gradient_mode == :riemannian || throw(
            ArgumentError(
                "solver=$solver does not use gradient_mode. Pass gradient_mode=:riemannian (default) or switch to a manifold solver.",
            ),
        )
    end

    model = JoinModel(
        A,
        r;
        geometry = geometry_eff,
        scale_by_lambda = scale_by_lambda,
        lambda_eps = lambda_eps,
        nonnegative = nonnegative,
        use_pullback_metric = (geometry_eff == :squaring_metric),
        pullback_eps = pullback_eps_eff,
    )
    p_solve = if init_eff isa ALSWarmStartInit && isnothing(p0) && solver ∈ manifold_solvers
        _cpd_als_warm_then_pack(
            model,
            init_eff;
            tol,
            normalization = als_normalization_eff,
            verbose,
            pullback_eps = pullback_eps_eff,
            kwargs...,
        )
    else
        _pack_cpd_explicit_p0(model, p0)
    end
    result_rgd = with_phase_progress() do
        _solve_model(
            model;
            init = init_eff,
            p0 = p_solve,
            solver = solver,
            maxiter = maxiter,
            stepsize = stepsize,
            tol = tol,
            gradient_mode = gradient_mode,
            normalization = normalization_eff,
            verbose = verbose,
            refinement_verbose = verbose,
            vector_transport_method = vector_transport_method,
            nonnegative,
            kwargs...,
        )
    end
    result = result_rgd
    if nonnegative && geometry_eff ∈ (:squaring_metric, :softplus_metric)
        result = _merge_res_solver_info(result, (nncp_pullback_eps = pullback_eps_eff,))
    end
    return _to_cpd_result(model, result, dims, r)
end

#### MAIN CPD ####

function cpd(A::AbstractArray{T,N}, r::Int, opts::CPDOpts) where {T,N}
    _validate_opts(opts)

    if opts.nonnegative
        return nncpd(
            A,
            r;
            solver = opts.solver,
            init = opts.init,
            geometry = opts.geometry,
            maxiter = opts.maxiter,
            stepsize = opts.stepsize,
            tol = opts.tol,
            gradient_mode = opts.gradient_mode,
            normalization = opts.normalization,
            pullback_eps = opts.pullback_eps,
            verbose = opts.verbose,
        )
    end

    return cpd(
        A,
        r;
        init = opts.init,
        solver = opts.solver,
        geometry = opts.geometry,
        maxiter = opts.maxiter,
        tol = opts.tol,
        gradient_mode = opts.gradient_mode,
        pullback_eps = opts.pullback_eps,
        verbose = opts.verbose,
    )
end

function cpd(A::AbstractArray{T,N}, r::Int; kwargs...) where {T,N}
    valid_keys = fieldnames(CPDOpts)
    for k in keys(kwargs)
        k in valid_keys || throw(ArgumentError("Unknown keyword argument: $k"))
    end
    opts = CPDOpts{eltype(A)}(; kwargs...)
    return cpd(A, r, opts)
end

function cpd(
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
    return cpd(A, r_eff; kwargs...)
end

"""
    cpd(A, r; kwargs...)

Computes a rank-`r` CP approximation of `A` in two steps: (1) the first step finds an initial point; (2) the second step refines the initial point. Returns a [`CPDResult`](@ref). 
If `r` is omitted, uses the smallest tensor mode as a heuristic rank.

## Main Options 
* `init = :auto`: Sets the algorithm to find the initial point. Possible options are:
    - `:auto`: Uses a default CPD initializer. For `solver = :als`, this uses `TuckerInit`; otherwise, it uses an ALS warm start.
    - `:alswarm`: Runs ALS first and uses the result as the initial point for refinement.
    - custom initializer objects, e.g. `TuckerInit(...)`.
    - `:tucker` (default when `solver = :als`): Uses a default Tucker initializer.
    - `:random`: Uses a random initial point.
    - `:hosvd`: Uses a HOSVD initial point.
* `solver = :rgd`: Sets the algorithm for refinement. Possible options are:
    - `rgd` (default): Riemannian gradient descent
    - `rgd_fixed`: Riemannian gradient descent with fixed step size
    - `rcg`: Riemannian conjugate gradient
    - `lbfgs`: Limited-memory BFGS
    - `als`: Alternating Least Squares
* `geometry = :canonical`: Sets the geometry of the manifold. Possible options are:
    - `:canonical`: Standard CPD parameterization with the usual Euclidean factors and canonical Riemannian gradient handling. Best default for general unconstrained CPD.
    - `:squaring_metric`: Nonnegative geometry based on squared latent coordinates. Enforces nonnegativity indirectly, but can become ill-conditioned near zero.
    - `:softplus_metric`: Nonnegative geometry uses a regularized pullback-inspired geometry induced by the softplus chart. Smoother and usually more stable near zero than `:squaring_metric`.
    - `:native`: Native CP manifold geometry using the model’s intrinsic CP/Segre representation not for nonnegative=true. Best for structured join layouts with `Manifolds.Segre` summands.


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
* `pullback_eps = 1e-8`:

## Example 
```julia-repl
julia> A = randn(20, 15, 10); r = 35
julia> res = cpd(A, r)
CPDResult{Float64}
  Order:        3
  Dimensions:   (20, 15, 10)
  Rank:         35
  Rel. error:   0.4359141301703327
```
"""
function cpd(
    A::AbstractArray{T,N},
    r::Int;
    init = :auto,
    p0 = nothing,
    warm_steps = 500,
    warm_init = TuckerInit(),
    solver = :rgd,
    geometry = :canonical,
    maxiter = 500,
    stepsize = 1.0,
    tol = 1e-6,
    gradient_mode = :riemannian,
    normalization = :auto,
    scale_by_lambda = true,
    lambda_eps = 1e-10,
    nonnegative::Bool = false,
    verbose = true,
    vector_transport_method = nothing,
    pullback_eps = 1e-8,
    kwargs...,
) where {T<:AbstractFloat,N}
    if nonnegative
        # Align effective defaults with nncpd() on the nonnegative route.
        # Explicitly passed non-default values are preserved.
        init_nn = init == :tucker ? :alswarm : init
        warm_steps_nn = warm_steps
        geometry_nn = if solver ∈ (:als,)
            :canonical
        elseif geometry == :canonical
            :softplus_metric
        else
            geometry
        end
        stepsize_nn = stepsize == 1.0 ? 0.01 : stepsize
        return nncpd(
            A,
            r;
            init = init_nn,
            p0 = p0,
            warm_steps = warm_steps_nn,
            warm_init = warm_init,
            solver = solver,
            geometry = geometry_nn,
            maxiter = maxiter,
            stepsize = stepsize_nn,
            tol = tol,
            gradient_mode = gradient_mode,
            normalization = normalization,
            scale_by_lambda = scale_by_lambda,
            lambda_eps = lambda_eps,
            pullback_eps = pullback_eps,
            verbose = verbose,
            vector_transport_method = vector_transport_method,
            kwargs...,
        )
    end
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
        nonnegative = false,
        pullback_eps = pullback_eps,
        verbose = verbose,
        vector_transport_method = vector_transport_method,
        kwargs...,
    )
end
