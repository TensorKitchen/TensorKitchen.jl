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

function _default_als_polish_max_steps(r::Int)
    r >= 1 || throw(ArgumentError("rank r must be >= 1, got r=$r"))
    return clamp(10 + 5 * r, 20, 120)
end

function _pullback_eps_value(::Type{T}, pullback_eps) where {T<:AbstractFloat}
    ε = T(pullback_eps)
    isfinite(ε) && ε > zero(T) ||
        throw(ArgumentError("pullback_eps must be finite and positive, got $pullback_eps."))
    return ε
end

"""
    _nncpd_manifold_snapshot(model, p, solver; kwargs...) -> NamedTuple

Build result-like statistics for a nonnegative CPD manifold point, typically
used to compare warm-start, RGD, and polished candidates.
"""
function _nncpd_manifold_snapshot(
    model::JoinModel,
    p,
    solver::Symbol;
    iterations::Int,
    converged::Bool,
    solver_info = (;),
)
    M = manifold(model)
    pl = _solver_point(M, p)
    cp = cpd_point(model, pl)
    X = reconstruct_cpd_rankr(lambda(cp), factors(cp))
    rel = rel_error(tensor(model), X)
    return (
        point = pl,
        cost = cost(model, pl),
        rel_error = rel,
        grad_norm = norm(M, pl, rgrad(model, pl)),
        iterations = iterations,
        converged = converged,
        solver = solver,
        solver_info = solver_info,
    )
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

"""
    _polish_nonneg_with_als(model, result, r; kwargs...) -> NamedTuple

Run chunked NNCP-ALS polish after manifold optimization and stop when the
relative-error gain stalls or the step budget is exhausted.
"""
function _polish_nonneg_with_als(
    model::JoinModel,
    result,
    r::Int;
    als_normalization,
    polish_max_steps::Int,
    polish_chunk::Int = 10,
    polish_rel_improve::Real = 1e-10,
)
    polish_max_steps <= 0 &&
        return (result = result, als_polish_steps = 0, als_polish_chunks = 0)
    polish_chunk < 1 && throw(ArgumentError("polish_chunk must be >= 1, got $polish_chunk"))
    M = manifold(model)
    base_iters = iterations(result)
    current = result
    total_als_iters = 0
    chunks = 0
    T = eltype(rel_error(current))
    floor_improve = T(polish_rel_improve)

    while total_als_iters < polish_max_steps
        step = min(polish_chunk, polish_max_steps - total_als_iters)
        step <= 0 && break
        cp = cpd_point(model, point(current))
        als_out = fit_cp_als(
            tensor(model),
            r;
            maxiter = step,
            tol = zero(eltype(lambda(cp))),
            init = RandomInit(),
            init_factors = (lambda(cp), factors(cp)),
            nonnegative = true,
            normalization = als_normalization,
            verbose = false,
            return_stats = true,
        )
        chunks += 1
        total_als_iters += iterations(als_out)
        cp_new = CPDPoint(weights(als_out), factors(als_out))
        p_new = _solver_point(M, pack_cpd_point(model, cp_new))
        rel_new =
            rel_error(tensor(model), reconstruct_cpd_rankr(lambda(cp_new), factors(cp_new)))
        rel_old = rel_error(current)
        if !(rel_new < rel_old)
            break
        end
        improvement = rel_old - rel_new
        current = (
            point = p_new,
            cost = cost(model, p_new),
            rel_error = rel_new,
            grad_norm = norm(M, p_new, rgrad(model, p_new)),
            iterations = base_iters + total_als_iters,
            converged = converged(result),
            solver = solver(result),
            solver_info = hasproperty(result, :solver_info) ? solver_info(result) : (;),
        )
        improvement <= floor_improve * max(rel_old, eps(T)) && break
    end
    si0 = hasproperty(result, :solver_info) ? solver_info(result) : (;)
    si = merge(
        si0,
        (
            als_polish_steps = total_als_iters,
            als_polish_chunks = chunks,
            als_polish_applied = total_als_iters > 0,
            als_polish_max_steps_cap = polish_max_steps,
        ),
    )
    current = (
        point = point(current),
        cost = cost(current),
        rel_error = rel_error(current),
        grad_norm = grad_norm(current),
        iterations = iterations(current),
        converged = converged(current),
        solver = solver(current),
        solver_info = si,
    )
    return (
        result = current,
        als_polish_steps = total_als_iters,
        als_polish_chunks = chunks,
    )
end

"""
    _nncpd_pick_best_candidate(model, warm_res, rgd_res, polished_res, polish_meta)

Select the best nonnegative CPD candidate by relative error and attach metadata
summarizing the candidate comparison.
"""
function _nncpd_pick_best_candidate(
    ::JoinModel,
    warm_res,
    rgd_res,
    polished_res,
    polish_meta,
)
    candidates = ((:warm, warm_res), (:rgd, rgd_res), (:polished, polished_res))
    best_tag, best_res = candidates[argmin([rel_error(c[2]) for c in candidates])]
    info = merge(
        polish_meta,
        (
            nncp_best_candidate = best_tag,
            nncp_rel_warm = rel_error(warm_res),
            nncp_rel_rgd = rel_error(rgd_res),
            nncp_rel_polished = rel_error(polished_res),
        ),
    )
    return _merge_res_solver_info(best_res, info)
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
    als_polish_max_steps = nothing,
    als_polish_chunk::Int = 10,
    als_polish_rel_improve = 1e-10,
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
    pullback_eps_eff = _pullback_eps_value(T, pullback_eps)

    r >= 1 || throw(ArgumentError("rank r must be >= 1, got r=$r"))
    nonnegative &&
        solver ∉ (:als, :rgd, :rcg) &&
        throw(
            ArgumentError(
                "nonnegative=true requires solver=:als, :rgd, or :rcg. Got solver=$solver.",
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
    p_solve =
        if nonnegative &&
           solver ∈ (:rgd, :rcg) &&
           init_eff isa ALSWarmStartInit &&
           isnothing(p0)
            initial_point(model, init_eff; verbose)
        else
            p0
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
    if nonnegative && solver ∈ (:rgd, :rcg) && init_eff isa ALSWarmStartInit
        p_warm = p_solve
        warm_iters = init_eff isa ALSWarmStartInit ? init_eff.nsteps : 0
        warm_res = _nncpd_manifold_snapshot(
            model,
            p_warm,
            result_rgd.solver;
            iterations = warm_iters,
            converged = false,
            solver_info = (nncp_snapshot = :warm,),
        )
        polish_cap = something(als_polish_max_steps, _default_als_polish_max_steps(r))
        polish_out = _polish_nonneg_with_als(
            model,
            result_rgd,
            r;
            als_normalization = normalization_eff,
            polish_max_steps = polish_cap,
            polish_chunk = als_polish_chunk,
            polish_rel_improve = als_polish_rel_improve,
        )
        polish_meta = (
            nncp_als_polish_steps = polish_out.als_polish_steps,
            nncp_als_polish_chunks = polish_out.als_polish_chunks,
        )
        result = _nncpd_pick_best_candidate(
            model,
            warm_res,
            result_rgd,
            polish_out.result,
            polish_meta,
        )
    end
    if nonnegative && geometry_eff ∈ (:squaring_metric, :softplus_metric)
        result = _merge_res_solver_info(result, (nncp_pullback_eps = pullback_eps_eff,))
    end
    return _to_cpd_result(model, result, dims, r)
end

#### MAIN CPD ####

function cpd(A::AbstractArray{T,N}, r::Int, opts::CPDOpts) where {T,N}
    _validate_opts(opts)

    # nonnegative=true 일 경우 내부적으로 자동 NNCPD 파이프라인으로 라우팅 (중복 제거)
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

    # Route through the public keyword API so dispatch/validation stays consistent.
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
If `r` is omitted, uses the smallest tensor mode as a conservative heuristic rank.

## Main Options 
* `init = :auto`: Sets the algorithm to find the initial point.
* `solver = :rgd`: Sets the algorithm for refinement.

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
julia> using TensorKitchen
julia> A = randn(20, 15, 10)
julia> r = 35
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
    als_polish_max_steps = nothing,
    als_polish_chunk::Int = 10,
    als_polish_rel_improve = 1e-10,
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
            als_polish_max_steps = als_polish_max_steps,
            als_polish_chunk = als_polish_chunk,
            als_polish_rel_improve = als_polish_rel_improve,
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
        als_polish_max_steps = als_polish_max_steps,
        als_polish_chunk = als_polish_chunk,
        als_polish_rel_improve = als_polish_rel_improve,
        kwargs...,
    )
end
