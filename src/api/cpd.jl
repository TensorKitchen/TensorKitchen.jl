# api/cpd.jl — user-facing CP decomposition entry points
export cpd

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
    warm_init = init.base_init == :auto ? :tucker : init.base_init
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
            progress_phase = :initialization,
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
        solver = ALSSolver(),
        maxiter = init.nsteps,
        stepsize = one(T),
        tol = tol,
        gradient_mode = :riemannian,
        normalization = normalization,
        verbose = verbose,
        vector_transport_method = nothing,
        nonnegative = inner.nonnegative,
        progress_phase = :initialization,
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

function _run_cpd_solver(
    model;
    init_eff,
    p0,
    solver::AbstractSolver,
    maxiter::Int,
    stepsize,
    tol,
    gradient_mode,
    normalization,
    warm_normalization,
    verbose::Bool,
    vector_transport_method,
    pullback_eps,
    kwargs...,
)
    p_solve = if init_eff isa ALSWarmStartInit && isnothing(p0) && !(solver isa ALSSolver)
        _cpd_als_warm_then_pack(
            model,
            init_eff;
            tol,
            normalization = warm_normalization,
            verbose,
            pullback_eps,
            kwargs...,
        )
    else
        _pack_cpd_explicit_p0(model, p0)
    end

    return _solve_model(
        model;
        init = init_eff,
        p0 = p_solve,
        solver = solver,
        maxiter,
        stepsize,
        tol,
        gradient_mode,
        normalization,
        verbose,
        refinement_verbose = verbose,
        vector_transport_method,
        kwargs...,
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

    solver_obj = _solver_object(solver, stepsize; kwargs...)
    init_resolved = init == :auto ? (solver_obj isa ALSSolver ? :tucker : :alswarm) : init
    init_eff =
        init_resolved == :alswarm ? ALSWarmStartInit(warm_steps; base_init = warm_init) :
        init_resolved
    geometry_eff = _is_native_rankr_geometry(geometry) ? :native : geometry
    pullback_eps_eff = _pullback_eps_value(T, pullback_eps)
    if normalization == :auto
        normalization_eff =
            solver_obj isa ALSSolver ?
            (nonnegative ? NoNormalization() : SeparateLambdaNormalization()) :
            NoNormalization()
        warm_normalization_eff =
            nonnegative ? NoNormalization() : SeparateLambdaNormalization()
    else
        normalization_eff = _normalization_policy(normalization)
        warm_normalization_eff = normalization_eff
    end

    r >= 1 || throw(ArgumentError("rank r must be >= 1, got r=$r"))
    solver_obj isa Union{ALSSolver,RGDSolver,RGDFixedSolver,RCGSolver} || throw(
        ArgumentError(
            "Unsupported CPD solver $(typeof(solver_obj)). Use :als, :rgd, :rgd_fixed, or :rcg.",
        ),
    )
    geometry_eff in (:native, :canonical, :squaring_metric, :softplus_metric) || throw(
        ArgumentError(
            "Unknown geometry=$geometry. Use :native, :canonical, :squaring_metric, or :softplus_metric.",
        ),
    )
    if geometry_eff in (:squaring_metric, :softplus_metric) && !nonnegative
        throw(ArgumentError("geometry=$geometry_eff requires nonnegative=true."))
    end
    if solver_obj isa ALSSolver
        geometry_eff == :canonical || throw(
            ArgumentError(
                "solver=:als does not use manifold geometry. Use geometry=:canonical.",
            ),
        )
        gradient_mode == :riemannian || throw(
            ArgumentError(
                "solver=:als does not use gradient_mode. Use gradient_mode=:riemannian.",
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

    raw_result = with_phase_progress() do
        _run_cpd_solver(
            model;
            init_eff,
            p0,
            solver = solver_obj,
            maxiter,
            stepsize,
            tol,
            gradient_mode,
            normalization = normalization_eff,
            warm_normalization = warm_normalization_eff,
            verbose,
            vector_transport_method,
            pullback_eps = pullback_eps_eff,
            nonnegative,
            kwargs...,
        )
    end

    result =
        nonnegative && geometry_eff in (:squaring_metric, :softplus_metric) ?
        _merge_res_solver_info(raw_result, (nncp_pullback_eps = pullback_eps_eff,)) :
        raw_result
    return _to_cpd_result(model, result, size(A), r)
end

#### MAIN CPD ####

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
    - customized initial point:
        - `:tucker` (default when `solver = :als`): Uses a default Tucker initializer.
        - `:random`: Uses a random initial point.
        - `:hosvd`: Uses a HOSVD initial point.
* `solver = :rgd`: Sets the algorithm for refinement. Possible options are:
    - `rgd` (default): Riemannian gradient descent
    - `rgd_fixed`: Riemannian gradient descent with fixed step size
    - `rcg`: Riemannian conjugate gradient
    - `als`: Alternating Least Squares

## Extended Options
* `p0 = nothing`: Explicit initial point. If provided, it overrides the default initial point.
* `:alswarm`: ALS warm start option.
    - `warm_init = TuckerInit()`: Before finding the warm start initial point, this sets the good starting point for ALS.
    - `warm_steps = 500`: Once finding the best initial point from warm_init, it runs this many ALS iterations to refine the initial point.
* `maxiter = 500`: Maximum number of Riemannian gradient descent iterations.
* `stepsize = 1.0`: Initial step size for line search in Riemannian gradient descent.
* `tol = 1e-6`: Convergence tolerance.
* `gradient_mode = :riemannian`: Gradient rule for manifold solvers. 
    - If the model has a direct rgrad, it uses that.
    - Otherwise it computes egrad and projects it to the tangent space.
    - This behavior is in src/solvers/abstract.jl (line 289).
* `geometry = :canonical`: Sets the geometry of the manifold. Possible options are:
    - `:canonical`: Standard CPD parameterization with the usual Euclidean factors and canonical Riemannian gradient handling. Best default for general unconstrained CPD.
    - `:squaring_metric`: Nonnegative geometry based on squared latent coordinates. Enforces nonnegativity indirectly, but can become ill-conditioned near zero.
    - `:softplus_metric`: Nonnegative geometry uses a regularized pullback-inspired geometry induced by the softplus chart. Smoother and usually more stable near zero than `:squaring_metric`.
    - `:native`: Native CP manifold geometry using the model’s intrinsic CP/Segre representation not for nonnegative=true. Best for structured join layouts with `Manifolds.Segre` summands.
* `verbose = true`: Enables progress output.
* `nonnegative::Bool = false`: Nonnegative CPD option to be selected by the user. (same as `nncpd`)
* `pullback_eps = 1e-8`: Regularization parameter for pullback-style nonnegative geometries.

## Notes
* `solver = :als` does not use manifold geometry. In that case:
    - `geometry` must be `:canonical`
    - `gradient_mode` is ignored except for validation
* `:squaring_metric` and `:softplus_metric` require `nonnegative = true`.
* When `nonnegative = true`, `cpd(...)` routes to `nncpd(...)`. In that route:
    - if `solver != :als` and `geometry` is left at `:canonical`, the effective geometry becomes `:softplus_metric`
    - if `stepsize` is left at `1.0`, the effective default becomes `0.01`
    - if `init = :tucker`, the effective initializer becomes `:alswarm`
    
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
        solver_obj = _solver_object(solver, stepsize; kwargs...)
        # Align effective defaults with nncpd() on the nonnegative route.
        # Explicitly passed non-default values are preserved.
        init_nn = init == :tucker ? :alswarm : init
        warm_steps_nn = warm_steps
        geometry_nn = if solver_obj isa ALSSolver
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
            solver = solver_obj,
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
