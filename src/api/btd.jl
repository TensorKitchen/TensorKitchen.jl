# api/btd.jl — Block-term decomposition API
export btd

"""
    _polish_btd_with_als(backend, result, manifold_solver; kwargs...) -> NamedTuple

Run a final BTD-ALS polish from a manifold-solver result and keep it only if it
improves relative error. Solver metadata records the polish budget and source.
"""
function _polish_btd_with_als(
    backend::BTDBackend,
    result,
    manifold_solver::Symbol;
    block_method::Symbol,
    block_maxiter::Int,
    polish_maxiter::Int,
    tol::Real,
)
    polish_maxiter <= 0 && return result
    als_res = fit_btd_als(
        backend.target,
        backend;
        p0 = point(result),
        maxiter = polish_maxiter,
        tol = tol,
        block_method = block_method,
        block_maxiter = block_maxiter,
        verbose = false,
        return_stats = true,
        max_stagnation_restarts = 0,
    )
    rel_error(als_res) < rel_error(result) || return result
    si0 = hasproperty(result, :solver_info) ? solver_info(result) : (;)
    si = merge(
        si0,
        (
            btd_als_polish_iters = iterations(als_res),
            btd_als_polish_applied = true,
            btd_manifold_solver = manifold_solver,
        ),
    )
    return (
        point = point(als_res),
        cost = cost(als_res),
        rel_error = rel_error(als_res),
        grad_norm = grad_norm(als_res),
        iterations = iterations(result) + iterations(als_res),
        converged = converged(als_res),
        solver = solver(result),
        solver_info = si,
    )
end

"""
    _resolve_btd_init(init, solver) -> initializer

Resolve `:auto` to the default BTD initializer for the requested solver family.
ALS uses multistart HOSVD; manifold solvers use ALS warm start.
"""
@inline function _resolve_btd_init(init, solver::Symbol)
    init == :auto || return init
    return solver == :als ? BTDHOSVDMultistartInit() : :alswarm
end

"""
    _btd_effective_init(solver, init, warm_steps, warm_init, warm_block_method, warm_block_maxiter)

Wrap non-ALS BTD initializers in `BTDALSWarmStartInit` so manifold solvers start
from a short BTD-ALS warm point.
"""
function _btd_effective_init(
    solver::Symbol,
    init,
    warm_steps::Int,
    warm_init,
    warm_block_method::Symbol,
    warm_block_maxiter::Int,
)
    if init == :alswarm
        return BTDALSWarmStartInit(
            warm_steps;
            base_init = warm_init,
            block_method = warm_block_method,
            block_maxiter = warm_block_maxiter,
        )
    elseif solver ∉ (:als,)
        return BTDALSWarmStartInit(
            warm_steps;
            base_init = init,
            block_method = warm_block_method,
            block_maxiter = warm_block_maxiter,
        )
    end
    return init
end

"""
    _btd_warm_start_result(model, backend, init, requested_solver; kwargs...)

Execute the BTD-ALS warm-start phase, return the warm point and metadata, and
optionally short-circuit when `warm_rel_error_gate` is exceeded.
"""
function _btd_warm_start_result(
    model::JoinModel{T,<:BTDBackend},
    backend::BTDBackend,
    init::BTDALSWarmStartInit,
    requested_solver::Symbol;
    verbose::Bool = true,
    warm_rel_error_gate,
) where {T<:AbstractFloat}
    p_base = initial_point(model, init.base_init)
    warm = fit_btd_als(
        backend.target,
        backend;
        p0 = p_base,
        maxiter = init.nsteps,
        tol = 1e-8,
        block_method = init.block_method,
        block_maxiter = init.block_maxiter,
        verbose = verbose,
        return_stats = true,
        progress_phase = :initialization,
    )
    gate = isnothing(warm_rel_error_gate) ? nothing : T(warm_rel_error_gate)
    gate_triggered = !isnothing(gate) && warm.rel_error > gate
    warm_info = (
        btd_als_warm_start_iters = warm.iterations,
        btd_als_warm_start_rel_error = Float64(warm.rel_error),
        btd_als_warm_start_requested_solver = requested_solver,
        btd_als_warm_start_gate = isnothing(gate) ? nothing : Float64(gate),
        btd_als_warm_start_gate_triggered = gate_triggered,
    )
    if gate_triggered
        short = (
            point = warm.point,
            cost = warm.cost,
            rel_error = warm.rel_error,
            grad_norm = warm.grad_norm,
            iterations = warm.iterations,
            converged = warm.converged,
            solver = requested_solver,
            solver_info = merge(
                warm.solver_info,
                warm_info,
                (
                    btd_skipped_manifold_polish = true,
                    btd_skip_reason = :warm_rel_error_gate,
                ),
            ),
        )
        return (
            p0 = warm.point,
            init = init.base_init,
            warm_info = warm_info,
            short_circuit = short,
        )
    end
    return (
        p0 = warm.point,
        init = init.base_init,
        warm_info = warm_info,
        short_circuit = nothing,
    )
end

function _merge_btd_solver_info(result, extra::NamedTuple)
    si0 = hasproperty(result, :solver_info) ? result.solver_info : (;)
    return (
        point = result.point,
        cost = result.cost,
        rel_error = result.rel_error,
        grad_norm = result.grad_norm,
        iterations = result.iterations,
        converged = result.converged,
        solver = result.solver,
        solver_info = merge(si0, extra),
    )
end

"""
    btd(A, blocks, ranks; kwargs...) -> BTDResult

Block-term decomposition (sum of Tucker blocks); default pipeline is
`BTDHOSVDMultistartInit(candidates=64, screening_steps=10, block_maxiter=12)`
followed by `ALSwarm(200)`, Riemannian gradient descent, and a final BTD-ALS
polish step. By default, manifold polish is skipped when the warm-stage
relative error exceeds `warm_rel_error_gate = 5e-2`.
"""
function btd(
    A::AbstractArray{T,N},
    blocks::Int,
    ranks::NTuple{N,Int};
    init = :auto,
    warm_steps = 200,
    warm_init = BTDHOSVDMultistartInit(64; screening_steps = 10, block_maxiter = 12),
    warm_block_method = :hooi,
    warm_block_maxiter = 20,
    warm_rel_error_gate = 5e-2,
    solver = :rgd,
    maxiter = 500,
    stepsize = 0.01,
    tol = 1e-6,
    gradient_mode = :riemannian,
    verbose = true,
    vector_transport_method = nothing,
    init_point = nothing,
    block_method = :hooi,
    block_maxiter = 30,
    btd_als_polish_maxiter = nothing,
    max_stagnation_restarts = 1,
    stagnation_rel_error = 1e-4,
    restart_candidates = 24,
    restart_screening_steps = 5,
    restart_block_maxiter = 20,
    restart_seed = nothing,
    kwargs...,
) where {T<:AbstractFloat,N}
    init_resolved = _resolve_btd_init(init, solver)
    init_eff =
        init_resolved == :alswarm ?
        BTDALSWarmStartInit(
            warm_steps;
            base_init = warm_init,
            block_method = warm_block_method,
            block_maxiter = warm_block_maxiter,
        ) : init_resolved
    blocks >= 1 || throw(ArgumentError("blocks must be >= 1, got $blocks"))
    manifolds = _as_join_manifold_tuple(TuckerJoin(size(A), ranks, blocks))
    b = _sum_backend_instance(BTDBackend, manifolds, A; init_point)
    model = JoinModel{T,typeof(b)}(b)
    solve_init = init_eff
    solve_p0 = nothing
    warm_info = (;)
    if solver ∉ (:als,) && init_eff isa BTDALSWarmStartInit
        warm = _btd_warm_start_result(
            model,
            b,
            init_eff,
            solver;
            verbose = verbose,
            warm_rel_error_gate,
        )
        solve_init = warm.init
        solve_p0 = warm.p0
        warm_info = warm.warm_info
        if !isnothing(warm.short_circuit)
            return _to_btd_result(model, warm.short_circuit)
        end
    end
    result = _solve_model(
        model;
        init = solve_init,
        p0 = solve_p0,
        solver = solver,
        maxiter = maxiter,
        stepsize = stepsize,
        tol = tol,
        gradient_mode = gradient_mode,
        normalization = NoNormalization(),
        verbose = verbose,
        vector_transport_method = vector_transport_method,
        block_method = block_method,
        block_maxiter = block_maxiter,
        max_stagnation_restarts = max_stagnation_restarts,
        stagnation_rel_error = stagnation_rel_error,
        restart_candidates = restart_candidates,
        restart_screening_steps = restart_screening_steps,
        restart_block_maxiter = restart_block_maxiter,
        restart_seed = restart_seed,
        kwargs...,
    )
    result =
        isempty(propertynames(warm_info)) ? result :
        _merge_btd_solver_info(result, warm_info)
    polish_n = if btd_als_polish_maxiter === nothing
        solver == :als ? 0 : clamp(maxiter ÷ 2, 20, 500)
    else
        btd_als_polish_maxiter
    end
    if polish_n > 0 && solver ∉ (:als,)
        result = _polish_btd_with_als(
            b,
            result,
            solver;
            block_method = block_method,
            block_maxiter = block_maxiter,
            polish_maxiter = polish_n,
            tol = tol,
        )
    end
    return _to_btd_result(model, result)
end

function btd(
    A::AbstractArray{T,N},
    blocks::Int,
    ranks::AbstractVector{<:Integer};
    kwargs...,
) where {T<:AbstractFloat,N}
    length(ranks) == N ||
        throw(ArgumentError("ranks length $(length(ranks)) must match ndims(A)=$N."))
    return btd(A, blocks, Tuple(Int.(ranks)); kwargs...)
end
