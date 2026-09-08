# api/btd.jl — Block-term decomposition API
export btd

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
    si0 = _result_solver_info(result)
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

_default_btd_init(::ALSSolver) = BTDHOSVDMultistartInit()
_default_btd_init(::AbstractSolver) = :alswarm

@inline function _resolve_btd_init(init, solver::AbstractSolver)
    init == :auto || return init
    return _default_btd_init(solver)
end

@inline _btd_solver_symbol(::ALSSolver) = :als
@inline _btd_solver_symbol(solver::AbstractSolver) = solver_symbol(solver)

_btd_uses_warm_start(::AbstractSolver, _) = false
_btd_uses_warm_start(::ALSSolver, ::BTDALSWarmStartInit) = false
_btd_uses_warm_start(::AbstractSolver, ::BTDALSWarmStartInit) = true

_btd_should_polish(::ALSSolver, ::Integer) = false
_btd_should_polish(::AbstractSolver, polish_n::Integer) = polish_n > 0

function _reject_unsupported_btd_solver(solver_obj)
    solver_obj isa LMSolver || return nothing
    throw(
        ArgumentError(
            "BTD currently does not support LM refinement because the required Manopt operator path is not yet available for nested Tucker layouts. Use :rgd, :rcg, :lbfgs, :als, or :btd_tsd instead.",
        ),
    )
end

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
    si0 = _result_solver_info(result)
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
    btd(A, blocks, ranks; init=:auto, warm_steps=200,
        warm_init=BTDHOSVDMultistartInit(64; screening_steps=10,
            block_maxiter=12),
        warm_block_method=:hooi, warm_block_maxiter=20,
        warm_rel_error_gate=5e-2, solver=:rgd, maxiter=500,
        stepsize=0.01, tol=1e-6, gradient_mode=:riemannian,
        verbose=true, vector_transport_method=nothing, init_point=nothing,
        block_method=:hooi, block_maxiter=30,
        btd_als_polish_maxiter=nothing, max_stagnation_restarts=1,
        stagnation_rel_error=1e-4, restart_candidates=24,
        restart_screening_steps=5, restart_block_maxiter=20,
        restart_seed=nothing, kwargs...)

Approximate `A` as a sum of Tucker blocks.

# Inputs

- `A`: numerical input tensor.
- `blocks`: number of Tucker blocks.
- `ranks`: multilinear rank tuple used for every block.

# Output

Returns a [`BTDResult`](@ref). Use `blocks` to inspect the fitted Tucker terms,
`reconstruct` to rebuild the approximation, and
`rel_error(A, result)` to measure reconstruction error.

# Common options

- `solver=:rgd` selects the refinement method. Supported symbols are `:als`,
  `:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`, and `:btd_tsd`; a compatible solver
  object may be passed instead. `:lm` is not supported for BTD.
- `init=:auto` selects `BTDHOSVDMultistartInit()` for direct ALS and an ALS warm
  start for manifold solvers. `init_point` supplies an explicit packed BTD point.
- `maxiter=500`, `stepsize=0.01`, and `tol=1e-6` control refinement.
- `verbose=true` displays progress.
- `gradient_mode=:riemannian` selects the public gradient route;
  `:egrad_project` is the alternative frontend route.
- `vector_transport_method=nothing` uses the solver's default vector transport.

# Initialization and ALS budgets

- `warm_steps=200` controls the ALS warm-start length.
- `warm_init=BTDHOSVDMultistartInit(64; screening_steps=10,
  block_maxiter=12)` selects its base initializer.
- `warm_rel_error_gate=5e-2` is a failure cutoff: if the warm-start relative
  error is larger, manifold refinement is skipped and the warm result is
  returned. Use `nothing` to disable the gate.
- `warm_block_method=:hooi` and `warm_block_maxiter=20` configure warm-start
  block updates. The supported block methods are `:hooi` and `:sthosvd`.
- `block_method=:hooi` and `block_maxiter=30` configure direct BTD-ALS block
  updates.
- `btd_als_polish_maxiter=nothing` uses the pipeline's automatic polishing
  budget (`clamp(maxiter ÷ 2, 20, 500)` for non-ALS solvers); an integer sets it
  explicitly and `0` disables polishing.

# Stagnation restarts

- `max_stagnation_restarts=1`: maximum number of retries after ALS stagnates at
  a poor fit.
- `stagnation_rel_error=1e-4`: error level above which small fit change is
  treated as poor stagnation; this is not an iteration-improvement tolerance.
- `restart_candidates=24`, `restart_screening_steps=5`, and
  `restart_block_maxiter=20`: multistart retry budgets.
- `restart_seed=nothing`: optional deterministic seed for retry generation.

BTD is nonconvex, so these controls improve search effort rather than guarantee
a globally optimal decomposition. `A` must have floating-point element type.

# Example

```julia
A = randn(20, 15, 10)
result = btd(A, 3, (5, 4, 3); verbose=false)
terms = blocks(result)
A_approx = reconstruct(result)
```
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
    solver_obj = _solver_object(solver, stepsize; kwargs...)
    _reject_unsupported_btd_solver(solver_obj)
    solver_sym = _btd_solver_symbol(solver_obj)
    init_resolved = _resolve_btd_init(init, solver_obj)
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
    short_circuited = Ref(false)
    result = with_phase_progress() do
        if _btd_uses_warm_start(solver_obj, init_eff)
            warm = _btd_warm_start_result(
                model,
                b,
                init_eff,
                solver_sym;
                verbose = verbose,
                warm_rel_error_gate,
            )
            solve_init = warm.init
            solve_p0 = warm.p0
            warm_info = warm.warm_info
            if !isnothing(warm.short_circuit)
                short_circuited[] = true
                return warm.short_circuit
            end
        end
        _solve_model(
            model;
            init = solve_init,
            p0 = solve_p0,
            solver = solver_obj,
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
    end
    if short_circuited[]
        return _to_btd_result(model, result)
    end
    result =
        isempty(propertynames(warm_info)) ? result :
        _merge_btd_solver_info(result, warm_info)
    polish_n = if btd_als_polish_maxiter === nothing
        solver_sym == :als ? 0 : clamp(maxiter ÷ 2, 20, 500)
    else
        btd_als_polish_maxiter
    end
    if _btd_should_polish(solver_obj, polish_n)
        result = _polish_btd_with_als(
            b,
            result,
            solver_sym;
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
