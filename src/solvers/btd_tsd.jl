# solvers/btd_tsd.jl — blockwise tangent-subspace descent for BTD

# This solver is inspired by tangent-subspace descent (TSD) in
# Gutman and Ho-Nguyen, "Coordinate Descent Without Coordinates:
# Tangent Subspace Descent on Riemannian Manifolds",
# Mathematics of Operations Research, 48(1):127–159, 2023.
#
# In the present implementation, the selected tangent subspace is the
# tangent space of one Tucker block inside the BTD join/product model.
# Thus this is a BTD-specialized blockwise TSD / Riemannian block-coordinate 
# descent method, rather than the exact algorithm studied for Stiefel/orthogonal 
# examples in the original TSD paper.

export BTDTSDSolver, TSDSolver

"""
    BTDTSDSolver(; stepsize=1.0, schedule=:cyclic, block_repeats=1,
        armijo_contraction=0.5, armijo_sufficient_decrease=1e-4, armijo_alpha_min=1e-12)

Blockwise tangent-subspace descent specialized to BTD. Each block update uses
the projected Tucker-block tangent direction and accepts it only when a block
Armijo decrease condition is satisfied.

This solver is inspired by tangent-subspace descent (TSD) in
Gutman and Ho-Nguyen, "Coordinate Descent Without Coordinates:
Tangent Subspace Descent on Riemannian Manifolds",
Mathematics of Operations Research, 48(1):127–159, 2023.
"""
struct BTDTSDSolver <: AbstractFirstOrderSolver
    stepsize::Float64
    schedule::Symbol
    block_repeats::Int
    armijo_contraction::Float64
    armijo_sufficient_decrease::Float64
    armijo_alpha_min::Float64
end

"""
    BTDTSDSolver(; schedule=:cyclic, block_repeats=1, ...)

Blockwise tangent-subspace descent for BTD.

The `schedule` keyword controls the order in which Tucker blocks are updated.

- `schedule = :cyclic`: updates blocks in the deterministic order
  `1, 2, ..., R` at every sweep.

- `schedule = :random`: updates all blocks once per sweep, but in a fresh
  random permutation. This is random reshuffling, not sampling with replacement.

The cyclic schedule is the default because it is deterministic and easiest to
analyze. The random schedule can be useful experimentally when different BTD
blocks compete strongly for the same residual structure.
"""
function BTDTSDSolver(;
    stepsize::Real = 1.0,
    schedule::Symbol = :cyclic,
    block_repeats::Int = 1, # number of times to repeat the block update sequence at each iteration
    armijo_contraction::Real = 0.5,
    armijo_sufficient_decrease::Real = 1e-4,
    armijo_alpha_min::Real = 1e-12,
)
    stepsize > 0 || throw(ArgumentError("stepsize must be > 0, got $stepsize"))
    block_repeats >= 1 ||
        throw(ArgumentError("block_repeats must be >= 1, got $block_repeats"))
    schedule in (:cyclic, :random) || throw(
        ArgumentError("Unsupported block schedule $schedule. Use :cyclic or :random."),
    )
    0 < armijo_contraction < 1 || throw(
        ArgumentError("armijo_contraction must be in (0, 1), got $armijo_contraction"),
    )
    0 < armijo_sufficient_decrease < 1 || throw(
        ArgumentError(
            "armijo_sufficient_decrease must be in (0, 1), got $armijo_sufficient_decrease",
        ),
    )
    armijo_alpha_min > 0 ||
        throw(ArgumentError("armijo_alpha_min must be > 0, got $armijo_alpha_min"))
    return BTDTSDSolver(
        Float64(stepsize),
        schedule,
        block_repeats,
        Float64(armijo_contraction),
        Float64(armijo_sufficient_decrease),
        Float64(armijo_alpha_min),
    )
end

solver_symbol(::BTDTSDSolver) = :btd_tsd

first_order_diagnostics_recorder(::BTDTSDSolver) =
    _SolverDiagnosticsRecorder(line_search_enabled = true)

@inline function _btd_block_schedule(r::Int, schedule::Symbol)
    schedule === :cyclic && return 1:r
    schedule === :random && return randperm(r)
    throw(ArgumentError("Unsupported block schedule $schedule. Use :cyclic or :random."))
end

@inline function _replace_block_part(p, b::Int, new_part)
    parts = collect(point_parts(p))
    parts[b] = new_part
    return wrap_like_point(p, Tuple(parts))
end

@inline function _btd_tangent_dot(a, b)
    s = sum(getproperty(a, :Ċ) .* getproperty(b, :Ċ))
    for (Am, Bm) in zip(getproperty(a, :U̇), getproperty(b, :U̇))
        s += sum(Am .* Bm)
    end
    return s
end

@inline function _btd_block_descent_direction(backend::BTDBackend, p, b::Int)
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "BTD block descent direction")
    pk = parts[b]
    eg_b = _btd_block_egrad(backend, parts, b)
    rg_b = egrad_to_rgrad(_backend_manifold(backend, b), pk, eg_b)
    decrease = _btd_tangent_dot(eg_b, rg_b)
    return pk, rg_b, decrease
end

function _btd_tsd_block_step(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    p,
    b::Int,
    c0,
    solver::BTDTSDSolver,
)
    backend = model.backend
    pk, direction, expected_decrease = _btd_block_descent_direction(backend, p, b)
    T = _scalar_eltype(p)
    if !isfinite(expected_decrease) || expected_decrease <= sqrt(eps(T))
        return p, c0, zero(T), 0, false
    end

    Mk = _backend_manifold(backend, b)
    α = T(solver.stepsize)
    α_min = T(solver.armijo_alpha_min)
    contraction = T(solver.armijo_contraction)
    sufficient_decrease = T(solver.armijo_sufficient_decrease)
    trials = 0

    while α >= α_min
        trials += 1
        qk = retract(Mk, pk, (-α) * direction)
        q = _replace_block_part(p, b, qk)
        cq = cost(model, q)
        if isfinite(cq) && cq <= c0 - sufficient_decrease * α * expected_decrease
            return q, cq, α, trials, true
        end
        α *= contraction
    end

    return p, c0, zero(T), trials, false
end

@inline function _btd_rel_error_from_cost(cost_val::T, normA2::T) where {T<:AbstractFloat}
    err2 = max(zero(T), T(2) * cost_val)
    normA2 <= eps(T) && return sqrt(err2)
    return sqrt(err2 / normA2)
end

function _btd_block_stats(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    p,
    solver::BTDTSDSolver;
    iterations::Int,
    converged::Bool,
    solver_info_extra = (;),
)
    backend = model.backend
    M = manifold(model)
    final_cost = cost(model, p)
    final_grad = rgrad(model, p)
    final_grad_norm = norm(M, p, final_grad)
    T = _scalar_eltype(p)
    normA2 = T(backend.target_normsq)
    rel_error = _btd_rel_error_from_cost(T(final_cost), normA2)
    solver_info = (
        total_iterations = iterations,
        schedule = solver.schedule,
        block_repeats = solver.block_repeats,
        block_count = backend.r,
        stepsize = solver.stepsize,
    )
    solver_info = merge(solver_info, solver_info_extra)
    return (
        point = p,
        cost = final_cost,
        rel_error = rel_error,
        grad_norm = final_grad_norm,
        iterations = iterations,
        converged = converged,
        solver = solver_symbol(solver),
        solver_info = solver_info,
    )
end

"""
    solve_btd_tsd(model, p0; kwargs...)
    solve_btd_tsd(model; init=:random, kwargs...)

Run blockwise tangent-subspace descent with per-block Armijo backtracking on a
BTD `JoinModel`.
"""
function solve_btd_tsd(
    model::JoinModel{<:AbstractFloat,<:BTDBackend},
    p0;
    maxiter::Int = 200,
    stepsize::Real = 1.0,
    tol::Real = 1e-6,
    verbose::Bool = true,
    return_stats::Bool = false,
    schedule::Symbol = :cyclic,
    block_repeats::Int = 1,
    armijo_contraction::Real = 0.5,
    armijo_sufficient_decrease::Real = 1e-4,
    armijo_alpha_min::Real = 1e-12,
)
    solver = BTDTSDSolver(;
        stepsize,
        schedule,
        block_repeats,
        armijo_contraction,
        armijo_sufficient_decrease,
        armijo_alpha_min,
    )
    backend = model.backend
    M = manifold(model)
    p = _solver_point(M, p0)
    T = _scalar_eltype(p)
    tol_T = T(tol)

    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))

    accepted_stepsize_history = Float64[]
    line_search_trial_history = Int[]
    failed_steps = 0
    accepted_steps = 0
    iters_done = 0
    converged = false
    c = cost(model, p)
    stop_reason = :maxiter
    progress =
        maxiter > 0 ?
        make_btd_tsd_progress(maxiter; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    for it = 1:maxiter
        g = rgrad(model, p)
        gnorm = norm(M, p, g)
        iters_done = it

        if verbose
            update_progress!(
                progress,
                it;
                showvalues = Any[
                    ("Cost", c),
                    ("Grad norm", gnorm),
                    ("Accepted steps", accepted_steps),
                    ("Failed steps", failed_steps),
                ],
            )
        end

        if !isfinite(c) || !isfinite(gnorm)
            stop_reason = :nonfinite
            break
        end
        if gnorm < tol_T
            converged = true
            stop_reason = :converged
            break
        end

        accepted_this_iter = 0
        for _ = 1:solver.block_repeats
            for b in _btd_block_schedule(backend.r, solver.schedule)
                p_new, c_new, α, trials, accepted =
                    _btd_tsd_block_step(model, p, b, c, solver)
                push!(accepted_stepsize_history, Float64(α))
                push!(line_search_trial_history, trials)
                if accepted
                    p = p_new
                    c = c_new
                    accepted_steps += 1
                    accepted_this_iter += 1
                else
                    failed_steps += 1
                end
            end
        end

        if accepted_this_iter == 0
            stop_reason = :stalled
            break
        end
    end

    final_grad_norm = norm(M, p, rgrad(model, p))
    converged |= final_grad_norm < tol_T
    converged && (stop_reason = :converged)

    if verbose
        status = if stop_reason == :converged
            "Converged"
        elseif stop_reason == :stalled
            "Stalled (no accepted block step)"
        elseif stop_reason == :nonfinite
            "Stopped (non-finite state)"
        else
            "Maximum iterations reached"
        end
        finish_progress!(
            progress;
            current = iters_done,
            showvalues = Any[("Status", status), ("Iterations", iters_done)],
        )
    end

    return_stats || return p
    solver_info_extra = (
        armijo_contraction = solver.armijo_contraction,
        armijo_sufficient_decrease = solver.armijo_sufficient_decrease,
        armijo_alpha_min = solver.armijo_alpha_min,
        accepted_steps = accepted_steps,
        failed_steps = failed_steps,
        accepted_stepsize_history = accepted_stepsize_history,
        line_search_trial_history = line_search_trial_history,
    )
    return _btd_block_stats(
        model,
        p,
        solver;
        iterations = iters_done,
        converged = converged,
        solver_info_extra = solver_info_extra,
    )
end

function solve_btd_tsd(
    model::JoinModel{<:AbstractFloat,<:BTDBackend};
    init::Union{Symbol,AbstractInitializer} = :random,
    verbose::Bool = true,
    kwargs...,
)
    p0 = initial_point(model, init; verbose)
    return solve_btd_tsd(model, p0; verbose, kwargs...)
end


function solve(
    solver::BTDTSDSolver,
    model::JoinModel{<:AbstractFloat,<:BTDBackend};
    init = :random,
    p0 = nothing,
    maxiter::Int = 500,
    tol::Real = 1e-6,
    gradient_mode = :riemannian,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = NoNormalization(),
    verbose::Bool = true,
    return_stats::Bool = false,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    kwargs...,
)
    gradient_mode_policy(gradient_mode) isa RiemannianGradientMode ||
        throw(ArgumentError("BTDTSDSolver supports only gradient_mode=:riemannian."))
    _normalization_policy(normalization) isa NoNormalization || throw(
        ArgumentError(
            "BTDTSDSolver does not support normalization. Use normalization=:none.",
        ),
    )
    p_start = isnothing(p0) ? initial_point(model, init; verbose) : p0
    return solve_btd_tsd(
        model,
        p_start;
        maxiter,
        stepsize = solver.stepsize,
        tol,
        verbose,
        return_stats,
        schedule = solver.schedule,
        block_repeats = solver.block_repeats,
        armijo_contraction = solver.armijo_contraction,
        armijo_sufficient_decrease = solver.armijo_sufficient_decrease,
        armijo_alpha_min = solver.armijo_alpha_min,
    )
end

function solve(::BTDTSDSolver, model::AbstractDecompositionModel; kwargs...)
    throw(
        ArgumentError(
            "BTDTSDSolver only supports JoinModel with BTDBackend, got $(typeof(model)).",
        ),
    )
end
