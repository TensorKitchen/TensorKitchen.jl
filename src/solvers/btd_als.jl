# solvers/btd_als.jl — Block coordinate ALS for BTD (sum of Tucker blocks)
export fit_btd_als

@inline function _btd_block_ranks(backend::BTDBackend, b::Int)
    M = _backend_manifold(backend, b)
    M isa Manifolds.Tucker || throw(
        ArgumentError(
            "BTD ALSSolver expects Tucker manifolds, got $(typeof(M)) at block $b.",
        ),
    )
    return multilinear_rank(M)
end

@inline function _btd_block_point(td::TuckerResult)
    return Manifolds.TuckerPoint(td.core, td.factors...)
end

@inline function _btd_block_tensor(p::Manifolds.TuckerPoint)
    core, factors = _tucker_data(p)
    return reconstruct_tucker(core, factors)
end

function _btd_tucker_point_to_result(p::Manifolds.TuckerPoint{T}) where {T<:AbstractFloat}
    core, factors = _tucker_data(p)
    N = length(factors)
    d = ndims(core)
    d == N || throw(DimensionMismatch("_btd_tucker_point_to_result: core ndims=$d, N=$N"))
    return TuckerResult{T,N}(core, collect(factors), collect(1:N), [T[] for _ = 1:N])
end

function _btd_block_fit_tucker(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int};
    method::Symbol = :hooi,
    block_maxiter::Int = 5,
    tol::Real = 1e-6,
    verbose::Bool = false,
    warm::Union{Nothing,Manifolds.TuckerPoint{T}} = nothing,
) where {T<:AbstractFloat,N}
    if method == :hooi
        hooi_init = if isnothing(warm)
            :sthosvd
        else
            td_w = _btd_tucker_point_to_result(warm)
            size(td_w.core) == ranks || throw(
                DimensionMismatch(
                    "BTD HOOI warm start: block Tucker ranks $(size(td_w.core)) != target ranks $ranks",
                ),
            )
            td_w
        end
        return hooi(
            A,
            ranks;
            maxiter = block_maxiter,
            tol = Float64(tol),
            init = hooi_init,
            verbose,
        )
    elseif method == :sthosvd
        return sthosvd(A, ranks)
    else
        throw(ArgumentError("Unknown block_method=$method. Use :hooi or :sthosvd."))
    end
end

function _btd_block_fit_tucker_projected(
    backend::BTDBackend{T,N},
    parts,
    b::Int,
    ranks::NTuple{N,Int};
    block_maxiter::Int = 5,
    tol::Real = 1e-6,
    warm::Manifolds.TuckerPoint{T},
) where {T<:AbstractFloat,N}
    warm_core, warm_factors = _tucker_data(warm)
    size(warm_core) == ranks || throw(
        DimensionMismatch(
            "BTD projected HOOI warm-start core size $(size(warm_core)) != target ranks $ranks",
        ),
    )

    factors = [copy(warm_factors[m]) for m = 1:N]
    singular_values = [T[] for _ = 1:N]
    residual_norm2 = _btd_residual_except_block_norm2(backend, parts, b)
    previous_rel_error = T(Inf)
    core = _btd_projected_residual_except_block_core(backend, parts, b, warm_factors)

    for _ = 1:block_maxiter
        factors_previous = copy(factors)
        @inbounds for mode = 1:N
            projection_factors = ntuple(k -> k < mode ? factors[k] : factors_previous[k], N)
            projected = _btd_projected_residual_except_block_mode(
                backend,
                parts,
                b,
                mode,
                projection_factors,
            )
            F = svd(unfold_mode(projected, mode))
            rank_mode = min(ranks[mode], size(F.U, 2))
            factors[mode] = Matrix(@view F.U[:, 1:rank_mode])
        end

        core = _btd_projected_residual_except_block_core(backend, parts, b, factors)
        residual_error2 = max(residual_norm2 - sum(abs2, core), zero(T))
        rel_error = _relative_error_frob_sq(residual_error2, residual_norm2)
        if previous_rel_error < T(Inf) && abs(previous_rel_error - rel_error) < T(tol)
            break
        end
        previous_rel_error = rel_error
    end

    return TuckerResult{T,N}(core, factors, collect(1:N), singular_values)
end

"""
    fit_btd_als(A, backend; init=:random, p0=nothing, maxiter=50,
        tol=1e-6, block_method=:hooi, block_maxiter=5, block_update=:auto,
        verbose=true,
        return_stats=false, max_stagnation_restarts=0,
        stagnation_rel_error=1e-4, restart_candidates=24,
        restart_screening_steps=5, restart_block_maxiter=10,
        restart_seed=nothing, progress_phase=:refinement)

Fit a BTD model with alternating block updates. `backend` defines the Tucker blocks and target layout.

`block_method` selects `:hooi` or `:sthosvd` for each block update. These are
finite approximate block solves, so the measured objective is not guaranteed to
decrease at every configured update. `block_update=:auto` uses projected,
residual-free updates for HOOI and the ambient reference path for ST-HOSVD.
Set `block_update=:ambient` to compare HOOI against the dense reference path.
`max_stagnation_restarts` controls optional reinitialization when a converged pass remains above `stagnation_rel_error`; the `restart_*` keywords configure that retry. With `return_stats=true`, return optimization statistics including the fitted point and relative error; otherwise return the fitted manifold point.

Most users should call [`btd`](@ref). This lower-level function is intended for experiments that already have a `BTDBackend`.
"""
function fit_btd_als(
    A::AbstractArray{T,N},
    backend::BTDBackend{T,N};
    init = :random,
    p0 = nothing,
    maxiter::Int = 50,
    tol::Real = 1e-6,
    block_method::Symbol = :hooi,
    block_maxiter::Int = 5,
    block_update::Symbol = :auto,
    verbose::Bool = true,
    return_stats::Bool = false,
    max_stagnation_restarts::Int = 0,
    stagnation_rel_error::Real = 1e-4,
    restart_candidates::Int = 24,
    restart_screening_steps::Int = 5,
    restart_block_maxiter::Int = 10,
    restart_seed = nothing,
    progress_phase::Symbol = :refinement,
) where {T<:AbstractFloat,N}
    block_update in (:auto, :projected, :ambient) ||
        throw(ArgumentError("block_update must be :auto, :projected, or :ambient"))
    block_update_eff =
        block_update == :auto ? (block_method == :hooi ? :projected : :ambient) :
        block_update
    block_update_eff == :projected &&
        block_method != :hooi &&
        throw(
            ArgumentError("block_update=:projected currently requires block_method=:hooi"),
        )
    !is_materialized(backend.target) &&
        block_update_eff != :projected &&
        throw(
            ArgumentError(
                "A lazy BTD target requires block_method=:hooi with " *
                "block_update=:projected. Set materialize=true to use ambient block updates.",
            ),
        )

    model = JoinModel{T,typeof(backend)}(backend)
    p0_eff =
        isnothing(p0) ?
        (
            block_method == :hooi ? initial_point(model, init; verbose) :
            initial_point(model, :random; verbose)
        ) : p0
    normA2 = backend.target_normsq
    restarts_done = 0
    total_iterations = 0
    restart_rel_errors = Float64[]

    function run_pass(p_start, iter_offset::Int)
        parts0 = point_parts(p_start)
        _check_parts_len(parts0, backend.r, "BTD ALS init")

        points = [parts0[k] for k = 1:backend.r]
        use_projected_updates = block_update_eff == :projected
        block_tensors =
            use_projected_updates ? nothing :
            [_btd_block_tensor(points[k]) for k = 1:backend.r]
        residual = use_projected_updates ? nothing : copy(A)
        if !use_projected_updates
            @inbounds for k = 1:backend.r
                residual .-= block_tensors[k]
            end
        end

        residual_norm2() =
            use_projected_updates ?
            max(T(2) * T(_btd_cost(backend, ArrayPartition(points...))), zero(T)) :
            T(sum(abs2, residual))

        prev_rel_error = T(Inf)
        converged = false
        stagnated = false
        iter_final = maxiter
        rel_error = _relative_error_frob_sq(residual_norm2(), T(normA2))
        progress =
            maxiter > 0 ?
            make_als_progress(
                maxiter;
                enabled = verbose,
                phase = progress_phase,
                dt = 0.2,
            ) : NoMethodProgress()

        for iter = 1:maxiter
            @inbounds for b = 1:backend.r
                ranks_b = _btd_block_ranks(backend, b)
                td = if use_projected_updates
                    _btd_block_fit_tucker_projected(
                        backend,
                        points,
                        b,
                        ranks_b;
                        block_maxiter,
                        tol,
                        warm = points[b],
                    )
                else
                    residual .+= block_tensors[b]
                    _btd_block_fit_tucker(
                        residual,
                        ranks_b;
                        method = block_method,
                        block_maxiter = block_maxiter,
                        tol = tol,
                        verbose = false,
                        warm = block_method == :hooi ? points[b] : nothing,
                    )
                end

                new_point = _btd_block_point(td)
                if !use_projected_updates
                    new_tensor = reconstruct(td)
                    residual .-= new_tensor
                    block_tensors[b] = new_tensor
                end
                points[b] = new_point
            end

            n_res = residual_norm2()
            rel_error = _relative_error_frob_sq(n_res, T(normA2))
            cost = T(0.5) * n_res
            fit_change = abs(prev_rel_error - rel_error)
            if verbose
                fit_change_display = isfinite(prev_rel_error) ? fit_change : "-"
                update_progress!(
                    progress,
                    iter;
                    showvalues = Any[
                        ("Total iter", iter_offset + iter),
                        ("Cost", cost),
                        ("RelErr", rel_error),
                        ("Δ RelErr", fit_change_display),
                    ],
                )
            end

            if fit_change < tol
                converged = true
                stagnated = rel_error > T(stagnation_rel_error)
                iter_final = iter
                break
            end
            prev_rel_error = rel_error
        end

        if verbose
            status =
                converged ? (stagnated ? "Stagnated" : "Converged") :
                "Maximum iterations reached"
            finish_progress!(
                progress;
                current = iter_final,
                showvalues = Any[
                    ("Status", status),
                    ("Pass iterations", iter_final),
                    ("Total iter", iter_offset + iter_final),
                ],
            )
        end

        n_fin = residual_norm2()
        point = ArrayPartition(points...)
        return (
            point = point,
            cost = T(0.5) * n_fin,
            rel_error = _relative_error_frob_sq(T(n_fin), T(normA2)),
            iterations = iter_final,
            converged = converged,
            stagnated = stagnated,
        )
    end

    result_pass = run_pass(p0_eff, total_iterations)
    total_iterations += result_pass.iterations
    while result_pass.stagnated && restarts_done < max_stagnation_restarts
        push!(restart_rel_errors, Float64(result_pass.rel_error))
        restarts_done += 1
        seed_eff = isnothing(restart_seed) ? nothing : Int(restart_seed) + restarts_done - 1
        restart_init = if is_materialized(backend.target)
            BTDHOSVDMultistartInit(
                restart_candidates;
                screening_steps = restart_screening_steps,
                block_method = block_method,
                block_maxiter = restart_block_maxiter,
                seed = seed_eff,
            )
        else
            BTDProjectedMultistartInit(
                restart_candidates;
                screening_steps = restart_screening_steps,
                block_maxiter = restart_block_maxiter,
                seed = seed_eff,
            )
        end
        p_restart = initial_point(model, restart_init; verbose)
        restarted = run_pass(p_restart, total_iterations)
        total_iterations += restarted.iterations
        if restarted.rel_error < result_pass.rel_error
            result_pass = restarted
        else
            break
        end
    end

    point = result_pass.point
    final_grad_norm = norm(manifold(model), point, rgrad(model, point))
    solver_info = (
        total_iterations = total_iterations,
        block_update = block_update_eff,
        stagnation_restarts = restarts_done,
        stagnation_rel_error = Float64(stagnation_rel_error),
        restart_candidates = restart_candidates,
        restart_rel_error_history = restart_rel_errors,
    )
    result = (
        point = point,
        cost = result_pass.cost,
        rel_error = result_pass.rel_error,
        grad_norm = final_grad_norm,
        iterations = total_iterations,
        converged = result_pass.converged && !result_pass.stagnated,
        solver = :btd_als,
        solver_info = solver_info,
    )
    return return_stats ? result : point
end

function solve(
    solver::ALSSolver,
    model::JoinModel{T,<:BTDBackend};
    maxiter::Int = 100,
    tol::Real = 1e-6,
    init = :random,
    p0 = nothing,
    verbose::Bool = true,
    return_stats::Bool = false,
    kwargs...,
) where {T<:AbstractFloat}
    backend = model.backend
    return fit_btd_als(
        backend.target,
        backend;
        init,
        p0,
        maxiter,
        tol,
        block_method = get(kwargs, :block_method, :hooi),
        block_maxiter = get(kwargs, :block_maxiter, 5),
        verbose,
        return_stats,
        max_stagnation_restarts = get(kwargs, :max_stagnation_restarts, 1),
        stagnation_rel_error = get(kwargs, :stagnation_rel_error, 1e-4),
        restart_candidates = get(kwargs, :restart_candidates, 24),
        restart_screening_steps = get(kwargs, :restart_screening_steps, 5),
        restart_block_maxiter = get(kwargs, :restart_block_maxiter, 10),
        restart_seed = get(kwargs, :restart_seed, nothing),
        progress_phase = :refinement,
    )
end
