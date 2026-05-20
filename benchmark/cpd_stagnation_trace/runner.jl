function run_case(
    case::BenchmarkCase,
    solver::Symbol,
    seed::Int,
    init_override::Symbol,
    opts::BenchmarkOptions,
)
    Random.seed!(100_000 + 1_000 * seed)
    init = init_override == :default ? case.default_init : init_override
    GC.gc()
    timed = @timed cpd(
        case.tensor,
        case.rank;
        solver,
        init,
        warm_steps = case.warm_steps,
        nonnegative = case.nonnegative,
        maxiter = opts.maxiter,
        tol = opts.tol,
        verbose = false,
        component_trace = true,
    )
    res = timed.value
    si = solver_info(res)

    cost_rel_tail = finite_median(
        tail(solver_info_value(si, :component_trace_cost_rel_change_history, Float64[])),
    )
    component_delta_tail = finite_median(
        tail(solver_info_value(si, :component_trace_max_delta_history, Float64[])),
    )
    rgrad_top1_tail = finite_median(
        tail(solver_info_value(si, :component_trace_rgrad_top1_share_history, Float64[])),
    )
    rgrad_top2_tail = finite_median(
        tail(solver_info_value(si, :component_trace_rgrad_top2_share_history, Float64[])),
    )
    rgrad_top3_tail = finite_median(
        tail(solver_info_value(si, :component_trace_rgrad_top3_share_history, Float64[])),
    )
    rgrad_effective_tail = finite_median(
        tail(
            solver_info_value(
                si,
                :component_trace_rgrad_effective_components_history,
                Float64[],
            ),
        ),
    )
    accepted_stepsize_tail =
        finite_median(tail(solver_info_value(si, :accepted_stepsize_history, Float64[])))
    line_search_trials_tail =
        finite_median(tail(solver_info_value(si, :line_search_trial_history, Float64[])))
    delta_history = solver_info_value(si, :component_trace_delta_history, Vector{Float64}[])
    share_history =
        solver_info_value(si, :component_trace_rgrad_share_history, Vector{Float64}[])
    argmax_history =
        solver_info_value(si, :component_trace_rgrad_argmax_component_history, Int[])
    argmax_stats = _trace_argmax_persistence(argmax_history)
    dominant_delta_vals = _trace_dominant_component_deltas(delta_history, argmax_history)
    dominant_delta_tail = finite_median(dominant_delta_vals)
    rgrad_share_by_component = _trace_per_component_tail_medians(share_history)
    component_deltas_by_component = _trace_per_component_tail_medians(delta_history)
    rgrad_share_entropy_tail = _trace_share_entropy_tail_median(share_history)
    rgrad_share_normalized_entropy_tail =
        _trace_share_normalized_entropy_tail_median(share_history)
    start_rel_error = Float64(solver_info_value(si, :component_trace_start_rel_error, NaN))
    movement_pattern = _dominant_movement_pattern(
        rgrad_top1_tail,
        dominant_delta_tail,
        cost_rel_tail,
        opts,
    )
    class = classify_stagnation(
        cost_rel_tail,
        component_delta_tail,
        Float64(grad_norm(res)),
        opts.cost_tol,
        opts.component_tol,
        opts.grad_tol,
    )

    return (
        scenario = case.name,
        seed = seed,
        solver = solver,
        dims = join(size(case.tensor), "x"),
        rank = case.rank,
        nonnegative = case.nonnegative,
        noise_level = case.noise_level,
        collinearity_noise = case.collinearity_noise,
        init = init,
        maxiter = opts.maxiter,
        time_seconds = timed.time,
        allocated_mb = timed.bytes / 1024^2,
        gc_seconds = timed.gctime,
        iterations = iterations(res),
        converged = converged(res),
        start_rel_error = start_rel_error,
        rel_error = Float64(rel_error(res)),
        rel_error_improvement = isfinite(start_rel_error) ?
                                start_rel_error - Float64(rel_error(res)) : NaN,
        cost = Float64(cost(res)),
        grad_norm = Float64(grad_norm(res)),
        trace_len = length(solver_info_value(si, :component_trace_iterations, Int[])),
        cost_rel_change_tail_median = cost_rel_tail,
        component_delta_tail_median = component_delta_tail,
        component_delta_final = solver_info_value(
            si,
            :component_trace_final_max_delta,
            NaN,
        ),
        rgrad_top1_share_tail_median = rgrad_top1_tail,
        rgrad_top2_share_tail_median = rgrad_top2_tail,
        rgrad_top3_share_tail_median = rgrad_top3_tail,
        rgrad_effective_components_tail_median = rgrad_effective_tail,
        rgrad_share_tail_median = _format_component_vector(rgrad_share_by_component),
        rgrad_share_entropy_tail_median = rgrad_share_entropy_tail,
        rgrad_share_normalized_entropy_tail_median = rgrad_share_normalized_entropy_tail,
        component_deltas_tail_median = _format_component_vector(
            component_deltas_by_component,
        ),
        rgrad_argmax_component_final = solver_info_value(
            si,
            :component_trace_rgrad_argmax_component_final,
            0,
        ),
        rgrad_argmax_tail_persistence = argmax_stats.persistence,
        rgrad_argmax_tail_unique_count = argmax_stats.unique_count,
        dominant_component_delta_tail_median = dominant_delta_tail,
        dominant_movement_pattern = movement_pattern,
        accepted_stepsize_tail_median = accepted_stepsize_tail,
        line_search_trials_tail_median = line_search_trials_tail,
        classification = class,
    )
end

function _precompile_cpd_case(
    A,
    rank,
    solver,
    init;
    nonnegative = false,
    warm_steps = nothing,
)
    kwargs = (
        solver = solver,
        init = init,
        maxiter = 1,
        verbose = false,
        component_trace = true,
        nonnegative = nonnegative,
    )
    if init == :alswarm
        cpd(A, rank; kwargs..., warm_steps = something(warm_steps, 1))
    else
        cpd(A, rank; kwargs...)
    end
    return nothing
end

function precompile_solver_paths(solvers, inits)
    rng = MersenneTwister(0)
    λ = [1.0, 0.7]
    U = [_rand_unit_matrix(rng, d, 2) for d in (4, 3, 2)]
    A = _cp_tensor(λ, U)
    U_nonnegative = [_rand_nonnegative_unit_matrix(rng, d, 2) for d in (4, 3, 2)]
    A_nonnegative = _cp_tensor(abs.(λ), U_nonnegative)
    for solver in solvers, init in inits
        init == :default && continue
        _precompile_cpd_case(A, 2, solver, init)
        _precompile_cpd_case(A_nonnegative, 2, solver, init; nonnegative = true)
    end
    return nothing
end

function main(args = ARGS)
    opts = parse_args(args)
    unsupported = setdiff(opts.solvers, collect(SUPPORTED_SOLVERS))
    isempty(unsupported) || throw(
        ArgumentError(
            "Unsupported solvers: $unsupported. Use $(join(string.(SUPPORTED_SOLVERS), ", ")).",
        ),
    )

    precompile_solver_paths(opts.solvers, opts.inits)

    rows = NamedTuple[]
    total =
        length(opts.scenarios) *
        length(opts.seeds) *
        length(opts.solvers) *
        length(opts.inits)
    run_id = 0
    for scenario in opts.scenarios
        for seed in opts.seeds
            case = make_case(scenario, seed)
            for solver in opts.solvers
                for init in opts.inits
                    run_id += 1
                    ctx = merge(
                        (; run_id, total, seed, solver, init),
                        scenario_log_context(scenario),
                    )
                    @info "Running CPD benchmark" ctx...
                    try
                        push!(rows, run_case(case, solver, seed, init, opts))
                    catch err
                        @warn "Run failed" scenario seed solver init err
                        continue
                    end
                end
            end
        end
    end

    tsv = write_tsv(opts.out, rows)
    summary = write_summary(opts.summary, rows, opts)
    println("Wrote TSV results to $tsv")
    println("Wrote Markdown summary to $summary")
    if !isnothing(opts.concentration_summary)
        conc = write_concentration_summary(opts.concentration_summary, rows, opts)
        println("Wrote concentration summary to $conc")
    end
    return rows
end
