const CONCENTRATION_OUTCOME_GROUPS = (
    :missing_rgrad_trace,
    :rank_degenerate,
    :concentrated_converged,
    :concentrated_cost_flat_grad_large,
    :concentrated_grad_large,
    :concentrated_maxiter,
    :nonconcentrated_converged,
    :nonconcentrated_grad_large,
    :other,
)

function _row_has_rgrad_trace(row)
    return row.trace_len > 0 &&
           isfinite(row.rgrad_top1_share_tail_median) &&
           isfinite(row.rgrad_top3_share_tail_median) &&
           isfinite(row.rgrad_effective_components_tail_median)
end

function _row_concentrated_with(row, top1::Float64, top3::Float64, effective_max::Float64)
    return _row_has_rgrad_trace(row) &&
           row.rgrad_top1_share_tail_median >= top1 &&
           row.rgrad_top3_share_tail_median >= top3 &&
           row.rgrad_effective_components_tail_median <= effective_max
end

function _row_concentrated(row, opts::BenchmarkOptions)
    θ = concentration_thresholds_for_row(row, opts)
    return _row_concentrated_with(row, θ.top1, θ.top3, θ.effective_max)
end

function _row_grad_large(row, opts::BenchmarkOptions)
    return isfinite(row.grad_norm) && row.grad_norm > opts.grad_tol
end

function _row_cost_flat(row, opts::BenchmarkOptions)
    return isfinite(row.cost_rel_change_tail_median) &&
           row.cost_rel_change_tail_median <= opts.cost_tol
end

function _row_good_fit(row, opts::BenchmarkOptions)
    return isfinite(row.rel_error) && row.rel_error <= opts.rel_error_tol
end

function _row_hit_maxiter(row)
    return !row.converged && row.iterations >= row.maxiter
end

function _row_benign_concentrated(row, opts::BenchmarkOptions)
    return _row_concentrated(row, opts) &&
           _row_good_fit(row, opts) &&
           !_row_grad_large(row, opts)
end

function _row_pathological_concentrated(row, opts::BenchmarkOptions)
    return _row_concentrated(row, opts) &&
           !_row_good_fit(row, opts) &&
           (
               _row_grad_large(row, opts) ||
               _row_cost_flat(row, opts) ||
               _row_hit_maxiter(row)
           )
end

function _concentration_eligible(row)
    return row.rank > 2 && _row_has_rgrad_trace(row)
end

function _concentration_outcome_group(row, opts::BenchmarkOptions)
    !_row_has_rgrad_trace(row) && return :missing_rgrad_trace
    row.rank <= 2 && return :rank_degenerate

    concentrated = _row_concentrated(row, opts)
    grad_large = _row_grad_large(row, opts)
    cost_flat = _row_cost_flat(row, opts)
    hit_maxiter = _row_hit_maxiter(row)

    if concentrated && row.converged
        return :concentrated_converged
    elseif concentrated && grad_large && cost_flat
        return :concentrated_cost_flat_grad_large
    elseif concentrated && grad_large
        return :concentrated_grad_large
    elseif concentrated && hit_maxiter
        return :concentrated_maxiter
    elseif !concentrated && row.converged
        return :nonconcentrated_converged
    elseif !concentrated && grad_large
        return :nonconcentrated_grad_large
    end

    return :other
end

function _group_median(rows, field)
    return finite_median([getproperty(row, field) for row in rows])
end

function _condition_heading(key)
    nn = key.nonnegative ? "nonnegative" : "signed"
    parts = String["$(key.scenario)", "rank=$(key.rank)"]
    if isfinite(key.noise_level)
        push!(parts, "noise_level=$(key.noise_level)")
    end
    if isfinite(key.collinearity_noise)
        push!(parts, "collinearity_noise=$(key.collinearity_noise)")
    end
    push!(parts, "solver=$(key.solver)", "init=$(key.init)", nn)
    return join(parts, ", ")
end

function _write_concentration_independent_counts(io, rows, opts::BenchmarkOptions)
    eligible = [row for row in rows if _concentration_eligible(row)]
    n = length(rows)
    n_eligible = length(eligible)
    println(io, "Independent counts (rank > 2 with rgrad trace: $n_eligible / $n runs):")
    if n_eligible == 0
        println(io)
        return
    end
    n_conc = count(row -> _row_concentrated(row, opts), eligible)
    n_conc_fixed = count(
        row -> _row_concentrated_with(
            row,
            opts.concentration_top1,
            FIXED_CONCENTRATION_TOP3,
            FIXED_CONCENTRATION_EFFECTIVE_MAX,
        ),
        eligible,
    )
    n_grad_large = count(row -> _row_grad_large(row, opts), eligible)
    n_cost_flat = count(row -> _row_cost_flat(row, opts), eligible)
    n_hit_maxiter = count(_row_hit_maxiter, eligible)
    n_conc_grad =
        count(row -> _row_concentrated(row, opts) && _row_grad_large(row, opts), eligible)
    n_conc_cost =
        count(row -> _row_concentrated(row, opts) && _row_cost_flat(row, opts), eligible)
    n_conc_conv = count(row -> _row_concentrated(row, opts) && row.converged, eligible)
    n_benign = count(row -> _row_benign_concentrated(row, opts), eligible)
    n_pathological = count(row -> _row_pathological_concentrated(row, opts), eligible)
    n_large_rgrad_small_delta = count(
        row -> row.dominant_movement_pattern == :large_rgrad_small_dominant_delta,
        eligible,
    )
    n_large_rgrad_large_delta_flat = count(
        row ->
            row.dominant_movement_pattern == :large_rgrad_large_dominant_delta_flat_cost,
        eligible,
    )
    n_argmax_persistent = count(
        row ->
            isfinite(row.rgrad_argmax_tail_persistence) &&
                row.rgrad_argmax_tail_persistence >= 0.8,
        eligible,
    )
    start_small = count(
        row ->
            isfinite(row.start_rel_error) && row.start_rel_error <= opts.rel_error_tol,
        eligible,
    )
    start_large = count(
        row ->
            isfinite(row.start_rel_error) && row.start_rel_error > opts.rel_error_tol,
        eligible,
    )
    println(io, "- Concentrated: $n_conc / $n_eligible")
    if opts.concentration_rank_aware
        println(
            io,
            "- Concentrated (fixed top3=$(FIXED_CONCENTRATION_TOP3), effective≤$(FIXED_CONCENTRATION_EFFECTIVE_MAX)): $n_conc_fixed / $n_eligible",
        )
    end
    println(io, "- Concentrated + grad_large: $n_conc_grad / $n_eligible")
    println(io, "- Concentrated + cost_flat: $n_conc_cost / $n_eligible")
    println(io, "- Concentrated + converged: $n_conc_conv / $n_eligible")
    println(io, "- Benign concentrated (good fit, small grad): $n_benign / $n_eligible")
    println(
        io,
        "- Pathological concentrated (bad fit or flat/maxiter + large grad): $n_pathological / $n_eligible",
    )
    println(io, "- start_rel_error ≤ rel_error_tol: $start_small / $n_eligible")
    println(io, "- start_rel_error > rel_error_tol: $start_large / $n_eligible")
    println(io, "- rgrad argmax tail persistence ≥ 0.8: $n_argmax_persistent / $n_eligible")
    println(
        io,
        "- large rgrad + small dominant delta: $n_large_rgrad_small_delta / $n_eligible",
    )
    println(
        io,
        "- large rgrad + large dominant delta + flat cost: $n_large_rgrad_large_delta_flat / $n_eligible",
    )
    println(io, "- grad_large (any): $n_grad_large / $n_eligible")
    println(io, "- cost_flat (any): $n_cost_flat / $n_eligible")
    println(io, "- hit_maxiter (any): $n_hit_maxiter / $n_eligible")
    println(io)
end

function _condition_threshold_summary(rows, opts::BenchmarkOptions)
    isempty(rows) && return "n/a"
    thresholds = unique(concentration_thresholds_for_row(row, opts) for row in rows)
    return join(
        (
            "top1=$(θ.top1), top3=$(θ.top3), effective≤$(θ.effective_max)" for
            θ in thresholds
        ),
        "; ",
    )
end

function write_concentration_summary(path::AbstractString, rows, opts::BenchmarkOptions)
    isempty(rows) && throw(ArgumentError("No benchmark rows to summarize."))
    mkpath(dirname(path))

    by_condition = Dict{NamedTuple,Vector{NamedTuple}}()
    for row in rows
        push!(get!(by_condition, condition_key(row), NamedTuple[]), row)
    end

    open(path, "w") do io
        println(io, "# Concentration × outcome summary")
        println(io)
        println(
            io,
            "Settings: `maxiter=$(opts.maxiter)`, `tol=$(opts.tol)`, `cost_tol=$(opts.cost_tol)`, `component_tol=$(opts.component_tol)`, `grad_tol=$(opts.grad_tol)`, `rel_error_tol=$(opts.rel_error_tol)`.",
        )
        println(
            io,
            "Concentrated (rank > 2 only): top1 ≥ $(opts.concentration_top1), top3/effective thresholds are evaluated per row when rank-aware mode is enabled.",
        )
        if opts.concentration_rank_aware
            println(
                io,
                "Rank-aware concentration thresholds are enabled (override with --concentration-top3 / --concentration-effective-max / --concentration-rank-aware=false).",
            )
        end
        println(
            io,
            "Fixed-threshold sensitivity reference: top3 ≥ $(FIXED_CONCENTRATION_TOP3), effective ≤ $(FIXED_CONCENTRATION_EFFECTIVE_MAX).",
        )
        println(
            io,
            "Rank ≤ 2 runs are labeled `rank_degenerate` (uniform two-component split can satisfy concentration thresholds).",
        )
        println(io)
        println(
            io,
            "Outcome groups (first match wins): `missing_rgrad_trace`, `rank_degenerate`, `concentrated_converged`, `concentrated_cost_flat_grad_large`, `concentrated_grad_large`, `concentrated_maxiter`, `nonconcentrated_converged`, `nonconcentrated_grad_large`, `other`.",
        )
        println(
            io,
            "`grad_large` / `cost_flat` use final `grad_norm` and tail cost relative change, not classification symbols.",
        )
        println(io)

        for key in sort!(collect(keys(by_condition)); by = x -> string(x))
            condition_rows = by_condition[key]
            grouped = Dict{Symbol,Vector{NamedTuple}}(
                group => NamedTuple[] for group in CONCENTRATION_OUTCOME_GROUPS
            )
            for row in condition_rows
                push!(grouped[_concentration_outcome_group(row, opts)], row)
            end

            println(io, "## $(_condition_heading(key))")
            println(io)
            println(io, "Total runs: $(length(condition_rows)).")
            println(
                io,
                "Concentration thresholds: $(_condition_threshold_summary(condition_rows, opts)).",
            )
            _write_concentration_independent_counts(io, condition_rows, opts)
            println(
                io,
                "| group | runs | median start_rel_error | median rel_error | median grad_norm | median iterations | median argmax persistence | median dominant delta | median step size | movement patterns |",
            )
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for group in CONCENTRATION_OUTCOME_GROUPS
                group_rows = grouped[group]
                n = length(group_rows)
                if n == 0
                    println(io, "| $group | 0 | — | — | — | — | — | — | — | — |")
                    continue
                end
                patterns = join(
                    sort!(
                        unique(string(row.dominant_movement_pattern) for row in group_rows),
                    ),
                    ", ",
                )
                println(
                    io,
                    "| $group | $n | $(_stringify(_group_median(group_rows, :start_rel_error))) | $(_stringify(_group_median(group_rows, :rel_error))) | $(_stringify(_group_median(group_rows, :grad_norm))) | $(_stringify(_group_median(group_rows, :iterations))) | $(_stringify(_group_median(group_rows, :rgrad_argmax_tail_persistence))) | $(_stringify(_group_median(group_rows, :dominant_component_delta_tail_median))) | $(_stringify(_group_median(group_rows, :accepted_stepsize_tail_median))) | $patterns |",
                )
            end
            println(io)
        end
    end
    return path
end
