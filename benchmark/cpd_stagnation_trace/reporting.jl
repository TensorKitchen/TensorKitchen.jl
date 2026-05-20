function write_tsv(path::AbstractString, rows)
    isempty(rows) && throw(ArgumentError("No benchmark rows to write."))
    mkpath(dirname(path))
    fields = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(fields, '\t'))
        for row in rows
            println(io, join((_stringify(getproperty(row, f)) for f in fields), '\t'))
        end
    end
    return path
end

function grouped_summary(rows)
    groups = Dict{NamedTuple,Vector{NamedTuple}}()
    for row in rows
        push!(get!(groups, condition_key(row), NamedTuple[]), row)
    end
    return sort!(collect(groups); by = x -> string(x[1]))
end

function write_summary(path::AbstractString, rows, opts::BenchmarkOptions)
    isempty(rows) && throw(ArgumentError("No benchmark rows to summarize."))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# CPD Stagnation Trace Benchmark")
        println(io)
        println(
            io,
            "- `cost_rel_change_tail_median`: median of the last $(TRACE_TAIL_LENGTH) relative cost changes.",
        )
        println(
            io,
            "- `component_delta_tail_median`: median of the last $(TRACE_TAIL_LENGTH) max rank-one term movements.",
        )
        println(
            io,
            "- `rgrad_top1_share_tail_median`: median of the last $(TRACE_TAIL_LENGTH) internal solver rgrad block-coordinate energy shares in the largest CP component.",
        )
        println(
            io,
            "- `rgrad_top3_share_tail_median`: median of the last $(TRACE_TAIL_LENGTH) internal solver rgrad block-coordinate energy shares in the largest three CP components.",
        )
        println(
            io,
            "- `rgrad_effective_components_tail_median`: effective number of CP components carrying internal solver rgrad block-coordinate energy (inverse HHI of the share vector).",
        )
        println(
            io,
            "- `rgrad_share_tail_median`: semicolon-separated tail median rgrad energy share per CP component (component 1;...;component r).",
        )
        println(
            io,
            "- `rgrad_share_entropy_tail_median`: tail median Shannon entropy of the rgrad share vector.",
        )
        println(
            io,
            "- `rgrad_share_normalized_entropy_tail_median`: tail median entropy divided by log(r), in [0,1] for uniform shares.",
        )
        println(
            io,
            "- `component_deltas_tail_median`: semicolon-separated tail median rank-one term movement per CP component.",
        )
        println(
            io,
            "- `start_rel_error`: relative error at the refinement start point (after init / ALS warm start, before RGD).",
        )
        println(
            io,
            "- `rgrad_argmax_tail_persistence`: fraction of tail iterations whose rgrad argmax matches the final argmax component.",
        )
        println(
            io,
            "- `dominant_component_delta_tail_median`: median movement of the tail rgrad-dominant CP component.",
        )
        println(
            io,
            "- `dominant_movement_pattern`: `large_rgrad_small_dominant_delta`, `large_rgrad_large_dominant_delta_flat_cost`, or `other`.",
        )
        println(
            io,
            "- For nonnegative pullback-style geometries, rgrad energy columns diagnose solver coordinates and should not be read as exact intrinsic pullback-metric component contributions.",
        )
        println(
            io,
            "- `converged_terms_stuck`: cost, rank-one terms, and gradient all stopped.",
        )
        println(
            io,
            "- `stagnated_terms_stuck_grad_large`: cost and rank-one terms stopped, but gradient is still large.",
        )
        println(
            io,
            "- `cost_flat_terms_moving`: cost is flat but at least one rank-one term still moves.",
        )
        println(io)
        println(
            io,
            "Settings: `maxiter=$(opts.maxiter)`, `tol=$(opts.tol)`, `cost_tol=$(opts.cost_tol)`, `component_tol=$(opts.component_tol)`, `grad_tol=$(opts.grad_tol)`.",
        )
        println(io)
        println(
            io,
            "| scenario | rank | noise | collinearity | solver | init | runs | median rel_error | median grad_norm | median time (s) | median cost tail | median component tail | median rgrad top1 | median rgrad top3 | median rgrad effective | classifications |",
        )
        println(
            io,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        )
        for (key, group) in grouped_summary(rows)
            rels = finite_median([row.rel_error for row in group])
            grads = finite_median([row.grad_norm for row in group])
            times = finite_median([row.time_seconds for row in group])
            cost_tail = finite_median([row.cost_rel_change_tail_median for row in group])
            component_tail =
                finite_median([row.component_delta_tail_median for row in group])
            rgrad_top1 = finite_median([row.rgrad_top1_share_tail_median for row in group])
            rgrad_top3 = finite_median([row.rgrad_top3_share_tail_median for row in group])
            rgrad_eff =
                finite_median([row.rgrad_effective_components_tail_median for row in group])
            classes = join(sort!(unique(string(row.classification) for row in group)), ", ")
            println(
                io,
                "| $(key.scenario) | $(key.rank) | $(_stringify(key.noise_level)) | $(_stringify(key.collinearity_noise)) | $(key.solver) | $(key.init) | $(length(group)) | $(_stringify(rels)) | $(_stringify(grads)) | $(_stringify(times)) | $(_stringify(cost_tail)) | $(_stringify(component_tail)) | $(_stringify(rgrad_top1)) | $(_stringify(rgrad_top3)) | $(_stringify(rgrad_eff)) | $classes |",
            )
        end
    end
    return path
end
