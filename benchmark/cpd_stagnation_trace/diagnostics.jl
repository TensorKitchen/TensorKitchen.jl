function tail(v, n::Int = TRACE_TAIL_LENGTH)
    isempty(v) && return v
    return v[max(1, length(v) - n + 1):end]
end

function finite_median(v)
    vals = sort!(Float64[x for x in v if isfinite(x)])
    isempty(vals) && return NaN
    mid = length(vals) ÷ 2
    return isodd(length(vals)) ? vals[mid+1] : (vals[mid] + vals[mid+1]) / 2
end

function solver_info_value(si, name::Symbol, default)
    return hasproperty(si, name) ? getproperty(si, name) : default
end

_stringify(x::AbstractFloat) = isfinite(x) ? @sprintf("%.8e", x) : string(x)
_stringify(x) = string(x)

function _format_component_vector(v::AbstractVector{<:Real})
    isempty(v) && return ""
    return join([_stringify(x) for x in v], ";")
end

function condition_key(row)
    return (
        scenario = row.scenario,
        rank = row.rank,
        nonnegative = row.nonnegative,
        noise_level = row.noise_level,
        collinearity_noise = row.collinearity_noise,
        solver = row.solver,
        init = row.init,
    )
end

function _aligned_trace_tail(delta_history, argmax_history, n::Int = TRACE_TAIL_LENGTH)
    m = min(length(delta_history), length(argmax_history), n)
    m == 0 && return (delta_history = Vector{Vector{Float64}}[], argmax_history = Int[])

    delta_tail = delta_history[(end-m+1):end]
    argmax_tail = argmax_history[(end-m+1):end]

    return (delta_history = delta_tail, argmax_history = argmax_tail)
end

function _trace_argmax_persistence(argmax_history::Vector{Int}, n::Int = TRACE_TAIL_LENGTH)
    t = tail(argmax_history, n)
    isempty(t) && return (persistence = NaN, unique_count = 0)
    valid = [k for k in t if k > 0]
    isempty(valid) && return (persistence = NaN, unique_count = 0)
    final = valid[end]
    persistence = count(==(final), valid) / length(valid)
    return (persistence = persistence, unique_count = length(unique(valid)))
end

function _trace_dominant_component_deltas(
    delta_history,
    argmax_history,
    n::Int = TRACE_TAIL_LENGTH,
)
    aligned = _aligned_trace_tail(delta_history, argmax_history, n)
    vals = Float64[]
    for (deltas, k) in zip(aligned.delta_history, aligned.argmax_history)
        if k > 0 && k <= length(deltas) && isfinite(deltas[k])
            push!(vals, deltas[k])
        end
    end
    return vals
end

function _trace_per_component_tail_medians(
    history::Vector{Vector{Float64}},
    n::Int = TRACE_TAIL_LENGTH,
)
    t = tail(history, n)
    isempty(t) && return Float64[]
    r = maximum(length, t)
    r == 0 && return Float64[]
    medians = Vector{Float64}(undef, r)
    for k = 1:r
        vals = Float64[]
        for row in t
            if k <= length(row) && isfinite(row[k])
                push!(vals, row[k])
            end
        end
        medians[k] = finite_median(vals)
    end
    return medians
end

function _share_entropy(shares::AbstractVector{<:Real})
    s = 0.0
    for p in shares
        if p > 0 && isfinite(p)
            s -= p * log(p)
        end
    end
    return s
end

function _share_normalized_entropy(shares::AbstractVector{<:Real})
    r = length(shares)
    (r <= 1 || all(!isfinite, shares)) && return NaN
    entropy = _share_entropy(shares)
    isfinite(entropy) || return NaN
    return entropy / log(r)
end

function _trace_share_entropy_tail_median(
    share_history::Vector{Vector{Float64}},
    n::Int = TRACE_TAIL_LENGTH,
)
    entropies = [_share_entropy(sh) for sh in tail(share_history, n)]
    return finite_median(entropies)
end

function _trace_share_normalized_entropy_tail_median(
    share_history::Vector{Vector{Float64}},
    n::Int = TRACE_TAIL_LENGTH,
)
    entropies = [_share_normalized_entropy(sh) for sh in tail(share_history, n)]
    return finite_median(entropies)
end

function concentration_thresholds_for_rank(
    rank::Int;
    top1 = DEFAULT_CONCENTRATION_TOP1,
    top3 = nothing,
    effective_max = nothing,
)
    uniform_top3 = min(3, rank) / rank
    top3_eff = something(top3, clamp(uniform_top3 + 0.15, 0.65, 0.95))
    effective_eff = something(effective_max, clamp(2.0 + 0.25 * (rank - 3), 2.0, 4.0))
    return (top1 = top1, top3 = top3_eff, effective_max = effective_eff)
end

function concentration_thresholds_for_row(row, opts::BenchmarkOptions)
    if !opts.concentration_rank_aware
        return (
            top1 = opts.concentration_top1,
            top3 = opts.concentration_top3,
            effective_max = opts.concentration_effective_max,
        )
    end

    defaults = concentration_thresholds_for_rank(row.rank; top1 = opts.concentration_top1)
    top3 = opts.concentration_top3_user_set ? opts.concentration_top3 : defaults.top3
    effective_max =
        opts.concentration_effective_max_user_set ? opts.concentration_effective_max :
        defaults.effective_max
    return (top1 = opts.concentration_top1, top3 = top3, effective_max = effective_max)
end

function _dominant_movement_pattern(
    rgrad_top1_tail,
    dominant_delta_tail,
    cost_rel_tail,
    opts::BenchmarkOptions,
)
    if !isfinite(rgrad_top1_tail) ||
       !isfinite(dominant_delta_tail) ||
       !isfinite(cost_rel_tail)
        return :unknown
    end
    rgrad_high = rgrad_top1_tail >= opts.concentration_top1
    delta_small = dominant_delta_tail <= opts.component_tol
    delta_large = dominant_delta_tail > opts.component_tol
    cost_flat = cost_rel_tail <= opts.cost_tol
    if rgrad_high && delta_small
        return :large_rgrad_small_dominant_delta
    elseif rgrad_high && delta_large && cost_flat
        return :large_rgrad_large_dominant_delta_flat_cost
    elseif rgrad_high && delta_large
        return :large_rgrad_large_dominant_delta
    end
    return :other
end

function classify_stagnation(
    cost_tail,
    component_tail,
    grad_norm,
    cost_tol,
    component_tol,
    grad_tol,
)
    if !isfinite(cost_tail) || !isfinite(component_tail) || !isfinite(grad_norm)
        return :no_trace
    end

    cost_flat = cost_tail <= cost_tol
    terms_stuck = component_tail <= component_tol
    grad_small = grad_norm <= grad_tol

    if cost_flat && terms_stuck && grad_small
        return :converged_terms_stuck
    elseif cost_flat && terms_stuck && !grad_small
        return :stagnated_terms_stuck_grad_large
    elseif cost_flat && !terms_stuck
        return :cost_flat_terms_moving
    elseif !cost_flat && grad_small
        return :small_grad_but_cost_changing
    else
        return :not_stagnated_or_still_descending
    end
end
