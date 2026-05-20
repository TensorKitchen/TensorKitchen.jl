function _parse_symbol_list(s::AbstractString)
    isempty(strip(s)) && return Symbol[]
    return Symbol.(strip.(split(s, ",")))
end

function _parse_int_list(s::AbstractString)
    isempty(strip(s)) && return Int[]
    return parse.(Int, strip.(split(s, ",")))
end

function _parse_float_list(s::AbstractString)
    isempty(strip(s)) && return Float64[]
    return parse.(Float64, strip.(split(s, ",")))
end

_cli_key(flag::AbstractString) = Symbol(replace(flag[3:end], "-" => "_"))

function _apply_quick!(opts::BenchmarkOptions)
    opts.quick = true
    opts.seeds = [1]
    opts.solvers = [:rgd]
    opts.scenarios =
        parse_scenarios((:exact_cp, :ill_conditioned_cp, :overranked_abs_randn))
    opts.maxiter = 100
    return opts
end

function _set_concentration_top3!(opts::BenchmarkOptions, value::AbstractString)
    opts.concentration_top3 = parse(Float64, value)
    opts.concentration_top3_user_set = true
    return nothing
end

function _set_concentration_effective_max!(opts::BenchmarkOptions, value::AbstractString)
    opts.concentration_effective_max = parse(Float64, value)
    opts.concentration_effective_max_user_set = true
    return nothing
end

function _cli_value(value, option::Symbol)
    isnothing(value) &&
        throw(ArgumentError("Expected `--$(replace(string(option), "_" => "-"))=value`."))
    return value
end

function _split_cli_arg(arg::AbstractString)
    arg == "-h" && return (:help, nothing)
    startswith(arg, "--") ||
        throw(ArgumentError("Unknown argument `$arg`. Use --help for usage."))
    parts = split(arg, "=", limit = 2)
    key = _cli_key(parts[1])
    value = length(parts) == 2 ? parts[2] : nothing
    return key, value
end

_apply_cli_option!(opts::BenchmarkOptions, ::Val{:help}, value) = (print_help(); exit(0))
_apply_cli_option!(opts::BenchmarkOptions, ::Val{:quick}, value) = _apply_quick!(opts)

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:noise_sweep_default}, value)
    opts.noise_sweep_default = true
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:out}, value)
    opts.out = _cli_value(value, :out)
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:summary}, value)
    opts.summary = _cli_value(value, :summary)
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:seeds}, value)
    opts.seeds = _parse_int_list(_cli_value(value, :seeds))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:solvers}, value)
    opts.solvers = _parse_symbol_list(_cli_value(value, :solvers))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:scenarios}, value)
    opts.scenarios = parse_scenarios(_parse_symbol_list(_cli_value(value, :scenarios)))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:init}, value)
    opts.inits = [Symbol(_cli_value(value, :init))]
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:inits}, value)
    opts.inits = _parse_symbol_list(_cli_value(value, :inits))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:maxiter}, value)
    opts.maxiter = parse(Int, _cli_value(value, :maxiter))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:tol}, value)
    opts.tol = parse(Float64, _cli_value(value, :tol))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:cost_tol}, value)
    opts.cost_tol = parse(Float64, _cli_value(value, :cost_tol))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:component_tol}, value)
    opts.component_tol = parse(Float64, _cli_value(value, :component_tol))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:grad_tol}, value)
    opts.grad_tol = parse(Float64, _cli_value(value, :grad_tol))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:rank}, value)
    opts.rank = parse(Int, _cli_value(value, :rank))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:overranked_rank}, value)
    opts.overranked_rank = parse(Int, _cli_value(value, :overranked_rank))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:replicates}, value)
    opts.replicates = parse(Int, _cli_value(value, :replicates))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:concentration_top1}, value)
    opts.concentration_top1 = parse(Float64, _cli_value(value, :concentration_top1))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:concentration_top3}, value)
    _set_concentration_top3!(opts, _cli_value(value, :concentration_top3))
    return opts
end

function _apply_cli_option!(
    opts::BenchmarkOptions,
    ::Val{:concentration_effective_max},
    value,
)
    _set_concentration_effective_max!(opts, _cli_value(value, :concentration_effective_max))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:concentration_rank_aware}, value)
    opts.concentration_rank_aware =
        parse(Bool, _cli_value(value, :concentration_rank_aware))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:concentration_summary}, value)
    opts.concentration_summary = _cli_value(value, :concentration_summary)
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:rel_error_tol}, value)
    opts.rel_error_tol = parse(Float64, _cli_value(value, :rel_error_tol))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:noise_level}, value)
    opts.noise_level = parse(Float64, _cli_value(value, :noise_level))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:noise_levels}, value)
    opts.noise_levels = _parse_float_list(_cli_value(value, :noise_levels))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:collinearity_noise}, value)
    opts.collinearity_noise = parse(Float64, _cli_value(value, :collinearity_noise))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{:collinearity_noises}, value)
    opts.collinearity_noises = _parse_float_list(_cli_value(value, :collinearity_noises))
    return opts
end

function _apply_cli_option!(opts::BenchmarkOptions, ::Val{K}, value) where {K}
    flag = replace(string(K), "_" => "-")
    throw(ArgumentError("Unknown argument `--$flag`. Use --help for usage."))
end

function _apply_cli_arg!(opts::BenchmarkOptions, arg::AbstractString)
    key, value = _split_cli_arg(arg)
    return _apply_cli_option!(opts, Val(key), value)
end

function parse_args(args)
    opts = BenchmarkOptions()

    for arg in args
        _apply_cli_arg!(opts, arg)
    end

    opts.scenarios = apply_rank_overrides(
        opts.scenarios;
        rank = opts.rank,
        overranked_rank = opts.overranked_rank,
    )
    opts.scenarios = apply_scenario_parameter_overrides(opts.scenarios, opts)
    normalize_noise_sweep!(opts)
    opts.scenarios = expand_parameter_sweeps(opts.scenarios, opts)
    if !isnothing(opts.replicates)
        opts.seeds = collect(1:opts.replicates)
    end

    validate_options!(opts)
    return opts
end

function print_help()
    println(
        """
CPD stagnation/component-motion benchmark

Usage:
  julia --project=. benchmark/cpd_stagnation_trace.jl [options]

Options:
  --quick                         Run a short smoke benchmark.
  --out=PATH                      TSV output path.
  --summary=PATH                  Markdown summary output path.
  --seeds=1,2,3                   Random seeds.
  --solvers=rgd,rcg               Solvers to run. Supported: $(join(string.(SUPPORTED_SOLVERS), ", ")).
  --scenarios=exact_cp,...        Scenarios to run.
  --init=default                  Single initializer. Use default, random, tucker, or alswarm.
  --inits=alswarm,random         Initializers to compare in one run.
  --maxiter=1000                  Solver iteration budget.
  --tol=1e-8                      Solver tolerance.
  --cost-tol=1e-8                 Tail cost-relative-change threshold.
  --component-tol=1e-7            Tail component-motion threshold.
  --grad-tol=1e-6                 Final gradient-norm threshold.
  --rank=10                       Override rank for exact_cp, ill_conditioned_cp, noisy_exact_cp, exact_nncp, border_rank_w.
  --overranked-rank=50            Override rank for overranked_abs_randn only.
  --noise-level=1e-3              Override noise level for noisy_exact_cp (0 = exact control).
  --noise-levels=0,1e-4,...       Expand noisy_exact_cp into one run per noise level (0 = σ=0 exact control).
  --noise-sweep-default           Use default noise sweep: 0,1e-4,1e-3,1e-2,1e-1.
  --collinearity-noise=1e-3       Override collinearity noise for ill_conditioned_cp.
  --collinearity-noises=1e-1,...  Expand ill_conditioned_cp into one run per collinearity noise.
  --replicates=50                 Shorthand for --seeds=1,2,...,50.
  --concentration-top1=0.5        Top-1 rgrad share threshold for concentration counts.
  --concentration-top3=0.9        Top-3 rgrad share threshold (overrides rank-aware default).
  --concentration-effective-max=2 Effective rgrad component threshold (overrides rank-aware default).
  --concentration-rank-aware=true Adjust top3/effective thresholds from rank (default true).
  --concentration-summary=PATH    Markdown table of concentration frequencies by condition.
  --rel-error-tol=1e-6            Good-fit rel_error threshold for benign/pathological counts.

Scenarios:
  exact_cp                        Exact signed CP tensor, rank 3 (override with --rank).
  ill_conditioned_cp              Exact CP tensor with nearly collinear terms.
  noisy_exact_cp                  Exact signed CP plus additive Gaussian noise (override with --noise-level or --noise-levels).
  exact_nncp                      Exact nonnegative CP tensor, rank 3.
  border_rank_w                   2x2x2 W-state tensor with border rank 2 and rank 3.
  overranked_abs_randn            B = abs.(randn(20, 15, 10)), rank 35 (override with --overranked-rank).
""",
    )
end
