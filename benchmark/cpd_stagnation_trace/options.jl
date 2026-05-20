Base.@kwdef mutable struct BenchmarkOptions
    quick::Bool = false
    out::String = joinpath(@__DIR__, "..", "results", "cpd_stagnation_trace.tsv")
    summary::String = joinpath(@__DIR__, "..", "results", "cpd_stagnation_trace.md")
    seeds::Vector{Int} = collect(1:3)
    solvers::Vector{Symbol} = collect(DEFAULT_SOLVERS)
    scenarios::Vector{AbstractCPDBenchmarkScenario} = parse_scenarios(DEFAULT_SCENARIOS)
    inits::Vector{Symbol} = [:default]
    maxiter::Int = 1000
    tol::Float64 = 1e-8
    cost_tol::Float64 = 1e-8
    component_tol::Float64 = 1e-7
    grad_tol::Float64 = 1e-6
    rank::Union{Int,Nothing} = nothing
    overranked_rank::Union{Int,Nothing} = nothing
    replicates::Union{Int,Nothing} = nothing
    concentration_top1::Float64 = DEFAULT_CONCENTRATION_TOP1
    concentration_top3::Float64 = FIXED_CONCENTRATION_TOP3
    concentration_effective_max::Float64 = FIXED_CONCENTRATION_EFFECTIVE_MAX
    concentration_rank_aware::Bool = true
    concentration_top3_user_set::Bool = false
    concentration_effective_max_user_set::Bool = false
    concentration_summary::Union{String,Nothing} = nothing
    rel_error_tol::Float64 = 1e-6
    noise_level::Union{Float64,Nothing} = nothing
    noise_levels::Vector{Float64} = Float64[]
    noise_sweep_default::Bool = false
    collinearity_noise::Union{Float64,Nothing} = nothing
    collinearity_noises::Vector{Float64} = Float64[]
end

function validate_options!(opts::BenchmarkOptions)
    isempty(opts.scenarios) && throw(ArgumentError("At least one scenario is required."))
    isempty(opts.solvers) && throw(ArgumentError("At least one solver is required."))
    isempty(opts.inits) && throw(ArgumentError("At least one init is required."))
    isempty(opts.seeds) && throw(ArgumentError("At least one seed is required."))

    opts.maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1."))
    opts.tol > 0 || throw(ArgumentError("tol must be positive."))
    opts.cost_tol > 0 || throw(ArgumentError("cost_tol must be positive."))
    opts.component_tol > 0 || throw(ArgumentError("component_tol must be positive."))
    opts.grad_tol > 0 || throw(ArgumentError("grad_tol must be positive."))
    opts.rel_error_tol > 0 || throw(ArgumentError("rel_error_tol must be positive."))
    opts.concentration_top3 >= opts.concentration_top1 ||
        @warn "concentration_top3 < concentration_top1 is unusual."

    if !isnothing(opts.rank)
        opts.rank >= 1 || throw(ArgumentError("rank must be >= 1."))
    end
    if !isnothing(opts.overranked_rank)
        opts.overranked_rank >= 1 || throw(ArgumentError("overranked_rank must be >= 1."))
    end
    if !isnothing(opts.replicates)
        opts.replicates >= 1 || throw(ArgumentError("replicates must be >= 1."))
    end

    0 <= opts.concentration_top1 <= 1 ||
        throw(ArgumentError("concentration_top1 must be in [0,1]."))
    0 <= opts.concentration_top3 <= 1 ||
        throw(ArgumentError("concentration_top3 must be in [0,1]."))
    opts.concentration_effective_max > 0 ||
        throw(ArgumentError("concentration_effective_max must be positive."))

    if !isnothing(opts.noise_level)
        opts.noise_level >= 0 || throw(ArgumentError("noise_level must be >= 0."))
    end
    all(>=(0), opts.noise_levels) ||
        throw(ArgumentError("noise_levels must all be >= 0 (use 0 for exact CP control)."))
    if !isnothing(opts.collinearity_noise)
        opts.collinearity_noise >= 0 ||
            throw(ArgumentError("collinearity_noise must be >= 0."))
    end
    all(>=(0), opts.collinearity_noises) ||
        throw(ArgumentError("collinearity_noises must all be >= 0."))

    return opts
end

function normalize_noise_sweep!(opts::BenchmarkOptions)
    opts.noise_sweep_default || return opts
    opts.noise_levels = collect(DEFAULT_NOISE_SWEEP_LEVELS)
    return opts
end

apply_parameter_override(scenario::AbstractCPDBenchmarkScenario, opts::BenchmarkOptions) =
    scenario

apply_parameter_override(scenario::NoisyExactCP, opts::BenchmarkOptions) =
    isnothing(opts.noise_level) ? scenario :
    set_scenario_noise_level(scenario, opts.noise_level)

apply_parameter_override(scenario::IllConditionedCP, opts::BenchmarkOptions) =
    isnothing(opts.collinearity_noise) ? scenario :
    set_scenario_collinearity_noise(scenario, opts.collinearity_noise)

expand_parameter_sweep(scenario::AbstractCPDBenchmarkScenario, opts::BenchmarkOptions) =
    AbstractCPDBenchmarkScenario[scenario]

expand_parameter_sweep(scenario::NoisyExactCP, opts::BenchmarkOptions) =
    isempty(opts.noise_levels) ? AbstractCPDBenchmarkScenario[scenario] :
    AbstractCPDBenchmarkScenario[
        set_scenario_noise_level(scenario, σ) for σ in opts.noise_levels
    ]

expand_parameter_sweep(scenario::IllConditionedCP, opts::BenchmarkOptions) =
    isempty(opts.collinearity_noises) ? AbstractCPDBenchmarkScenario[scenario] :
    AbstractCPDBenchmarkScenario[
        set_scenario_collinearity_noise(scenario, η) for η in opts.collinearity_noises
    ]

function apply_scenario_parameter_overrides(
    scenarios::Vector{AbstractCPDBenchmarkScenario},
    opts::BenchmarkOptions,
)
    return apply_parameter_override.(scenarios, Ref(opts))
end

function expand_parameter_sweeps(
    scenarios::Vector{AbstractCPDBenchmarkScenario},
    opts::BenchmarkOptions,
)
    return vcat(expand_parameter_sweep.(scenarios, Ref(opts))...)
end
