abstract type AbstractBenchmarkScenario end
abstract type AbstractCPDBenchmarkScenario <: AbstractBenchmarkScenario end

struct ExactCP <: AbstractCPDBenchmarkScenario
    dims::NTuple{3,Int}
    rank::Int
    warm_steps::Int
end

struct IllConditionedCP <: AbstractCPDBenchmarkScenario
    dims::NTuple{3,Int}
    rank::Int
    collinearity_noise::Float64
    warm_steps::Int
end

struct NoisyExactCP <: AbstractCPDBenchmarkScenario
    dims::NTuple{3,Int}
    rank::Int
    noise_level::Float64
    warm_steps::Int
end

struct ExactNNCP <: AbstractCPDBenchmarkScenario
    dims::NTuple{3,Int}
    rank::Int
    warm_steps::Int
end

struct BorderRankW <: AbstractCPDBenchmarkScenario
    rank::Int
    warm_steps::Int
end

struct OverrankedAbsRandn <: AbstractCPDBenchmarkScenario
    dims::NTuple{3,Int}
    rank::Int
    warm_steps::Int
end

ExactCP(; dims = (20, 15, 10), rank = 3, warm_steps = 100) = ExactCP(dims, rank, warm_steps)
IllConditionedCP(;
    dims = (20, 15, 10),
    rank = 3,
    collinearity_noise = 1e-3,
    warm_steps = 100,
) = IllConditionedCP(dims, rank, collinearity_noise, warm_steps)
NoisyExactCP(; dims = (20, 15, 10), rank = 3, noise_level = 1e-3, warm_steps = 100) =
    NoisyExactCP(dims, rank, noise_level, warm_steps)
ExactNNCP(; dims = (20, 15, 10), rank = 3, warm_steps = 100) =
    ExactNNCP(dims, rank, warm_steps)
BorderRankW(; rank = 2, warm_steps = 100) = BorderRankW(rank, warm_steps)
OverrankedAbsRandn(; dims = (20, 15, 10), rank = 35, warm_steps = 100) =
    OverrankedAbsRandn(dims, rank, warm_steps)

parse_scenario(::Val{:exact_cp}) = ExactCP()
parse_scenario(::Val{:ill_conditioned_cp}) = IllConditionedCP()
parse_scenario(::Val{:noisy_exact_cp}) = NoisyExactCP()
parse_scenario(::Val{:exact_nncp}) = ExactNNCP()
parse_scenario(::Val{:border_rank_w}) = BorderRankW()
parse_scenario(::Val{:overranked_abs_randn}) = OverrankedAbsRandn()
parse_scenario(::Val{S}) where {S} =
    throw(ArgumentError("Unknown scenario=$S. Use one of $(KNOWN_SCENARIOS)."))
parse_scenario(name::Symbol) = parse_scenario(Val(name))
parse_scenarios(names) =
    AbstractCPDBenchmarkScenario[parse_scenario(name) for name in names]

function set_scenario_rank(scenario::ExactCP, rank::Int)
    return ExactCP(scenario.dims, rank, scenario.warm_steps)
end

function set_scenario_rank(scenario::IllConditionedCP, rank::Int)
    return IllConditionedCP(
        scenario.dims,
        rank,
        scenario.collinearity_noise,
        scenario.warm_steps,
    )
end

function set_scenario_rank(scenario::NoisyExactCP, rank::Int)
    return NoisyExactCP(scenario.dims, rank, scenario.noise_level, scenario.warm_steps)
end

function set_scenario_rank(scenario::ExactNNCP, rank::Int)
    return ExactNNCP(scenario.dims, rank, scenario.warm_steps)
end

function set_scenario_rank(scenario::BorderRankW, rank::Int)
    return BorderRankW(rank, scenario.warm_steps)
end

function set_scenario_rank(scenario::OverrankedAbsRandn, rank::Int)
    return OverrankedAbsRandn(scenario.dims, rank, scenario.warm_steps)
end

set_scenario_noise_level(scenario::NoisyExactCP, noise_level::Float64) =
    NoisyExactCP(scenario.dims, scenario.rank, noise_level, scenario.warm_steps)

set_scenario_collinearity_noise(scenario::IllConditionedCP, collinearity_noise::Float64) =
    IllConditionedCP(scenario.dims, scenario.rank, collinearity_noise, scenario.warm_steps)

scenario_name(::ExactCP) = :exact_cp
scenario_name(::IllConditionedCP) = :ill_conditioned_cp
scenario_name(::NoisyExactCP) = :noisy_exact_cp
scenario_name(::ExactNNCP) = :exact_nncp
scenario_name(::BorderRankW) = :border_rank_w
scenario_name(::OverrankedAbsRandn) = :overranked_abs_randn

case_metadata(::ExactCP) = (; noise_level = 0.0)
case_metadata(::OverrankedAbsRandn) = NamedTuple()
case_metadata(::ExactNNCP) = (; nonnegative = true)
case_metadata(::BorderRankW) = (; default_init = :random)
case_metadata(scenario::IllConditionedCP) =
    (; collinearity_noise = scenario.collinearity_noise)
case_metadata(scenario::NoisyExactCP) = (; noise_level = scenario.noise_level)

scenario_log_fields(::AbstractCPDBenchmarkScenario) = NamedTuple()
scenario_log_fields(scenario::NoisyExactCP) = (; noise_level = scenario.noise_level)
scenario_log_fields(scenario::IllConditionedCP) =
    (; collinearity_noise = scenario.collinearity_noise)
scenario_log_context(scenario::AbstractCPDBenchmarkScenario) =
    merge((; scenario = scenario_name(scenario)), scenario_log_fields(scenario))

apply_rank_override(scenario::OverrankedAbsRandn, rank, overranked_rank) =
    isnothing(overranked_rank) ? scenario : set_scenario_rank(scenario, overranked_rank)

apply_rank_override(scenario::AbstractCPDBenchmarkScenario, rank, overranked_rank) =
    isnothing(rank) ? scenario : set_scenario_rank(scenario, rank)

function apply_rank_overrides(
    scenarios::Vector{AbstractCPDBenchmarkScenario};
    rank = nothing,
    overranked_rank = nothing,
)
    return apply_rank_override.(scenarios, Ref(rank), Ref(overranked_rank))
end
