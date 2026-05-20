struct BenchmarkCase
    name::Symbol
    tensor::Array{Float64,3}
    rank::Int
    nonnegative::Bool
    default_init::Symbol
    warm_steps::Int
    noise_level::Float64
    collinearity_noise::Float64
end

function BenchmarkCase(;
    name,
    tensor,
    rank,
    nonnegative = false,
    default_init = :alswarm,
    warm_steps = 100,
    noise_level = NaN,
    collinearity_noise = NaN,
)
    return BenchmarkCase(
        name,
        tensor,
        rank,
        nonnegative,
        default_init,
        warm_steps,
        noise_level,
        collinearity_noise,
    )
end

function _build_benchmark_case(scenario::AbstractCPDBenchmarkScenario, tensor)
    return BenchmarkCase(;
        name = scenario_name(scenario),
        tensor,
        rank = scenario.rank,
        warm_steps = scenario.warm_steps,
        case_metadata(scenario)...,
    )
end

function _rand_unit_matrix(rng::AbstractRNG, d::Int, r::Int)
    U = randn(rng, d, r)
    for k = 1:r
        nrm = norm(view(U, :, k))
        if nrm > 0
            @views U[:, k] ./= nrm
        else
            U[1, k] = 1.0
        end
    end
    return U
end

function _rand_nonnegative_unit_matrix(rng::AbstractRNG, d::Int, r::Int)
    U = rand(rng, d, r)
    for k = 1:r
        @views U[:, k] ./= max(norm(view(U, :, k)), eps(Float64))
    end
    return U
end

function _ill_conditioned_matrix(rng::AbstractRNG, d::Int, r::Int; noise = 1e-3)
    base = randn(rng, d)
    base ./= max(norm(base), eps(Float64))
    U = Matrix{Float64}(undef, d, r)
    for k = 1:r
        U[:, k] .= base .+ noise .* randn(rng, d)
        @views U[:, k] ./= max(norm(view(U, :, k)), eps(Float64))
    end
    return U
end

function _cp_tensor(λ::Vector{Float64}, factors::Vector{Matrix{Float64}})
    return reconstruct_cpd_rankr(λ, factors)
end

function _random_cp_tensor(
    rng::AbstractRNG,
    dims::NTuple{3,Int},
    rank::Int;
    factor_fn = _rand_unit_matrix,
)
    λ = 0.5 .+ rand(rng, rank)
    factors = [factor_fn(rng, d, rank) for d in dims]
    return λ, _cp_tensor(λ, factors)
end

function _add_gaussian_noise(
    A_clean::AbstractArray{Float64,3},
    rng::AbstractRNG,
    noise_level::Float64,
)
    E = randn(rng, size(A_clean)...)
    scale = noise_level * norm(A_clean) / max(norm(E), eps(Float64))
    E .*= scale
    return A_clean .+ E
end

function rank1_tensor(u, v, w)
    return reshape(u, :, 1, 1) .* reshape(v, 1, :, 1) .* reshape(w, 1, 1, :)
end

function border_rank_w_state()
    e1 = [1.0, 0.0]
    e2 = [0.0, 1.0]
    return rank1_tensor(e2, e1, e1) .+ rank1_tensor(e1, e2, e1) .+ rank1_tensor(e1, e1, e2)
end

function make_case(scenario::ExactCP, seed::Int)
    rng = MersenneTwister(seed)
    _, tensor = _random_cp_tensor(rng, scenario.dims, scenario.rank)
    return _build_benchmark_case(scenario, tensor)
end

function make_case(scenario::IllConditionedCP, seed::Int)
    rng = MersenneTwister(seed)
    factor_fn(rng, d, rank) =
        _ill_conditioned_matrix(rng, d, rank; noise = scenario.collinearity_noise)
    _, tensor = _random_cp_tensor(rng, scenario.dims, scenario.rank; factor_fn)
    return _build_benchmark_case(scenario, tensor)
end

function make_case(scenario::NoisyExactCP, seed::Int)
    rng = MersenneTwister(seed)
    _, A_clean = _random_cp_tensor(rng, scenario.dims, scenario.rank)
    tensor = _add_gaussian_noise(A_clean, rng, scenario.noise_level)
    return _build_benchmark_case(scenario, tensor)
end

function make_case(scenario::ExactNNCP, seed::Int)
    rng = MersenneTwister(seed)
    _, tensor = _random_cp_tensor(
        rng,
        scenario.dims,
        scenario.rank;
        factor_fn = _rand_nonnegative_unit_matrix,
    )
    return _build_benchmark_case(scenario, tensor)
end

function make_case(scenario::BorderRankW, _seed::Int)
    return _build_benchmark_case(scenario, border_rank_w_state())
end

function make_case(scenario::OverrankedAbsRandn, seed::Int)
    rng = MersenneTwister(seed)
    A = randn(rng, scenario.dims...)
    return _build_benchmark_case(scenario, abs.(A))
end
