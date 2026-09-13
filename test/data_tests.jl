struct _NormCountingArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    norm_calls::Base.RefValue{Int}
    norm_block_lengths::Vector{Int}
end

Base.size(A::_NormCountingArray) = size(A.data)
Base.axes(A::_NormCountingArray) = axes(A.data)
Base.IndexStyle(::Type{<:_NormCountingArray{T,N,A}}) where {T,N,A} = Base.IndexStyle(A)
Base.getindex(A::_NormCountingArray, I...) = getindex(A.data, I...)
Base.similar(A::_NormCountingArray, ::Type{T}, dims::Dims) where {T} =
    similar(A.data, T, dims)

function TensorKitchen.observation_norm2(
    A::_NormCountingArray;
    block_length::Int = 65_536,
    kwargs...,
)
    A.norm_calls[] += 1
    push!(A.norm_block_lengths, block_length)
    return sum(abs2, A.data)
end

@testset "storage and compute precision separation" begin
    raw = reshape(Int16.(-12:11), 4, 3, 2)

    lazy = prepare_tensor(raw; compute_type = Float32)
    @test lazy isa ComputeArray{Float32,3}
    @test parent(lazy) === raw
    @test storage_type(lazy) === Int16
    @test compute_type(lazy) === Float32
    @test !is_materialized(lazy)
    @test size(lazy) == size(raw)
    @test lazy[2, 2, 2] === Float32(raw[2, 2, 2])
    @test collect(lazy) == Float32.(raw)

    # Preparing an already compatible tensor is allocation-free at the object
    # level and preserves the original array identity.
    native = rand(Float32, 3, 2)
    @test prepare_tensor(native) === native
    @test storage_type(native) === Float32
    @test compute_type(native) === Float32
    @test is_materialized(native)

    dense =
        prepare_tensor(raw; compute_type = Float32, materialize = true, block_length = 5)
    @test dense isa Array{Float32,3}
    @test dense == Float32.(raw)
    @test dense !== raw

    @test materialize_tensor(lazy; block_length = 7) == Float32.(raw)
    @test prepare_tensor(lazy; compute_type = Float32) === lazy
    @test prepare_tensor(lazy; compute_type = Float64) isa ComputeArray{Float64,3}

    @test_throws ArgumentError prepare_tensor(raw; compute_type = Int32)
    @test_throws ArgumentError materialize_tensor(raw, Float32; block_length = 0)
end

@testset "observation-preserving implicit kernels" begin
    raw = reshape(Int16.(-30:29), 5, 4, 3)
    dense = Float32.(raw)
    lazy = prepare_tensor(raw; compute_type = Float32)

    @test observation_norm2(raw; compute_type = Float32, block_length = 7) ==
          sum(abs2, dense)
    @test observation_norm2(lazy; block_length = 9) == sum(abs2, dense)

    stats = observation_stats(lazy; block_length = 8)
    @test stats.norm2 == sum(abs2, dense)
    @test !stats.has_nonfinite
    @test stats.minimum == minimum(dense)
    @test stats.has_negative

    nonfinite = Float32[1, Inf, NaN]
    nonfinite_stats = observation_stats(nonfinite; block_length = 2)
    @test nonfinite_stats.has_nonfinite
    @test !nonfinite_stats.has_negative
    @test observation_stats(Float32[]).minimum === nothing

    # A Float32 scalar accumulator cannot add unit increments after 2^24. The
    # streaming reductions use a wider accumulator and round only once.
    reduction_values = vcat(Int16(4096), ones(Int16, 1_000))
    reduction_lazy = prepare_tensor(reduction_values; compute_type = Float32)
    reduction_expected = Float32(sum(abs2, Float64.(reduction_values)))
    stable_norm2 = observation_norm2(reduction_lazy; block_length = 37)
    stable_stats = observation_stats(reduction_lazy; block_length = 41)
    @test stable_norm2 isa Float32
    @test stable_norm2 == reduction_expected
    @test stable_stats.norm2 isa Float32
    @test stable_stats.norm2 == reduction_expected

    residual_raw = ones(Int16, length(reduction_values))
    residual_lazy = prepare_tensor(residual_raw; compute_type = Float32)
    residual_norm2 = observation_norm2(residual_lazy)
    residual_weights = Float32[1]
    residual_factors = [reshape(Float32.(residual_raw) .+ Float32.(reduction_values), :, 1)]
    residual_stats = TensorKitchen.cp_residual_stats_explicit(
        residual_lazy,
        residual_norm2,
        residual_weights,
        residual_factors,
    )
    @test residual_stats[1] isa Float32
    @test residual_stats[1] == reduction_expected
    @test residual_stats[2] == Float32(0.5) * reduction_expected

    rng = MersenneTwister(91)
    factors = [randn(rng, Float32, size(raw, mode), 2) for mode = 1:3]
    for mode = 1:3
        expected = mttkrp(dense, factors, mode; method = :direct)
        actual_raw = implicit_mttkrp(raw, factors, mode; block_length = 7)
        actual_lazy = implicit_mttkrp(lazy, factors, mode; block_length = 9)
        dispatched_raw = mttkrp(raw, factors, mode; method = :auto)
        dispatched_lazy = mttkrp(lazy, factors, mode; method = :auto)

        @test actual_raw ≈ expected rtol = 2.0f-5 atol = 2.0f-5
        @test actual_lazy ≈ expected rtol = 2.0f-5 atol = 2.0f-5
        @test dispatched_raw ≈ expected rtol = 2.0f-5 atol = 2.0f-5
        @test dispatched_lazy ≈ expected rtol = 2.0f-5 atol = 2.0f-5

        output = zeros(Float32, size(raw, mode), 2)
        @test mttkrp!(output, raw, factors, mode; method = :implicit) === output
        @test output ≈ expected rtol = 2.0f-5 atol = 2.0f-5
    end

    workspace = TensorKitchen.CPALSWorkspace(lazy, size(lazy), 2; mttkrp_method = :auto)
    @test all(isnothing, workspace.mttkrp_tmp_work)
    @test all(isnothing, workspace.mttkrp_kr_work)
    @test all(isnothing, workspace.mttkrp_kr_work2)

    initial_factors = [randn(rng, Float32, size(raw, mode), 2) for mode = 1:3]
    lazy_fit = fit_cp_als(
        lazy,
        2;
        init_factors = (ones(Float32, 2), deepcopy(initial_factors)),
        maxiter = 2,
        verbose = false,
        return_stats = true,
    )
    dense_fit = fit_cp_als(
        dense,
        2;
        init_factors = (ones(Float32, 2), deepcopy(initial_factors)),
        maxiter = 2,
        mttkrp_method = :implicit,
        verbose = false,
        return_stats = true,
    )
    @test lazy_fit.rel_error ≈ dense_fit.rel_error rtol = 2.0f-5 atol = 2.0f-5
    @test lazy_fit.weights ≈ dense_fit.weights rtol = 2.0f-5 atol = 2.0f-5
    @test all(
        isapprox(
            lazy_fit.factors[mode],
            dense_fit.factors[mode];
            rtol = 2.0f-5,
            atol = 2.0f-5,
        ) for mode = 1:3
    )

    exact_weights = Float32[2]
    exact_factors = [reshape(Float32[1, 2], 2, 1), reshape(Float32[3, 1], 2, 1)]
    exact_raw = Int16.(reconstruct_cpd_rankr(exact_weights, exact_factors))
    exact_lazy = prepare_tensor(exact_raw; compute_type = Float32)
    exact_norm2 = observation_norm2(exact_lazy)
    exact_stats = TensorKitchen.cp_residual_stats_explicit(
        exact_lazy,
        exact_norm2,
        exact_weights,
        exact_factors,
    )
    @test exact_stats[1] == 0.0f0
    @test exact_stats[2] == 0.0f0
    @test exact_stats[3] == 0.0f0
    @test_throws ArgumentError copy(exact_lazy)

    fallback_weights = Float32[0.75]
    fallback_factors = [reshape(Float32[1, 0.5], 2, 1), reshape(Float32[0.25, 2], 2, 1)]
    lazy_fallback = TensorKitchen.cp_residual_stats_explicit(
        exact_lazy,
        exact_norm2,
        fallback_weights,
        fallback_factors,
    )
    dense_fallback = TensorKitchen.cp_residual_stats_explicit(
        Float32.(exact_raw),
        exact_norm2,
        fallback_weights,
        fallback_factors,
    )
    @test all(
        isapprox(lazy_fallback[i], dense_fallback[i]; rtol = 2.0f-6, atol = 2.0f-6) for
        i in eachindex(lazy_fallback)
    )

    mode_matrix = randn(rng, Float32, 2, size(raw, 2))
    implicit_product =
        TensorKitchen._implicit_mode_product(lazy, mode_matrix, 2; block_columns = 5)
    @test implicit_product ≈ mode_n_product(dense, mode_matrix, 2) rtol = 2.0f-5 atol =
        2.0f-5
    @test mode_n_product(lazy, mode_matrix, 2; block_columns = 5) ≈ implicit_product

    comparison = randn(rng, Float32, 7, size(raw, 2), size(raw, 3))
    implicit_cross =
        TensorKitchen._implicit_mode_cross(lazy, comparison, 1; block_columns = 5)
    expected_cross = unfold_mode(dense, 1) * transpose(unfold_mode(comparison, 1))
    @test implicit_cross ≈ expected_cross rtol = 2.0f-5 atol = 2.0f-5

    @test_throws ArgumentError observation_norm2(raw; block_length = 0)
    @test_throws DimensionMismatch implicit_mttkrp(raw, factors[1:2], 1)
    @test_throws ArgumentError mttkrp(raw, factors, 1; method = :khatri_rao)
end

@testset "CP lazy objective primitives match dense" begin
    raw = reshape(Int16.(1:24), 4, 3, 2)
    lazy = prepare_tensor(raw; compute_type = Float32)
    dense = Float32.(raw)
    dims = size(raw)
    rank = 2

    lazy_model = TensorKitchen.JoinModel(lazy, rank; geometry = :canonical)
    dense_model = TensorKitchen.JoinModel(dense, rank; geometry = :canonical)
    point = TensorKitchen.initial_point(dense_model, RandomInit(); verbose = false)
    normA2 = observation_norm2(lazy)
    lazy_cost, lazy_egrad = TensorKitchen.model_cost_egrad_functions(lazy_model, normA2)
    dense_cost, dense_egrad = TensorKitchen.model_cost_egrad_functions(dense_model, normA2)

    @test lazy_cost(manifold(lazy_model), point) ≈ dense_cost(manifold(dense_model), point) rtol =
        2.0f-5 atol = 2.0f-5
    lazy_gλ, lazy_gU =
        unpack_point_rankr(lazy_egrad(manifold(lazy_model), point), dims, rank)
    dense_gλ, dense_gU =
        unpack_point_rankr(dense_egrad(manifold(dense_model), point), dims, rank)
    @test lazy_gλ ≈ dense_gλ rtol = 2.0f-5 atol = 2.0f-5
    @test all(
        isapprox(lazy_gU[mode], dense_gU[mode]; rtol = 2.0f-5, atol = 2.0f-5) for
        mode in eachindex(lazy_gU)
    )

    lazy_nn_model =
        TensorKitchen.JoinModel(lazy, rank; geometry = :softplus_metric, nonnegative = true)
    dense_nn_model = TensorKitchen.JoinModel(
        dense,
        rank;
        geometry = :softplus_metric,
        nonnegative = true,
    )
    nn_point = TensorKitchen.initial_point(dense_nn_model, RandomInit(); verbose = false)
    lazy_nn_cost, lazy_nn_egrad =
        TensorKitchen.model_cost_egrad_functions(lazy_nn_model, normA2)
    dense_nn_cost, dense_nn_egrad =
        TensorKitchen.model_cost_egrad_functions(dense_nn_model, normA2)

    @test lazy_nn_cost(manifold(lazy_nn_model), nn_point) ≈
          dense_nn_cost(manifold(dense_nn_model), nn_point) rtol = 2.0f-5 atol = 2.0f-5
    lazy_nn_gradient = lazy_nn_egrad(manifold(lazy_nn_model), nn_point)
    dense_nn_gradient = dense_nn_egrad(manifold(dense_nn_model), nn_point)
    @test all(
        isapprox(lazy_nn_gradient[i], dense_nn_gradient[i]; rtol = 2.0f-5, atol = 2.0f-5)
        for i in eachindex(lazy_nn_gradient)
    )
end

@testset "public preprocessing routes" begin
    raw = reshape(Int16.(1:60), 5, 4, 3)
    dense = Float32.(raw)
    rng = MersenneTwister(203)
    initial =
        CPDPoint(ones(Float32, 2), [rand(rng, Float32, size(raw, mode), 2) for mode = 1:3])

    lazy_cp = cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :als,
        p0 = initial,
        maxiter = 2,
        verbose = false,
    )
    dense_cp = cpd(dense, 2; solver = :als, p0 = initial, maxiter = 2, verbose = false)
    @test eltype(weights(lazy_cp)) === Float32
    @test weights(lazy_cp) ≈ weights(dense_cp) rtol = 2.0f-5 atol = 2.0f-5
    @test all(
        isapprox(
            factors(lazy_cp)[mode],
            factors(dense_cp)[mode];
            rtol = 2.0f-5,
            atol = 2.0f-5,
        ) for mode = 1:3
    )

    float64_initial =
        CPDPoint(ones(Float64, 2), [Float64.(factor) for factor in factors(initial)])
    converted_cp = cpd(
        dense,
        2;
        compute_type = Float64,
        solver = :als,
        p0 = float64_initial,
        maxiter = 1,
        verbose = false,
    )
    @test eltype(weights(converted_cp)) === Float64

    positive_initial =
        CPDPoint(ones(Float32, 2), [rand(rng, Float32, size(raw, mode), 2) for mode = 1:3])
    lazy_nn = nncpd(
        raw,
        2;
        compute_type = Float32,
        solver = :als,
        p0 = positive_initial,
        maxiter = 2,
        verbose = false,
    )
    dense_nn =
        nncpd(dense, 2; solver = :als, p0 = positive_initial, maxiter = 2, verbose = false)
    @test eltype(weights(lazy_nn)) === Float32
    @test weights(lazy_nn) ≈ weights(dense_nn) rtol = 2.0f-5 atol = 2.0f-5
    @test all(
        isapprox(
            factors(lazy_nn)[mode],
            factors(dense_nn)[mode];
            rtol = 2.0f-5,
            atol = 2.0f-5,
        ) for mode = 1:3
    )

    lazy_tucker = tucker(
        raw,
        (2, 2, 2);
        compute_type = Float32,
        svd_backend = :randomized,
        oversampling = 1,
        power_iterations = 0,
        block_columns = 5,
        rng = MersenneTwister(204),
    )
    dense_tucker = tucker(
        dense,
        (2, 2, 2);
        svd_backend = :randomized,
        oversampling = 1,
        power_iterations = 0,
        block_columns = 5,
        rng = MersenneTwister(204),
    )
    @test eltype(core(lazy_tucker)) === Float32
    @test reconstruct(lazy_tucker) ≈ reconstruct(dense_tucker) rtol = 2.0f-5 atol = 2.0f-5

    @test_throws ArgumentError cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :lm,
        init = :random,
        maxiter = 0,
        verbose = false,
    )
    @test_nowarn cpd(raw, 2; compute_type = Float32, maxiter = 1, verbose = false)
    @test_nowarn nncpd(raw, 2; compute_type = Float32, maxiter = 1, verbose = false)
    @test_throws ArgumentError tucker(raw, (2, 2, 2); compute_type = Float32)
    @test_throws ArgumentError tucker(
        raw,
        (2, 2, 2);
        compute_type = Float32,
        method = :hooi,
    )

    materialized_tucker = tucker(raw, (2, 2, 2); compute_type = Float32, materialize = true)
    @test eltype(core(materialized_tucker)) === Float32

    @test_throws ArgumentError nncpd(
        reshape(Int16[-1, 2, 3, 4], 2, 2),
        1;
        compute_type = Float32,
        solver = :als,
        init = :random,
        maxiter = 0,
        verbose = false,
    )
    @test_throws ArgumentError nncpd(
        reshape(Float32[1, Inf, 3, 4], 2, 2),
        1;
        solver = :als,
        init = :random,
        maxiter = 0,
        verbose = false,
    )
end

function _test_lazy_manifold_cpd_solver(solver; component_trace::Bool = false)
    raw = reshape(Int16.(1:24), 4, 3, 2)
    dense = Float32.(raw)
    rng = MersenneTwister(410 + Int(solver === :rcg) + 2 * Int(solver === :lbfgs))

    rank2_start =
        CPDPoint(ones(Float32, 2), [rand(rng, Float32, size(raw, mode), 2) for mode = 1:3])
    lazy_cp = cpd(
        raw,
        2;
        compute_type = Float32,
        solver,
        p0 = rank2_start,
        maxiter = 1,
        component_trace,
        verbose = false,
    )
    dense_cp = cpd(
        dense,
        2;
        solver,
        p0 = rank2_start,
        maxiter = 1,
        component_trace,
        verbose = false,
    )
    @test rel_error(lazy_cp) ≈ rel_error(dense_cp) rtol = 5.0f-4 atol = 5.0f-4
    @test weights(lazy_cp) ≈ weights(dense_cp) rtol = 5.0f-4 atol = 5.0f-4
    @test all(
        isapprox(
            factors(lazy_cp)[mode],
            factors(dense_cp)[mode];
            rtol = 5.0f-4,
            atol = 5.0f-4,
        ) for mode = 1:3
    )

    rank1_start =
        CPDPoint(ones(Float32, 1), [rand(rng, Float32, size(raw, mode), 1) for mode = 1:3])
    lazy_nn = nncpd(
        raw,
        1;
        compute_type = Float32,
        solver,
        p0 = rank1_start,
        maxiter = 1,
        verbose = false,
    )
    dense_nn = nncpd(dense, 1; solver, p0 = rank1_start, maxiter = 1, verbose = false)
    @test rel_error(lazy_nn) ≈ rel_error(dense_nn) rtol = 5.0f-4 atol = 5.0f-4
    @test weights(lazy_nn) ≈ weights(dense_nn) rtol = 5.0f-4 atol = 5.0f-4
    @test all(
        isapprox(
            factors(lazy_nn)[mode],
            factors(dense_nn)[mode];
            rtol = 5.0f-4,
            atol = 5.0f-4,
        ) for mode = 1:3
    )

    return lazy_cp
end

@testset "lazy CP RGD paths" begin
    rgd_result = _test_lazy_manifold_cpd_solver(:rgd; component_trace = true)
    @test isfinite(rgd_result.solver_info.component_trace_start_rel_error)
    @test !isempty(rgd_result.solver_info.component_trace_cost_history)
    _test_lazy_manifold_cpd_solver(:rgd_fixed)
end

@testset "lazy CP RCG path" begin
    _test_lazy_manifold_cpd_solver(:rcg)
end

@testset "lazy CP L-BFGS path" begin
    _test_lazy_manifold_cpd_solver(:lbfgs)
end

@testset "lazy CP initialization boundary" begin
    raw = reshape(Int16.(1:24), 4, 3, 2)
    random_result = cpd(
        raw,
        2;
        compute_type = Float32,
        solver = RGDSolver(0.1),
        init = RandomInit(),
        maxiter = 1,
        verbose = false,
    )
    @test solver(random_result) == :rgd
    @test_nowarn cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :rgd,
        init = ALSWarmStartInit(1; base_init = RandomInit()),
        maxiter = 1,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :rgd,
        init = TuckerInit(),
        maxiter = 0,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :rgd,
        init = TuckerDiagInit(),
        maxiter = 0,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        raw,
        2;
        compute_type = Float32,
        solver = :rgd,
        init = ALSWarmStartInit(1; base_init = TuckerInit()),
        maxiter = 0,
        verbose = false,
    )
end

@testset "CP target norm is cached for one solver run" begin
    data = reshape(Float32.(1:24), 4, 3, 2)
    norm_calls = Ref(0)
    norm_block_lengths = Int[]
    counted = _NormCountingArray(data, norm_calls, norm_block_lengths)
    rng = MersenneTwister(501)
    start =
        CPDPoint(ones(Float32, 2), [rand(rng, Float32, size(data, mode), 2) for mode = 1:3])

    for solver_name in (:als, :rgd, :rgd_fixed, :rcg, :lbfgs)
        norm_calls[] = 0
        empty!(norm_block_lengths)
        cpd(
            counted,
            2;
            solver = solver_name,
            p0 = start,
            maxiter = 2,
            component_trace = solver_name == :rgd,
            conversion_block_length = 7,
            verbose = false,
        )
        @test norm_calls[] == 1
        @test norm_block_lengths == [7]
    end

    norm_calls[] = 0
    empty!(norm_block_lengths)
    cpd(
        counted,
        2;
        solver = :rgd,
        init = ALSWarmStartInit(1; base_init = RandomInit()),
        maxiter = 1,
        verbose = false,
    )
    @test norm_calls[] == 1

    norm_calls[] = 0
    empty!(norm_block_lengths)
    nncpd(
        counted,
        2;
        solver = :rgd,
        p0 = start,
        maxiter = 2,
        component_trace = true,
        verbose = false,
    )
    @test norm_calls[] == 0

    @test_throws ArgumentError cpd(
        data,
        2;
        observation_norm2_cache = sum(abs2, data),
        p0 = start,
        maxiter = 0,
        verbose = false,
    )
    @test_throws ArgumentError nncpd(
        data,
        2;
        observation_norm2_cache = sum(abs2, data),
        p0 = start,
        maxiter = 0,
        verbose = false,
    )
end
