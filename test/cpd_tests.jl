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

@testset "low-level rank-1 solve: unpack_point_rank1" begin
    A = randn(6, 5, 4)
    model = JoinModel(A, 1)
    out = solve(
        RGDSolver(1.0),
        model;
        init = RandomInit(),
        maxiter = 50,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    λ, U = unpack_point_rank1(TensorKitchen.point(out), size(A))
    @test λ isa Number
    @test length(U) == 3
    @test length(U[1]) == 6 && length(U[2]) == 5 && length(U[3]) == 4
end

# =========================================================================
# low-level rank-r solve + packed-point helpers
# =========================================================================

@testset "low-level rank-r solve: unpack_point_rankr, reconstruct_cp_rankr" begin
    A = randn(6, 5, 4)
    r = 2
    model = JoinModel(A, r; geometry = :canonical)
    out = solve(
        RGDSolver(1.0),
        model;
        init = TuckerInit(),
        maxiter = 30,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    λ, U = unpack_point_rankr(TensorKitchen.point(out), size(A), r)
    @test length(λ) == r
    @test length(U) == 3
    for m = 1:3
        @test size(U[m]) == (size(A, m), r)
    end
    Ahat = reconstruct_cp_rankr(λ, U)
    @test size(Ahat) == size(A)
    @test hasproperty(out, :solver_info)
    @test out.solver_info.total_iterations == out.iterations
    @test length(out.solver_info.accepted_stepsize_history) == out.iterations
    @test length(out.solver_info.line_search_trial_history) == out.iterations
    @test out.solver_info.function_evaluations >= 0
    @test out.solver_info.gradient_evaluations >= 1
end


@testset "LM CPD rank-r Jacobian matches finite differences across geometries" begin
    A = randn(5, 4, 3)
    r = 2
    cases = (
        (JoinModel(A, r; geometry = :canonical), 1e-7),
        (JoinModel(A, r; geometry = :native), 1e-7),
        (JoinModel(abs.(A), r; nonnegative = true, geometry = :squaring_metric), 5e-6),
        (JoinModel(abs.(A), r; nonnegative = true, geometry = :softplus_metric), 5e-6),
    )

    for (model, tol_fd) in cases
        M = TensorKitchen.manifold(model)
        p = TensorKitchen._solver_point(
            M,
            TensorKitchen.initial_point(model, :random; verbose = false),
        )
        basis = ManifoldsBase.DefaultOrthonormalBasis()
        J = TensorKitchen._lm_raw_jacobian_matrix(model, M, p; basis)
        @test all(isfinite, J)

        retraction_method = TensorKitchen._solver_retraction_method(M, p)
        ϵ = 1e-6
        d = manifold_dimension(M)
        for j = 1:min(d, 3)
            coeff = zeros(Float64, d)
            coeff[j] = 1.0
            Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
            p_plus = ManifoldsBase.retract(M, p, ϵ * Xj, retraction_method)
            p_minus = ManifoldsBase.retract(M, p, -ϵ * Xj, retraction_method)
            r_plus = TensorKitchen._lm_raw_residual_vector(model, p_plus)
            r_minus = TensorKitchen._lm_raw_residual_vector(model, p_minus)
            fd = (r_plus .- r_minus) ./ (2 * ϵ)
            @test maximum(abs.(fd .- J[:, j])) ≤ tol_fd
        end
    end
end


@testset "LM CPD Jacobian finite differences across direct parameterizations" begin
    dims = (5, 4, 3)
    A = randn(dims...)
    r = 2
    cases = (
        ("rank1 native", TensorKitchen.Rank1CPDModel(A), 5e-7),
        ("rank1 squared", TensorKitchen.Rank1CPDModel(abs.(A); nonnegative = true), 5e-6),
        (
            "rank1 softplus",
            TensorKitchen.Rank1CPDModel(
                abs.(A);
                nonnegative = true,
                use_softplus_metric = true,
            ),
            5e-6,
        ),
        ("rankr native", TensorKitchen.RankRCPDModel(A, r; geometry = :native), 5e-7),
        ("rankr canonical", TensorKitchen.RankRCPDModel(A, r; geometry = :canonical), 5e-7),
        (
            "rankr squared",
            TensorKitchen.RankRCPDModel(
                abs.(A),
                r;
                nonnegative = true,
                geometry = :squaring_metric,
            ),
            5e-6,
        ),
        (
            "rankr softplus",
            TensorKitchen.RankRCPDModel(
                abs.(A),
                r;
                nonnegative = true,
                geometry = :softplus_metric,
            ),
            5e-6,
        ),
    )

    for (label, model, tol_fd) in cases
        M = TensorKitchen.manifold(model)
        p = TensorKitchen._solver_point(
            M,
            TensorKitchen.initial_point(model, :random; verbose = false),
        )
        basis = ManifoldsBase.DefaultOrthonormalBasis()
        J = TensorKitchen._lm_raw_jacobian_matrix(model, M, p; basis)
        @testset "$label" begin
            @test all(isfinite, J)
            retraction_method = TensorKitchen._solver_retraction_method(M, p)
            ϵ = 1e-6
            d = manifold_dimension(M)
            r0 = TensorKitchen._lm_raw_residual_vector(model, p)
            for j = 1:min(d, 3)
                coeff = zeros(Float64, d)
                coeff[j] = 1.0
                Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
                p_plus = ManifoldsBase.retract(M, p, ϵ * Xj, retraction_method)
                r_plus = TensorKitchen._lm_raw_residual_vector(model, p_plus)
                fd = (r_plus .- r0) ./ ϵ
                @test maximum(abs.(fd .- J[:, j])) ≤ tol_fd
            end
        end
    end
end

@testset "CP parameterization tangent decode helpers" begin
    dims = (3, 2, 2)
    r = 2
    λ̃ = [1.5, -0.4]
    Ũ = [randn(dims[m], r) for m = 1:length(dims)]
    λ̇̃ = randn(r)
    U̇̃ = [randn(dims[m], r) for m = 1:length(dims)]
    p = TensorKitchen.pack_point_rankr(λ̃, Ũ, r)
    X = TensorKitchen.pack_point_rankr(λ̇̃, U̇̃, r)

    λ_sq, U_sq, λ̇_sq, U̇_sq = TensorKitchen._cp_rankr_decode_tangent_factors(
        TensorKitchen.SquaredNNCPParam(),
        dims,
        r,
        p,
        X,
    )
    @test λ_sq ≈ λ̃ .^ 2
    @test all(U_sq[m] ≈ Ũ[m] .^ 2 for m in eachindex(U_sq))
    @test λ̇_sq ≈ 2 .* λ̃ .* λ̇̃
    @test all(U̇_sq[m] ≈ 2 .* Ũ[m] .* U̇̃[m] for m in eachindex(U̇_sq))

    λ_sp, U_sp, λ̇_sp, U̇_sp = TensorKitchen._cp_rankr_decode_tangent_factors(
        TensorKitchen.SoftplusNNCPParam(),
        dims,
        r,
        p,
        X,
    )
    @test λ_sp ≈ TensorKitchen._softplus_value.(λ̃)
    @test all(U_sp[m] ≈ TensorKitchen._softplus_value.(Ũ[m]) for m in eachindex(U_sp))
    @test λ̇_sp ≈ TensorKitchen._softplus_derivative.(λ̃) .* λ̇̃
    @test all(
        U̇_sp[m] ≈ TensorKitchen._softplus_derivative.(Ũ[m]) .* U̇̃[m] for m in eachindex(U̇_sp)
    )

    model_sp = TensorKitchen.RankRCPDModel(
        randn(dims...),
        r;
        nonnegative = true,
        geometry = :softplus_metric,
    )
    q_zero =
        CPDPoint(zeros(Float64, r), [zeros(Float64, dims[m], r) for m = 1:length(dims)])
    p_zero = TensorKitchen.pack_cpd_point(model_sp, q_zero)
    λ_lat, U_lat = TensorKitchen.unpack_point_rankr(p_zero, dims, r)
    @test all(isfinite, λ_lat)
    @test all(F -> all(isfinite, F), U_lat)
end

@testset "cpd/approx accept LMSolver" begin
    A = randn(5, 4, 3)
    res_cpd_symbol = cpd(A, 2; solver = :lm, maxiter = 2, tol = 1e-6, verbose = false)
    @test res_cpd_symbol.solver == :lm

    res_cpd_object = cpd(
        A,
        2;
        solver = LMSolver(damping_term_min = 1e-2),
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
    @test res_cpd_object.solver == :lm

    res_cpd_alswarm_lm = cpd(
        A,
        2;
        solver = :lm,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :tucker,
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
    @test res_cpd_alswarm_lm.solver == :lm
    @test isfinite(res_cpd_alswarm_lm.rel_error)

    target = [1.2, -0.4, 0.8]
    res_approx =
        approx(Manifolds.Sphere(2), target; solver = :lm, maxiter = 2, verbose = false)
    @test res_approx.solver == :lm
end


@testset "cp_rank.jl: low-level rank-r solve, cost_segre, egrad_segre, cost_secant_rankr, egrad_secant_rankr" begin
    dims = (5, 4, 3)
    r = 2
    A = randn(dims...)
    model = JoinModel(A, r; geometry = :canonical)
    out = solve(
        RGDSolver(1.0),
        model;
        init = TuckerInit(),
        maxiter = 20,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    λ, U = unpack_point_rankr(TensorKitchen.point(out), dims, r)
    @test length(λ) == r && length(U) == 3
    for m = 1:3
        @test size(U[m]) == (dims[m], r)
    end
    M1 = Manifolds.Segre(dims)
    p1 = pack_point_rank1(1.0, [U[m][:, 1] for m = 1:3])
    c1 = cost_segre(A, dims)
    @test c1(M1, p1) isa Float64
end


@testset "low-level rank-r solve: unpack_point_rankr for cpd reconstruction" begin
    A = randn(6, 5, 4)
    r = 2
    model = JoinModel(A, r; geometry = :canonical)
    out = solve(
        RGDSolver(1.0),
        model;
        init = TuckerInit(),
        maxiter = 30,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    λ, U = unpack_point_rankr(TensorKitchen.point(out), size(A), r)
    @test length(λ) == r && length(U) == 3
    Ahat = reconstruct_cpd_rankr(λ, U)
    @test size(Ahat) == size(A)
end

# =========================================================================
# cpd/cpd.jl
# =========================================================================

@testset "cpd.jl: cpd(), CPDResult, reconstruct" begin
    dims = (6, 5, 4)
    r = 2
    core = randn(r, r, r)
    factors = [randn(dims[k], r) for k = 1:3]
    A = reconstruct_tucker(core, factors)
    res = cpd(A, r; verbose = false)
    @test res isa CPDResult
    @test TensorKitchen.solver(res) == :rgd
    @test length(TensorKitchen.weights(res)) == r && length(TensorKitchen.factors(res)) == 3
    Ahat = reconstruct(res)
    @test size(Ahat) == size(A)
    @test rel_error(A, res) == TensorKitchen.relative_frobenius_error(A, Ahat)
    @test rel_error(A, Ahat) == rel_error(A, res)

    model = JoinModel(A, r; geometry = :canonical)
    p = TensorKitchen.initial_point(model, :random)
    comps = TensorKitchen.extract_components(model, p)
    @test length(comps) == r
    @test comps[1] isa TensorKitchen.CPDComponent
    @test !(:tensor in fieldnames(typeof(comps[1])))
    @test comps[1].point !== p
    @test comps[1].kind == :Segre
    @test size(comps[1].tensor) == size(A)
    Xparts = zero(A)
    for c in comps
        Xparts .+= c.tensor
    end
    @test TensorKitchen.cost(model, p) ≈ 0.5 * sum(abs2, A .- Xparts)
end


@testset "cpd.jl: initializer objects and explicit p0" begin
    dims = (6, 5, 4)
    r = 2
    comps = [RankOneTensor(randn(), [randn(d) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)

    model = JoinModel(A, r; geometry = :canonical)
    p_hosvd = TensorKitchen.initial_point(model, HOSVDInit())
    p_hosvd_sym = TensorKitchen.initial_point(model, :hosvd)
    λ_hosvd, U_hosvd = unpack_point_rankr(p_hosvd, dims, r)
    λ_hosvd_sym, U_hosvd_sym = unpack_point_rankr(p_hosvd_sym, dims, r)
    @test length(λ_hosvd) == r
    @test size(U_hosvd[1]) == (dims[1], r)
    @test λ_hosvd_sym == λ_hosvd
    @test U_hosvd_sym == U_hosvd

    p0 = TensorKitchen.initial_point(model, TuckerDiagInit())
    p_base = TensorKitchen.initial_point(model, RandomInit())
    p_warm = TensorKitchen.initial_point(
        model,
        ALSWarmStartInit(2; base_init = PointInit(p_base)),
    )
    p_warm_sym = TensorKitchen.initial_point(model, :alswarm)
    @test TensorKitchen.cost(model, p_warm) <= TensorKitchen.cost(model, p_base) + 1e-10
    @test isfinite(TensorKitchen.cost(model, p_warm_sym))
    generic_join = JoinModel((Manifolds.Segre(dims), Manifolds.Segre(dims)), A)
    p_warm_canonical_match = TensorKitchen.initial_point(
        model,
        ALSWarmStartInit(2; base_init = TuckerInit());
        verbose = false,
    )
    p_join_warm = TensorKitchen.initial_point(
        generic_join,
        ALSWarmStartInit(2; base_init = TuckerInit());
        verbose = false,
    )
    p_join_from_canonical =
        TensorKitchen.canonical_to_joinpoint(p_warm_canonical_match, dims, r)
    A_join_warm = reconstruct_cpd_rankr(
        components_from_factors(TensorKitchen.unpack_rankr_native(p_join_warm, dims, r)...),
    )
    A_join_from_canonical = reconstruct_cpd_rankr(
        components_from_factors(
            TensorKitchen.unpack_rankr_native(p_join_from_canonical, dims, r)...,
        ),
    )
    @test A_join_warm ≈ A_join_from_canonical

    res_p0 = cpd(A, r; solver = :rgd, p0 = p0, maxiter = 5, tol = 1e-6, verbose = false)
    @test res_p0 isa CPDResult
    @test isfinite(res_p0.rel_error)

    res_init_obj = cpd(
        A,
        r;
        solver = :als,
        init = PointInit(p0),
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
    @test res_init_obj isa CPDResult
    @test isfinite(res_init_obj.rel_error)

    res_init_sym =
        cpd(A, r; solver = :rgd, init = :tucker, maxiter = 5, tol = 1e-6, verbose = false)
    @test res_init_sym isa CPDResult
    @test isfinite(res_init_sym.rel_error)
    @test res_init_sym.solver_info.total_iterations == res_init_sym.iterations
    @test length(res_init_sym.solver_info.accepted_stepsize_history) ==
          res_init_sym.iterations
    @test length(res_init_sym.solver_info.line_search_trial_history) ==
          res_init_sym.iterations
    @test res_init_sym.solver_info.function_evaluations >= 0
    @test res_init_sym.solver_info.gradient_evaluations >= 1

    res_trace = @test_logs min_level = Base.CoreLogging.Warn cpd(
        A,
        r;
        solver = :rgd,
        init = :tucker,
        maxiter = 5,
        tol = 1e-6,
        verbose = false,
        component_trace = true,
    )
    trace_info = res_trace.solver_info
    @test hasproperty(trace_info, :component_trace_iterations)
    @test hasproperty(trace_info, :component_trace_max_delta_history)
    @test hasproperty(trace_info, :component_trace_delta_history)
    @test hasproperty(trace_info, :component_trace_rgrad_top1_share_history)
    @test hasproperty(trace_info, :component_trace_rgrad_top3_share_history)
    @test hasproperty(trace_info, :component_trace_rgrad_effective_components_history)
    @test hasproperty(trace_info, :component_trace_coordinate_rgrad_energy_history)
    @test hasproperty(trace_info, :component_trace_metric_rgrad_energy_history)
    @test hasproperty(trace_info, :component_trace_ambient_component_velocity_history)
    @test hasproperty(trace_info, :component_trace_metric_rgrad_top1_share_history)
    @test hasproperty(trace_info, :component_trace_ambient_velocity_top1_share_history)
    @test hasproperty(trace_info, :component_trace_metric_rgrad_argmax_component_history)
    @test hasproperty(
        trace_info,
        :component_trace_ambient_velocity_argmax_component_history,
    )
    @test hasproperty(trace_info, :component_trace_metric_rgrad_argmax_component_final)
    @test hasproperty(trace_info, :component_trace_ambient_velocity_argmax_component_final)
    @test hasproperty(trace_info, :component_trace_start_rel_error)
    @test hasproperty(trace_info, :component_trace_rgrad_failed_count)
    @test isfinite(trace_info.component_trace_start_rel_error)
    @test trace_info.component_trace_rgrad_failed_count == 0
    @test length(trace_info.component_trace_iterations) == res_trace.iterations
    @test length(trace_info.component_trace_cost_history) == res_trace.iterations
    @test length(trace_info.component_trace_max_delta_history) == res_trace.iterations
    @test length(trace_info.component_trace_iterations) ==
          length(trace_info.component_trace_max_delta_history)
    @test length(trace_info.component_trace_delta_history) ==
          length(trace_info.component_trace_max_delta_history)
    @test length(trace_info.component_trace_rgrad_top1_share_history) ==
          length(trace_info.component_trace_iterations)
    @test length(trace_info.component_trace_rgrad_top3_share_history) ==
          length(trace_info.component_trace_iterations)
    @test length(trace_info.component_trace_rgrad_effective_components_history) ==
          length(trace_info.component_trace_iterations)
    @test length(trace_info.component_trace_metric_rgrad_top1_share_history) ==
          length(trace_info.component_trace_iterations)
    @test length(trace_info.component_trace_ambient_velocity_top1_share_history) ==
          length(trace_info.component_trace_iterations)
    @test all(length(deltas) == r for deltas in trace_info.component_trace_delta_history)
    @test all(
        length(energies) == r for
        energies in trace_info.component_trace_metric_rgrad_energy_history
    )
    @test all(
        length(vel) == r for
        vel in trace_info.component_trace_ambient_component_velocity_history
    )
    @test all(isfinite, trace_info.component_trace_max_delta_history)
    @test all(
        x -> isnan(x) || -1e-12 <= x <= 1 + 1e-12,
        trace_info.component_trace_rgrad_top1_share_history,
    )
    @test all(
        x -> isnan(x) || -1e-12 <= x <= 1 + 1e-12,
        trace_info.component_trace_rgrad_top3_share_history,
    )
    @test all(
        zip(
            trace_info.component_trace_rgrad_top1_share_history,
            trace_info.component_trace_rgrad_top3_share_history,
        ),
    ) do (top1, top3)
        isnan(top1) || isnan(top3) || top1 <= top3 + 1e-12
    end
    @test all(
        x -> isnan(x) || 1 <= x <= r,
        trace_info.component_trace_rgrad_effective_components_history,
    )
    @test trace_info.component_trace_rgrad_argmax_component_final ==
          trace_info.component_trace_rgrad_argmax_component_history[end]
    @test trace_info.component_trace_metric_rgrad_argmax_component_final ==
          trace_info.component_trace_metric_rgrad_argmax_component_history[end]
    @test trace_info.component_trace_ambient_velocity_argmax_component_final ==
          trace_info.component_trace_ambient_velocity_argmax_component_history[end]
    @test 1 <= trace_info.component_trace_metric_rgrad_argmax_component_final <= r
    @test 1 <= trace_info.component_trace_ambient_velocity_argmax_component_final <= r
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :als,
        init = :tucker,
        maxiter = 1,
        tol = 1e-6,
        verbose = false,
        component_trace = true,
    )

    res_alswarm_obj = cpd(
        A,
        r;
        solver = :rgd,
        init = ALSWarmStartInit(2; base_init = RandomInit()),
        maxiter = 5,
        tol = 1e-6,
        verbose = false,
    )
    @test res_alswarm_obj isa CPDResult
    @test isfinite(res_alswarm_obj.rel_error)

    res_alswarm_sym = cpd(
        A,
        r;
        solver = :rgd,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :random,
        maxiter = 5,
        tol = 1e-6,
        verbose = false,
    )
    @test res_alswarm_sym isa CPDResult
    @test isfinite(res_alswarm_sym.rel_error)

    res_als_warm = cpd(
        A,
        r;
        solver = :als,
        init = TuckerInit(),
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    res_alswarm_zero = cpd(
        A,
        r;
        solver = :rgd,
        init = :alswarm,
        warm_steps = 3,
        warm_init = TuckerInit(),
        maxiter = 0,
        tol = 1e-6,
        verbose = false,
    )
    res_manual_zero = cpd(
        A,
        r;
        solver = :rgd,
        p0 = res_als_warm,
        maxiter = 0,
        tol = 1e-6,
        verbose = false,
    )
    @test res_alswarm_zero.rel_error ≈ res_manual_zero.rel_error atol = 1e-12

    res_rcg =
        cpd(A, r; solver = :rcg, init = :tucker, maxiter = 5, tol = 1e-6, verbose = false)
    @test res_rcg.solver_info.total_iterations == res_rcg.iterations
    @test length(res_rcg.solver_info.accepted_stepsize_history) == res_rcg.iterations
    @test length(res_rcg.solver_info.line_search_trial_history) == res_rcg.iterations
    @test res_rcg.solver_info.function_evaluations >= 0
    @test res_rcg.solver_info.gradient_evaluations >= 1
end


@testset "cp_als.jl: CP-ALS stability on low-rank tensor" begin
    rng = MersenneTwister(1234)
    dims = (20, 18, 15)
    r = 3
    comps = [RankOneTensor(randn(rng), [randn(rng, d) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)
    A .+= 0.05 .* randn(rng, size(A)...)

    out = fit_cp_als(
        A,
        r;
        maxiter = 30,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
        return_stats = true,
    )
    @test isfinite(out.rel_error)
    @test all(isfinite, TensorKitchen.weights(out))
    @test out.rel_error < 0.2
end


@testset "cpd.jl: normalization policies and explicit CPDPoint" begin
    dims = (7, 6, 5)
    r = 2
    λ = [2.0, -0.75]
    U = [randn(dims[m], r) for m = 1:3]
    A_ref = reconstruct_cpd_rankr(components_from_factors(λ, U))

    q_sep = normalize_components(CPDPoint(λ, U), :lambda_separate)
    @test reconstruct_cpd_rankr(q_sep.lambda, q_sep.factors) ≈ A_ref
    @test all(
        isapprox(norm(q_sep.factors[m][:, k]), 1; atol = 1e-10) for m = 1:3 for k = 1:r
    )
    U_sep, λ_sep = normalize_components(U, λ, SeparateLambdaNormalization())
    @test U_sep isa Vector{Matrix{Float64}}
    @test λ_sep isa Vector{Float64}
    @test reconstruct_cpd_rankr(λ_sep, U_sep) ≈ A_ref

    @test_throws ArgumentError normalize_components(CPDPoint(λ, U), :last_mode)
    @test_throws ArgumentError normalize_components(CPDPoint(λ, U), :distribute_evenly)

    A_pos = abs.(A_ref)
    @test_throws ArgumentError cpd(
        A_pos,
        r;
        solver = :rgd,
        nonnegative = true,
        normalization = :lambda_separate,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )

    out_als = fit_cp_als(
        A_ref,
        r;
        maxiter = 3,
        tol = 1e-6,
        init = RandomInit(),
        normalization = :lambda_separate,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(out_als.rel_error)
end


@testset "cpd.jl: ProductManifold(Manifolds.Segre(...), ...) geometry" begin
    rng = MersenneTwister(77)
    dims = (8, 7, 6)
    r = 2
    comps = [RankOneTensor(randn(rng), [randn(rng, d) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)

    res_native = cpd(
        A,
        r;
        solver = :rgd,
        geometry = :native,
        maxiter = 10,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test isfinite(res_native.rel_error)
    @test length(TensorKitchen.weights(res_native)) == r
    @test length(TensorKitchen.factors(res_native)) == length(dims)
    res_native_rcg = cpd(
        A,
        r;
        solver = :rcg,
        geometry = :native,
        maxiter = 5,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test isfinite(res_native_rcg.rel_error)
    @test res_native_rcg.solver == :rcg

    model_native = TensorKitchen.RankRCPDModel(A, r; geometry = :native)
    @test hasproperty(TensorKitchen.manifold(model_native), :manifolds)
    @test length(TensorKitchen.manifold(model_native).manifolds) == r
    p_native = TensorKitchen.initial_point(model_native, TuckerDiagInit())
    g_native = grad(model_native, p_native)
    @test isnothing(
        ManifoldsBase.check_vector(
            TensorKitchen.manifold(model_native),
            p_native,
            g_native,
        ),
    )
    g_native_exact = TensorKitchen.rgrad_exact(model_native, p_native)
    M_native = TensorKitchen.manifold(model_native)
    basis_native = ManifoldsBase.DefaultOrthonormalBasis()
    d_native = manifold_dimension(M_native)
    # Finite-diff gradient check (one direction)
    e_1 = zeros(Float64, d_native)
    e_1[1] = 1.0
    X_1 = ManifoldsBase.get_vector(M_native, p_native, e_1, basis_native)
    X_1 ./= max(norm(M_native, p_native, X_1), eps(Float64))
    ϵ = 1e-5
    p_plus = ManifoldsBase.retract(
        M_native,
        p_native,
        ϵ * X_1,
        ManifoldsBase.ExponentialRetraction(),
    )
    p_minus = ManifoldsBase.retract(
        M_native,
        p_native,
        -ϵ * X_1,
        ManifoldsBase.ExponentialRetraction(),
    )
    fd =
        (
            TensorKitchen.cost(model_native, p_plus) -
            TensorKitchen.cost(model_native, p_minus)
        ) / (2 * ϵ)
    ip = ManifoldsBase.inner(M_native, p_native, g_native_exact, X_1)
    # Use relative tolerance when scale is large, else absolute (finite-diff noise)
    @test abs(fd - ip) < max(1e-4 * max(abs(fd), abs(ip)), 1e-8)

    g_native_proj = TensorKitchen.egrad_to_rgrad(
        M_native,
        p_native,
        TensorKitchen.egrad(model_native, p_native),
    )
    @test norm(M_native, p_native, g_native_exact - g_native_proj) < 1e-10

    res_exact = cpd(
        A,
        r;
        solver = :rgd,
        geometry = :native,
        gradient_mode = :exact_native,
        maxiter = 10,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test isfinite(res_exact.cost)
    res_native_eproj = cpd(
        A,
        r;
        solver = :rgd,
        geometry = :native,
        gradient_mode = :egrad_project,
        maxiter = 10,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test isfinite(res_native_eproj.cost)
end


@testset "cpd exact synthetic regression" begin
    rels = Float64[]
    for seed = 1:3
        A = _make_cp_tensor(seed; noisy = false)
        Random.seed!(10_000 + seed)
        res = cpd(
            A,
            3;
            solver = :als,
            init = :tucker,
            maxiter = 80,
            tol = 1e-8,
            verbose = false,
        )
        push!(rels, Float64(res.rel_error))
        @test res.converged
        @test res.rel_error < 1e-7
        @test res.iterations <= 80
        @test isfinite(res.grad_norm)
    end
    @test median(rels) < 1e-8
end

@testset "cpd noisy synthetic regression" begin
    rels = Float64[]
    for seed = 1:3
        A = _make_cp_tensor(seed; noisy = true)
        Random.seed!(20_000 + seed)
        res = cpd(
            A,
            3;
            solver = :als,
            init = :tucker,
            maxiter = 80,
            tol = 1e-8,
            verbose = false,
        )
        push!(rels, Float64(res.rel_error))
        @test res.converged
        @test res.rel_error < 0.02
        @test res.iterations <= 80
        @test isfinite(res.grad_norm)
    end
    @test median(rels) < 0.01
end
