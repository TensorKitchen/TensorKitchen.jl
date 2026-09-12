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

    mode_matrix = randn(rng, Float32, 2, size(raw, 2))
    implicit_product =
        TensorKitchen._implicit_mode_product(lazy, mode_matrix, 2; block_columns = 5)
    @test implicit_product ≈ mode_n_product(dense, mode_matrix, 2) rtol = 2.0f-5 atol =
        2.0f-5

    comparison = randn(rng, Float32, 7, size(raw, 2), size(raw, 3))
    implicit_cross =
        TensorKitchen._implicit_mode_cross(lazy, comparison, 1; block_columns = 5)
    expected_cross = unfold_mode(dense, 1) * transpose(unfold_mode(comparison, 1))
    @test implicit_cross ≈ expected_cross rtol = 2.0f-5 atol = 2.0f-5

    @test_throws ArgumentError observation_norm2(raw; block_length = 0)
    @test_throws DimensionMismatch implicit_mttkrp(raw, factors[1:2], 1)
    @test_throws ArgumentError mttkrp(raw, factors, 1; method = :khatri_rao)
end
