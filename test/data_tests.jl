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

        @test actual_raw ≈ expected rtol = 2.0f-5 atol = 2.0f-5
        @test actual_lazy ≈ expected rtol = 2.0f-5 atol = 2.0f-5

        output = zeros(Float32, size(raw, mode), 2)
        @test implicit_mttkrp!(output, raw, factors, mode; block_length = 8) === output
        @test output ≈ expected rtol = 2.0f-5 atol = 2.0f-5
    end

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
end
