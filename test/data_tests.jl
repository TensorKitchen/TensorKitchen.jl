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
