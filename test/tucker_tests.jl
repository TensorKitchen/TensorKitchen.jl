@testset "hosvd.jl: tucker_hosvd, reconstruct_tucker, reconstruction_error" begin
    A = randn(6, 5, 4)
    ranks = (3, 3, 2)
    core, factors = tucker_hosvd(A, ranks)
    @test size(core) == ranks
    @test length(factors) == 3
    for m = 1:3
        @test size(factors[m], 1) == size(A, m)
        @test size(factors[m], 2) == ranks[m]
    end
    Ahat = reconstruct_tucker(core, factors)
    @test size(Ahat) == size(A)
    Ahat_inplace = similar(Ahat)
    @test reconstruct_tucker!(Ahat_inplace, core, factors) === Ahat_inplace
    @test Ahat_inplace ≈ Ahat
    @test reconstruction_error(A, core, factors) >= 0
    @test reconstruction_error(A, core, factors) <= 1 + 1e-10
end

# =========================================================================
# tucker/sthosvd.jl
# =========================================================================

@testset "sthosvd.jl: sthosvd, thosvd, TuckerResult, relative_error" begin
    @test optimal_mode_order((120, 40, 30), (10, 20, 20)) == [1, 2, 3]
    @test optimal_mode_order((120, 40, 30)) == [3, 2, 1]

    dims = (10, 8, 6)
    r = (4, 3, 3)
    core = randn(r...)
    factors = [randn(dims[k], r[k]) for k = 1:3]
    A = reconstruct_tucker(core, factors)
    A2 = reconstruct_tucker(core, factors)
    B2 = A2 .+ 1e-4 .* randn(size(A2)...)
    nA = sum(abs2, A2)
    nδ = sum(abs2, A2 .- B2)
    rel_ref = nA > 0 ? sqrt(max(nδ, 0) / nA) : sqrt(max(nδ, 0))
    @test TensorKitchen.relative_frobenius_error(A2, B2) ≈ rel_ref rtol = 1e-12 atol = 1e-12
    @test rel_error(A2, B2) ≈ rel_ref rtol = 1e-12 atol = 1e-12
    td = sthosvd(A, r)
    @test processing_order(td) == optimal_mode_order(dims, r)
    @test td isa TuckerResult
    @test size(td.core) == r
    @test length(td.factors) == 3
    @test relative_error(A, td) < 1e-10
    @test rel_error(A, td) == relative_error(A, td)
    @test norm(A - reconstruct(td)) / norm(A) < 1e-10
    A_rand = randn(8, 6, 5)
    td_rand = sthosvd(A_rand, (3, 3, 2))
    @test relative_error(A_rand, td_rand) >= 0 &&
          relative_error(A_rand, td_rand) <= 1 + 1e-10
    td_th = thosvd(A, r)
    @test td_th isa TuckerResult
    @test size(td_th.core) == r
end


@testset "implicit randomized ST-HOSVD" begin
    rng = MersenneTwister(1701)

    # Generic tensor contractions must agree with the historical unfolding-based
    # implementations for every mode, including tensors above order three.
    A4 = randn(rng, Float64, 7, 6, 5, 4)
    for mode = 1:4
        U = randn(rng, Float64, 3, size(A4, mode))
        implicit_product =
            TensorKitchen._implicit_mode_product(A4, U, mode; block_columns = 5)
        @test implicit_product ≈ mode_n_product(A4, U, mode)

        Bdims = ntuple(m -> m == mode ? 2 : size(A4, m), 4)
        B4 = randn(rng, Float64, Bdims)
        implicit_cross = TensorKitchen._implicit_mode_cross(A4, B4, mode; block_columns = 5)
        explicit_cross = unfold_mode(A4, mode) * transpose(unfold_mode(B4, mode))
        @test implicit_cross ≈ explicit_cross
    end

    # With one block, the implicit Gaussian projection is exactly the explicit
    # mode unfolding multiplied by the same random matrix.
    A3 = randn(rng, Float32, 9, 8, 7)
    for mode = 1:3
        sketch_rank = 4
        implicit_rng = MersenneTwister(900 + mode)
        explicit_rng = MersenneTwister(900 + mode)
        implicit_sketch = TensorKitchen._implicit_mode_sketch(
            A3,
            mode,
            sketch_rank,
            implicit_rng;
            block_columns = length(A3),
        )
        other_modes = [m for m = 1:3 if m != mode]
        omega_dims = Tuple(vcat([size(A3, m) for m in other_modes], sketch_rank))
        omega = randn(explicit_rng, Float32, omega_dims)
        explicit_sketch = unfold_mode(A3, mode) * reshape(omega, :, sketch_rank)
        @test implicit_sketch ≈ explicit_sketch rtol = 5e-6 atol = 5e-6
    end

    # The public backend is deterministic for a fixed RNG and remains close to
    # deterministic ST-HOSVD on a low-rank tensor with small dense noise.
    dims = (36, 28, 20)
    ranks = (7, 6, 5)
    latent_core = randn(rng, Float32, ranks)
    latent_factors =
        [Matrix(qr(randn(rng, Float32, dims[m], ranks[m])).Q[:, 1:ranks[m]]) for m = 1:3]
    target = reconstruct_tucker(latent_core, latent_factors)
    target .+= 1.0f-3 .* randn(rng, Float32, dims)

    exact = sthosvd(target, ranks; processing_order = [2, 3, 1])
    randomized1 = sthosvd(
        target,
        ranks;
        processing_order = [2, 3, 1],
        svd_backend = :randomized,
        oversampling = 6,
        power_iterations = 1,
        block_columns = 128,
        rng = MersenneTwister(44),
    )
    randomized2 = sthosvd(
        target,
        ranks;
        processing_order = [2, 3, 1],
        svd_backend = :randomized,
        oversampling = 6,
        power_iterations = 1,
        block_columns = 128,
        rng = MersenneTwister(44),
    )

    @test size(core(randomized1)) == ranks
    @test core(randomized1) ≈ core(randomized2)
    @test all(factors(randomized1)[m] ≈ factors(randomized2)[m] for m = 1:3)
    @test all(
        isapprox(
            transpose(factors(randomized1)[m]) * factors(randomized1)[m],
            I;
            rtol = 2e-5,
            atol = 2e-5,
        ) for m = 1:3
    )
    @test relative_error(target, randomized1) <= 1.05 * relative_error(target, exact)
    @test all(isempty, singular_values(randomized1))
    @test_throws ArgumentError error_bound(randomized1)

    @test_throws ArgumentError sthosvd(target, ranks; svd_backend = :unknown)
    @test_throws ArgumentError sthosvd(
        target,
        ranks;
        svd_backend = :randomized,
        oversampling = -1,
    )
    @test_throws ArgumentError sthosvd(
        target,
        ranks;
        svd_backend = :randomized,
        block_columns = 0,
    )
end

# =========================================================================
# tucker/hooi.jl
# =========================================================================

@testset "hooi.jl: hooi (TuckerResult), init :sthosvd" begin
    dims = (8, 6, 5)
    ranks = (3, 3, 2)
    core = randn(ranks...)
    factors = [randn(dims[k], ranks[k]) for k = 1:3]
    A = reconstruct_tucker(core, factors)
    td = hooi(A, ranks; maxiter = 30, verbose = false)
    @test td isa TuckerResult
    @test size(td.core) == ranks
    @test reconstruction_error(A, td.core, td.factors) < 1e-9
    td_zero = hooi(A, ranks; maxiter = 0, verbose = false)
    @test td_zero isa TuckerResult
    @test size(td_zero.core) == ranks
    @test reconstruction_error(A, td_zero.core, td_zero.factors) < 1e-9
    td_st = hooi(A, ranks; init = :sthosvd, maxiter = 20, verbose = false)
    @test td_st isa TuckerResult
    @test_throws ErrorException hooi(A, ranks; init = :hosvd, maxiter = 20, verbose = false)
    @test_throws ErrorException hooi(
        A,
        ranks;
        init = :thosvd,
        maxiter = 20,
        verbose = false,
    )
end

# =========================================================================
# low-level rank-1 solve + packed-point helpers
# =========================================================================

@testset "Tucker retraction preserves point scalar type" begin
    rng = MersenneTwister(811)
    dims = (5, 4, 3)
    ranks = (2, 2, 2)
    M = Manifolds.Tucker(dims, ranks)
    make_point = function (::Type{T}) where {T<:AbstractFloat}
        factors = ntuple(3) do mode
            Q = qr(randn(rng, T, dims[mode], ranks[mode])).Q
            Matrix(Q[:, 1:ranks[mode]])
        end
        return Manifolds.TuckerPoint(randn(rng, T, ranks...), factors...)
    end

    @test 0.1 isa Float64
    for T in (Float32, Float64)
        p = make_point(T)
        X = rand(rng, M; vector_at = p)
        method = TensorKitchen._solver_retraction_method(M, p)
        q = ManifoldsBase.retract_fused(M, p, X, 0.1, method)

        @test eltype(q.hosvd.core) === T
        @test all(eltype(U) === T for U in q.hosvd.U)
    end

    M_product = ProductManifold(M, M)
    p_product = ArrayPartition(make_point(Float32), make_point(Float32))
    X_product = rand(rng, M_product; vector_at = p_product)
    method_product = TensorKitchen._solver_retraction_method(M_product, p_product)
    q_product =
        ManifoldsBase.retract_fused(M_product, p_product, X_product, 0.1, method_product)

    for q_part in TensorKitchen.point_parts(q_product)
        @test eltype(q_part.hosvd.core) === Float32
        @test all(eltype(U) === Float32 for U in q_part.hosvd.U)
    end
end


@testset "tucker exact synthetic regression" begin
    methods = (:sthosvd, :hooi)
    for method in methods
        for seed = 1:3
            A = _make_tucker_tensor(seed; noisy = false)
            kwargs = method == :hooi ? (; maxiter = 20, tol = 1e-8, verbose = false) : (;)
            td = tucker(A, (4, 3, 2); method = method, kwargs...)
            @test rel_error(A, td) < 1e-12
        end
    end
end

@testset "tucker noisy synthetic regression" begin
    for seed = 1:3
        A = _make_tucker_tensor(seed; noisy = true)
        td_sthosvd = tucker(A, (4, 3, 2); method = :sthosvd)
        td_hooi =
            tucker(A, (4, 3, 2); method = :hooi, maxiter = 20, tol = 1e-8, verbose = false)
        err_sthosvd = rel_error(A, td_sthosvd)
        err_hooi = rel_error(A, td_hooi)
        @test err_sthosvd < 0.02
        @test err_hooi < 0.02
        @test err_hooi <= err_sthosvd + 1e-8
    end
end
