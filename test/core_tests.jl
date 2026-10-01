@testset "manifolds: Segre / Tucker / Secant constructors" begin
    dims = (7, 6, 5)
    r = 3
    Ms = Manifolds.Segre(dims)
    @test factor_dims(Ms) == dims
    @test dim(Ms) > 0

    mlrank = (3, 2, 2)
    Mt = Manifolds.Tucker(dims, mlrank)
    @test factor_dims(Mt) == dims
    @test multilinear_rank(Mt) == mlrank
    @test dim(Mt) > 0

    Jt = TuckerJoin(dims, mlrank, 2)
    @test Jt isa ProductManifold
    @test length(Jt.manifolds) == 2
    @test length(join_product(Mt, 1).manifolds) == 1

    base_product = ProductManifold(Manifolds.Sphere(1), Manifolds.Sphere(2))
    Jp = join_product(base_product, 2)
    @test Jp isa ProductManifold
    @test length(Jp.manifolds) == 4
    @test_throws ArgumentError join_product(Mt, 0)

    Mt_vec = Manifolds.Tucker(collect(dims), collect(mlrank))
    @test factor_dims(Mt_vec) == dims
    @test multilinear_rank(Mt_vec) == mlrank
    @test dim(Mt_vec) > 0
end

# =========================================================================
# low-level rank-r solve + cpd packed-point helpers
# =========================================================================

@testset "utils: pack_point_rankr, unpack_point_rankr, cp_init_tucker" begin
    dims = (5, 4, 3)
    r = 2
    λ = [1.0, -0.5]
    U = [randn(dims[m], r) for m = 1:3]
    p = pack_point_rankr(λ, U, r)
    λ2, U2 = unpack_point_rankr(p, dims, r)
    @test λ2 ≈ λ && all(U2[m] ≈ U[m] for m = 1:3)

    p_native = TensorKitchen.pack_rankr_native(λ, U, r)
    λn, Un = TensorKitchen.unpack_rankr_native(p_native, dims, r)
    # ProductManifold(Manifolds.Segre(...), ...) packing performs gauge normalization.
    # Check representation-invariant equality (reconstructed tensor), not raw factors.
    A_in = reconstruct_cpd_rankr(components_from_factors(λ, U))
    A_native = reconstruct_cpd_rankr(components_from_factors(λn, Un))
    @test all(λn .>= 0)
    @test A_native ≈ A_in
    p_canonical = TensorKitchen.pack_rankr_canonical(λ, U, r)
    p_join = TensorKitchen.canonical_to_joinpoint(p_canonical, dims, r)
    p_canonical_roundtrip = TensorKitchen.joinpoint_to_canonical(p_join, dims, r)
    λ_join, U_join = TensorKitchen.unpack_rankr_native(p_join, dims, r)
    λ_canon_rt, U_canon_rt =
        TensorKitchen.unpack_rankr_canonical(p_canonical_roundtrip, dims, r)
    A_join = reconstruct_cpd_rankr(components_from_factors(λ_join, U_join))
    A_canon_rt = reconstruct_cpd_rankr(components_from_factors(λ_canon_rt, U_canon_rt))
    @test A_join ≈ A_in
    @test A_canon_rt ≈ A_in

    A = randn(8, 6, 5)
    λ0, U0 = cp_init_tucker(A, 3)
    @test length(λ0) == 3 && length(U0) == 3
    for m = 1:3
        for k = 1:3
            @test norm(U0[m][:, k]) ≈ 1 atol = 1e-10
        end
    end

    # TuckerInit should now differ from TuckerDiagInit on non-diagonal Tucker cores.
    dims_t = (6, 5, 4)
    r_t = 2
    λc = [1.0, 0.8]
    Cfac = [
        [1.0 0.6; 0.0 sqrt(1 - 0.6^2)],
        [1.0 -0.4; 0.0 sqrt(1 - 0.4^2)],
        [1.0 0.5; 0.0 sqrt(1 - 0.5^2)],
    ]
    core = reconstruct_cpd_rankr(λc, [Matrix(F) for F in Cfac])
    Q = [Matrix(qr(randn(dims_t[m], r_t)).Q[:, 1:r_t]) for m = 1:3]
    A_t = reconstruct_tucker(core, Q)
    # Tiny ambient perturbation so Tucker-diagonal weights and LS weights on HOSVD factors
    # are not identical (otherwise Frobenius errors can match to machine precision).
    A_t = A_t .+ 1e-5 .* randn(size(A_t))
    λ_diag, U_diag = TensorKitchen.init_cpd_factors(A_t, r_t; init = :tucker_diag)
    λ_tuck, U_tuck = TensorKitchen.init_cpd_factors(A_t, r_t; init = :tucker)
    err_diag = norm(reconstruct_cpd_rankr(λ_diag, U_diag) - A_t)
    err_tuck = norm(reconstruct_cpd_rankr(λ_tuck, U_tuck) - A_t)
    @test err_tuck <= err_diag + 1e-8

    # The public CPD initialization path should preserve the Tucker/TuckerDiag distinction.
    model_t = JoinModel(A_t, r_t; geometry = :canonical)
    p_diag = TensorKitchen.initial_point(model_t, TuckerDiagInit())
    p_tuck = TensorKitchen.initial_point(model_t, TuckerInit())
    point_diag = TensorKitchen.cpd_point(TensorKitchen.unwrap_model(model_t), p_diag)
    point_tuck = TensorKitchen.cpd_point(TensorKitchen.unwrap_model(model_t), p_tuck)
    err_diag_model = norm(
        reconstruct_cpd_rankr(
            TensorKitchen.lambda(point_diag),
            TensorKitchen.factors(point_diag),
        ) - A_t,
    )
    err_tuck_model = norm(
        reconstruct_cpd_rankr(
            TensorKitchen.lambda(point_tuck),
            TensorKitchen.factors(point_tuck),
        ) - A_t,
    )
    @test err_tuck_model <= err_diag_model + 1e-8

    # mttkrp: direct path should match explicit Khatri-Rao path
    U3 = [randn(8, 3), randn(6, 3), randn(5, 3)]
    for mode = 1:3
        G_kr = mttkrp(A, U3, mode; method = :khatri_rao)
        G_dir = mttkrp(A, U3, mode; method = :direct)
        @test G_dir ≈ G_kr atol = 1e-10
    end

    A4 = randn(7, 5, 4, 3)
    U4 = [randn(7, 2), randn(5, 2), randn(4, 2), randn(3, 2)]
    for mode = 1:4
        G_kr = mttkrp(A4, U4, mode; method = :khatri_rao)
        G_dir = mttkrp(A4, U4, mode; method = :direct)
        @test G_dir ≈ G_kr atol = 1e-10
    end

    A5 = randn(4, 3, 2, 3, 2)
    U5 = [randn(4, 2), randn(3, 2), randn(2, 2), randn(3, 2), randn(2, 2)]
    for mode = 1:5
        G_kr = mttkrp(A5, U5, mode; method = :khatri_rao)
        G_dir = mttkrp(A5, U5, mode; method = :direct)
        @test G_dir ≈ G_kr atol = 1e-10
    end

    ws_direct3 = TensorKitchen.CPALSWorkspace(A, size(A), 3; mttkrp_method = :direct)
    @test all(isnothing, ws_direct3.mttkrp_kr_work)
    @test all(isnothing, ws_direct3.mttkrp_kr_work2)
    @test all(x -> !isnothing(x), ws_direct3.mttkrp_tmp_work)

    ws_kr = TensorKitchen.CPALSWorkspace(A, size(A), 3; mttkrp_method = :khatri_rao)
    @test all(x -> !isnothing(x), ws_kr.mttkrp_kr_work)
    @test all(x -> !isnothing(x), ws_kr.mttkrp_kr_work2)

    ws_direct5 = TensorKitchen.CPALSWorkspace(A5, size(A5), 2; mttkrp_method = :direct)
    @test all(isnothing, ws_direct5.mttkrp_kr_work)
    @test all(isnothing, ws_direct5.mttkrp_kr_work2)
    @test all(isnothing, ws_direct5.mttkrp_tmp_work)

    out_buf = zeros(size(A, 1), size(U3[1], 2))
    @test_throws ArgumentError TensorKitchen.mttkrp!(
        out_buf,
        A,
        Matrix{Float64}[],
        1;
        method = :direct,
    )
    @test_throws DimensionMismatch TensorKitchen.mttkrp!(
        out_buf,
        A,
        [U3[1], U3[2]],
        1;
        method = :direct,
    )
    @test_throws DimensionMismatch TensorKitchen.mttkrp!(
        out_buf,
        A,
        [randn(7, 3), U3[2], U3[3]],
        1;
        method = :direct,
    )
    @test_throws DimensionMismatch TensorKitchen.mttkrp!(
        out_buf,
        A,
        [U3[1], randn(6, 2), U3[3]],
        1;
        method = :direct,
    )
end


@testset "utils: cross_component, build_cross_matrix, grad_lambda_cp, cp_rankr_cost_value, cross_term_gradU, gradU_column_cp" begin
    U = [randn(4, 2), randn(3, 2), randn(5, 2)]
    r = 2
    λ = [1.0, 0.5]
    comps = [RankOneTensor(λ[k], [U[m][:, k] for m = 1:length(U)]) for k = 1:r]
    @test cross_component(comps[1], comps[2]) isa Float64
    cm = build_cross_matrix_unit(comps)
    @test size(cm) == (2, 2) && cm[1, 2] == cm[2, 1]
    inner = [0.3, 0.2]
    gλ = grad_lambda_cp(λ, inner, cm)
    @test length(gλ) == 2
    c = cp_rankr_cost_value(10.0, λ, inner, cm)
    @test c isa Float64 && c >= 0
    ct = cross_term_gradU(comps, 1, 1)
    @test length(ct) == 4
    col = gradU_column_cp(
        TensorKitchen.λ(comps[1]),
        TensorKitchen.vectors(comps[1])[1],
        randn(4),
        ct,
    )
    @test length(col) == 4
end


@testset "squaring_metric.jl: 2D benchmark objectives" begin
    p = [2.0, -3.0]
    @test sm_cost_nn_quadratic(p; a = 1.0, b = 1.0) ≈ 36.5
    @test sm_egrad_nn_quadratic(p; a = 1.0, b = 1.0) ≈ [12.0, -48.0]

    @test sm_cost_nn_rank1([1.0, 1.0]; A = 2.0) ≈ 0.5
    @test sm_egrad_nn_rank1([1.0, 1.0]; A = 2.0) ≈ [-2.0, -2.0]

    @test sm_cost_repelling_test([0.0, 0.0]) ≈ 1.0
    @test sm_egrad_repelling_test([0.0, 0.0]) ≈ [0.0, 0.0]

    r0 = sm_double_circle_at_zero()
    @test r0.C1 ≈ 49.0
    @test r0.C2 ≈ 26.25
    @test r0.value ≈ 1286.25
    @test r0.gradient ≈ [-115.5, -752.5]
end


@testset "norm re-exported" begin
    @test norm([1.0, 0.0, 0.0]) ≈ 1.0
end
