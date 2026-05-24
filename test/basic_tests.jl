

# =========================================================================
# tucker/hosvd.jl
# =========================================================================
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

# =========================================================================
# cpd/cp_rank.jl (cost/egrad functions)
# =========================================================================
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

@testset "frontend defaults through public APIs" begin
    A = randn(6, 5, 4)
    cpd_res = cpd(A, 2; maxiter = 1, verbose = false)
    @test cpd_res isa CPDResult
    @test cpd_res.solver == :rgd

    nncpd_res = nncpd(abs.(A), 2; maxiter = 1, verbose = false)
    @test nncpd_res isa CPDResult
    @test nncpd_res.solver == :rgd

    cpd_rgd_object = cpd(
        A,
        2;
        solver = RGDSolver(),
        init = :alswarm,
        warm_steps = 2,
        maxiter = 1,
        verbose = false,
    )
    @test cpd_rgd_object isa CPDResult
    @test cpd_rgd_object.solver == :rgd

    cpd_als_object =
        cpd(A, 2; solver = ALSSolver(), init = :tucker, maxiter = 1, verbose = false)
    @test cpd_als_object isa CPDResult
    @test cpd_als_object.solver == :cp_als
    @test_throws ArgumentError cpd(
        A,
        2;
        solver = ALSSolver(),
        gradient_mode = :egrad_project,
        maxiter = 1,
        verbose = false,
    )
    @test_throws ArgumentError cpd(A, 2; solver = "rgd", maxiter = 1, verbose = false)

    nncpd_als_object = nncpd(
        abs.(A),
        2;
        solver = ALSSolver(),
        init = :tucker,
        maxiter = 1,
        verbose = false,
    )
    @test nncpd_als_object isa CPDResult
    @test nncpd_als_object.solver == :cp_als

    btd_res = btd(A, 2, (3, 2, 2); maxiter = 1, verbose = false)
    @test btd_res isa BTDResult
    @test btd_res.solver == :rgd

    approx_res = approx(JoinModel(A, 2); maxiter = 1, verbose = false)
    @test approx_res isa ApproxResult
    @test approx_res.solver == :rgd

    generic_segre = JoinModel(Manifolds.Segre((6, 5, 4)), A)
    @test generic_segre isa JoinModel

    sphere_target = [1.2, 0.4, -0.3]
    sphere_single = approx(Manifolds.Sphere(2), sphere_target; maxiter = 1, verbose = false)
    @test sphere_single isa ApproxResult
    @test isfinite(sphere_single.rel_error)

    sphere_pair = approx(
        (Manifolds.Sphere(2), Manifolds.Sphere(2)),
        sphere_target;
        maxiter = 1,
        verbose = false,
    )
    @test sphere_pair isa ApproxResult
    @test isfinite(sphere_pair.rel_error)
end

@testset "solver helpers accept omitted normA2" begin
    M = Euclidean(2)
    p0 = [1.0, -2.0]
    model_cost = (M, p) -> 0.5 * sum(abs2, p)
    model_egrad = (M, p) -> p

    res_rgd = TensorKitchen.solve_rgd(
        model_cost,
        model_egrad,
        M,
        p0;
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(res_rgd.cost)
    @test isfinite(res_rgd.rel_error)

    res_rgd_fixed = TensorKitchen.solve_rgd_fixed(
        model_cost,
        model_egrad,
        M,
        p0;
        maxiter = 2,
        stepsize = 0.1,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(res_rgd_fixed.cost)
    @test isfinite(res_rgd_fixed.rel_error)

    res_rcg = TensorKitchen.solve_rcg(
        model_cost,
        model_egrad,
        M,
        p0;
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(res_rcg.cost)
    @test isfinite(res_rcg.rel_error)
end

@testset "cpd.jl: nonnegative CPD keeps cost nonnegative on larger tensors" begin
    Random.seed!(1)
    A = abs.(randn(60, 50, 40))
    res = cpd(A, 5; solver = :rgd, nonnegative = true, maxiter = 3, verbose = false)
    @test isfinite(res.cost)
    @test res.cost >= 0
    @test isfinite(res.rel_error)
    @test res.rel_error > 0
end

@testset "cpd.jl: nonnegative analytic cost matches explicit residual" begin
    Random.seed!(7)
    dims = (8, 6, 5)
    r = 3
    A = abs.(randn(Float64, dims...))
    model = JoinModel(A, r; nonnegative = true)
    p = TensorKitchen.initial_point(model, RandomInit())
    λ̃, Ũ = unpack_point_rankr(p, dims, r)
    λ = λ̃ .^ 2
    U = [Um .^ 2 for Um in Ũ]
    normA2 = sum(abs2, A)
    M1 = mttkrp(A, U, 1; method = :auto)
    inner = TensorKitchen._inner_from_mttkrp_first_mode(U, M1)
    grams = TensorKitchen._gram_matrices(U)
    cross_mat = TensorKitchen._cross_unit_from_grams(grams)
    analytic_cost = cp_rankr_cost_value(normA2, λ, inner, cross_mat)
    _, explicit_cost, _ = TensorKitchen.cp_residual_stats_explicit(A, normA2, λ, U)
    @test analytic_cost ≈ explicit_cost atol = 1e-8 rtol = 1e-8
end

@testset "cpd.jl: nonnegative solver outputs match explicit residual" begin
    explicit_stats(A, res) = begin
        X = reconstruct_cpd_rankr(TensorKitchen.weights(res), TensorKitchen.factors(res))
        cost = 0.5 * sum(abs2, X .- A)
        rel = norm(A) > 0 ? norm(X .- A) / norm(A) : norm(X .- A)
        cost, rel
    end
    public_columns_unit(res) = all(
        isapprox(norm(TensorKitchen.factors(res)[m][:, k]), 1; atol = 1e-8, rtol = 1e-8) for m in eachindex(TensorKitchen.factors(res)) for
        k in eachindex(TensorKitchen.weights(res))
    )

    Random.seed!(42)
    A1 = abs.(randn(18, 14, 10))
    for solver in (:rgd, :rcg)
        res = cpd(A1, 1; solver = solver, nonnegative = true, maxiter = 4, verbose = false)
        cost, rel = explicit_stats(A1, res)
        @test res.cost ≈ cost atol = 1e-8 rtol = 1e-8
        @test res.rel_error ≈ rel atol = 1e-8 rtol = 1e-8
        @test public_columns_unit(res)
    end

    Random.seed!(42)
    Ar = abs.(randn(20, 16, 12))
    for solver in (:als, :rgd, :rcg)
        res = cpd(Ar, 3; solver = solver, nonnegative = true, maxiter = 4, verbose = false)
        cost, rel = explicit_stats(Ar, res)
        @test res.cost ≈ cost atol = 1e-8 rtol = 1e-8
        @test res.rel_error ≈ rel atol = 1e-8 rtol = 1e-8
        @test public_columns_unit(res)
    end
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

    res_trace = cpd(
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

@testset "cpd.jl: NonnegativeSeparateLambdaNormalization" begin
    dims = (8, 7, 6)
    r = 3
    λ = [1.0e6, 1.0e-4, 2.0]
    U = [
        [1.0e3 1.0e-2 3.0; 2.0e2 4.0e-3 1.0; 5.0e1 2.0e-2 0.5],
        [8.0e2 3.0e-3 2.0; 1.0e2 1.0e-2 4.0; 4.0e1 5.0e-3 1.0],
        [6.0e2 2.0e-2 1.0; 9.0e1 4.0e-3 3.0; 3.0e1 1.0e-2 2.0],
    ]
    A_ref = reconstruct_cpd_rankr(components_from_factors(λ, U))

    q = normalize_components(CPDPoint(λ, U), NonnegativeSeparateLambdaNormalization())
    @test reconstruct_cpd_rankr(q.lambda, q.factors) ≈ A_ref
    @test all(q.lambda .>= 0)
    @test all(F -> all(F .>= 0), q.factors)
    @test all(isapprox(norm(q.factors[m][:, k]), 1; atol = 1e-10) for m = 1:3 for k = 1:r)

    rng = MersenneTwister(77)
    comps =
        [RankOneTensor(abs(randn(rng)), [abs.(randn(rng, d)) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)
    res_auto = cpd(
        A,
        r;
        solver = :rgd,
        nonnegative = true,
        geometry = :softplus_metric,
        normalization = :auto,
        maxiter = 5,
        tol = 1e-6,
        verbose = false,
    )
    @test res_auto isa CPDResult
    @test isfinite(res_auto.rel_error)

    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :rgd,
        nonnegative = true,
        normalization = :lambda_separate,
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
end

@testset "cpd.jl: nonnegative ALS auto normalization uses no normalization" begin
    rng = MersenneTwister(91)
    dims = (12, 10, 8)
    r = 3
    comps =
        [RankOneTensor(abs(randn(rng)), [abs.(randn(rng, d)) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)

    Random.seed!(2026)
    res_auto = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        init = :random,
        normalization = :auto,
        verbose = false,
    )
    Random.seed!(2026)
    res_none = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        init = :random,
        normalization = :none,
        verbose = false,
    )
    Random.seed!(2026)
    res_sep = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        init = :random,
        normalization = :lambda_separate,
        verbose = false,
    )

    @test res_auto.rel_error ≈ res_none.rel_error atol = 1e-12 rtol = 1e-12
    @test res_auto.grad_norm ≈ res_none.grad_norm atol = 1e-12 rtol = 1e-12
    @test !(
        isapprox(res_auto.rel_error, res_sep.rel_error; atol = 1e-12, rtol = 1e-12) &&
        isapprox(res_auto.grad_norm, res_sep.grad_norm; atol = 1e-12, rtol = 1e-12)
    )

    Random.seed!(2026)
    res_default_update = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        maxiter = 40,
        tol = 1e-6,
        init = :random,
        normalization = :auto,
        verbose = false,
    )
    @test res_default_update.rel_error ≈ res_auto.rel_error atol = 1e-12 rtol = 1e-12
    @test res_default_update.grad_norm ≈ res_auto.grad_norm atol = 1e-12 rtol = 1e-12
end

@testset "cpd.jl: nonnegative CPD option" begin
    rng = MersenneTwister(2026)
    dims = (16, 14, 12)
    r = 3
    comps =
        [RankOneTensor(abs(randn(rng)), [abs.(randn(rng, d)) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)
    A .+= 0.01 .* abs.(randn(rng, size(A)...))

    out_nn = fit_cp_als(
        A,
        r;
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
        return_stats = true,
        nonnegative = true,
    )
    @test all(w -> w >= -1e-12, TensorKitchen.weights(out_nn))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(out_nn))
    @test isfinite(out_nn.rel_error)

    res_nn = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn))
    @test isfinite(res_nn.rel_error)

    res_nn_api = nncpd(
        A,
        r;
        solver = :als,
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_api))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_api))
    @test isfinite(res_nn_api.rel_error)

    res_nn_als_warm = nncpd(
        A,
        r;
        solver = :als,
        maxiter = 3,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    res_nn_alswarm_rgd = nncpd(
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
    res_nn_manual_rgd = nncpd(
        A,
        r;
        solver = :rgd,
        p0 = res_nn_als_warm,
        maxiter = 0,
        tol = 1e-6,
        verbose = false,
    )
    @test res_nn_alswarm_rgd.rel_error ≈ res_nn_manual_rgd.rel_error atol = 1e-12

    res_nn_api_heur = nncpd(
        A;
        solver = :als,
        maxiter = 2,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test length(TensorKitchen.weights(res_nn_api_heur)) == minimum(size(A))
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_api_heur))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_api_heur))

    res_nn_mu = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :mu,
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test isfinite(res_nn_mu.rel_error)
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_mu))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_mu))

    res_nn_hals = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :hals,
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test isfinite(res_nn_hals.rel_error)
    @test isfinite(res_nn_hals.grad_norm)
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_hals))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_hals))

    res_nn_nnls = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 30,
        tol = 1e-6,
        init = RandomInit(),
        verbose = false,
    )
    @test isfinite(res_nn_nnls.rel_error)
    @test isfinite(res_nn_nnls.grad_norm)
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_nnls))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_nnls))

    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :bad,
        maxiter = 2,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :als,
        nn_update = :mu,
        maxiter = 2,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :als,
        geometry = :native,
        maxiter = 2,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :als,
        gradient_mode = :exact_native,
        maxiter = 2,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :rals,
        geometry = :native,
        maxiter = 2,
        verbose = false,
    )

    res_nn_rgd = cpd(A, r; solver = :rgd, nonnegative = true, maxiter = 50, verbose = false)
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_rgd))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_rgd))
    @test isfinite(res_nn_rgd.rel_error)

    # geometry=:squaring_metric selects SqEuclidean manifold.
    model_sm = TensorKitchen.RankRCPDModel(
        A,
        r;
        nonnegative = true,
        geometry = :squaring_metric,
        lambda_eps = 1e-8,
    )
    @test TensorKitchen.manifold(model_sm) isa ProductManifold
    @test all(m -> m isa SqEuclidean, TensorKitchen.manifold(model_sm).manifolds)
    join_model_sm = JoinModel(A, r; nonnegative = true, geometry = :squaring_metric)
    @test all(
        m -> m isa SqEuclidean,
        TensorKitchen.manifold(TensorKitchen.cpd_model(join_model_sm)).manifolds,
    )
    @test getproperty(model_sm, :scale_by_lambda) == false
    res_nn_sm = cpd(
        A,
        r;
        solver = :rgd,
        nonnegative = true,
        geometry = :squaring_metric,
        maxiter = 20,
        verbose = false,
    )
    @test isfinite(res_nn_sm.rel_error)
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn_sm))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn_sm))
    res_nn_sm_r1 = cpd(
        A,
        1;
        solver = :rgd,
        nonnegative = true,
        geometry = :squaring_metric,
        maxiter = 20,
        verbose = false,
    )
    @test isfinite(res_nn_sm_r1.rel_error)
    model_sm_r1 =
        TensorKitchen.Rank1CPDModel(A; nonnegative = true, use_pullback_metric = true)
    @test TensorKitchen.manifold(model_sm_r1) isa ProductManifold
    @test all(m -> m isa SqEuclidean, TensorKitchen.manifold(model_sm_r1).manifolds)
    @test getproperty(model_sm_r1, :scale_by_lambda) == false

    # Pullback regularization is public for nonnegative manifold solvers.
    res_sp_eps = nncpd(
        A,
        r;
        solver = :rgd,
        init = :tucker,
        geometry = :softplus_metric,
        pullback_eps = 1e-10,
        maxiter = 1,
        verbose = false,
    )
    @test res_sp_eps.solver_info.nncp_pullback_eps ≈ 1e-10

    model_sp = TensorKitchen.RankRCPDModel(
        A,
        r;
        nonnegative = true,
        geometry = :softplus_metric,
        pullback_eps = 1e-10,
    )
    @test TensorKitchen.manifold(model_sp) isa ProductManifold
    @test all(m -> m isa SoftplusEuclidean, TensorKitchen.manifold(model_sp).manifolds)
    @test all(
        m -> getproperty(m, :ε) ≈ 1e-10,
        getproperty(TensorKitchen.manifold(model_sp), :manifolds),
    )

    model_sp_r1 = TensorKitchen.Rank1CPDModel(
        A;
        nonnegative = true,
        use_softplus_metric = true,
        pullback_eps = 1e-10,
    )
    @test all(m -> m isa SoftplusEuclidean, TensorKitchen.manifold(model_sp_r1).manifolds)
    @test all(
        m -> getproperty(m, :ε) ≈ 1e-10,
        getproperty(TensorKitchen.manifold(model_sp_r1), :manifolds),
    )

    # Dense strictly-positive exact rank-2 tensor that both pullback geometries
    # should recover nearly exactly with the default full nncpd() pipeline.
    let
        dims_easy = (10, 8, 6)
        r_easy = 2
        rng_easy = MersenneTwister(3)
        λ_easy = rand(rng_easy, r_easy) .+ 0.5
        U_easy = [rand(rng_easy, dims_easy[m], r_easy) .+ 0.2 for m = 1:length(dims_easy)]
        A_easy = reconstruct_cpd_rankr(λ_easy, U_easy)

        Random.seed!(777)
        res_sq_easy = nncpd(
            A_easy,
            r_easy;
            solver = :rgd,
            geometry = :squaring_metric,
            maxiter = 200,
            tol = 1e-10,
            verbose = false,
        )
        @test isfinite(res_sq_easy.rel_error)
        @test res_sq_easy.rel_error < 1e-4

        Random.seed!(777)
        res_sp_easy = nncpd(
            A_easy,
            r_easy;
            solver = :rgd,
            geometry = :softplus_metric,
            maxiter = 200,
            tol = 1e-10,
            verbose = false,
        )
        @test isfinite(res_sp_easy.rel_error)
        @test res_sp_easy.rel_error < 1e-4
    end

    # Sparse exact rank-3 tensor: softplus ALSWarm path should preserve the
    # correct warm-start geometry instead of squaring latent coordinates.
    let
        dims_sparse = (30, 24, 18)
        r_sparse = 3
        λ_sparse = [2.5, 1.7, 1.2]
        U1 = zeros(dims_sparse[1], r_sparse)
        U2 = zeros(dims_sparse[2], r_sparse)
        U3 = zeros(dims_sparse[3], r_sparse)

        U1[1:6, 1] .= [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
        U1[11:18, 2] .= [1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.44]
        U1[23:30, 3] .= [1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37]

        U2[1:5, 1] .= [1.0, 0.86, 0.72, 0.58, 0.44]
        U2[9:15, 2] .= [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
        U2[18:24, 3] .= [1.0, 0.89, 0.78, 0.67, 0.56, 0.45, 0.34]

        U3[1:4, 1] .= [1.0, 0.82, 0.64, 0.46]
        U3[7:12, 2] .= [1.0, 0.88, 0.76, 0.64, 0.52, 0.40]
        U3[14:18, 3] .= [1.0, 0.85, 0.70, 0.55, 0.40]

        A_sparse = reconstruct_cpd_rankr(λ_sparse, [U1, U2, U3])

        Random.seed!(777)
        res_sp_sparse = nncpd(
            A_sparse,
            r_sparse;
            solver = :rgd,
            geometry = :softplus_metric,
            warm_steps = 500,
            maxiter = 200,
            verbose = false,
        )
        @test isfinite(res_sp_sparse.rel_error)
        @test res_sp_sparse.rel_error < 1e-4
    end

    @test_throws ArgumentError nncpd(
        A,
        r;
        solver = :rgd,
        init = :tucker,
        geometry = :softplus_metric,
        pullback_eps = 0.0,
        maxiter = 1,
        verbose = false,
    )
    @test_throws ArgumentError cpd(
        A,
        r;
        solver = :rgd,
        geometry = :squaring_metric,
        nonnegative = false,
        maxiter = 3,
        verbose = false,
    )

    # Regularized squaring metric (SqEuclidean): positive definite everywhere,
    # inner uses the diagonal metric G(p).
    M_pb = SqEuclidean(r * (1 + sum(dims)))
    @test M_pb isa SqEuclidean
    n_params = r * (1 + sum(dims))
    p_test = randn(rng, n_params) .+ 0.5  # stay positive
    X_test = randn(rng, n_params)
    inner_val = ManifoldsBase.inner(M_pb, p_test, X_test, X_test)
    @test inner_val > 0 && isfinite(inner_val)

    # Regularized squaring geometry: directional derivative should match the
    # Riemannian inner product with the converted gradient.
    let
        function _parts(x)
            return hasproperty(x, :x) ? Tuple(getproperty(x, :x)) : Tuple(x)
        end
        function _add_scaled(p, X, h)
            pp = _parts(p)
            XX = _parts(X)
            return ntuple(i -> pp[i] .+ h .* XX[i], length(pp))
        end
        function _rand_like(rng, p)
            pp = _parts(p)
            return ntuple(i -> randn(rng, size(pp[i])...), length(pp))
        end
        function _product_inner(M, p, X, Y)
            pp = _parts(p)
            XX = _parts(X)
            YY = _parts(Y)
            return sum(
                ManifoldsBase.inner(M.manifolds[i], pp[i], XX[i], YY[i]) for
                i = 1:length(M.manifolds)
            )
        end

        rng_pull = MersenneTwister(17)
        A_pull = abs.(randn(rng_pull, 8, 7, 6))

        model_r1 = TensorKitchen.Rank1CPDModel(
            A_pull;
            nonnegative = true,
            use_pullback_metric = true,
        )
        M_r1 = TensorKitchen.manifold(model_r1)
        p_r1 = TensorKitchen.initial_point(model_r1, :tucker; verbose = false)
        g_r1 = TensorKitchen.model_rgrad_function(model_r1)(M_r1, p_r1)
        X_r1 = _rand_like(rng_pull, p_r1)
        f_r1(q) = cost(model_r1, q)
        deriv_r1 = _product_inner(M_r1, p_r1, g_r1, X_r1)
        fd_r1 = (f_r1(_add_scaled(p_r1, X_r1, 1e-7)) - f_r1(p_r1)) / 1e-7
        @test isapprox(fd_r1, deriv_r1; rtol = 1e-4, atol = 1e-4)

        model_rr = TensorKitchen.RankRCPDModel(
            A_pull,
            2;
            nonnegative = true,
            geometry = :squaring_metric,
        )
        M_rr = TensorKitchen.manifold(model_rr)
        p_rr = TensorKitchen.initial_point(model_rr, :tucker; verbose = false)
        g_rr = TensorKitchen.model_rgrad_function(model_rr)(M_rr, p_rr)
        X_rr = _rand_like(rng_pull, p_rr)
        f_rr(q) = cost(model_rr, q)
        deriv_rr = _product_inner(M_rr, p_rr, g_rr, X_rr)
        fd_rr = (f_rr(_add_scaled(p_rr, X_rr, 1e-7)) - f_rr(p_rr)) / 1e-7
        @test isapprox(fd_rr, deriv_rr; rtol = 1e-4, atol = 1e-4)
    end
end

@testset "cp_als.jl: nonnegative HOSVD init stays strictly positive" begin
    rng = MersenneTwister(77)
    dims = (12, 10, 8)
    r = 3
    comps =
        [RankOneTensor(abs(randn(rng)), [abs.(randn(rng, d)) for d in dims]) for _ = 1:r]
    A = reconstruct_cpd_rankr(comps)

    out = fit_cp_als(
        A,
        r;
        maxiter = 0,
        tol = 1e-6,
        init = HOSVDInit(),
        verbose = false,
        return_stats = true,
        nonnegative = true,
    )
    @test all(w -> w > 0, TensorKitchen.weights(out))
    @test all(F -> all(F .> 0), TensorKitchen.factors(out))
    @test isfinite(out.grad_norm)
    @test out.grad_norm > 0

    out_early = fit_cp_als(
        A,
        r;
        maxiter = 1,
        tol = 1e6,
        init = RandomInit(),
        verbose = false,
        return_stats = true,
        nonnegative = true,
    )
    @test !out_early.converged
    @test isfinite(out_early.grad_norm)
    @test out_early.grad_norm > 0
end

@testset "nncp_updates.jl: row NNLS update reduces local quadratic objective" begin
    T = Float64
    V = T[2.0 0.3 0.1; 0.3 1.7 0.2; 0.1 0.2 1.4]
    g = T[0.8, 0.4, 0.6]
    x = T[0.2, 0.05, 0.1]
    work = similar(x)
    obj(z) = 0.5 * dot(z, V, z) - dot(g, z)
    before = obj(x)
    TensorKitchen._nncp_nnls_row_update!(x, g, V, work)
    after = obj(x)
    @test all(x .> 0)
    @test after <= before + 1e-12
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

@testset "Join pipeline: Sphere join, gradient modes, CPD routing" begin
    model = JoinModel(Manifolds.Sphere(1), 2, [1.2, 0.4])
    M = TensorKitchen.manifold(model)

    # Gradient conversions
    p0 = TensorKitchen.initial_point(model, :deterministic)
    eg0 = TensorKitchen.egrad(model, p0)
    gp0 = TensorKitchen.egrad_to_rgrad(M, p0, eg0)
    gd0 = TensorKitchen.rgrad(model, p0)
    @test isnothing(
        ManifoldsBase.check_vector(M, ArrayPartition(p0...), ArrayPartition(gp0...)),
    )
    @test norm(
        (hasproperty(gp0, :x) ? vcat(gp0.x[1], gp0.x[2]) : vcat(gp0[1], gp0[2])) -
        (hasproperty(gd0, :x) ? vcat(gd0.x[1], gd0.x[2]) : vcat(gd0[1], gd0[2])),
    ) < 1e-12

    # RGD convergence (Sphere join: sum of 2 points on S¹ approximates target)
    res = solve(
        RGDSolver(1.0),
        model;
        gradient_mode = :riemannian,
        init = :deterministic,
        maxiter = 600,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(res.cost) && res.cost >= 0
    @test res.iterations > 0
    @test res.solver_info.gradient_source == :state
    @test haskey(pairs(res.solver_info), :has_converged_state)
    @test haskey(pairs(res.solver_info), :converged_by_gradient_threshold)

    res_lbfgs = solve(
        LBFGSSolver(memory_size = 5),
        model;
        gradient_mode = :riemannian,
        init = :deterministic,
        maxiter = 30,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test res_lbfgs.solver == :lbfgs
    @test isfinite(res_lbfgs.cost) && res_lbfgs.cost >= 0
    @test res_lbfgs.solver_info.memory_size == 5

    # Gradient modes equivalent (exact_join, exact_join_basis, riemannian)
    res_r = solve(
        RGDSolver(1.0),
        model;
        gradient_mode = :riemannian,
        init = :deterministic,
        maxiter = 120,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )
    res_x = solve(
        RGDSolver(1.0),
        model;
        gradient_mode = :exact_join,
        init = :deterministic,
        maxiter = 120,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )
    res_b = solve(
        RGDSolver(1.0),
        model;
        gradient_mode = :exact_join_basis,
        init = :deterministic,
        maxiter = 120,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )
    @test isapprox(res_x.cost, res_r.cost; atol = 1e-12, rtol = 1e-12)
    @test isapprox(res_b.cost, res_r.cost; atol = 1e-12, rtol = 1e-12)

    @test_throws ArgumentError solve(
        RGDSolver(1.0),
        model;
        gradient_mode = :exact_native,
        init = :deterministic,
        maxiter = 60,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )

    # CPD through JoinModel
    rng = MersenneTwister(909)
    A = randn(rng, 6, 5, 4)
    jm_cpd = JoinModel(A, 2; geometry = :canonical)
    @test TensorKitchen.unwrap_model(jm_cpd) isa TensorKitchen.RankRCPDModel
    res_jm = solve(
        RGDSolver(1.0),
        jm_cpd;
        gradient_mode = :riemannian,
        init = TuckerInit(),
        maxiter = 20,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(res_jm.cost) && isfinite(res_jm.rel_error)
end

@testset "Core approx + cpd/btd API split" begin
    rng = MersenneTwister(111)

    # approx(manifolds, target)
    target = [1.2, 0.4]
    manifolds = (Manifolds.Sphere(1), Manifolds.Sphere(1))
    res_j = approx(
        manifolds,
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 80,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_j isa ApproxResult
    @test length(res_j.components) == 2
    @test isfinite(res_j.cost) && isfinite(res_j.rel_error)

    # approx(manifolds, target) accepts multiple manifold container types
    res_a_tuple = approx(
        manifolds,
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 30,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_a_tuple isa ApproxResult
    @test isfinite(res_a_tuple.cost)

    res_a_vec = approx(
        collect(manifolds),
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 30,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_a_vec isa ApproxResult
    @test isfinite(res_a_vec.cost)

    Mprod = ProductManifold(manifolds...)
    res_a_prod = approx(
        Mprod,
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 30,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_a_prod isa ApproxResult
    @test isfinite(res_a_prod.cost)

    res_a_base_r = approx(
        Manifolds.Sphere(1),
        2,
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 30,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_a_base_r isa ApproxResult
    @test isfinite(res_a_base_r.cost)

    res_a_base = approx(
        Manifolds.Sphere(1),
        target;
        init = :deterministic,
        solver = :rgd,
        maxiter = 30,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )
    @test res_a_base isa ApproxResult
    @test isfinite(res_a_base.cost)

    # approx auto-routes uniform Segre inputs to CPD.
    dims_segre = (4, 3, 2)
    target_segre = randn(rng, dims_segre...)
    segre_manifolds = (Manifolds.Segre(dims_segre), Manifolds.Segre(dims_segre))
    res_segre_cpd = approx(
        segre_manifolds,
        target_segre;
        init = TuckerInit(),
        solver = :rgd,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    @test res_segre_cpd isa CPDResult
    @test isfinite(res_segre_cpd.cost)

    @test_throws ArgumentError approx(
        manifolds,
        target;
        dispatch = :cpd,
        init = :deterministic,
        solver = :rgd,
        maxiter = 3,
        tol = 1e-8,
        gradient_mode = :exact_join,
        verbose = false,
    )

    err_bad = try
        approx(
            :not_a_manifold_spec,
            target;
            init = :deterministic,
            solver = :rgd,
            maxiter = 1,
            tol = 1e-8,
            gradient_mode = :exact_join,
            verbose = false,
        )
        nothing
    catch e
        e
    end
    @test err_bad isa ArgumentError
    @test occursin("Unsupported manifolds specification", sprint(showerror, err_bad))

    # Mixed joins share a flattened ambient vector space even when individual
    # manifolds use different natural ambient shapes.
    M_tucker = Manifolds.Tucker(dims_segre, (2, 2, 2))
    M_sphere = Manifolds.Sphere(prod(dims_segre) - 1)
    target_mixed = randn(rng, prod(dims_segre))
    res_mixed = approx(
        (M_tucker, M_sphere),
        target_mixed;
        init = :random,
        solver = :rgd_fixed,
        stepsize = 1e-2,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    @test res_mixed isa ApproxResult
    @test length(res_mixed.components) == 2
    @test eltype(components(res_mixed)) <: TensorKitchen.DecompositionComponent{Float64,1}
    @test size(res_mixed.components[1].tensor) == size(target_mixed)
    @test size(res_mixed.components[2].tensor) == size(target_mixed)
    @test isfinite(res_mixed.cost)

    # cpd(A, r)
    dims = (5, 4, 3)
    A = randn(rng, dims...)
    res_cpd = cpd(
        A,
        2;
        geometry = :canonical,
        solver = :rgd,
        maxiter = 20,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test res_cpd isa CPDResult
    @test isfinite(res_cpd.cost) && isfinite(res_cpd.rel_error)

    # CPD high-level ALS path should work through JoinModel as well.
    res_cpd_als = cpd(
        A,
        2;
        solver = :als,
        maxiter = 2,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )
    @test res_cpd_als isa CPDResult
    @test isfinite(res_cpd_als.cost) && isfinite(res_cpd_als.rel_error)
    @test_throws ArgumentError cpd(
        A,
        2;
        solver = :rals,
        maxiter = 2,
        tol = 1e-6,
        init = TuckerInit(),
        verbose = false,
    )

    tucker_manifolds =
        (Manifolds.Tucker(size(A), (2, 2, 2)), Manifolds.Tucker(size(A), (2, 2, 2)))
    res_tucker_auto = approx(
        tucker_manifolds,
        A;
        solver = :rgd,
        maxiter = 3,
        tol = 1e-6,
        init = :sthosvd,
        verbose = false,
    )
    @test res_tucker_auto isa BTDResult
    @test isfinite(res_tucker_auto.cost)

    res_tucker_auto_default =
        approx(tucker_manifolds, A; maxiter = 3, tol = 1e-6, verbose = false)
    @test res_tucker_auto_default isa BTDResult
    @test res_tucker_auto_default.solver == :rgd

    # btd
    res_btd = btd(
        A,
        2,
        (2, 2, 2);
        solver = :rgd,
        maxiter = 8,
        tol = 1e-6,
        init = :sthosvd,
        verbose = false,
    )
    @test res_btd isa BTDResult && length(res_btd.components) == 2
    @test eltype(components(res_btd)) <: TensorKitchen.DecompositionComponent{Float64,3}

    res_btd_default = btd(A, 2, (2, 2, 2); maxiter = 3, tol = 1e-6, verbose = false)
    @test res_btd_default isa BTDResult
    @test res_btd_default.solver == :rgd

    res_btd_warm_als = btd(
        A,
        2,
        (2, 2, 2);
        solver = :als,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :sthosvd,
        warm_block_method = :hooi,
        warm_block_maxiter = 2,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_warm_als isa BTDResult
    @test isfinite(res_btd_warm_als.rel_error)

    res_btd_warm_rgd = btd(
        A,
        2,
        (2, 2, 2);
        solver = :rgd,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :sthosvd,
        warm_block_method = :hooi,
        warm_block_maxiter = 2,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_warm_rgd isa BTDResult
    @test isfinite(res_btd_warm_rgd.rel_error)

    res_btd_warm_rcg = btd(
        A,
        2,
        (2, 2, 2);
        solver = :rcg,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :sthosvd,
        warm_block_method = :hooi,
        warm_block_maxiter = 2,
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_warm_rcg isa BTDResult
    @test isfinite(res_btd_warm_rcg.rel_error)

    res_btd_warm_lbfgs = btd(
        A,
        2,
        (2, 2, 2);
        solver = :lbfgs,
        init = :alswarm,
        warm_steps = 2,
        warm_init = :sthosvd,
        warm_block_method = :hooi,
        warm_block_maxiter = 2,
        maxiter = 5,
        tol = 1e-6,
        verbose = false,
        memory_size = 5,
    )
    @test res_btd_warm_lbfgs isa BTDResult
    @test res_btd_warm_lbfgs.solver == :lbfgs
    @test isfinite(res_btd_warm_lbfgs.rel_error)

    res_btd_tsd = btd(
        A,
        2,
        (2, 2, 2);
        solver = :btd_tsd,
        init = :hosvd_multistart,
        maxiter = 3,
        stepsize = 1.0,
        schedule = :cyclic,
        block_repeats = 1,
        btd_als_polish_maxiter = 0,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_tsd isa BTDResult
    @test res_btd_tsd.solver == :btd_tsd
    @test res_btd_tsd.solver_info.accepted_steps >= 0
    @test length(res_btd_tsd.solver_info.accepted_stepsize_history) ==
          length(res_btd_tsd.solver_info.line_search_trial_history)
    @test isfinite(res_btd_tsd.rel_error)
    @test res_btd_tsd.rel_error ≈ norm(A - reconstruct(res_btd_tsd)) / norm(A)
    res_btd_tsd_object = btd(
        A,
        2,
        (2, 2, 2);
        solver = BTDTSDSolver(stepsize = 1.0),
        init = :hosvd_multistart,
        maxiter = 1,
        schedule = :cyclic,
        btd_als_polish_maxiter = 0,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_tsd_object isa BTDResult
    @test res_btd_tsd_object.solver == :btd_tsd
    @test_throws ArgumentError btd(
        A,
        2,
        (2, 2, 2);
        solver = :tsd,
        maxiter = 1,
        verbose = false,
    )

    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(size(A), (2, 2, 2), 2))
    backend = TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, A)
    model_btd = JoinModel{Float64,typeof(backend)}(backend)
    M_btd = TensorKitchen.manifold(model_btd)
    p_btd =
        TensorKitchen._solver_point(M_btd, TensorKitchen.initial_point(model_btd, :random))
    parts_btd = TensorKitchen.point_parts(p_btd)
    residual_btd = TensorKitchen._join_residual!(backend, p_btd)
    tangent_dot_btd(a, b) = begin
        s = sum(getproperty(a, :Ċ) .* getproperty(b, :Ċ))
        for (Am, Bm) in zip(getproperty(a, :U̇), getproperty(b, :U̇))
            s += sum(Am .* Bm)
        end
        s
    end
    for b = 1:backend.r
        fast_eg = TensorKitchen._btd_block_egrad(backend, parts_btd, b)
        residual_eg =
            TensorKitchen._tucker_egrad(backend.manifolds[b], parts_btd[b], residual_btd)
        @test norm(getproperty(fast_eg, :Ċ) - getproperty(residual_eg, :Ċ)) < 1e-10
        @test all(
            norm(F - R) < 1e-10 for
            (F, R) in zip(getproperty(fast_eg, :U̇), getproperty(residual_eg, :U̇))
        )

        _, block_grad, _ = TensorKitchen._btd_block_descent_direction(backend, p_btd, b)
        h = 1e-6
        q_btd = TensorKitchen._replace_block_part(
            p_btd,
            b,
            retract(backend.manifolds[b], parts_btd[b], (-h) * block_grad),
        )
        fd =
            (TensorKitchen.cost(model_btd, q_btd) - TensorKitchen.cost(model_btd, p_btd)) /
            h
        theory = -tangent_dot_btd(fast_eg, block_grad)
        @test fd ≈ theory rtol = 1e-4 atol = 1e-4
    end
    @test backend.workspace.tensor_slot1 isa TensorKitchen._WorkspaceTensorCache{Float64,3}
    @test backend.workspace.tensor_slot2 isa TensorKitchen._WorkspaceTensorCache{Float64,3}
    @test backend.workspace.perm_in isa TensorKitchen._WorkspaceTensorCache{Float64,3}
    @test backend.workspace.perm_out isa TensorKitchen._WorkspaceTensorCache{Float64,3}
    @test backend.workspace.persist isa TensorKitchen._WorkspaceTensorCache{Float64,3}
    model = TensorKitchen.JoinModel{Float64,typeof(backend)}(backend)
    Random.seed!(2028)
    p_base_btd = TensorKitchen.initial_point(model, :random)
    p_warm_btd = TensorKitchen.initial_point(
        model,
        BTDALSWarmStartInit(
            2;
            base_init = PointInit(p_base_btd),
            block_method = :hooi,
            block_maxiter = 2,
        ),
    )
    @test TensorKitchen.cost(model, p_warm_btd) <=
          TensorKitchen.cost(model, p_base_btd) + 1e-8
    p0 = TensorKitchen.initial_point(model, :sthosvd)
    comps = TensorKitchen.extract_components(model, p0)
    Xhat = zero(A)
    for c in comps
        Xhat .+= c.tensor
    end
    @test isapprox(
        TensorKitchen.cost(model, p0),
        0.5 * sum(abs2, A .- Xhat);
        atol = 1e-10,
        rtol = 1e-10,
    )

    p_ms = TensorKitchen.initial_point(
        model,
        BTDHOSVDMultistartInit(3; screening_steps = 0, include_sequential = true),
    )
    @test TensorKitchen.cost(model, p_ms) <= TensorKitchen.cost(model, p0) + 1e-8
    p_ms_sym = TensorKitchen.initial_point(model, :hosvd_multistart)
    @test isfinite(TensorKitchen.cost(model, p_ms_sym))
    @test_throws ArgumentError TensorKitchen.initial_point(model, :hosvd)
    @test_throws ArgumentError TensorKitchen.initial_point(model, :thosvd)
    @test_throws ArgumentError BTDHOSVDMultistartInit(0)

    eg = TensorKitchen.egrad(model, p0)
    rg = TensorKitchen.rgrad(model, p0)
    M = TensorKitchen.manifold(model)
    rg_from_eg = TensorKitchen.egrad_to_rgrad(M, p0, eg)

    function _tucker_tangent_distance(x, y)
        dc = norm(getproperty(x, :Ċ) .- getproperty(y, :Ċ))
        xf = getproperty(x, :U̇)
        yf = getproperty(y, :U̇)
        df = zero(dc)
        for k = 1:length(xf)
            df += norm(xf[k] .- yf[k])
        end
        return dc + df
    end

    rg_parts = TensorKitchen.point_parts(rg)
    rg_ref_parts = TensorKitchen.point_parts(rg_from_eg)
    @test sum(
        _tucker_tangent_distance(rg_parts[k], rg_ref_parts[k]) for k = 1:length(rg_parts)
    ) ≤ 1e-8
end

@testset "gradient interface: grad/egrad_to_rgrad" begin
    rng = MersenneTwister(2027)
    dims = (6, 5, 4)
    A = randn(rng, dims...)

    # Built-in Segre path (upstream-only): no project(M,p,⋅) for structured Segre tangent.
    model = TensorKitchen.Rank1CPDModel(A)
    M = TensorKitchen.manifold(model)
    p = TensorKitchen.initial_point(model, :random)
    eg = TensorKitchen.egrad(model, p)
    g_model = grad(model, p)
    g_api = grad(M, p, eg)
    g_e2r = egrad_to_rgrad(M, p, eg)
    @test_throws MethodError ManifoldsBase.project(M, p, eg)

    gλ_model, gU_model = unpack_point_rank1(g_model, dims)
    gλ_api, gU_api = unpack_point_rank1(g_api, dims)
    gλ_e2r, gU_e2r = unpack_point_rank1(g_e2r, dims)
    λp, Up = unpack_point_rank1(p, dims)
    gλ_eg, gU_eg = unpack_point_rank1(eg, dims)
    gU_ref = [gU_eg[m] .- sum(Up[m] .* gU_eg[m]) .* Up[m] for m = 1:length(dims)]

    @test gλ_model ≈ gλ_eg
    @test gλ_api ≈ gλ_eg
    @test gλ_e2r ≈ gλ_eg
    for m = 1:length(dims)
        @test gU_model[m] ≈ gU_ref[m]
        @test gU_api[m] ≈ gU_ref[m]
        @test gU_e2r[m] ≈ gU_ref[m]
    end
    g_dir = rgrad(model, p)
    gλ_dir, gU_dir = unpack_point_rank1(g_dir, dims)
    @test gλ_dir ≈ gλ_eg
    for m = 1:length(dims)
        @test gU_dir[m] ≈ gU_ref[m]
    end
    @test supports_rgrad(model)

    # Rank-r canonical direct rgrad path.
    model_c = TensorKitchen.RankRCPDModel(A, 2; geometry = :canonical)
    p_c = TensorKitchen.initial_point(model_c, :random)
    g_c_proj = grad(model_c, p_c)
    g_c_dir = rgrad(model_c, p_c)
    gλ_c_proj, gU_c_proj = TensorKitchen.unpack_rankr_canonical(g_c_proj, dims, 2)
    gλ_c_dir, gU_c_dir = TensorKitchen.unpack_rankr_canonical(g_c_dir, dims, 2)
    @test gλ_c_dir ≈ gλ_c_proj
    for m = 1:length(dims)
        @test gU_c_dir[m] ≈ gU_c_proj[m]
    end
    @test supports_rgrad(model_c)

    # Pullback metric path: grad = G(p)^{-1} * egrad
    r = 2
    M_pb = SqEuclidean(r * (1 + sum(dims)))
    n_params = r * (1 + sum(dims))
    p_pb = randn(rng, n_params) .+ 0.5
    eg_pb = randn(rng, n_params)
    g_pb = grad(M_pb, p_pb, eg_pb)
    g_pb_ref = pullback_metric_inverse(M_pb, p_pb, eg_pb)
    @test g_pb ≈ g_pb_ref
    @test_throws ArgumentError egrad_to_rgrad(M_pb, (p_pb,), eg_pb)

    # Segre ambient-vector path: direct ambient vectors are rejected; join-style
    # code must first convert them to the native (λ, u₁, …, u_d) Euclidean
    # gradient representation.
    M_seg = Manifolds.Segre(dims)
    p_seg = TensorKitchen.pack_point_rank1_segre(
        1.0,
        [
            begin
                v = randn(rng, dims[m])
                v ./= norm(v)
                v
            end for m = 1:length(dims)
        ],
    )
    Φ = randn(rng, prod(dims))
    @test_throws ArgumentError egrad_to_rgrad(M_seg, p_seg, Φ)
    eg_seg = TensorKitchen._manifold_egrad(M_seg, p_seg, Φ)
    g_seg = egrad_to_rgrad(M_seg, p_seg, eg_seg)
    @test isnothing(ManifoldsBase.check_vector(M_seg, p_seg, g_seg))
    @test length(TensorKitchen.point_parts(eg_seg)) == length(dims) + 1

    # Generic manifolds without a specialized projection path should fail
    # instead of silently using the embedded Euclidean basis helper.
    struct _NoProjectManifold <: AbstractManifold{ManifoldsBase.ℝ} end
    ManifoldsBase.manifold_dimension(::_NoProjectManifold) = 1
    M_noproj = _NoProjectManifold()
    @test_throws ArgumentError egrad_to_rgrad(M_noproj, [1.0], [2.0])
    @test_throws ArgumentError egrad_to_rgrad!(M_noproj, [0.0], [1.0], [2.0])

    # If project is applicable, internal MethodError should still propagate.
    struct _ThrowingProjectManifold <: AbstractManifold{ManifoldsBase.ℝ} end
    ManifoldsBase.manifold_dimension(::_ThrowingProjectManifold) = 1
    ManifoldsBase.project(::_ThrowingProjectManifold, p_local, x_local) =
        throw(MethodError(:internal_project_bug, (p_local, x_local)))
    M_throw = _ThrowingProjectManifold()
    @test_throws MethodError egrad_to_rgrad(M_throw, [1.0], [2.0])
    @test_throws MethodError egrad_to_rgrad!(M_throw, [0.0], [1.0], [2.0])

    # Function lifting: (M, p) -> egrad  ->  (M, p) -> grad
    egrad_fn = (M_local, p_local) -> 2 .* p_local
    grad_fn = grad(egrad_fn)
    @test grad_fn(M_pb, p_pb) ≈ pullback_metric_inverse(M_pb, p_pb, 2 .* p_pb)

    # Solver-level mode switch.
    out_proj = solve(
        RGDSolver(),
        model_c;
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
        gradient_mode = :egrad_project,
    )
    out_riem = solve(
        RGDSolver(),
        model_c;
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
        gradient_mode = :riemannian,
    )
    @test isfinite(out_proj.cost)
    @test isfinite(out_riem.cost)

    # geometry=:join removed from pipeline
    @test_throws ArgumentError TensorKitchen.RankRCPDModel(A, 2; geometry = :join)

    # Upstream-only mode does not provide :exact_native gradient mode.
    @test_throws ArgumentError solve(
        RGDSolver(),
        model_c;
        maxiter = 1,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
        gradient_mode = :exact_native,
    )
end

@testset "public example smoke tests" begin
    rng = MersenneTwister(4242)
    A = randn(rng, 8, 6, 5)
    r = 3
    ranks = (4, 3, 2)

    # CPD example
    res_cpd_example = cpd(
        A,
        r;
        init = TuckerInit(),
        solver = :rgd,
        maxiter = 20,
        tol = 1e-6,
        verbose = false,
    )
    @test res_cpd_example isa CPDResult
    @test size(reconstruct(res_cpd_example)) == size(A)
    @test isfinite(res_cpd_example.rel_error)

    # Tucker methods example coverage
    td_st = tucker(A, ranks; method = :sthosvd)
    td_ho = tucker(A, ranks; method = :hooi, maxiter = 10, tol = 1e-6, verbose = false)
    @test td_st isa TuckerResult
    @test td_ho isa TuckerResult
    @test size(reconstruct(td_st)) == size(A)
    @test size(reconstruct(td_ho)) == size(A)
    @test_throws ArgumentError tucker(A, ranks; method = :thosvd)
    @test_throws ArgumentError tucker(A, ranks; method = :hosvd)

    # Join example
    target = [1.2, 0.4]
    res_join_example = approx(
        Manifolds.Sphere(1),
        2,
        target;
        solver = :rgd,
        init = :deterministic,
        maxiter = 80,
        tol = 1e-8,
        verbose = false,
    )
    @test res_join_example isa ApproxResult
    @test isfinite(res_join_example.cost)

    # JoinModel + solve example
    model_example = JoinModel((Manifolds.Sphere(1), Manifolds.Sphere(1)), target)
    out_example = solve(
        RGDSolver(1.0),
        model_example;
        gradient_mode = :riemannian,
        init = :deterministic,
        maxiter = 120,
        tol = 1e-8,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(out_example.cost)
    @test isfinite(out_example.rel_error)

    # Native CPD exact-native example
    λ_ex = [1.2, 0.8]
    U_ex = [randn(rng, 4, 2), randn(rng, 4, 2), randn(rng, 3, 2)]
    for m = 1:3, k = 1:2
        U_ex[m][:, k] ./= norm(U_ex[m][:, k])
    end
    A_native = reconstruct_cpd_rankr(λ_ex, U_ex)
    res_native_example = cpd(
        A_native,
        2;
        solver = :rgd,
        geometry = :native,
        gradient_mode = :exact_native,
        init = TuckerInit(),
        maxiter = 20,
        tol = 1e-6,
        verbose = false,
    )
    @test res_native_example isa CPDResult
    @test isfinite(res_native_example.rel_error)

    # Canonical gradient mode examples
    res_eproj = cpd(
        A,
        r;
        solver = :rgd,
        geometry = :canonical,
        gradient_mode = :egrad_project,
        init = TuckerInit(),
        maxiter = 10,
        tol = 1e-6,
        verbose = false,
    )
    res_riem = cpd(
        A,
        r;
        solver = :rgd,
        geometry = :canonical,
        gradient_mode = :riemannian,
        init = TuckerInit(),
        maxiter = 10,
        tol = 1e-6,
        verbose = false,
    )
    @test isfinite(res_eproj.rel_error)
    @test isfinite(res_riem.rel_error)

    # Uniform manifold routing examples
    A_small = randn(rng, 3, 4, 5)
    segres = (Manifolds.Segre((3, 4, 5)), Manifolds.Segre((3, 4, 5)))
    tuckers =
        (Manifolds.Tucker((3, 4, 5), (2, 2, 2)), Manifolds.Tucker((3, 4, 5), (2, 2, 2)))
    res_route_cpd = approx(
        segres,
        A_small;
        solver = :rgd,
        init = TuckerInit(),
        maxiter = 3,
        tol = 1e-6,
        verbose = false,
    )
    res_route_btd = approx(tuckers, A_small; maxiter = 3, tol = 1e-6, verbose = false)
    @test res_route_cpd isa CPDResult
    @test res_route_btd isa BTDResult

    # BTD example
    res_btd_example = btd(A, 2, (3, 2, 2); maxiter = 5, tol = 1e-6, verbose = false)
    @test res_btd_example isa BTDResult
    @test size(reconstruct(res_btd_example)) == size(A)
    @test isfinite(res_btd_example.rel_error)

    A_zero_btd = zeros(Float64, 6, 5, 4)
    res_btd_zero = btd(
        A_zero_btd,
        2,
        (2, 2, 2);
        solver = :als,
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
    @test res_btd_zero isa BTDResult
    @test isfinite(res_btd_zero.rel_error)
    @test res_btd_zero.rel_error ≥ 0
end

@testset "README workflow examples stay executable" begin
    rng = MersenneTwister(5151)
    A = randn(rng, 8, 6, 5)
    r = 3

    res = cpd(A, r; init = TuckerInit(), maxiter = 5, verbose = false)
    @test res isa CPDResult
    λ = weights(res)
    U = factors(res)
    @test length(λ) == r
    @test length(U) == ndims(A)
    Â = reconstruct(res)
    @test size(Â) == size(A)
    @test cpd(A, r; solver = :als, maxiter = 3, verbose = false) isa CPDResult

    mlrank = (4, 3, 2)
    tucker_res = tucker(A, mlrank)
    @test tucker_res isa TuckerResult
    @test size(core(tucker_res)) == mlrank
    @test length(factors(tucker_res)) == ndims(A)

    B = abs.(A)
    nn_res = nncpd(B, r; solver = :als, maxiter = 3, verbose = false)
    @test nn_res isa CPDResult
    @test length(weights(nn_res)) == r
    @test length(factors(nn_res)) == ndims(B)
    @test all(w -> w >= -1e-12, weights(nn_res))
    @test all(F -> all(x -> x >= -1e-12, F), factors(nn_res))
    @test cpd(B, r; nonnegative = true, solver = :als, maxiter = 3, verbose = false) isa
          CPDResult

    block_count = 2
    block_rank = (3, 2, 2)
    btd_res = btd(
        A,
        block_count,
        block_rank;
        solver = :als,
        init = BTDHOSVDMultistartInit(2; screening_steps = 0, block_maxiter = 1),
        maxiter = 2,
        block_maxiter = 1,
        verbose = false,
    )
    @test btd_res isa BTDResult
    btd_blocks = blocks(btd_res)
    @test length(btd_blocks) == block_count
    blk = btd_blocks[1]
    @test size(core(blk)) == block_rank
    @test length(factors(blk)) == ndims(A)

    p = [1.2, 0.4]
    S = Manifolds.Sphere(1)
    join_res = approx(
        (S, S),
        p;
        init = :deterministic,
        solver = :rgd,
        maxiter = 60,
        verbose = false,
    )
    @test join_res isa ApproxResult
    @test length(components(join_res)) == 2
    @test size(reconstruct(join_res)) == size(p)
end

@testset "result conversion consistency for CPD, NNCPD, and BTD" begin
    rng = MersenneTwister(6262)
    A = randn(rng, 6, 5, 4)
    r = 2

    res_cpd = cpd(A, r; solver = :als, init = TuckerInit(), maxiter = 4, verbose = false)
    Ahat_cpd = reconstruct(res_cpd)
    @test Ahat_cpd ≈ reconstruct_cpd_rankr(weights(res_cpd), factors(res_cpd))
    @test rel_error(A, res_cpd) ≈ rel_error(A, Ahat_cpd)
    @test length(components(res_cpd)) == r
    @test factors(res_cpd) == TensorKitchen.factors_from_components(components(res_cpd))

    B = abs.(A)
    res_nn = nncpd(B, r; solver = :als, init = TuckerInit(), maxiter = 4, verbose = false)
    Ahat_nn = reconstruct(res_nn)
    @test Ahat_nn ≈ reconstruct_cpd_rankr(weights(res_nn), factors(res_nn))
    @test rel_error(B, res_nn) ≈ rel_error(B, Ahat_nn)
    @test length(components(res_nn)) == r
    @test all(w -> w >= -1e-12, weights(res_nn))
    @test all(F -> all(x -> x >= -1e-12, F), factors(res_nn))

    res_btd = btd(
        A,
        2,
        (2, 2, 2);
        solver = :als,
        init = BTDHOSVDMultistartInit(2; screening_steps = 0, block_maxiter = 1),
        maxiter = 3,
        block_maxiter = 1,
        verbose = false,
    )
    Ahat_btd = reconstruct(res_btd)
    block_sum = zero(A)
    for blk in blocks(res_btd)
        @test size(core(blk)) == (2, 2, 2)
        @test length(factors(blk)) == ndims(A)
        block_sum .+= tensor(blk)
    end
    @test Ahat_btd ≈ block_sum
    @test rel_error(A, res_btd) ≈ rel_error(A, Ahat_btd)
end

# =========================================================================
# utils: pack_unpack, cp_init_tucker, tensor_contractions
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
