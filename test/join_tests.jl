@testset "Join container boundary and point-aware residual" begin
    rng = MersenneTwister(118)

    # Only ProductManifold levels are adapted. Native component points remain
    # opaque even when their representation itself is a tuple.
    component = SymmetricRankOne(3, 2)
    V = manifold(component)
    x = normalize([1.0, 2.0, -1.0])
    y = normalize([-1.0, 0.5, 2.0])
    p1, p2 = ([1.2], x), ([-0.7], y)
    P1 = ProductManifold(V)
    p_single = TensorKitchen._solver_point(P1, (p1,))
    @test p_single isa ArrayPartition
    @test p_single.x == (p1,)
    @test is_point(P1, p_single)
    @test TensorKitchen._scale_solver_tangent(zero_vector(P1, p_single), 0.5).x[1] isa Tuple

    P = ProductManifold(V, V)
    p = TensorKitchen._solver_point(P, (p1, p2))
    @test p isa ArrayPartition
    @test TensorKitchen.join_parts(P, p) == (p1, p2)
    @test p.x[1] isa Tuple
    @test p.x[2] isa Tuple
    @test is_point(P, p)
    @test zero_vector(P, p).x[1] isa Tuple

    A =
        TensorKitchen._symcpd_embed_coordinates(V, p1) +
        TensorKitchen._symcpd_embed_coordinates(V, p2)
    model = JoinModel(component, 2, A)
    @test cost(model, p) ≈ 0 atol = 1.0e-14

    q1 = ([0.8], normalize([1.0, 1.0, 0.2]))
    q = TensorKitchen._solver_point(P, (q1, p2))
    @test cost(model, q) ≈
          0.5 * sum(
        abs2,
        TensorKitchen._symcpd_embed_coordinates(V, q1) +
        TensorKitchen._symcpd_embed_coordinates(V, p2) - A,
    )

    # Every ordering must evaluate at the requested point, even if an optimizer
    # mutates a previously cached point object before a line-search cost.
    TensorKitchen.rgrad(model, p)
    @test cost(model, q) ≈
          0.5 * sum(
        abs2,
        TensorKitchen._symcpd_embed_coordinates(V, q1) +
        TensorKitchen._symcpd_embed_coordinates(V, p2) - A,
    )
    TensorKitchen.rgrad(model, q)
    @test cost(model, p) ≈ 0 atol = 1.0e-14
    TensorKitchen.rgrad(model, p)
    p.x[1][1][1] += 0.1
    @test cost(model, p) ≈
          0.5 * sum(
        abs2,
        TensorKitchen._symcpd_embed_coordinates(V, p.x[1]) +
        TensorKitchen._symcpd_embed_coordinates(V, p.x[2]) - A,
    )

    # The same outer adaptation also preserves Segre and Tucker native points.
    S = Manifolds.Segre((3, 3))
    segre_model = JoinModel(S, 2, zeros(3, 3))
    segre_native = TensorKitchen.initial_point(segre_model, :deterministic)
    segre_solver = TensorKitchen._solver_point(manifold(segre_model), segre_native)
    @test typeof(segre_solver.x[1]) === typeof(segre_native.x[1])
    @test is_point(manifold(segre_model), segre_solver)

    T = Manifolds.Tucker((3, 3), (1, 1))
    tucker_model = JoinModel(T, 2, zeros(3, 3))
    tucker_component = Manifolds.TuckerPoint(
        reshape([1.0], 1, 1),
        reshape([1.0, 0.0, 0.0], 3, 1),
        reshape([0.0, 1.0, 0.0], 3, 1),
    )
    tucker_native = TensorKitchen.join_point(
        manifold(tucker_model),
        (deepcopy(tucker_component), deepcopy(tucker_component)),
    )
    tucker_solver = TensorKitchen._solver_point(manifold(tucker_model), tucker_native)
    @test tucker_solver.x[1] isa Manifolds.TuckerPoint
    @test is_point(manifold(tucker_model), tucker_solver)
end
@testset "JoinModel: generic (Segre, Segre, ...) backend" begin
    dims = (5, 4, 3)
    rng = MersenneTwister(4242)
    A = randn(rng, dims...)
    segres = (Manifolds.Segre(dims), Manifolds.Segre(dims), Manifolds.Segre(dims))

    model = JoinModel(segres, A)
    @test model isa JoinModel
    @test model.backend isa TensorKitchen.JoinBackend
    @test model.backend.r == 3
    @test tensor(model) == A
    @test length(model.backend.components) == 3
    @test map(TensorKitchen.manifold, model.backend.components) == segres

    M = TensorKitchen.manifold(model)
    @test M isa ProductManifold
    @test length(M.manifolds) == 3
    @test all(Mk -> Mk isa Manifolds.Segre && factor_dims(Mk) == dims, M.manifolds)

    model_repeat = JoinModel(Manifolds.Segre(dims), 3, A)
    @test model_repeat.backend.r == 3
    @test map(TensorKitchen.manifold, model_repeat.backend.components) == segres

    model_component =
        JoinModel((TensorKitchen.JoinComponent(segres[1]), segres[2], segres[3]), A)
    @test model_component.backend.components[1] isa TensorKitchen.JoinComponent
    @test map(TensorKitchen.manifold, model_component.backend.components) == segres
    @test TensorKitchen.component_manifold(model_component.backend.components[1]) ==
          segres[1]
    @test TensorKitchen.component_embedding(model_component.backend.components[1]) isa
          TensorKitchen.DefaultJoinEmbedding

    point_parts = ntuple(3) do _
        factors = [normalize(randn(rng, d)) for d in dims]
        TensorKitchen.pack_point_rank1_segre(1.0, factors)
    end
    p = TensorKitchen.join_point(M, point_parts)
    @test length(TensorKitchen.point_parts(p)) == 3

    f = cost(model, p)
    g = TensorKitchen.egrad(model, p)
    rg = rgrad(model, p)
    @test isfinite(f) && f >= 0
    @test isfinite(norm(M, p, g))
    @test isfinite(norm(M, p, rg))

    rec = similar(model.backend.work_rec)
    TensorKitchen._join_reconstruct!(rec, model.backend, p)
    @test f ≈ 0.5 * sum(abs2, rec .- vec(A))

    comps = TensorKitchen.extract_components(model, p)
    @test length(comps) == 3
    @test all(c -> c.manifold isa Manifolds.Segre, comps)
    @test all(c -> size(c.tensor) == dims, comps)

    initial_normalized_cost = f / sum(abs2, A)
    p_before = deepcopy(p)
    out = solve(
        RGDSolver(1.0e-2),
        model;
        p0 = p,
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
        return_stats = true,
    )
    @test isfinite(out.cost) && isfinite(out.rel_error)
    @test out.cost <= initial_normalized_cost
    @test TensorKitchen._join_cache_point_equal(p, p_before)
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
