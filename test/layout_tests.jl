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
    segre_native = TensorKitchen.initial_point(segre_model, :random)
    segre_solver = TensorKitchen._solver_point(manifold(segre_model), segre_native)
    @test typeof(segre_solver.x[1]) === typeof(segre_native.x[1])
    @test is_point(manifold(segre_model), segre_solver)

    T = Manifolds.Tucker((3, 3), (1, 1))
    tucker_model = JoinModel(T, 2, zeros(3, 3))
    tucker_native = TensorKitchen.initial_point(tucker_model, :random)
    tucker_solver = TensorKitchen._solver_point(manifold(tucker_model), tucker_native)
    @test tucker_solver.x[1] isa Manifolds.TuckerPoint
    @test is_point(manifold(tucker_model), tucker_solver)
end
