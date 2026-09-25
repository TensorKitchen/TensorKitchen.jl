# Small full-tensor oracle used only for numerical validation.
function _symcpd_full_term(λ, x, d)
    A = Array{typeof(λ)}(undef, ntuple(_ -> length(x), d))
    for I in CartesianIndices(A)
        A[I] = λ * prod(x[i] for i in Tuple(I))
    end
    return A
end

_symcpd_coordinates(M, p) = TensorKitchen._symcpd_embed_coordinates(M, p)

@testset "SymCPD compressed representation" begin
    indices = symmetric_multiindices(3, 4)
    @test length(indices) == binomial(6, 4)
    @test first(indices) == [4, 0, 0]
    @test last(indices) == [0, 0, 4]
    @test all(alpha -> sum(alpha) == 4, indices)
    @test length(unique(indices)) == length(indices)
    @test sum(multinomial_multiplicity, indices) == 3^4
    @test multinomial_multiplicity([2, 1, 1]) == 12
    @test_throws ArgumentError symmetric_multiindices(0, 3)
    @test_throws ArgumentError multinomial_multiplicity([2, -1, 2])
    @test_throws ArgumentError SymmetricRankOne(Manifolds.Sphere(2))

    x = normalize([1.0, -2.0, 0.5])
    y = normalize([0.4, 1.0, -1.0])
    for d in (2, 3, 4)
        V = TensorKitchen._symcpd_manifold(3, d)
        p = ([-1.3], x)
        q = ([0.7], y)
        a = _symcpd_coordinates(V, p)
        A = _symcpd_full_term(p[1][1], x, d)
        @test length(a) == binomial(3 + d - 1, d)
        @test compress_symmetric_tensor(A) ≈ a atol = 1e-12
        @test expand_symmetric_tensor(a, 3, d) ≈ A atol = 1e-12
        @test dot(a, _symcpd_coordinates(V, q)) ≈ sum(A .* _symcpd_full_term(q[1][1], y, d)) atol =
            1e-12
        representative = ([(-1)^d * p[1][1]], -x)
        @test _symcpd_coordinates(V, representative) ≈ a atol = 1e-12
    end

    A = _symcpd_full_term(1.4, x, 3) + _symcpd_full_term(-0.6, y, 3)
    A[1, 2, 3] += 0.2
    a = compress_symmetric_tensor(A)
    S = expand_symmetric_tensor(a, 3, 3)
    @test S[1, 2, 3] ≈
          (A[1, 2, 3] + A[1, 3, 2] + A[2, 1, 3] + A[2, 3, 1] + A[3, 1, 2] + A[3, 2, 1]) / 6
    @test compress_symmetric_tensor(S) ≈ a
    @test_throws ArgumentError symcpd(A, 1; maxiter = 0, verbose = false)

    # A full tensor here would have 2^24 entries. The compressed target has 25
    # entries, while the model side uses only factors and scalar kernels.
    Vlarge = TensorKitchen._symcpd_manifold(2, 24)
    plarge = ([1.1], normalize([0.6, 0.8]))
    target_large = _symcpd_coordinates(Vlarge, plarge)
    component_large = SymmetricRankOne(2, 24)
    model_large = JoinModel(component_large, 1, target_large)
    @test component_large isa AbstractJoinComponent
    @test manifold(component_large) === component_large.manifold
    @test model_large isa JoinModel
    @test model_large.backend isa SymmetricCPDBackend
    outer_large = TensorKitchen.join_point(manifold(model_large), (plarge,))
    cost(model_large, outer_large)
    TensorKitchen.rgrad(model_large, outer_large)
    @test @allocated(cost(model_large, outer_large)) < 16_000_000
    @test @allocated(TensorKitchen.rgrad(model_large, outer_large)) < 1_000_000
end

@testset "SymCPD target operator interface" begin
    x = normalize([1.0, -2.0, 0.5])
    y = normalize([0.4, 1.0, -1.0])
    d = 3
    A = _symcpd_full_term(1.4, x, d) + _symcpd_full_term(-0.6, y, d)
    dense = DenseSymmetricTarget(A)
    compressed = CompressedSymmetricTarget(compress_symmetric_tensor(A), 3, d)
    functional = FunctionalSymmetricTarget(
        3,
        d,
        target_norm2(dense);
        evaluate = z -> evaluate(dense, z),
        contract = z -> contract(dense, z),
    )

    z = normalize([0.3, -0.7, 0.2])
    @test target_norm2(dense) ≈ target_norm2(compressed)
    @test target_norm2(dense) ≈ target_norm2(functional)
    @test evaluate(dense, z) ≈ evaluate(compressed, z) atol = 2e-15
    @test evaluate(dense, z) ≈ evaluate(functional, z) atol = 2e-15
    @test contract(dense, z) ≈ contract(compressed, z) atol = 2e-15
    @test contract(dense, z) ≈ contract(functional, z) atol = 2e-15
    @test_throws DimensionMismatch evaluate(functional, ones(2))
    @test_throws ArgumentError FunctionalSymmetricTarget(
        3,
        d,
        -1.0;
        evaluate = _ -> 0.0,
        contract = _ -> zeros(3),
    )

    p0 = (([1.2], normalize(x + [0.02, -0.01, 0.03])),)
    for target in (dense, compressed, functional)
        model = SymCPDModel(target, 1)
        initial_cost = cost(model, TensorKitchen.join_point(manifold(model), deepcopy(p0)))
        result = symcpd(
            target,
            1;
            p0 = deepcopy(p0),
            solver = :gn_cg,
            maxiter = 20,
            tol = 1e-10,
            verbose = false,
        )
        @test result isa SymCPDResult
        @test isfinite(cost(result))
        @test cost(result) < initial_cost
    end

    first_order = symcpd(
        functional,
        1;
        p0 = deepcopy(p0),
        solver = :lbfgs,
        maxiter = 20,
        verbose = false,
    )
    @test isfinite(cost(first_order))
end

@testset "SymCPD matrix-free cost and intrinsic gradient" begin
    x = normalize([1.0, -2.0, 0.5])
    y = normalize([0.4, 1.0, -1.0])
    d = 3
    V = TensorKitchen._symcpd_manifold(3, d)
    A = _symcpd_full_term(1.4, x, d) + _symcpd_full_term(-0.6, y, d)
    component = SymmetricRankOne(3, d)
    model = JoinModel(component, 2, DenseSymmetricTarget(A))
    @test SymCPDModel(DenseSymmetricTarget(A), 2) isa JoinModel
    @test JoinModel(component, DenseSymmetricTarget(A)).backend.rank == 1
    @test kind(component) == :Veronese
    M = manifold(model)
    p = TensorKitchen.join_point(M, (([1.2], x), ([-0.4], y)))
    residual_full = _symcpd_full_term(1.2, x, d) + _symcpd_full_term(-0.4, y, d) - A
    @test cost(model, p) ≈ 0.5 * sum(abs2, residual_full) atol = 1e-12

    u = [0.3, 0.1, -0.2]
    u .-= dot(u, x) .* x
    v = [-0.1, 0.2, 0.15]
    v .-= dot(v, y) .* y
    X = TensorKitchen.join_point(M, (([0.2], u), ([-0.15], v)))
    g = TensorKitchen.rgrad(model, p)
    h = 1e-6
    fd =
        (
            cost(model, retract(M, p, TensorKitchen._scale_solver_tangent(X, h))) -
            cost(model, retract(M, p, TensorKitchen._scale_solver_tangent(X, -h)))
        ) / (2h)
    @test fd ≈ inner(M, p, g, X) atol = 1e-7 rtol = 1e-6
    tangent_1 = zeros(binomial(3 + d - 1, d))
    tangent_2 = similar(tangent_1)
    TensorKitchen._symcpd_embed_coordinates!(tangent_1, V, p.x[1], X.x[1])
    TensorKitchen._symcpd_embed_coordinates!(tangent_2, V, p.x[2], X.x[2])
    @test inner(M, p, X, X) ≈ sum(abs2, tangent_1) + sum(abs2, tangent_2) atol = 1e-12
    @test all(c -> c isa SymCPDComponent, TensorKitchen.extract_components(model, p))
end

@testset "SymCPD analytic normal operator" begin
    x = normalize([1.0, -2.0, 0.5])
    y = normalize([0.4, 1.0, -1.0])
    d = 3
    A = _symcpd_full_term(1.4, x, d) + _symcpd_full_term(-0.6, y, d)
    component = SymmetricRankOne(3, d)
    dense_model = JoinModel(component, 2, DenseSymmetricTarget(A))
    compressed_model = JoinModel(
        component,
        2,
        CompressedSymmetricTarget(compress_symmetric_tensor(A), 3, d),
    )
    M = manifold(dense_model)
    p = TensorKitchen.join_point(M, (([1.2], x), ([-0.4], y)))
    u = [0.3, 0.1, -0.2]
    u .-= dot(u, x) .* x
    v = [-0.1, 0.2, 0.15]
    v .-= dot(v, y) .* y
    X = TensorKitchen.join_point(M, (([0.2], u), ([-0.15], v)))

    # The explicit compressed-coordinate JVP is the derivative of the join
    # embedding along the manifold retraction.
    h = 1e-6
    p_plus = retract(M, p, TensorKitchen._scale_solver_tangent(X, h))
    p_minus = retract(M, p, TensorKitchen._scale_solver_tangent(X, -h))
    embedded_plus = zeros(TensorKitchen.ambient_length(component))
    embedded_minus = similar(embedded_plus)
    for part in TensorKitchen.join_parts(M, p_plus)
        embedded_plus .+= TensorKitchen._symcpd_embed_coordinates(component.manifold, part)
    end
    for part in TensorKitchen.join_parts(M, p_minus)
        embedded_minus .+= TensorKitchen._symcpd_embed_coordinates(component.manifold, part)
    end
    jvp = differential_action(dense_model, p, X)
    @test jvp ≈ (embedded_plus - embedded_minus) / (2h) atol = 2e-10 rtol = 2e-10

    # JVP and VJP are adjoints in the induced Riemannian metric.
    ambient_covector = collect(range(-0.4, 0.7; length = length(jvp)))
    vjp = pullback(dense_model, p, ambient_covector)
    @test dot(ambient_covector, jvp) ≈ inner(M, p, vjp, X) atol = 2e-13 rtol = 2e-13

    # Dense and compressed storage change only target evaluation.
    @test cost(dense_model, p) ≈ cost(compressed_model, p) atol = 2e-15
    @test norm(
        M,
        p,
        TensorKitchen.rgrad(dense_model, p) - TensorKitchen.rgrad(compressed_model, p),
    ) < 2e-14

    # The analytic kernel action equals the explicit J'J reference at machine
    # precision. The production path does not call pushforward or pullback.
    Y_operator = normal_operator(dense_model, p, X)
    Y_reference = pullback(dense_model, p, differential_action(dense_model, p, X))
    @test norm(M, p, Y_operator - Y_reference) < 2e-14

    H_operator = dense_normal_matrix(dense_model, p)
    H_reference = dense_normal_matrix(dense_model, p; reference = true)
    @test H_operator ≈ H_reference atol = 3e-14 rtol = 3e-14
    @test H_operator ≈ transpose(H_operator) atol = 3e-14 rtol = 3e-14
    X_coordinates = get_coordinates(M, p, X, ManifoldsBase.DefaultOrthonormalBasis())
    NX_coordinates =
        get_coordinates(M, p, Y_operator, ManifoldsBase.DefaultOrthonormalBasis())
    @test H_operator * X_coordinates ≈ NX_coordinates atol = 3e-14 rtol = 3e-14

    # The intrinsic normal action is self-adjoint in the Riemannian metric.
    rng = MersenneTwister(912)
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    tangent_dim = manifold_dimension(M)
    for _ = 1:5
        X_random = get_vector(M, p, randn(rng, tangent_dim), basis)
        Y_random = get_vector(M, p, randn(rng, tangent_dim), basis)
        NX = normal_operator(dense_model, p, X_random)
        NY = normal_operator(dense_model, p, Y_random)
        @test inner(M, p, X_random, NY) ≈ inner(M, p, NX, Y_random) atol = 5e-13 rtol =
            5e-13
    end

    # CG and a dense solve agree for the same single damped GN system.
    damping = 1e-3
    gradient = TensorKitchen.rgrad(dense_model, p)
    rhs = TensorKitchen._scale_solver_tangent(gradient, -1.0)
    cg_step, _, cg_converged = TensorKitchen._symcpd_cg(
        dense_model,
        p,
        rhs,
        damping;
        tol = 1e-13,
        maxiter = 10 * tangent_dim,
    )
    H_damped = dense_normal_matrix(dense_model, p)
    H_damped[diagind(H_damped)] .+= damping
    dense_coordinates = -(H_damped \ get_coordinates(M, p, gradient, basis))
    dense_step = get_vector(M, p, dense_coordinates, basis)
    @test cg_converged
    @test norm(M, p, cg_step - dense_step) < 2e-10
end

@testset "SymCPD inexact CG diagnostics and nearly-collinear damping" begin
    x1 = normalize([1.0, 0.2, -0.1])
    orthogonal = normalize([-x1[2], x1[1], 0.0])
    x2 = normalize(0.99 .* x1 .+ sqrt(1 - 0.99^2) .* orthogonal)
    @test dot(x1, x2) ≈ 0.99 atol = 2e-15
    A = _symcpd_full_term(1.0, x1, 3) + _symcpd_full_term(0.8, x2, 3)
    component = SymmetricRankOne(3, 3)
    model = JoinModel(component, 2, DenseSymmetricTarget(A))
    M = manifold(model)
    p0 = TensorKitchen.join_point(M, (([0.85], x1), ([0.65], x2)))

    # Positive damping keeps the CG curvature positive even when the two
    # components make the undamped normal operator poorly conditioned.
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    direction = get_vector(M, p0, collect(1.0:manifold_dimension(M)), basis)
    damping = 1e-3
    damped_action =
        normal_operator(model, p0, direction) +
        TensorKitchen._scale_solver_tangent(direction, damping)
    @test inner(M, p0, direction, damped_action) > 0

    initial_cost = cost(model, p0)
    result = symcpd(
        A,
        2;
        p0 = (([0.85], x1), ([0.65], x2)),
        solver = :gn_cg,
        maxiter = 3,
        damping = damping,
        cg_tol = 1e-16,
        cg_maxiter = 1,
        verbose = false,
    )
    info = solver_info(result)
    @test isfinite(cost(result))
    @test cost(result) < initial_cost
    @test !isempty(info.cg_converged_history)
    @test info.cg_failed_count == count(!, info.cg_converged_history)
    @test info.total_cg_iterations == sum(info.cg_iterations_history)
    @test info.cg_failed_count > 0
    trial_count = length(info.damping_history)
    @test trial_count == length(info.predicted_reduction_history)
    @test trial_count == length(info.actual_reduction_history)
    @test trial_count == length(info.rho_history)
    @test trial_count == length(info.step_accepted_history)
    @test info.accepted_steps == count(identity, info.step_accepted_history)
    @test info.rejected_steps == count(!, info.step_accepted_history)
    for k = 1:trial_count
        predicted = info.predicted_reduction_history[k]
        actual = info.actual_reduction_history[k]
        rho = info.rho_history[k]
        if isfinite(predicted) && predicted > 0 && isfinite(actual)
            @test rho ≈ actual / predicted
        end
        if info.step_accepted_history[k]
            @test rho >= info.acceptance_ratio
        end
    end

    @test_throws ArgumentError symcpd(
        A,
        2;
        solver = :gn_cg,
        maxiter = 0,
        acceptance_ratio = 0.3,
        poor_step_ratio = 0.2,
        verbose = false,
    )

    # A tiny damped step is a stagnation condition, not proof that the outer
    # gradient tolerance has been met.
    stalled = symcpd(
        A,
        2;
        p0 = (([0.85], x1), ([0.65], x2)),
        solver = :gn_cg,
        maxiter = 2,
        tol = 1e-3,
        damping = 1e8,
        cg_tol = 1e-12,
        verbose = false,
    )
    @test solver_info(stalled).termination_reason == :small_step
    @test !converged(stalled)
    @test grad_norm(stalled) > 1e-3
end

@testset "SymCPD first-order solvers and result" begin
    x = normalize([1.0, 0.8, -0.3])
    x0 = normalize(x + [0.05, -0.03, 0.04])
    λ = -1.5
    A = _symcpd_full_term(λ, x, 2)
    model = JoinModel(SymmetricRankOne(3, 2), 1, compress_symmetric_tensor(A))
    p0 = (([λ * 0.9], x0),)
    initial_cost = cost(model, TensorKitchen.join_point(manifold(model), p0))
    for method in (:rgd, :rcg, :lbfgs)
        result = symcpd(
            A,
            1;
            p0 = deepcopy(p0),
            solver = method,
            maxiter = 200,
            tol = 1e-12,
            verbose = false,
        )
        @test result isa SymCPDResult
        @test solver(result) == method
        @test cost(result) < initial_cost
        @test rel_error(A, result) < 1e-5
        @test compressed_coordinates(result) ≈
              compress_symmetric_tensor(reconstruct(result)) atol = 1e-12
        @test size(factors(result)) == (3, 1)
        @test length(components(result)) == 1
        @test point(components(result)[1]) isa Tuple
        @test weights(result)[1] < 0
    end


    for method in (:gn_cg, :gn_dense)
        result = symcpd(
            A,
            1;
            p0 = deepcopy(p0),
            solver = method,
            maxiter = 30,
            tol = 1e-10,
            verbose = false,
        )
        @test solver(result) == method
        @test cost(result) < initial_cost
        @test rel_error(A, result) < 1e-8
        @test solver_info(result).materializes_jacobian == false
        @test solver_info(result).matrix_free_normal == (method == :gn_cg)
    end

    y = normalize([0.2, -0.5, 1.0])
    B = _symcpd_full_term(1.2, x, 3) + _symcpd_full_term(-0.7, y, 3)
    p2 = (([1.1], x), ([-0.65], y))
    model2 = JoinModel(SymmetricRankOne(3, 3), 2, compress_symmetric_tensor(B))
    initial2 = cost(model2, TensorKitchen.join_point(manifold(model2), p2))
    for method in (:rgd, :rcg, :lbfgs)
        result = symcpd(
            B,
            2;
            p0 = deepcopy(p2),
            solver = method,
            maxiter = 120,
            tol = 1e-10,
            verbose = false,
        )
        @test cost(result) < initial2
        @test rel_error(B, result) < 1e-4
        @test length(components(result)) == 2
    end
end
