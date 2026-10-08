@testset "cpd.jl: nonnegative CPD keeps cost nonnegative on larger tensors" begin
    A = abs.(_test_randn(1591, 60, 50, 40))
    res = cpd(A, 5; solver = :rgd, nonnegative = true, maxiter = 3, verbose = false)
    @test isfinite(res.cost)
    @test res.cost >= 0
    @test isfinite(res.rel_error)
    @test res.rel_error > 0
end
@testset "cpd.jl: nonnegative analytic cost matches explicit residual" begin
    dims = (8, 6, 5)
    r = 3
    A = abs.(_test_randn(1601, Float64, dims...))
    model = JoinModel(A, r; nonnegative = true)
    p = TensorKitchen.initial_point(model, HOSVDInit())
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
    expected_solver_cost(solver, cost, rel) = solver == :als ? cost : 0.5 * rel^2
    public_columns_unit(res) = all(
        isapprox(norm(TensorKitchen.factors(res)[m][:, k]), 1; atol = 1e-8, rtol = 1e-8) for m in eachindex(TensorKitchen.factors(res)) for
        k in eachindex(TensorKitchen.weights(res))
    )

    dims1 = (18, 14, 10)
    A1 = abs.(_test_randn(1633, dims1...))
    init_rng1 = MersenneTwister(1634)
    p01 = CPDPoint([1.0], [rand(init_rng1, d, 1) .+ 0.1 for d in dims1])
    for solver in (:rgd, :rcg)
        res = cpd(A1, 1; solver, nonnegative = true, p0 = p01, maxiter = 4, verbose = false)
        cost, rel = explicit_stats(A1, res)
        @test res.cost ≈ expected_solver_cost(solver, cost, rel) atol = 1e-8 rtol = 1e-8
        @test res.rel_error ≈ rel atol = 1e-8 rtol = 1e-8
        @test public_columns_unit(res)
    end

    dimsr = (20, 16, 12)
    Ar = abs.(_test_randn(1643, dimsr...))
    init_rngr = MersenneTwister(1644)
    p0r = CPDPoint(ones(3), [rand(init_rngr, d, 3) .+ 0.1 for d in dimsr])
    for solver in (:als, :rgd, :rcg)
        res = cpd(Ar, 3; solver, nonnegative = true, p0 = p0r, maxiter = 4, verbose = false)
        cost, rel = explicit_stats(Ar, res)
        @test res.cost ≈ expected_solver_cost(solver, cost, rel) atol = 1e-8 rtol = 1e-8
        @test res.rel_error ≈ rel atol = 1e-8 rtol = 1e-8
        @test public_columns_unit(res)
    end
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
    init_rng = MersenneTwister(2026)
    p0 = CPDPoint(ones(r), [rand(init_rng, dims[m], r) .+ 0.1 for m = 1:length(dims)])

    res_auto = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        p0,
        normalization = :auto,
        verbose = false,
    )
    res_none = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        p0,
        normalization = :none,
        verbose = false,
    )
    res_sep = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        nn_update = :nnls,
        maxiter = 40,
        tol = 1e-6,
        p0,
        normalization = :lambda_separate,
        verbose = false,
    )

    @test res_auto.rel_error ≈ res_none.rel_error atol = 1e-12 rtol = 1e-12
    @test res_auto.grad_norm ≈ res_none.grad_norm atol = 1e-12 rtol = 1e-12
    @test !(
        isapprox(res_auto.rel_error, res_sep.rel_error; atol = 1e-12, rtol = 1e-12) &&
        isapprox(res_auto.grad_norm, res_sep.grad_norm; atol = 1e-12, rtol = 1e-12)
    )

    res_default_update = cpd(
        A,
        r;
        solver = :als,
        nonnegative = true,
        maxiter = 40,
        tol = 1e-6,
        p0,
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
    init_rng = MersenneTwister(2128)
    λ0 = ones(r)
    U0 = [rand(init_rng, d, r) .+ 0.1 for d in dims]
    p0_nn = CPDPoint(λ0, U0)

    out_nn = fit_cp_als(
        A,
        r;
        maxiter = 30,
        tol = 1e-6,
        init_factors = (λ0, U0),
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
        p0 = p0_nn,
        verbose = false,
    )
    @test all(w -> w >= -1e-12, TensorKitchen.weights(res_nn))
    @test all(F -> all(F .>= -1e-12), TensorKitchen.factors(res_nn))
    @test isfinite(res_nn.rel_error)

    res_nn_api =
        nncpd(A, r; solver = :als, maxiter = 30, tol = 1e-6, p0 = p0_nn, verbose = false)
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

    r_heur = minimum(size(A))
    p0_heur = CPDPoint(ones(r_heur), [rand(init_rng, d, r_heur) .+ 0.1 for d in dims])
    res_nn_api_heur =
        nncpd(A; solver = :als, maxiter = 2, tol = 1e-6, p0 = p0_heur, verbose = false)
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
        p0 = p0_nn,
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
        p0 = p0_nn,
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
        p0 = p0_nn,
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
        init = HOSVDInit(),
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
