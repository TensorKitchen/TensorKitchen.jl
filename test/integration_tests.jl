@testset "frontend defaults through public APIs" begin
    A = _test_randn(1445, 6, 5, 4)
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

    cpd_lbfgs_symbol = cpd(
        A,
        2;
        solver = :lbfgs,
        init = :alswarm,
        warm_steps = 2,
        maxiter = 2,
        verbose = false,
    )
    @test cpd_lbfgs_symbol isa CPDResult
    @test cpd_lbfgs_symbol.solver == :lbfgs
    @test cpd_lbfgs_symbol.solver_info.memory_size == 1

    cpd_lbfgs_object = cpd(
        A,
        2;
        solver = LBFGSSolver(memory_size = 3),
        init = :alswarm,
        warm_steps = 2,
        maxiter = 2,
        verbose = false,
    )
    @test cpd_lbfgs_object isa CPDResult
    @test cpd_lbfgs_object.solver == :lbfgs
    @test cpd_lbfgs_object.solver_info.memory_size == 3

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
    @test length(backend.components) == 2
    @test all(c -> c isa TensorKitchen.JoinComponent, backend.components)
    @test map(TensorKitchen._component_manifold, backend.components) == manifolds
    model_btd = JoinModel{Float64,typeof(backend)}(backend)
    M_btd = TensorKitchen.manifold(model_btd)
    p_btd = TensorKitchen._solver_point(
        M_btd,
        TensorKitchen._btd_random_point(MersenneTwister(3179), backend),
    )
    basis_btd = ManifoldsBase.DefaultOrthonormalBasis()
    J_btd =
        TensorKitchen._lm_raw_jacobian_matrix(model_btd, M_btd, p_btd; basis = basis_btd)
    @test all(isfinite, J_btd)
    @test sum(
        TensorKitchen.component_tangent_dimension(
            TensorKitchen._backend_component(backend, k),
            TensorKitchen.point_parts(p_btd)[k],
        ) for k = 1:backend.r
    ) == manifold_dimension(M_btd)
    J_btd_ref = _reference_join_jacobian_from_product_basis(
        model_btd,
        M_btd,
        p_btd;
        basis = basis_btd,
    )
    @test maximum(abs.(J_btd .- J_btd_ref)) ≤ 1e-12

    retraction_method_btd = TensorKitchen._solver_retraction_method(M_btd, p_btd)
    residual0_btd = TensorKitchen._lm_raw_residual_vector(model_btd, p_btd)
    ϵ_btd = 1e-6
    for j = 1:min(manifold_dimension(M_btd), 2)
        coeff = zeros(Float64, manifold_dimension(M_btd))
        coeff[j] = 1.0
        Xj = ManifoldsBase.get_vector(M_btd, p_btd, coeff, basis_btd)
        p_plus = TensorKitchen._independent_retract(
            M_btd,
            p_btd,
            ϵ_btd * Xj,
            retraction_method_btd,
        )
        r_plus = TensorKitchen._lm_raw_residual_vector(model_btd, p_plus)
        fd = (r_plus .- residual0_btd) ./ ϵ_btd
        @test maximum(abs.(fd .- J_btd[:, j])) ≤ 5e-6
    end

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
        residual_eg = TensorKitchen._tucker_egrad(
            TensorKitchen._backend_manifold(backend, b),
            parts_btd[b],
            residual_btd,
        )
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
            TensorKitchen._independent_retract(
                TensorKitchen._backend_manifold(backend, b),
                parts_btd[b],
                (-h) * block_grad,
            ),
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
    p_base_btd = TensorKitchen._btd_random_point(MersenneTwister(2028), backend)
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
@testset "save_result and load_result preserve experiment records" begin
    A = _test_randn(3869, 8, 6, 4)
    ranks = (3, 2, 2)
    result = tucker(A, ranks)
    record = (
        result = result,
        method = :tucker,
        ranks = ranks,
        input_size = size(A),
        preprocessing = :none,
        julia_version = string(VERSION),
        tensorkitchen_version = string(pkgversion(TensorKitchen)),
    )

    mktemp() do path, io
        close(io)
        @test save_result(path, record) == path
        loaded = load_result(path)
        @test loaded.ranks == ranks
        @test loaded.input_size == size(A)
        @test reconstruct(loaded.result) ≈ reconstruct(result)
        @test processing_order(loaded.result) == processing_order(result)
    end
end
