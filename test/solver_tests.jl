@testset "Tangent CG solver contract" begin
    for T in (Float32, Float64)
        M = Euclidean(2)
        p = zeros(T, 2)
        A = T[4 1; 1 3]
        b = T[1, 2]
        action = X -> A * X
        tolerance = T(1e-5)
        solution, iterations, converged, info =
            TensorKitchen._tangent_cg(M, p, action, b; tol = tolerance, maxiter = 4)
        @test converged
        @test iterations <= 2
        @test solution ≈ A \ b atol = T(1e-5)
        @test eltype(solution) === T
        @test info.termination_reason == :converged
        @test info.relative_residual <= tolerance

        for scale in (T(1e-10), T(1e10))
            scaled_b = scale .* b
            scaled_solution, _, scaled_converged, scaled_info =
                TensorKitchen._tangent_cg(M, p, action, scaled_b; tol = tolerance, maxiter = 4)
            @test scaled_converged
            @test scaled_solution / scale ≈ A \ b rtol = tolerance
            @test scaled_info.relative_residual <= tolerance
        end

        _, limited_iterations, limited_converged, limited_info =
            TensorKitchen._tangent_cg(M, p, action, b; tol = tolerance, maxiter = 0)
        @test limited_iterations == 0
        @test !limited_converged
        @test limited_info.termination_reason == :maxiter
        @test_throws ArgumentError TensorKitchen._tangent_cg(
            M, p, action, b; tol = tolerance, maxiter = -1,
        )

        for (bad_action, reason) in ((X -> T(NaN) .* X, :nonfinite_curvature),
                                     (X -> T(Inf) .* X, :nonfinite_curvature))
            _, _, bad_converged, bad_info =
                TensorKitchen._tangent_cg(M, p, bad_action, b; tol = tolerance, maxiter = 4)
            @test !bad_converged
            @test bad_info.termination_reason == reason
        end
        _, _, bad_rhs_converged, bad_rhs_info = TensorKitchen._tangent_cg(
            M, p, action, T[NaN, 1]; tol = tolerance, maxiter = 4,
        )
        @test !bad_rhs_converged
        @test bad_rhs_info.termination_reason == :nonfinite_residual

        zero_solution, zero_iterations, zero_converged, zero_info =
            TensorKitchen._tangent_cg(
                M,
                p,
                action,
                zeros(T, 2);
                tol = tolerance,
                maxiter = 4,
            )
        @test iszero(zero_solution)
        @test zero_iterations == 0
        @test zero_converged
        @test zero_info.termination_reason == :initial_residual

        _, stalled_iterations, stalled_converged, stalled_info =
            TensorKitchen._tangent_cg(M, p, X -> -X, b; tol = tolerance, maxiter = 4)
        @test stalled_iterations == 0
        @test !stalled_converged
        @test stalled_info.termination_reason == :nonpositive_curvature
    end
end

@testset "Manopt normalized_objective controls objective units" begin
    A = _test_randn(255, 6, 5, 4)
    r = 2
    model = JoinModel(A, r; geometry = :canonical)
    p0 = TensorKitchen.initial_point(model, TuckerInit(); verbose = false)
    target_norm = norm(A)

    rel_out = solve(
        RGDFixedSolver(0.0),
        model;
        p0,
        maxiter = 1,
        tol = 0.0,
        verbose = false,
        return_stats = true,
        normalized_objective = true,
    )
    abs_out = solve(
        RGDFixedSolver(0.0),
        model;
        p0,
        maxiter = 1,
        tol = 0.0,
        verbose = false,
        return_stats = true,
        normalized_objective = false,
    )

    @test isapprox(2 * rel_out.cost, rel_out.rel_error^2; rtol = 1e-12, atol = 1e-12)
    @test isapprox(abs_out.rel_error, rel_out.rel_error; rtol = 1e-12, atol = 1e-12)
    @test isapprox(abs_out.cost, rel_out.cost * target_norm^2; rtol = 1e-12, atol = 1e-12)
    @test isapprox(
        abs_out.grad_norm,
        rel_out.grad_norm * target_norm^2;
        rtol = 1e-10,
        atol = 1e-10,
    )
end
@testset "LM residual/Jacobian smoke check matches gradient" begin
    cases = (
        JoinModel(_test_randn(366, 5, 4, 3), 2; geometry = :canonical),
        JoinModel(
            abs.(_test_randn(367, 5, 4, 3)),
            2;
            geometry = :softplus_metric,
            nonnegative = true,
        ),
    )
    for model in cases
        M = TensorKitchen.manifold(model)
        p = TensorKitchen._solver_point(
            M,
            TensorKitchen.initial_point(model, HOSVDInit(); verbose = false),
        )
        basis = ManifoldsBase.DefaultOrthonormalBasis()
        r = TensorKitchen._lm_raw_residual_vector(model, p)
        J = TensorKitchen._lm_raw_jacobian_matrix(model, M, p; basis)
        g_coord = transpose(J) * r
        g_from_J = ManifoldsBase.get_vector(M, p, g_coord, basis)
        g_model = TensorKitchen.rgrad(model, p)
        @test norm(M, p, g_from_J - g_model) ≤ 1e-7 * max(1.0, norm(M, p, g_model))
    end
end
@testset "Operator interface matches Jacobian and gradient" begin
    A = _test_randn(386, 5, 4, 3)
    cases = (
        (
            "generic_join",
            JoinModel((Manifolds.Segre((5, 4, 3)), Manifolds.Segre((5, 4, 3))), A),
            :deterministic,
        ),
        ("cp_canonical", JoinModel(A, 2; geometry = :canonical), HOSVDInit()),
        (
            "cp_softplus",
            JoinModel(abs.(A), 2; geometry = :softplus_metric, nonnegative = true),
            HOSVDInit(),
        ),
    )
    for (label, model, init) in cases
        M = TensorKitchen.manifold(model)
        p = TensorKitchen._solver_point(
            M,
            TensorKitchen.initial_point(model, init; verbose = false),
        )
        basis = ManifoldsBase.DefaultOrthonormalBasis()
        r = TensorKitchen.residual(model, p)
        J = TensorKitchen._lm_raw_jacobian_matrix(model, M, p; basis)
        @testset "$label" begin
            @test r ≈ TensorKitchen._lm_raw_residual_vector(model, p)
            d = manifold_dimension(M)
            for j = 1:min(d, 3)
                coeff = zeros(Float64, d)
                coeff[j] = 1.0
                Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
                @test TensorKitchen.differential_action(model, p, Xj) ≈ J[:, j]
            end
            g_adj = TensorKitchen.adjoint_action(model, p, r; basis)
            g_model = TensorKitchen.rgrad(model, p)
            @test norm(M, p, g_adj - g_model) ≤ 1e-7 * max(1.0, norm(M, p, g_model))
        end
    end
end
function _reference_join_jacobian_from_product_basis(model, M, p; basis)
    backend = model.backend
    parts = TensorKitchen.point_parts(p)
    T = TensorKitchen._scalar_eltype(p)
    ambient_dim = length(TensorKitchen.tensor(model))
    d = manifold_dimension(M)
    J = Matrix{T}(undef, ambient_dim, d)
    coeff = zeros(T, d)
    col = Vector{T}(undef, ambient_dim)
    buf = Vector{T}(undef, ambient_dim)
    for j = 1:d
        fill!(coeff, zero(T))
        coeff[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
        xparts = TensorKitchen.point_parts(Xj)
        fill!(col, zero(T))
        for k = 1:backend.r
            TensorKitchen.component_ambient_pushforward!(
                buf,
                TensorKitchen._backend_component(backend, k),
                parts[k],
                xparts[k],
            )
            col .+= buf
        end
        J[:, j] .= col
    end
    return J
end
@testset "LM generic join Jacobian uses component pushforwards" begin
    A = _test_randn(453, 5, 4, 3)
    model = JoinModel((Manifolds.Segre((5, 4, 3)), Manifolds.Segre((5, 4, 3))), A)
    M = TensorKitchen.manifold(model)
    p = TensorKitchen._solver_point(
        M,
        TensorKitchen.initial_point(model, :deterministic; verbose = false),
    )
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    J = TensorKitchen._lm_raw_jacobian_matrix(model, M, p; basis)
    @test all(isfinite, J)
    @test sum(
        TensorKitchen.component_tangent_dimension(
            TensorKitchen._backend_component(model.backend, k),
            TensorKitchen.point_parts(p)[k],
        ) for k = 1:model.backend.r
    ) == manifold_dimension(M)
    J_ref = _reference_join_jacobian_from_product_basis(model, M, p; basis)
    @test maximum(abs.(J .- J_ref)) ≤ 1e-12

    parts = TensorKitchen.point_parts(p)
    c1 = TensorKitchen._backend_component(model.backend, 1)
    ξ1 = TensorKitchen.component_basis_vector(c1, parts[1], 1; basis)
    buf = similar(model.backend.work_rec)
    TensorKitchen.component_ambient_pushforward!(buf, c1, parts[1], ξ1)
    @test maximum(abs.(buf .- J[:, 1])) ≤ 1e-10

    c2 = TensorKitchen._backend_component(model.backend, 2)
    ξ2 = TensorKitchen.component_basis_vector(c2, parts[2], 1; basis)
    TensorKitchen.component_ambient_pushforward!(buf, c2, parts[2], ξ2)
    offset2 = TensorKitchen.component_tangent_dimension(c1, parts[1]) + 1
    @test maximum(abs.(buf .- J[:, offset2])) ≤ 1e-10

    fill!(buf, NaN)
    TensorKitchen.component_ambient_pushforward!(buf, c1, parts[1], ξ1)
    fresh = similar(buf)
    TensorKitchen.component_ambient_pushforward!(fresh, c1, parts[1], ξ1)
    @test buf == fresh

    retraction_method = TensorKitchen._solver_retraction_method(M, p)
    ϵ = 1e-6
    buf_plus = similar(model.backend.work_rec)
    buf_minus = similar(model.backend.work_rec)
    d = manifold_dimension(M)
    for j = 1:min(d, 3)
        coeff = zeros(Float64, d)
        coeff[j] = 1.0
        Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
        p_plus = TensorKitchen._independent_retract(M, p, ϵ * Xj, retraction_method)
        p_minus = TensorKitchen._independent_retract(M, p, -ϵ * Xj, retraction_method)
        TensorKitchen._join_reconstruct!(buf_plus, model.backend, p_plus)
        TensorKitchen._join_reconstruct!(buf_minus, model.backend, p_minus)
        fd = (buf_plus .- buf_minus) ./ (2 * ϵ)
        @test maximum(abs.(fd .- J[:, j])) ≤ 1e-7
    end
end
@testset "LM normalized and unnormalized objectives take the same step" begin
    A = _test_randn(613, 6, 5, 4)
    model = JoinModel(A, 2; geometry = :canonical)
    p0 = TensorKitchen._solver_point(
        TensorKitchen.manifold(model),
        TensorKitchen.initial_point(model, HOSVDInit(); verbose = false),
    )
    res_rel = solve(
        LMSolver(),
        model;
        p0,
        maxiter = 1,
        tol = 0.0,
        verbose = false,
        return_stats = true,
        normalized_objective = true,
    )
    res_abs = solve(
        LMSolver(),
        model;
        p0,
        maxiter = 1,
        tol = 0.0,
        verbose = false,
        return_stats = true,
        normalized_objective = false,
    )
    rawmodel = TensorKitchen.cpd_model(model)
    X_rel = TensorKitchen.embed_point(rawmodel, TensorKitchen.point(res_rel))
    X_abs = TensorKitchen.embed_point(rawmodel, TensorKitchen.point(res_abs))
    @test maximum(abs.(X_rel .- X_abs)) ≤ 1e-10
    @test isapprox(res_rel.rel_error, res_abs.rel_error; rtol = 1e-10, atol = 1e-10)
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
@testset "gradient interface: grad/egrad_to_rgrad" begin
    rng = MersenneTwister(2027)
    dims = (6, 5, 4)
    A = randn(rng, dims...)

    # Built-in Segre path (upstream-only): no project(M,p,⋅) for structured Segre tangent.
    model = TensorKitchen.Rank1CPDModel(A)
    M = TensorKitchen.manifold(model)
    p = TensorKitchen.initial_point(model, HOSVDInit())
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
    p_c = TensorKitchen.initial_point(model_c, HOSVDInit())
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
