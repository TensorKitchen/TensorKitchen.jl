@testset "BTD exposes LM residual/Jacobian hooks on nested Tucker layouts" begin
    A = randn(7, 6, 5)
    ranks = (2, 2, 2)
    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(size(A), ranks, 2))
    backend = TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, A)
    model = TensorKitchen.JoinModel{Float64,typeof(backend)}(backend)
    p0 = TensorKitchen.initial_point(model, :random; verbose = false)
    M = TensorKitchen.manifold(model)
    p0_solver = TensorKitchen._solver_point(M, p0)
    basis = ManifoldsBase.DefaultOrthonormalBasis()

    @test p0 isa ArrayPartition
    @test TensorKitchen.point_parts(p0)[1] isa Manifolds.TuckerPoint
    @test p0_solver isa ArrayPartition
    @test TensorKitchen.point_parts(p0_solver)[1] isa Manifolds.TuckerPoint

    residual0 = TensorKitchen._lm_raw_residual_vector(model, p0_solver)
    J0 = TensorKitchen._lm_raw_jacobian_matrix(model, M, p0_solver; basis = basis)
    @test length(residual0) == length(A)
    @test size(J0) == (length(A), manifold_dimension(M))
    @test all(isfinite, residual0)
    @test all(isfinite, J0)
    reconstructed = zeros(size(A))
    for part in TensorKitchen.point_parts(p0_solver)
        reconstructed .+= TensorKitchen._btd_block_tensor(part)
    end
    @test residual0 ≈ vec(reconstructed .- A)

    coeff = zeros(Float64, manifold_dimension(M))
    coeff[1] = 1.0
    X = ManifoldsBase.get_vector(M, p0_solver, coeff, basis)
    JX = TensorKitchen.differential_action(model, p0_solver, X)
    ambient = randn(size(A))
    lhs = dot(JX, vec(ambient))
    rhs = ManifoldsBase.inner(
        M,
        p0_solver,
        X,
        TensorKitchen.adjoint_action(model, p0_solver, vec(ambient)),
    )
    @test isapprox(lhs, rhs; atol = 1e-8, rtol = 1e-8)

    exact_basis_gradient =
        TensorKitchen.model_exact_join_basis_function(model)(M, p0_solver)
    direct_gradient = TensorKitchen.rgrad(model, p0_solver)
    @test norm(M, p0_solver, exact_basis_gradient - direct_gradient) ≤ 1e-12
end


@testset "BTD backend construction is ambient-workspace-free" begin
    dims = (5, 4, 3)
    ranks = (2, 2, 2)
    data = reshape(Float32.(1:prod(dims)), dims)
    target = _NoSimilarArray(data)
    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(dims, ranks, 2))

    backend =
        TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, target)
    @test backend.target === target
    @test backend.target_normsq == observation_norm2(target)
    @test !hasfield(typeof(backend), :target_flat)
    @test !hasfield(typeof(backend), :work_rec)
    @test !hasfield(typeof(backend), :work_residual)
    @test !hasfield(typeof(backend), :component_bufs)
    @test all(
        isempty(cache.bufs) for cache in (
            backend.workspace.tensor_slot1,
            backend.workspace.tensor_slot2,
            backend.workspace.perm_in,
            backend.workspace.perm_out,
            backend.workspace.persist,
        )
    )

    lazy = prepare_tensor(Int16.(data); compute_type = Float32)
    lazy_backend =
        TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, lazy)
    @test lazy_backend.target === lazy
    @test !is_materialized(lazy_backend.target)
    @test lazy_backend.target_normsq == observation_norm2(lazy)
    @test all(
        isempty(cache.bufs) for cache in (
            lazy_backend.workspace.tensor_slot1,
            lazy_backend.workspace.tensor_slot2,
            lazy_backend.workspace.perm_in,
            lazy_backend.workspace.perm_out,
            lazy_backend.workspace.persist,
        )
    )

    lazy_model = TensorKitchen.JoinModel{Float32,typeof(lazy_backend)}(lazy_backend)
    lazy_point = TensorKitchen.initial_point(lazy_model, :random)
    @test TensorKitchen.cost(lazy_model, lazy_point) isa Float32
    @test TensorKitchen.rgrad(lazy_model, lazy_point) isa ArrayPartition
    @test TensorKitchen.model_exact_join_basis_function(lazy_model)(
        TensorKitchen.manifold(lazy_model),
        lazy_point,
    ) isa ArrayPartition
    @test_throws ArgumentError TensorKitchen.initial_point(lazy_model, :sthosvd)
    @test_throws ArgumentError TensorKitchen.residual(lazy_model, lazy_point)

    projected_init =
        BTDProjectedMultistartInit(2; screening_steps = 1, block_maxiter = 1, seed = 812)
    projected_lazy = TensorKitchen.initial_point(lazy_model, projected_init)
    dense_backend =
        TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, data)
    dense_model = TensorKitchen.JoinModel{Float32,typeof(dense_backend)}(dense_backend)
    projected_dense = TensorKitchen.initial_point(dense_model, projected_init)
    @test TensorKitchen.cost(lazy_model, projected_lazy) ≈
          TensorKitchen.cost(dense_model, projected_dense) rtol = 2e-5 atol = 2e-4

    @test_throws ArgumentError BTDProjectedMultistartInit(0)
    @test_throws ArgumentError BTDProjectedMultistartInit(1; screening_steps = -1)
    @test_throws ArgumentError BTDProjectedMultistartInit(1; block_maxiter = -1)

    shared_point = TensorKitchen._btd_random_point(MersenneTwister(913), dense_backend)
    dense_result = btd(
        data,
        2,
        ranks;
        solver = :als,
        init = PointInit(deepcopy(shared_point)),
        maxiter = 1,
        tol = 0.0,
        block_method = :hooi,
        block_maxiter = 1,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    lazy_result = btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = false,
        solver = :als,
        init = PointInit(deepcopy(shared_point)),
        maxiter = 1,
        tol = 0.0,
        block_method = :hooi,
        block_maxiter = 1,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    @test lazy_result isa BTDResult
    @test eltype(core(first(blocks(lazy_result)))) === Float32
    @test lazy_result.solver_info.block_update == :projected
    @test lazy_result.cost ≈ dense_result.cost rtol = 2e-5 atol = 2e-4
    @test lazy_result.rel_error ≈ dense_result.rel_error rtol = 2e-5 atol = 2e-5
    @test reconstruct(lazy_result) ≈ reconstruct(dense_result) rtol = 2e-5 atol = 2e-4

    auto_result = btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = false,
        solver = :als,
        maxiter = 0,
        block_maxiter = 1,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    @test auto_result isa BTDResult
    @test isfinite(auto_result.rel_error)
    @test auto_result.solver_info.block_update == :projected

    lazy_rgd_result = btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = false,
        solver = :rgd,
        init = BTDALSWarmStartInit(
            1;
            base_init = projected_init,
            block_method = :hooi,
            block_maxiter = 1,
        ),
        maxiter = 1,
        btd_als_polish_maxiter = 0,
        warm_rel_error_gate = nothing,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    @test lazy_rgd_result isa BTDResult
    @test lazy_rgd_result.solver == :rgd
    @test isfinite(lazy_rgd_result.rel_error)

    for solver_name in (:rcg, :lbfgs, :rgd_fixed, :btd_tsd)
        solver_result = btd(
            Int16.(data),
            2,
            ranks;
            compute_type = Float32,
            materialize = false,
            solver = solver_name,
            init = BTDProjectedMultistartInit(1; screening_steps = 0, seed = 812),
            maxiter = 1,
            btd_als_polish_maxiter = 0,
            max_stagnation_restarts = 0,
            verbose = false,
        )
        @test solver_result isa BTDResult
        @test solver_result.solver == solver_name
        @test isfinite(solver_result.rel_error)
    end

    @test_throws ArgumentError btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = false,
        solver = :als,
        init = BTDHOSVDMultistartInit(1; screening_steps = 0),
        maxiter = 0,
        verbose = false,
    )
    @test_throws ArgumentError btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = false,
        solver = :als,
        init = PointInit(shared_point),
        block_method = :sthosvd,
        maxiter = 0,
        verbose = false,
    )
    materialized_result = btd(
        Int16.(data),
        2,
        ranks;
        compute_type = Float32,
        materialize = true,
        solver = :als,
        init = BTDHOSVDMultistartInit(1; screening_steps = 0),
        maxiter = 0,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    @test materialized_result isa BTDResult
    @test eltype(core(first(blocks(materialized_result)))) === Float32

    norm_calls = Ref(0)
    counted = _NormCountingArray(data, norm_calls, Int[])
    counted_backend =
        TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, counted)
    @test counted_backend.target === counted
    @test norm_calls[] == 1

    norm_calls[] = 0
    empty!(counted.norm_block_lengths)
    btd(
        counted,
        2,
        ranks;
        solver = :als,
        init = PointInit(deepcopy(shared_point)),
        maxiter = 0,
        conversion_block_length = 7,
        max_stagnation_restarts = 0,
        verbose = false,
    )
    @test norm_calls[] == 1
    @test counted.norm_block_lengths == [7]

    @test_throws ArgumentError btd(
        data,
        2,
        ranks;
        observation_norm2_cache = sum(abs2, data),
        solver = :als,
        init = PointInit(shared_point),
        maxiter = 0,
        verbose = false,
    )

    join_backend =
        TensorKitchen._sum_backend_instance(TensorKitchen.JoinBackend, manifolds, data)
    @test length(join_backend.work_rec) == length(data)
    @test length(join_backend.work_residual) == length(data)
    @test length(join_backend.component_bufs) == length(manifolds)
end


@testset "BTD rejects LMSolver until nested Tucker LM support lands" begin
    A = randn(7, 6, 5)
    ranks = (2, 2, 2)

    @test_throws ArgumentError btd(
        A,
        2,
        ranks;
        solver = :lm,
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )

    @test_throws ArgumentError btd(
        A,
        2,
        ranks;
        solver = LMSolver(),
        maxiter = 2,
        tol = 1e-6,
        verbose = false,
    )
end


@testset "BTD projected block residual matches explicit ambient residual" begin
    rng = MersenneTwister(713)
    dims = (6, 5, 4)
    ranks = (2, 2, 2)
    A = randn(rng, dims...)
    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(dims, ranks, 3))
    backend = TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, A)
    model = TensorKitchen.JoinModel{Float64,typeof(backend)}(backend)
    parts = TensorKitchen.point_parts(TensorKitchen.initial_point(model, :random))
    target_before = copy(A)

    for b = 1:backend.r
        residual_without_b = copy(A)
        for c = 1:backend.r
            c == b && continue
            residual_without_b .-= TensorKitchen._btd_block_tensor(parts[c])
        end

        for mode = 1:length(dims)
            explicit_projection = TensorKitchen._tucker_project_target_except_mode(
                parts[b],
                residual_without_b,
                mode,
            )
            implicit_projection = TensorKitchen._btd_projected_residual_except_block_mode(
                backend,
                parts,
                b,
                mode,
            )
            @test implicit_projection ≈ explicit_projection rtol = 1e-12 atol = 1e-12

            projection_point = parts[mod1(b + 1, backend.r)]
            explicit_alternate = TensorKitchen._tucker_project_target_except_mode(
                projection_point,
                residual_without_b,
                mode,
            )
            implicit_alternate = TensorKitchen._btd_projected_residual_except_block_mode(
                backend,
                parts,
                b,
                mode,
                projection_point,
            )
            @test implicit_alternate ≈ explicit_alternate rtol = 1e-12 atol = 1e-12
        end

        explicit_core = TensorKitchen._tucker_project_target(parts[b], residual_without_b)
        implicit_core =
            TensorKitchen._btd_projected_residual_except_block_core(backend, parts, b)
        @test implicit_core ≈ explicit_core rtol = 1e-12 atol = 1e-12
        @test TensorKitchen._btd_residual_except_block_norm2(backend, parts, b) ≈
              sum(abs2, residual_without_b) rtol = 1e-12 atol = 1e-12
    end

    @test A == target_before
    @test_throws BoundsError TensorKitchen._btd_projected_residual_except_block_mode(
        backend,
        parts,
        backend.r + 1,
        1,
    )
    @test_throws ArgumentError TensorKitchen._btd_projected_residual_except_block_mode(
        backend,
        parts,
        1,
        length(dims) + 1,
    )
end


@testset "BTD projected HOOI block solve matches dense residual solve" begin
    rng = MersenneTwister(714)
    dims = (7, 6, 5)
    ranks = (2, 2, 2)
    A = randn(rng, dims...)
    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(dims, ranks, 3))
    backend = TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, A)
    model = TensorKitchen.JoinModel{Float64,typeof(backend)}(backend)
    parts = TensorKitchen.point_parts(TensorKitchen.initial_point(model, :random))

    for b = 1:backend.r
        residual_without_b = copy(A)
        for c = 1:backend.r
            c == b && continue
            residual_without_b .-= TensorKitchen._btd_block_tensor(parts[c])
        end

        dense = TensorKitchen._btd_block_fit_tucker(
            residual_without_b,
            ranks;
            method = :hooi,
            block_maxiter = 2,
            tol = 0.0,
            warm = parts[b],
        )
        projected = TensorKitchen._btd_block_fit_tucker_projected(
            backend,
            parts,
            b,
            ranks;
            block_maxiter = 2,
            tol = 0.0,
            warm = parts[b],
        )

        @test reconstruct(projected) ≈ reconstruct(dense) rtol = 1e-11 atol = 1e-11
        @test projected.core ≈ dense.core rtol = 1e-11 atol = 1e-11
        for mode = 1:length(dims)
            dense_projector = dense.factors[mode] * transpose(dense.factors[mode])
            projected_projector =
                projected.factors[mode] * transpose(projected.factors[mode])
            @test projected_projector ≈ dense_projector rtol = 1e-11 atol = 1e-11
        end
    end
end


@testset "BTD projected ALS pass matches ambient reference" begin
    rng = MersenneTwister(715)
    dims = (7, 6, 5)
    ranks = (2, 2, 2)
    A = randn(rng, dims...)
    manifolds = TensorKitchen._as_join_manifold_tuple(TuckerJoin(dims, ranks, 2))
    backend = TensorKitchen._sum_backend_instance(TensorKitchen.BTDBackend, manifolds, A)
    model = TensorKitchen.JoinModel{Float64,typeof(backend)}(backend)
    p0 = TensorKitchen.initial_point(model, :random)

    ambient = fit_btd_als(
        A,
        backend;
        p0,
        maxiter = 2,
        tol = 0.0,
        block_method = :hooi,
        block_maxiter = 2,
        block_update = :ambient,
        verbose = false,
        return_stats = true,
    )
    projected = fit_btd_als(
        A,
        backend;
        p0,
        maxiter = 2,
        tol = 0.0,
        block_method = :hooi,
        block_maxiter = 2,
        verbose = false,
        return_stats = true,
    )

    @test projected.solver_info.block_update == :projected
    @test projected.cost ≈ ambient.cost rtol = 1e-10 atol = 1e-10
    @test projected.rel_error ≈ ambient.rel_error rtol = 1e-10 atol = 1e-10
    projected_parts = TensorKitchen.point_parts(projected.point)
    ambient_parts = TensorKitchen.point_parts(ambient.point)
    for b = 1:length(projected_parts)
        @test TensorKitchen._btd_block_tensor(projected_parts[b]) ≈
              TensorKitchen._btd_block_tensor(ambient_parts[b]) rtol = 1e-10 atol = 1e-10
    end

    @test_throws ArgumentError fit_btd_als(
        A,
        backend;
        p0,
        maxiter = 0,
        block_update = :invalid,
        verbose = false,
    )
    @test_throws ArgumentError fit_btd_als(
        A,
        backend;
        p0,
        maxiter = 0,
        block_method = :sthosvd,
        block_update = :projected,
        verbose = false,
    )
end

# =========================================================================
# cpd/cp_rank.jl (cost/egrad functions)
# =========================================================================
