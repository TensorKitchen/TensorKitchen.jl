# A quadratic model with a normal action but no tensor/residual/Jacobian API.
struct _NormalOnlyGNModel{T<:AbstractFloat,R} <: AbstractDecompositionModel{T}
    H::Matrix{T}
    solution::Vector{T}
    reference::R
end
TensorKitchen.manifold(::_NormalOnlyGNModel) = Euclidean(2)
TensorKitchen.supports_rgrad(::_NormalOnlyGNModel) = true
TensorKitchen.model_norm2(model::_NormalOnlyGNModel) = model.reference
TensorKitchen.cost(model::_NormalOnlyGNModel, p) =
    dot(p - model.solution, model.H * (p - model.solution)) / 2
TensorKitchen.rgrad(model::_NormalOnlyGNModel, p) = model.H * (p - model.solution)
TensorKitchen.normal_operator(model::_NormalOnlyGNModel, p, X) = model.H * X

@testset "Generic GN without tensor or residual storage" begin
    for T in (Float32, Float64), reference in (nothing, T(4))
        model = _NormalOnlyGNModel(T[2 0; 0 5], T[1, -2], reference)
        p0 = zeros(T, 2)
        @test model_scalar_type(model) === T
        @test dense_normal_matrix(model, p0) ≈ model.H
        prepared = TensorKitchen._prepare_solver_problem(model; init = PointInit(p0))
        @test prepared.normA2 === reference
        @test prepared.T === T
        for method in (:cg, :dense), init in (PointInit(p0), FunctionInit(_ -> copy(p0)))
            result = solve(
                GaussNewtonSolver(linear_solver = method),
                model;
                init,
                maxiter = 10,
                tol = T(1e-6),
                return_stats = true,
                verbose = false,
            )
            @test result.point ≈ model.solution atol = T(1e-5)
            @test eltype(result.point) === T
            @test result.cost < T(1e-8)
            @test result.converged
            @test result.solver_info.inner.solve_count > 0
            @test result.solver_info.error_measure ==
                  (isnothing(reference) ? :absolute_residual : :relative_residual)
            @test all(iszero, p0)
        end
        initial = solve(
            GaussNewtonSolver(),
            model;
            p0,
            init = FunctionInit(_ -> error("p0 must take precedence")),
            maxiter = 0,
            return_stats = true,
            verbose = false,
        )
        expected = sqrt(2 * cost(model, p0) / (isnothing(reference) ? one(T) : reference))
        @test initial.rel_error ≈ expected
        @test initial.solver_info.requested_init == :explicit
        @test initial.point !== p0
        # The shared first-order preparation also avoids tensor(model).
        first_order = solve(
            RGDSolver(),
            model;
            init = PointInit(p0),
            maxiter = 2,
            verbose = false,
            return_stats = true,
        )
        @test first_order.cost < cost(model, p0)
    end
end

@testset "SymCPD shared initialization" begin
    x = normalize([1.0, 0.4, -0.2])
    A = [1.2 * x[i] * x[j] * x[k] for i = 1:3, j = 1:3, k = 1:3]
    model = JoinModel(SymmetricRankOne(3, 3), 1, DenseSymmetricTarget(A))
    p0 = (([0.6], copy(x)),)
    snapshot = deepcopy(p0)
    for method in (:cg, :dense)
        solver = GaussNewtonSolver(linear_solver = method)
        baseline =
            solve(solver, model; p0, maxiter = 2, return_stats = true, verbose = false)
        for init in (PointInit(p0), FunctionInit(_ -> p0))
            result = solve(
                solver,
                model;
                init,
                maxiter = 2,
                return_stats = true,
                verbose = false,
            )
            @test result.cost ≈ baseline.cost
            @test result.solver_info.resolved_init ==
                  (init isa PointInit ? :explicit : :function)
        end
        @test TensorKitchen._join_cache_point_equal(p0, snapshot)
        default = solve(solver, model; maxiter = 0, return_stats = true, verbose = false)
        @test default.solver_info.requested_init == :auto
        @test default.solver_info.resolved_init == :sshopm
    end
end

# Nonlinear residual forces damping retries at a fixed outer point.
struct _RetryGNModel <: AbstractDecompositionModel{Float64}
    assemblies::Base.RefValue{Int}
end
TensorKitchen.manifold(::_RetryGNModel) = Euclidean(1)
TensorKitchen.supports_rgrad(::_RetryGNModel) = true
TensorKitchen.model_norm2(::_RetryGNModel) = nothing
TensorKitchen.cost(::_RetryGNModel, p) = (p[1]^2 - 1)^2 / 2
TensorKitchen.rgrad(::_RetryGNModel, p) = [2p[1] * (p[1]^2 - 1)]
TensorKitchen.normal_operator(::_RetryGNModel, p, X) = 4p[1]^2 * X
function TensorKitchen.dense_normal_matrix(model::_RetryGNModel, p)
    model.assemblies[] += 1
    return reshape([4p[1]^2], 1, 1)
end

@testset "GN dense damping retries reuse the unshifted normal matrix" begin
    model = _RetryGNModel(Ref(0))
    p0 = [0.1]
    results = map((:dense, :cg)) do method
        solve(
            GaussNewtonSolver(linear_solver = method),
            model;
            p0,
            maxiter = 1,
            verbose = false,
            return_stats = true,
        )
    end
    dense, cg = results
    info = dense.solver_info
    @test model.assemblies[] == 1
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    inner = TensorKitchen._solve_gn_inner(
        GaussNewtonSolver(linear_solver = :dense),
        model,
        p0,
        rgrad(model, p0),
        1e-6;
        tolerance = nothing,
        cache = (normal = reshape([4p0[1]^2], 1, 1), gradient = rgrad(model, p0), basis),
    )
    @test isnothing(inner.iterations)
    @test isnothing(inner.tolerance)
    @test info.rejected_steps > 1
    @test info.accepted_steps == 1
    @test info.inner.solve_count == info.accepted_steps + info.rejected_steps
    @test length(info.damping_history) ==
          length(info.rho_history) ==
          length(info.predicted_reduction_history) ==
          length(info.actual_reduction_history) ==
          length(info.step_accepted_history) ==
          info.inner.solve_count
    @test info.step_accepted_history == cg.solver_info.step_accepted_history
    @test info.damping_history ≈ cg.solver_info.damping_history
    @test dense.point ≈ cg.point
    @test dense.point ≈ p0 - rgrad(model, p0) / (4p0[1]^2 + last(info.damping_history))
    @test p0 == [0.1]
end
