@testset "LM zero-start CR and inner tolerance policies" begin
    for T in (Float32, Float64)
        M = Euclidean(2)
        p = zeros(T, 2)
        A = T[2 0; 0 5]
        b = T[-1, -2]
        calls = Ref(0)
        objective = TensorKitchen.Manopt.SymmetricLinearSystemObjective(
            (M, p, X) -> (calls[] += 1; A * X),
            (M, p) -> b,
        )
        options = LMInnerOptions(tolerance = RelativeResidualTolerance(1e-5), maxiter = 5)
        trace = TensorKitchen._LMInnerTrace(T, true)
        rng = copy(Random.default_rng())
        problem, state = TensorKitchen._lm_cr_state(M, p, objective, options, one(T), trace)
        @test calls[] == 0
        @test iszero(state.X)
        @test !state.warm_start
        @test rand(copy(Random.default_rng())) == rand(rng)
        TensorKitchen.Manopt.solve!(problem, state)
        @test state.X ≈ -(A \ b) rtol = T(1e-5)
        @test trace.iterations == [2]
        @test trace.final_residual[1] / trace.initial_residual[1] <= T(1e-5)
        @test trace.reason[1] in (:relative_tolerance, :zero_residual)
        # Every reuse starts from zero, including at a new tangent base point.
        state.X .= T(10)
        TensorKitchen.Manopt.solve!(problem, state)
        @test trace.iterations == [2, 2]
        @test state.X ≈ -(A \ b) rtol = T(1e-5)

        zero_objective = TensorKitchen.Manopt.SymmetricLinearSystemObjective(
            (M, p, X) -> A * X,
            (M, p) -> zeros(T, 2),
        )
        zero_trace = TensorKitchen._LMInnerTrace(T, false)
        zero_problem, zero_state =
            TensorKitchen._lm_cr_state(M, p, zero_objective, options, one(T), zero_trace)
        TensorKitchen.Manopt.solve!(zero_problem, zero_state)
        @test zero_trace.iterations == [0]
        @test zero_trace.reason == [:zero_residual]
        @test all(isfinite, zero_state.X)

        limited_trace = TensorKitchen._LMInnerTrace(T, false)
        limited_options =
            LMInnerOptions(tolerance = RelativeResidualTolerance(1e-6), maxiter = 1)
        limited_problem, limited_state = TensorKitchen._lm_cr_state(
            M,
            p,
            objective,
            limited_options,
            one(T),
            limited_trace,
        )
        TensorKitchen.Manopt.solve!(limited_problem, limited_state)
        @test limited_trace.iterations == [1]
        @test limited_trace.reason == [:maxiter]
        policy = AdaptiveResidualTolerance()
        @test TensorKitchen._inner_tolerance(policy, T(1)) ≈ T(1e-2)
        @test TensorKitchen._inner_tolerance(policy, T(1e-12)) < T(1e-2)
        @test TensorKitchen._inner_tolerance(policy, zero(T)) >= eps(T)
    end
    @test_throws ArgumentError RelativeResidualTolerance(0)
    @test_throws ArgumentError AdaptiveResidualTolerance(minimum = 0.1, maximum = 0.01)
    @test_throws ArgumentError AbsoluteResidualTolerance(Inf)
    @test_throws ArgumentError LMInnerOptions(maxiter = 0)
end

struct _LMCountingModel{T<:AbstractFloat} <: AbstractDecompositionModel{T}
    target::Vector{T}
    J::Matrix{T}
    calls::Vector{Int}
end
TensorKitchen.manifold(::_LMCountingModel) = Euclidean(2)
TensorKitchen.tensor(model::_LMCountingModel) = model.target
TensorKitchen.cost(model::_LMCountingModel, p) = sum(abs2, model.J * p - model.target) / 2
TensorKitchen.supports_rgrad(::_LMCountingModel) = true
TensorKitchen.rgrad(model::_LMCountingModel, p) = model.J' * (model.J * p - model.target)
TensorKitchen.residual(model::_LMCountingModel, p) =
    (model.calls[1] += 1; model.J * p - model.target)
TensorKitchen.differential_action!(out::AbstractVector, model::_LMCountingModel, p, X) =
    (model.calls[2] += 1; mul!(out, model.J, X))
TensorKitchen.adjoint_action(model::_LMCountingModel, p, z::AbstractVector; kwargs...) =
    (model.calls[3] += 1; model.J' * z)

@testset "LM operator diagnostics and deterministic execution" begin
    for T in (Float32, Float64)
        model = _LMCountingModel(T[1, 2, 3], T[1 0; 0 2; 1 1], zeros(Int, 3))
        p0 = T[0.2, 0.3]
        snapshot = copy(p0)
        opts = LMInnerOptions(tolerance = RelativeResidualTolerance(1e-4))
        solver = LMSolver(inner = opts, diagnostics = true)
        Random.seed!(22)
        rng = copy(Random.default_rng())
        result = solve(solver, model; p0, maxiter = 2, verbose = false, return_stats = true)
        @test rand(copy(Random.default_rng())) == rand(rng)
        info = solver_info(result)
        @test collect(values(info.operator_calls)) == model.calls
        @test info.operator_calls.adjoint <=
              2 * iterations(result) + info.operator_calls.differential + 3
        @test length(info.inner.iterations) == iterations(result)
        @test sum(info.inner.iterations) == info.inner.total_iterations
        @test all(>=(0), values(info.operator_seconds))
        @test all(>=(0), info.inner.seconds)
        @test cost(result) < cost(model, p0)
        @test p0 == snapshot
        @test eltype(point(result)) === T
        Random.seed!(87)
        plain = solve(
            LMSolver(inner = opts),
            model;
            p0,
            maxiter = 2,
            verbose = false,
            return_stats = true,
        )
        @test point(plain) ≈ point(result)
        @test isnothing(solver_info(plain).operator_calls)
        @test isnothing(solver_info(plain).inner.seconds)
        adaptive = LMSolver(diagnostics = true, damping_reduction_threshold = 0.8)
        scaled =
            solve(adaptive, model; p0, maxiter = 1, verbose = false, return_stats = true)
        raw = solve(
            adaptive,
            model;
            p0,
            maxiter = 1,
            normalized_objective = false,
            verbose = false,
            return_stats = true,
        )
        @test solver_info(scaled).inner.tolerances ≈ solver_info(raw).inner.tolerances
        @test solver_info(scaled).damping_reduction_threshold ≈ T(0.8)
        stopped =
            solve(adaptive, model; p0, maxiter = 0, verbose = false, return_stats = true)
        @test isempty(solver_info(stopped).inner.iterations)
        @test solver_info(stopped).inner.total_iterations == 0
        # Conversion/scaling does not mutate the caller's residual storage.
        a = T[1, 2]
        @test TensorKitchen._lm_scaled_vector(a, T, one(T)) === a
        @test TensorKitchen._lm_scaled_vector(a, T, T(2)) == T[2, 4]
        @test a == T[1, 2]
    end
end

@testset "SymCPD explicit residual and LM" begin
    x = normalize([1.0, 0.2, -0.1])
    A = reshape(kron(x, kron(x, x)), 3, 3, 3)
    p0 = (([0.7], copy(x)),)
    component = SymmetricRankOne(3, 3)
    dense = JoinModel(component, 1, DenseSymmetricTarget(A))
    compressed = JoinModel(
        component,
        1,
        CompressedSymmetricTarget(compress_symmetric_tensor(A), 3, 3),
    )
    for model in (dense, compressed)
        p = TensorKitchen.join_solver_point(manifold(model), p0)
        r = residual(model, p)
        @test sum(abs2, r) / 2 ≈ cost(model, p)
        @test norm(manifold(model), p, adjoint_action(model, p, r) - rgrad(model, p)) <
              1e-12
    end
    result = symcpd(
        A,
        1;
        p0,
        solver = LMSolver(diagnostics = true),
        maxiter = 3,
        verbose = false,
    )
    @test cost(result) < cost(dense, p0)
    @test solver_info(result).operator_calls.differential > 0
    result32 = symcpd(
        Float32.(A),
        1;
        p0 = ((Float32[0.7], Float32.(x)),),
        solver = LMSolver(),
        maxiter = 3,
        verbose = false,
    )
    @test cost(result32) < 0.045f0
    @test solver_info(result32).uses_lm_subproblem_adapter
    gn_inner = InnerSolveOptions(tolerance = RelativeResidualTolerance(1e-3), maxiter = 1)
    gn_result =
        symcpd(A, 1; p0, solver = :gn_cg, inner = gn_inner, maxiter = 1, verbose = false)
    gn_info = solver_info(gn_result)
    @test gn_info.inner.tolerance_policy === gn_inner.tolerance
    @test gn_info.inner.max_iterations == 1
    @test gn_info.inner.total_iterations == sum(gn_info.inner.iterations)
    @test gn_info.inner.failed_count == count(!, gn_info.inner.converged)
    functional = JoinModel(
        component,
        1,
        FunctionalSymmetricTarget(
            3,
            3,
            1.0;
            evaluate = y -> dot(x, y)^3,
            contract = y -> dot(x, y)^2 .* x,
        ),
    )
    @test_throws ArgumentError residual(functional, p0)
end
