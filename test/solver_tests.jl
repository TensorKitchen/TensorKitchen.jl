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
