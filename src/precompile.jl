if Base.VERSION >= v"1.9"
    using PrecompileTools: @compile_workload, @setup_workload

    @setup_workload begin
        rng = MersenneTwister(7)
        A = randn(rng, 4, 3, 2)
        B = abs.(A)

        model = JoinModel(A, 2; geometry = :canonical)
        p0 = _solver_point(manifold(model), initial_point(model, :random; verbose = false))

        # Keep this workload intentionally small: we want to cover the operator LM
        # path and the public CP/NNCP frontends without forcing every user to pay
        # a very large package precompile cost up front.
        @compile_workload begin
            # Direct LM solve on a canonical CP model covers the core operator
            # least-squares path without routing through the public frontend.
            solve(
                LMSolver(),
                model;
                init = :random,
                p0 = p0,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
                return_stats = true,
            )

            # Warm-started CPD is the most common public LM entry point.
            cpd(
                A,
                2;
                solver = :lm,
                init = :alswarm,
                warm_steps = 2,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )

            # Nonnegative LM still goes through a distinct parameterization path.
            nncpd(
                B,
                2;
                solver = :lm,
                init = :alswarm,
                warm_steps = 2,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )
        end
    end
end
