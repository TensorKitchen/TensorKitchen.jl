if Base.VERSION >= v"1.9"
    using PrecompileTools: @compile_workload, @setup_workload

    @setup_workload begin
        rng = MersenneTwister(7)
        A = randn(rng, 4, 3, 2)
        B = abs.(A)
        A_med = randn(rng, 6, 5, 4)

        model = JoinModel(A, 2; geometry = :canonical)
        p0 = _solver_point(
            manifold(model),
            initial_point(model, :random; verbose = false),
        )
        model_med = JoinModel(A_med, 2; geometry = :canonical)
        p0_med = _solver_point(
            manifold(model_med),
            initial_point(model_med, :random; verbose = false),
        )

        nn_model = JoinModel(B, 2; nonnegative = true, geometry = :softplus_metric)
        p0_nn = _solver_point(
            manifold(nn_model),
            initial_point(nn_model, :random; verbose = false),
        )

        # The first LM call is dominated by JIT compilation in the operator-action
        # path. Precompiling a minimal canonical and softplus NNCP workload moves
        # that cost to package precompile time instead of first user execution.
        @compile_workload begin
            # Direct LM solve path on an already-prepared canonical CP model.
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
            solve(
                LMSolver(),
                model_med;
                init = :random,
                p0 = p0_med,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
                return_stats = true,
            )
            cpd(
                A,
                2;
                solver = :lm,
                init = :random,
                p0 = p0,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )
            cpd(
                A_med,
                2;
                solver = :lm,
                init = :random,
                p0 = p0_med,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )
            # Frontend LM route with ALS warm start, which is the path users hit
            # most often when they request cpd(...; solver=:lm).
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
            cpd(
                A_med,
                2;
                solver = :lm,
                init = :alswarm,
                warm_steps = 2,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )
            nncpd(
                B,
                2;
                solver = :lm,
                init = :random,
                p0 = p0_nn,
                maxiter = 1,
                tol = 1e-6,
                verbose = false,
            )
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
