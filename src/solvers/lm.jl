# solvers/lm.jl — Riemannian Levenberg-Marquardt via Manopt nonlinear least squares
export LMSolver

struct LMSolver <: AbstractSecondOrderROSolver
    η::Float64
    damping_term_min::Float64
    β::Float64
    expect_zero_residual::Bool
    linear_subsolver::Any
end

function LMSolver(;
    η::Real = 0.2,
    damping_term_min::Real = 0.1,
    β::Real = 5.0,
    expect_zero_residual::Bool = false,
    linear_subsolver = Manopt.default_lm_lin_solve!,
)
    0 < η < 1 || throw(ArgumentError("η must satisfy 0 < η < 1, got $η"))
    damping_term_min > 0 ||
        throw(ArgumentError("damping_term_min must be > 0, got $damping_term_min"))
    β > 1 || throw(ArgumentError("β must be > 1, got $β"))
    return LMSolver(
        Float64(η),
        Float64(damping_term_min),
        Float64(β),
        expect_zero_residual,
        linear_subsolver,
    )
end

solver_symbol(::LMSolver) = :lm

@inline function _lm_scaling_factor(::Type{T}, normA2, normalized_objective::Bool) where {T}
    return normalized_objective && !isnothing(normA2) && normA2 > 0 ?
           one(T) / sqrt(T(normA2)) : one(T)
end

_lm_raw_residual_vector(model::AbstractDecompositionModel, p) = residual(model, p)

function _lm_raw_jacobian_matrix(
    model::AbstractDecompositionModel,
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
)
    T = _scalar_eltype(p)
    ambient_dim = length(tensor(model))
    d = manifold_dimension(M)
    J = Matrix{T}(undef, ambient_dim, d)
    coeff = zeros(T, d)
    column = Vector{T}(undef, ambient_dim)
    @inbounds for j = 1:d
        fill!(coeff, zero(T))
        coeff[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p, coeff, basis)
        differential_action!(column, model, p, Xj)
        J[:, j] .= column
    end
    return J
end

function _lm_residual_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return (M, p) -> scale .* _lm_raw_residual_vector(model, p)
end

function _lm_differential_action_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return (M, p, X) -> scale .* differential_action(model, p, X)
end

function _lm_adjoint_action_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return (M, p, a) -> adjoint_action(model, p, scale .* a)
end

function _lm_vector_differential_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    ambient_dim = length(tensor(model))
    residual_f = _lm_residual_function(model, T, normA2, normalized_objective)
    differential_f =
        _lm_differential_action_function(model, T, normA2, normalized_objective)
    adjoint_f = _lm_adjoint_action_function(model, T, normA2, normalized_objective)
    return Manopt.VectorDifferentialFunction(
        residual_f,
        differential_f,
        adjoint_f,
        ambient_dim;
        evaluation = Manopt.AllocatingEvaluation(),
        function_type = Manopt.FunctionVectorialType(),
        jacobian_type = Manopt.FunctionVectorialType(),
        adjoint_jacobian_type = Manopt.FunctionVectorialType(),
    )
end

function solve_lm(
    model,
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 1000,
    tol::Real = 1e-6,
    verbose::Bool = true,
    return_stats::Bool = false,
    normA2 = nothing,
    model_grad = nothing,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback = nothing,
    diagnostics_recorder = nothing,
    iteration_callbacks = (),
    η::Real = 0.2,
    damping_term_min::Real = 0.1,
    β::Real = 5.0,
    expect_zero_residual::Bool = false,
    linear_subsolver = Manopt.default_lm_lin_solve!,
    grad_tol = nothing,
    normalized_objective::Bool = true,
)
    setup = _prepare_manopt_solver_functions(
        model_cost,
        model_egrad,
        M,
        p0;
        normA2,
        model_grad,
        tol,
        grad_tol,
        normalized_objective,
    )
    p0_local = setup.p0
    T = setup.T
    vdf = _lm_vector_differential_function(model, T, normA2, setup.uses_relative_objective)
    initial_residual_values = residual(model, p0_local)
    scale = _lm_scaling_factor(T, normA2, setup.uses_relative_objective)
    if scale != one(T)
        initial_residual_values .*= scale
    end
    nlso = Manopt.ManifoldNonlinearLeastSquaresObjective(
        vdf,
        Manopt.ComponentwiseRobustifierFunction(Manopt.IdentityRobustifier()),
    )
    initial_jacobian_matrices = fill(nothing, 1)
    sub_objective = Manopt.construct_lm_subobjective(
        false,
        nlso,
        damping_term_min,
        1.0e-6,
        :Strict,
        initial_residual_values,
        initial_jacobian_matrices,
    )
    sub_state = Manopt.ConjugateResidualState(
        TangentSpace(M, p0_local),
        sub_objective;
        stopping_criterion = StopAfterIteration(max(20 * manifold_dimension(M), 200)) |
                             StopWhenGradientNormLess(T(1e-16)),
    )
    retraction_method = _solver_retraction_method(M, p0_local)
    stopping = StopWhenAny(
        StopAfterIteration(maxiter),
        StopWhenGradientNormLess(setup.grad_stop_tol),
        StopWhenStepsizeLess(T(tol)),
        StopWhenCostRelChangeAndGradientLess(T(tol), setup.dual_grad_tol),
    )
    callbacks = _manopt_callbacks(
        n -> make_manopt_family_progress(
            n;
            enabled = verbose,
            phase = :refinement,
            method = "LM",
            dt = 0.2,
        ),
        maxiter,
        verbose,
        setup.solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )
    state = Manopt.LevenbergMarquardt(
        M,
        nlso,
        p0_local;
        retraction_method = retraction_method,
        stopping_criterion = stopping,
        initial_residual_values = initial_residual_values,
        candidate_acceptance_threshold = η,
        damping_increase_factor = β,
        damping_increase_threshold = η,
        damping_reduction_threshold = expect_zero_residual ? η : Inf,
        damping_reduction_factor = inv(T(β)),
        damping_term_min = damping_term_min,
        initial_damping_term = damping_term_min,
        use_unified_basis = false,
        sub_objective = sub_objective,
        sub_state = sub_state,
        debug = callbacks.debug_actions,
        callbacks = callbacks.solver_callbacks,
        return_state = true,
    )

    return _manopt_finish_result(
        _tk_get_solver_result(state),
        state,
        callbacks.progress,
        diagnostics_recorder,
        setup.solver_cost,
        setup.solver_grad,
        M,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :lm,
        tiny_grad_tol = setup.dual_grad_tol,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
        solver_info_extra = (
            η = Float64(η),
            damping_term_min = Float64(damping_term_min),
            β = Float64(β),
            expect_zero_residual = expect_zero_residual,
            uses_operator_jacobian = true,
            uses_direct_adjoint_action = true,
            uses_coordinate_linear_solver = false,
            uses_user_linear_subsolver = linear_subsolver !== Manopt.default_lm_lin_solve!,
            uses_vector_transport = !isnothing(vector_transport_method),
        ),
    )
end

function run_second_order_solver(
    solver::LMSolver,
    setup;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
    return_stats::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback,
    diagnostics_recorder,
    iteration_callbacks,
    grad_tol = nothing,
    normalized_objective::Bool = true,
)
    return solve_lm(
        setup.model,
        setup.model_cost,
        setup.model_egrad,
        setup.M,
        setup.p0;
        maxiter,
        tol,
        verbose,
        return_stats,
        normA2 = setup.normA2,
        model_grad = setup.model_grad,
        vector_transport_method,
        post_step_callback,
        diagnostics_recorder,
        iteration_callbacks,
        η = solver.η,
        damping_term_min = solver.damping_term_min,
        β = solver.β,
        expect_zero_residual = solver.expect_zero_residual,
        linear_subsolver = solver.linear_subsolver,
        grad_tol,
        normalized_objective,
    )
end
