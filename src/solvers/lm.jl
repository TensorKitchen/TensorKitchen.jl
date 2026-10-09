# solvers/lm.jl — Riemannian Levenberg-Marquardt via Manopt nonlinear least squares
export LMSolver

"""
    LMSolver(; η=0.2, damping_term_min=0.1, β=5.0,
        expect_zero_residual=false, inner=LMInnerOptions(), diagnostics=false,
        damping_reduction_threshold=nothing)

Configure the Riemannian Levenberg--Marquardt solver used for nonlinear
least-squares models. `η` controls step acceptance, `damping_term_min` is the
minimum damping scale, and `β` controls damping updates. Set
`expect_zero_residual=true` only when the model is expected to fit the target
exactly.

The operator-based Manopt path uses its tangent conjugate-residual subproblem
solver with a zero start at every outer iteration. `inner` configures its
accuracy and iteration cap. `diagnostics=true` records residual/JVP/VJP call
counts and elapsed times; inner iteration and residual histories are always
reported. `damping_reduction_threshold=nothing` preserves the policy selected
by `expect_zero_residual`; a finite override enables controlled comparisons.
"""
struct LMSolver{I<:InnerSolveOptions} <: AbstractSecondOrderROSolver
    η::Float64
    damping_term_min::Float64
    β::Float64
    expect_zero_residual::Bool
    inner::I
    diagnostics::Bool
    damping_reduction_threshold::Union{Nothing,Float64}
end

function LMSolver(;
    η::Real = 0.2,
    damping_term_min::Real = 0.1,
    β::Real = 5.0,
    expect_zero_residual::Bool = false,
    inner::InnerSolveOptions = InnerSolveOptions(),
    diagnostics::Bool = false,
    damping_reduction_threshold::Union{Nothing,Real} = nothing,
)
    0 < η < 1 || throw(ArgumentError("η must satisfy 0 < η < 1, got $η"))
    damping_term_min > 0 ||
        throw(ArgumentError("damping_term_min must be > 0, got $damping_term_min"))
    β > 1 || throw(ArgumentError("β must be > 1, got $β"))
    if !isnothing(damping_reduction_threshold)
        (η <= damping_reduction_threshold <= 1 || damping_reduction_threshold == Inf) ||
            throw(
                ArgumentError("Damping reduction threshold must lie in [η, 1] or be Inf."),
            )
    end
    threshold =
        isnothing(damping_reduction_threshold) ? nothing :
        Float64(damping_reduction_threshold)
    return LMSolver(
        Float64(η),
        Float64(damping_term_min),
        Float64(β),
        expect_zero_residual,
        inner,
        diagnostics,
        threshold,
    )
end

LMSolver(η::Real, damping_term_min::Real, β::Real, expect_zero_residual::Bool) =
    LMSolver(; η, damping_term_min, β, expect_zero_residual)

solver_symbol(::LMSolver) = :lm

@inline function _lm_scaling_factor(::Type{T}, normA2, normalized_objective::Bool) where {T}
    return normalized_objective && !isnothing(normA2) && normA2 > 0 ?
           one(T) / sqrt(T(normA2)) : one(T)
end

_lm_raw_residual_vector(model::AbstractDecompositionModel, p) = residual(model, p)

# Identity scaling needs no copy. Fuse conversion with scaling otherwise;
# never mutate model-provided residuals or Manopt's adjoint input.
_lm_scaled_vector(a, ::Type{T}, scale) where {T} =
    eltype(a) === T && scale == one(T) ? a : T.(scale .* a)

function _lm_raw_jacobian_matrix(
    model::AbstractDecompositionModel,
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
)
    T = _scalar_eltype(p)
    ambient_dim = residual_dimension(model)
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
    return (M, p) -> _lm_scaled_vector(_lm_raw_residual_vector(model, p), T, scale)
end

# Cache the fixed target coordinates once per callback, never in the model.
function _lm_residual_function(
    model::JoinModel{S,B},
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {S,T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    target_coordinates = _symcpd_target_coordinates(backend.component, backend.target)
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return (M, p) ->
        _lm_scaled_vector(_symcpd_residual(model, p, target_coordinates), T, scale)
end

function _lm_differential_action_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return (M, p, X) -> _lm_scaled_vector(differential_action(model, p, X), T, scale)
end

function _lm_adjoint_action_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return function (M, p, a)
        return adjoint_action(model, p, _lm_scaled_vector(a, T, scale))
    end
end

function _lm_adjoint_action_function!(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    return function (M, out, p, a)
        return adjoint_action!(out, model, p, _lm_scaled_vector(a, T, scale))
    end
end

function _lm_vector_differential_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool;
    operator_stats = nothing,
    residual_f = _lm_residual_function(model, T, normA2, normalized_objective),
) where {T<:AbstractFloat}
    ambient_dim = residual_dimension(model)
    scale = _lm_scaling_factor(T, normA2, normalized_objective)
    residual_f! = (M, out, p) -> copyto!(out, residual_f(M, p))
    differential_f! = function (M, out, p, X)
        differential_action!(out, model, p, X)
        out .*= scale
        return out
    end
    adjoint_f! = _lm_adjoint_action_function!(model, T, normA2, normalized_objective)
    return Manopt.VectorDifferentialFunction(
        _measure_lm(residual_f!, operator_stats, 1),
        _measure_lm(differential_f!, operator_stats, 2),
        _measure_lm(adjoint_f!, operator_stats, 3),
        ambient_dim;
        evaluation = Manopt.InplaceEvaluation(),
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
    inner::InnerSolveOptions = InnerSolveOptions(),
    diagnostics::Bool = false,
    damping_reduction_threshold::Union{Nothing,Real} = nothing,
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
    p0_local = _independent_solver_point(setup.p0)
    T = setup.T
    η_T = T(η)
    damping_term_min_T = T(damping_term_min)
    β_T = T(β)
    reduction_threshold =
        isnothing(damping_reduction_threshold) ? (expect_zero_residual ? η_T : T(Inf)) :
        T(damping_reduction_threshold)
    damping_policy = DampingPolicy(
        initial = damping_term_min_T,
        minimum = damping_term_min_T,
        increase_factor = β_T,
        reduction_factor = inv(β_T),
        acceptance_threshold = η_T,
        increase_threshold = η_T,
        reduction_threshold = reduction_threshold,
    )
    operator_stats = diagnostics ? _LMOperatorStats() : nothing
    inner_trace = _LMInnerTrace(T, diagnostics)
    residual_f = _lm_residual_function(model, T, normA2, setup.uses_relative_objective)
    vdf = _lm_vector_differential_function(
        model,
        T,
        normA2,
        setup.uses_relative_objective;
        operator_stats,
        residual_f,
    )
    initial_residual = _measure_lm(residual_f, operator_stats, 1)
    initial_residual_values = copy(initial_residual(M, p0_local))
    nlso = _lm_nonlinear_least_squares_objective(vdf, initial_residual_values)
    initial_jacobian_matrices = fill(nothing, 1)
    sub_objective = Manopt.construct_lm_subobjective(
        false,
        nlso,
        damping_term_min_T,
        T(1.0e-6),
        :Strict,
        initial_residual_values,
        initial_jacobian_matrices,
    )
    M_subproblem = _lm_subproblem_manifold(M)
    sub_problem, sub_state =
        _lm_cr_state(M, p0_local, sub_objective, inner, setup.objective_scale, inner_trace)
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
        candidate_acceptance_threshold = T(damping_policy.acceptance_threshold),
        damping_increase_factor = T(damping_policy.increase_factor),
        damping_increase_threshold = T(damping_policy.increase_threshold),
        damping_reduction_threshold = T(damping_policy.reduction_threshold),
        damping_reduction_factor = T(damping_policy.reduction_factor),
        damping_term_min = T(damping_policy.minimum),
        damping_term_max = T(damping_policy.maximum),
        initial_damping_term = T(damping_policy.initial),
        scaling_threshold = T(1.0e-6),
        minimum_acceptable_model_improvement = eps(T),
        use_unified_basis = false,
        sub_objective = sub_objective,
        sub_problem = sub_problem,
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
        solver_info_extra = merge(
            (
                η = Float64(η),
                damping_term_min = Float64(damping_term_min),
                β = Float64(β),
                expect_zero_residual = expect_zero_residual,
                uses_operator_jacobian = true,
                uses_direct_adjoint_action = true,
                uses_coordinate_linear_solver = false,
                uses_lm_subproblem_adapter = M_subproblem !== M,
                uses_vector_transport = !isnothing(vector_transport_method),
                damping_reduction_threshold = reduction_threshold,
                damping_policy = damping_policy,
                diagnostics = diagnostics,
            ),
            _lm_inner_info(
                inner_trace,
                inner,
                _inner_maxiter(inner, manifold_dimension(M)),
            ),
            _lm_operator_info(operator_stats),
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
        inner = solver.inner,
        diagnostics = solver.diagnostics,
        damping_reduction_threshold = solver.damping_reduction_threshold,
        grad_tol,
        normalized_objective,
    )
end
