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

@inline function _lm_objective_scale(
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T}
    return normalized_objective && !isnothing(normA2) && normA2 > 0 ?
           one(T) / sqrt(T(normA2)) : one(T)
end

function _ambient_tangent_vector!(out::AbstractVector, M, p, X)
    emb = ManifoldsBase.embed(M, p, X)
    length(emb) == length(out) || throw(
        DimensionMismatch(
            "Tangent embedding length $(length(emb)) does not match output length $(length(out)).",
        ),
    )
    copyto!(out, vec(emb))
    return out
end

function _ambient_tangent_vector!(out::AbstractVector, M::Manifolds.Segre, p, X)
    copyto!(out, _segre_tangent_tensorvec(p, X))
    return out
end

function _join_tangent_ambient_vector!(
    out::AbstractVector,
    backend::Union{JoinBackend,BTDBackend},
    p,
    X,
)
    parts = point_parts(p)
    xparts = point_parts(X)
    _check_parts_len(parts, backend.r, "_join_tangent_ambient_vector!")
    _check_parts_len(xparts, backend.r, "_join_tangent_ambient_vector!")
    fill!(out, zero(eltype(out)))
    @inbounds for k = 1:backend.r
        _ambient_tangent_vector!(
            backend.component_bufs[k],
            backend.manifolds[k],
            parts[k],
            xparts[k],
        )
        out .+= backend.component_bufs[k]
    end
    return out
end

function _lm_raw_residual_vector(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
    p,
)
    return copy(_join_residual!(model.backend, p))
end

function _lm_raw_jacobian_matrix(
    model::JoinModel{<:AbstractFloat,<:Union{JoinBackend,BTDBackend}},
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
)
    p_work = _solver_point(M, p)
    T = _scalar_eltype(p_work)
    ambient_dim = length(tensor(model))
    d = manifold_dimension(M)
    J = Matrix{T}(undef, ambient_dim, d)
    coeff = zeros(T, d)
    column = similar(model.backend.work_rec, T, ambient_dim)
    @inbounds for j = 1:d
        fill!(coeff, zero(T))
        coeff[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p_work, coeff, basis)
        _join_tangent_ambient_vector!(column, model.backend, p_work, Xj)
        J[:, j] .= column
    end
    return J
end

@inline function _cp_scaled_tangent_factors(λ, U, λ̇, U̇, ::Val{:identity})
    return λ, U, λ̇, U̇
end

@inline function _cp_scaled_tangent_factors(λ̃, Ũ, λ̇̃, U̇̃, ::Val{:square})
    λ = λ̃ .^ 2
    U = [Ũ[m] .^ 2 for m in eachindex(Ũ)]
    λ̇ = 2 .* λ̃ .* λ̇̃
    U̇ = [2 .* Ũ[m] .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end

@inline function _cp_scaled_tangent_factors(λ̃, Ũ, λ̇̃, U̇̃, ::Val{:softplus})
    λ = _softplus_value.(λ̃)
    U = [_softplus_value.(Ũ[m]) for m in eachindex(Ũ)]
    λ̇ = _softplus_derivative.(λ̃) .* λ̇̃
    U̇ = [_softplus_derivative.(Ũ[m]) .* U̇̃[m] for m in eachindex(Ũ)]
    return λ, U, λ̇, U̇
end

function _cp_rankr_tangent_tensorvec!(
    out::AbstractVector{T},
    λ::AbstractVector{T},
    U::Vector{<:AbstractMatrix{T}},
    λ̇::AbstractVector{T},
    U̇::Vector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    r = length(λ)
    @inbounds for k = 1:r
        comp = ([λ[k]], [Vector(@view U[m][:, k]) for m in eachindex(U)]...)
        xcomp = ([λ̇[k]], [Vector(@view U̇[m][:, k]) for m in eachindex(U̇)]...)
        out .+= _segre_tangent_tensorvec(comp, xcomp)
    end
    return out
end

function _cp_rank1_tangent_tensorvec!(
    out::AbstractVector{T},
    λ::T,
    U::Vector{<:AbstractVector{T}},
    λ̇::T,
    U̇::Vector{<:AbstractVector{T}},
) where {T<:AbstractFloat}
    comp = ([λ], U...)
    xcomp = ([λ̇], U̇...)
    copyto!(out, _segre_tangent_tensorvec(comp, xcomp))
    return out
end

function _lm_raw_residual_vector(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p)
    return _lm_raw_residual_vector(cpd_model(model), p)
end

function _lm_raw_jacobian_matrix(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
)
    return _lm_raw_jacobian_matrix(cpd_model(model), M, p; basis)
end

function _lm_raw_residual_vector(model::Rank1CPDModel{T}, p) where {T<:AbstractFloat}
    return vec(embed_point(model, p)) .- vec(model.A)
end

function _lm_raw_jacobian_matrix(
    model::Rank1CPDModel{T},
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
) where {T<:AbstractFloat}
    p_work = _solver_point(M, p)
    ambient_dim = length(model.A)
    d = manifold_dimension(M)
    J = Matrix{T}(undef, ambient_dim, d)
    coeff = zeros(T, d)
    λp, Up = unpack_point_rank1(p_work, model.dims)
    column = Vector{T}(undef, ambient_dim)
    @inbounds for j = 1:d
        fill!(coeff, zero(T))
        coeff[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p_work, coeff, basis)
        if model.nonnegative
            λ̇p, U̇p = unpack_point_rank1(Xj, model.dims)
            kind = _rank1_uses_softplus_metric(model.M) ? Val(:softplus) : Val(:square)
            λ, U, λ̇, U̇ = _cp_scaled_tangent_factors(
                [λp],
                [reshape(u, :, 1) for u in Up],
                [λ̇p],
                [reshape(u, :, 1) for u in U̇p],
                kind,
            )
            U_vec = [Vector(@view U[m][:, 1]) for m in eachindex(U)]
            U̇_vec = [Vector(@view U̇[m][:, 1]) for m in eachindex(U̇)]
            _cp_rank1_tangent_tensorvec!(column, λ[1], U_vec, λ̇[1], U̇_vec)
        else
            copyto!(column, vec(ManifoldsBase.embed(M, p_work, Xj)))
        end
        J[:, j] .= column
    end
    return J
end

function _lm_raw_residual_vector(model::RankRCPDModel{T}, p) where {T<:AbstractFloat}
    return vec(embed_point(model, p)) .- vec(model.A)
end

function _lm_raw_jacobian_matrix(
    model::RankRCPDModel{T},
    M,
    p;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
) where {T<:AbstractFloat}
    p_work = _solver_point(M, p)
    ambient_dim = length(model.A)
    d = manifold_dimension(M)
    J = Matrix{T}(undef, ambient_dim, d)
    coeff = zeros(T, d)
    column = Vector{T}(undef, ambient_dim)
    if model.geometry == :native && !model.nonnegative
        pparts = point_parts(p_work)
        @inbounds for j = 1:d
            fill!(coeff, zero(T))
            coeff[j] = one(T)
            Xj = ManifoldsBase.get_vector(M, p_work, coeff, basis)
            xparts = point_parts(Xj)
            fill!(column, zero(T))
            for k = 1:model.r
                column .+= _segre_tangent_tensorvec(pparts[k], xparts[k])
            end
            J[:, j] .= column
        end
        return J
    end

    λp, Up =
        model.nonnegative ? unpack_point_rankr(p_work, model.dims, model.r) :
        unpack_rankr_canonical(p_work, model.dims, model.r)
    kind =
        model.nonnegative ?
        (model.geometry == :softplus_metric ? Val(:softplus) : Val(:square)) :
        Val(:identity)
    @inbounds for j = 1:d
        fill!(coeff, zero(T))
        coeff[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p_work, coeff, basis)
        λ̇p, U̇p =
            model.nonnegative ? unpack_point_rankr(Xj, model.dims, model.r) :
            unpack_rankr_canonical(Xj, model.dims, model.r)
        λ, U, λ̇, U̇ = _cp_scaled_tangent_factors(λp, Up, λ̇p, U̇p, kind)
        _cp_rankr_tangent_tensorvec!(column, λ, U, λ̇, U̇)
        J[:, j] .= column
    end
    return J
end

function _lm_raw_residual_vector(model::AbstractDecompositionModel, p)
    throw(ArgumentError("LMSolver residual is not implemented for model $(typeof(model))."))
end

function _lm_raw_jacobian_matrix(model::AbstractDecompositionModel, M, p; basis)
    throw(ArgumentError("LMSolver Jacobian is not implemented for model $(typeof(model))."))
end

function _lm_residual_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool,
) where {T<:AbstractFloat}
    scale = _lm_objective_scale(T, normA2, normalized_objective)
    return (M, p) -> scale .* _lm_raw_residual_vector(model, p)
end

function _lm_jacobian_function(
    model::AbstractDecompositionModel,
    ::Type{T},
    normA2,
    normalized_objective::Bool;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
) where {T<:AbstractFloat}
    scale = _lm_objective_scale(T, normA2, normalized_objective)
    return (M, p) -> scale .* _lm_raw_jacobian_matrix(model, M, p; basis)
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
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    residual = _lm_residual_function(model, T, normA2, setup.uses_relative_objective)
    jacobian = _lm_jacobian_function(model, T, normA2, setup.uses_relative_objective; basis)
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
        residual,
        jacobian,
        p0_local;
        evaluation = Manopt.AllocatingEvaluation(),
        function_type = Manopt.FunctionVectorialType(),
        jacobian_type = Manopt.CoordinateVectorialType(basis),
        retraction_method = retraction_method,
        stopping_criterion = stopping,
        η = η,
        damping_term_min = damping_term_min,
        β = β,
        expect_zero_residual = expect_zero_residual,
        linear_subsolver! = linear_subsolver,
        debug = callbacks.debug_actions,
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
