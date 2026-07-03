# join/cpd_backend.jl — CPD-specialized JoinModel backend

function JoinModel(
    A::AbstractArray{T,N},
    r::Int;
    geometry::Symbol = :canonical,
    scale_by_lambda::Bool = true,
    lambda_eps::Real = 1e-10,
    nonnegative::Bool = false,
    use_pullback_metric::Bool = false,
    pullback_eps::Real = 1e-8,
) where {T<:AbstractFloat,N}
    geometry_eff = _is_native_rankr_geometry(geometry) ? :native : geometry
    inner =
        r == 1 ?
        Rank1CPDModel(
            A;
            scale_by_lambda = scale_by_lambda,
            lambda_eps = lambda_eps,
            nonnegative = nonnegative,
            use_pullback_metric = (geometry_eff == :squaring_metric) || use_pullback_metric,
            use_softplus_metric = (geometry_eff == :softplus_metric),
            pullback_eps = pullback_eps,
        ) :
        RankRCPDModel(
            A,
            r;
            geometry = geometry_eff,
            scale_by_lambda = scale_by_lambda,
            lambda_eps = lambda_eps,
            nonnegative = nonnegative,
            use_pullback_metric = (geometry_eff == :squaring_metric) || use_pullback_metric,
            pullback_eps = pullback_eps,
        )
    b = CPDBackend(inner)
    return JoinModel{T,typeof(b)}(b)
end

@inline cpd_model(model::JoinModel{<:AbstractFloat,<:CPDBackend}) = model.backend.model

manifold(model::JoinModel{<:AbstractFloat,<:CPDBackend}) = manifold(cpd_model(model))
tensor(model::JoinModel{<:AbstractFloat,<:CPDBackend}) = tensor(cpd_model(model))
initial_point(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    init::Union{Symbol,BuiltinInitializer};
    kwargs...,
) = initial_point(cpd_model(model), init; kwargs...)
initial_point(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    init::ALSWarmStartInit;
    kwargs...,
) = initial_point(cpd_model(model), init; kwargs...)
initial_point(model::JoinModel{<:AbstractFloat,<:CPDBackend}, init::PointInit; kwargs...) =
    initial_point(cpd_model(model), init; kwargs...)
initial_point(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    init::FunctionInit;
    kwargs...,
) = initial_point(cpd_model(model), init; kwargs...)
cost(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p) = cost(cpd_model(model), p)
egrad(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p) = egrad(cpd_model(model), p)
residual(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p) = residual(cpd_model(model), p)
differential_action!(
    out::AbstractVector,
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    p,
    X,
) = differential_action!(out, cpd_model(model), p, X)
adjoint_action(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    p,
    a::AbstractVector;
    kwargs...,
) = adjoint_action(cpd_model(model), p, a; kwargs...)
supports_rgrad(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    supports_rgrad(cpd_model(model))
rgrad(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p) = rgrad(cpd_model(model), p)
unwrap_model(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    unwrap_model(cpd_model(model))
supports_egrad_project(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    supports_egrad_project(cpd_model(model))
supports_exact_native(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    supports_exact_native(cpd_model(model))
supports_exact_join_basis(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    supports_exact_join_basis(cpd_model(model))
supports_normalization_policy(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    policy::AbstractNormalizationPolicy,
) = supports_normalization_policy(cpd_model(model), policy)
cpd_point(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p) =
    cpd_point(cpd_model(model), p)
pack_cpd_point(model::JoinModel{<:AbstractFloat,<:CPDBackend}, point::CPDPoint) =
    pack_cpd_point(cpd_model(model), point)
post_step!(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p; kwargs...) =
    post_step!(cpd_model(model), p; kwargs...)
model_exact_native_function(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    model_exact_native_function(cpd_model(model))
model_exact_join_basis_function(model::JoinModel{<:AbstractFloat,<:CPDBackend}) =
    model_exact_join_basis_function(cpd_model(model))
cp_als_data(model::JoinModel{<:AbstractFloat,<:CPDBackend}) = cp_als_data(cpd_model(model))

const _CP_ALS_FAMILY_SOLVERS = (:cp_als, :rals, :rals_mix)

function _public_cpd_factors(m, λ, U)
    if m.nonnegative
        U_norm, λ_norm = normalize_components(U, λ, SeparateLambdaNormalization())
        return λ_norm, U_norm
    end
    return λ, U
end

@inline _cpd_solver_point(m, p, solver_sym::Symbol) = cpd_point(m, p)

function _cpd_solver_point(
    m::RankRCPDModel{T},
    p,
    solver_sym::Symbol,
) where {T<:AbstractFloat}
    if m.nonnegative && (solver_sym in _CP_ALS_FAMILY_SOLVERS)
        λ, U = unpack_rankr_canonical(p, m.dims, m.r)
        return CPDPoint(λ, U)
    end
    return cpd_point(m, p)
end

function _cpd_result(model::JoinModel{<:AbstractFloat,<:CPDBackend}, result, dims, r)
    m = cpd_model(model)
    solver_sym = _result_solver_symbol(solver(result))
    si = _result_solver_info(result)
    q = _cpd_solver_point(m, point(result), solver_sym)
    λ_raw = lambda(q)
    U_raw = factors(q)

    rel_err = rel_error(result)
    if !isfinite(rel_err)
        Xhat = reconstruct_cpd_rankr(λ_raw, U_raw)
        rel_err = rel_error(m.A, Xhat)
    end
    λ_pub, U_pub = _public_cpd_factors(m, λ_raw, U_raw)
    return CPDResult(
        λ_pub,
        U_pub,
        cost(result),
        rel_err,
        grad_norm(result),
        iterations(result),
        converged(result),
        solver_sym,
        si,
    )
end

function extract_components(model::JoinModel{<:AbstractFloat,<:CPDBackend}, p)
    return _extract_cpd_components(cpd_model(model), p)
end

function _extract_cpd_components(m::Union{Rank1CPDModel,RankRCPDModel}, p)
    q = cpd_point(m, p)
    comps = components_from_factors(lambda(q), factors(q))
    return [CPDComponent(pack_point_rank1(c.λ, c.vectors), c) for c in comps]
end

function _extract_cpd_components(m, p)
    throw(
        ArgumentError(
            "Unsupported CPD backend model $(typeof(m)) for component extraction.",
        ),
    )
end
