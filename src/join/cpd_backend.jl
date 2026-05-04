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
            use_pullback_metric = use_pullback_metric,
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

@inline function _decode_nonnegative_cpd(m, λ̃, Ũ)
    use_softplus =
        hasproperty(m, :geometry) ? (m.geometry == :softplus_metric) :
        (hasproperty(m, :M) && _rank1_uses_softplus_metric(m.M))
    if use_softplus
        return _softplus_value.(λ̃), [_softplus_value.(Ũ[j]) for j = 1:length(Ũ)]
    end
    return λ̃ .^ 2, [Ũ[j] .^ 2 for j = 1:length(Ũ)]
end

function _cpd_result(model::JoinModel{<:AbstractFloat,<:CPDBackend}, result, dims, r)
    m = cpd_model(model)
    solver_sym = solver(result) isa Symbol ? solver(result) : :unknown
    si = hasproperty(result, :solver_info) ? solver_info(result) : (;)
    als_family = solver_sym in _CP_ALS_FAMILY_SOLVERS

    if r == 1
        λ̃, U_vec = unpack_point_rank1(point(result), dims)
        if m.nonnegative && !als_family
            λ_vec, U_sq = _decode_nonnegative_cpd(m, [λ̃], U_vec)
            λ = λ_vec[1]
        else
            λ = λ̃
            U_sq = U_vec
        end
        λ_pub, U_pub = _public_cpd_factors(m, [λ], [reshape(u, :, 1) for u in U_sq])
        return CPDResult(
            λ_pub,
            U_pub,
            cost(result),
            rel_error(result),
            grad_norm(result),
            iterations(result),
            converged(result),
            solver_sym,
            si,
        )
    end

    comps = if m.nonnegative && !als_family
        λ̃, Ũ = unpack_point_rankr(point(result), dims, r)
        λ, U = _decode_nonnegative_cpd(m, λ̃, Ũ)
        components_from_factors(λ, U)
    else
        unpack_point_rankr_components(point(result), dims, r)
    end

    rel_err = rel_error(result)
    if !isfinite(rel_err)
        Xhat = reconstruct_cpd_rankr([c.λ for c in comps], factors_from_components(comps))
        rel_err = rel_error(m.A, Xhat)
    end
    λ_pub, U_pub =
        _public_cpd_factors(m, [c.λ for c in comps], factors_from_components(comps))
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
    m = cpd_model(model)
    if m isa RankRCPDModel
        comps =
            m.nonnegative ? begin
                λ̃, Ũ = unpack_point_rankr(p, m.dims, m.r)
                λ, U = _decode_nonnegative_cpd(m, λ̃, Ũ)
                components_from_factors(λ, U)
            end : unpack_point_rankr_components(p, m.dims, m.r)
        return [
            (
                kind = :Segre,
                point = p,
                weights = [c.λ],
                factors = [reshape(c.vectors[j], :, 1) for j = 1:length(c.vectors)],
                tensor = reconstruct_cp_rank1(c.λ, c.vectors),
            ) for c in comps
        ]
    elseif m isa Rank1CPDModel
        λ, U = unpack_point_rank1(p, m.dims)
        if m.nonnegative
            if _rank1_uses_softplus_metric(m.M)
                λ = _softplus_value(λ)
                U = [_softplus_value.(u) for u in U]
            else
                λ = λ^2
                U = [u .^ 2 for u in U]
            end
        end
        return [(
            kind = :Segre,
            point = p,
            weights = [λ],
            factors = [reshape(U[j], :, 1) for j = 1:length(U)],
            tensor = reconstruct_cp_rank1(λ, U),
        )]
    end
    throw(
        ArgumentError(
            "Unsupported CPD backend model $(typeof(m)) for component extraction.",
        ),
    )
end
