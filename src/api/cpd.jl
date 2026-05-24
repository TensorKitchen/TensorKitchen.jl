# api/cpd.jl — user-facing CP decomposition entry points
export cpd

function _pullback_eps_value(::Type{T}, pullback_eps) where {T<:AbstractFloat}
    ε = T(pullback_eps)
    isfinite(ε) && ε > zero(T) ||
        throw(ArgumentError("pullback_eps must be finite and positive, got $pullback_eps."))
    return ε
end

function _merge_res_solver_info(res, patch::NamedTuple)
    si0 = hasproperty(res, :solver_info) ? solver_info(res) : (;)
    return (
        point = point(res),
        cost = cost(res),
        rel_error = rel_error(res),
        grad_norm = grad_norm(res),
        iterations = iterations(res),
        converged = converged(res),
        solver = solver(res),
        solver_info = merge(si0, patch),
    )
end

Base.@kwdef mutable struct _CPDComponentTraceHistory
    iterations::Vector{Int} = Int[]
    cost_history::Vector{Float64} = Float64[]
    cost_rel_change_history::Vector{Float64} = Float64[]
    max_component_delta_history::Vector{Float64} = Float64[]
    component_delta_history::Vector{Vector{Float64}} = Vector{Float64}[]
    coordinate_rgrad_energy_history::Vector{Vector{Float64}} = Vector{Float64}[]
    coordinate_rgrad_share_history::Vector{Vector{Float64}} = Vector{Float64}[]
    coordinate_rgrad_top1_share_history::Vector{Float64} = Float64[]
    coordinate_rgrad_top2_share_history::Vector{Float64} = Float64[]
    coordinate_rgrad_top3_share_history::Vector{Float64} = Float64[]
    coordinate_rgrad_effective_components_history::Vector{Float64} = Float64[]
    coordinate_rgrad_argmax_component_history::Vector{Int} = Int[]
    metric_rgrad_energy_history::Vector{Vector{Float64}} = Vector{Float64}[]
    metric_rgrad_share_history::Vector{Vector{Float64}} = Vector{Float64}[]
    metric_rgrad_top1_share_history::Vector{Float64} = Float64[]
    metric_rgrad_top2_share_history::Vector{Float64} = Float64[]
    metric_rgrad_top3_share_history::Vector{Float64} = Float64[]
    metric_rgrad_effective_components_history::Vector{Float64} = Float64[]
    metric_rgrad_argmax_component_history::Vector{Int} = Int[]
    ambient_component_velocity_history::Vector{Vector{Float64}} = Vector{Float64}[]
    ambient_velocity_share_history::Vector{Vector{Float64}} = Vector{Float64}[]
    ambient_velocity_top1_share_history::Vector{Float64} = Float64[]
    ambient_velocity_top2_share_history::Vector{Float64} = Float64[]
    ambient_velocity_top3_share_history::Vector{Float64} = Float64[]
    ambient_velocity_effective_components_history::Vector{Float64} = Float64[]
    ambient_velocity_argmax_component_history::Vector{Int} = Int[]
    rgrad_failed_count::Int = 0
end

Base.@kwdef mutable struct _CPDComponentTraceRecorder{M}
    model::M
    previous::Union{Nothing,CPDPoint} = nothing
    previous_cost::Float64 = NaN
    start_rel_error::Float64 = NaN
    history::_CPDComponentTraceHistory = _CPDComponentTraceHistory()
end

_CPDComponentTraceRecorder(model) = _CPDComponentTraceRecorder(; model)

function _rankone_norm2(λ, U, k::Int)
    val = abs2(λ[k])
    @inbounds for m in eachindex(U)
        val *= sum(abs2, @view U[m][:, k])
    end
    return Float64(val)
end

function _rankone_inner(λa, Ua, λb, Ub, k::Int)
    val = λa[k] * λb[k]
    @inbounds for m in eachindex(Ua)
        val *= dot(@view(Ua[m][:, k]), @view(Ub[m][:, k]))
    end
    return Float64(val)
end

function _cpd_component_deltas(prev::CPDPoint, curr::CPDPoint)
    λ_prev = lambda(prev)
    U_prev = factors(prev)
    λ_curr = lambda(curr)
    U_curr = factors(curr)
    r = length(λ_curr)
    deltas = Vector{Float64}(undef, r)
    @inbounds for k in eachindex(deltas)
        n_prev = _rankone_norm2(λ_prev, U_prev, k)
        n_curr = _rankone_norm2(λ_curr, U_curr, k)
        cross = _rankone_inner(λ_prev, U_prev, λ_curr, U_curr, k)
        delta = sqrt(max(n_prev + n_curr - 2 * cross, 0.0))
        deltas[k] = delta / max(sqrt(max(n_prev, 0.0)), 1.0)
    end
    return deltas
end

function _component_energy_summary(energies::AbstractVector{<:Real})
    r = length(energies)
    if r == 0
        return (
            shares = Float64[],
            top1 = NaN,
            top2 = NaN,
            top3 = NaN,
            effective = NaN,
            argmax_component = 0,
        )
    end
    total = sum(energies)
    if !isfinite(total) || total <= 0
        return (
            shares = fill(NaN, r),
            top1 = NaN,
            top2 = NaN,
            top3 = NaN,
            effective = NaN,
            argmax_component = 0,
        )
    end
    shares = Float64.(energies ./ total)
    ordered = sort(shares; rev = true)
    hhi = sum(abs2, shares)
    return (
        shares = shares,
        top1 = ordered[1],
        top2 = sum(@view ordered[1:min(2, r)]),
        top3 = sum(@view ordered[1:min(3, r)]),
        effective = hhi > 0 ? inv(hhi) : NaN,
        argmax_component = argmax(shares),
    )
end

function _cpd_coordinate_rgrad_energy_from_blocks(gλ, gU)
    r = length(gλ)
    energies = zeros(Float64, r)
    @inbounds for k in eachindex(energies)
        energies[k] += abs2(Float64(gλ[k]))
        for m in eachindex(gU)
            energies[k] += sum(abs2, @view gU[m][:, k])
        end
    end
    return energies
end

function _cpd_component_coordinate_rgrad_energies(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    g,
)
    return _cpd_component_coordinate_rgrad_energies(cpd_model(model), g)
end

function _cpd_component_coordinate_rgrad_energies(model::RankRCPDModel, g)
    gλ, gU = unpack_point_rankr(g, model.dims, model.r)
    return _cpd_coordinate_rgrad_energy_from_blocks(gλ, gU)
end

function _cpd_component_coordinate_rgrad_energies(model::Rank1CPDModel, g)
    gλ, gU = unpack_point_rank1(g, model.dims)
    energy = abs2(Float64(gλ))
    @inbounds for u in gU
        energy += sum(abs2, u)
    end
    return [energy]
end

_cpd_component_coordinate_rgrad_energies(_, _) = Float64[]

function _cpd_product_block_metric_energy(M::ProductManifold, p, g, block_idx::Int)
    factors = M.manifolds
    pp = point_parts(p)
    gp = point_parts(g)
    return Float64(
        ManifoldsBase.inner(
            factors[block_idx],
            pp[block_idx],
            gp[block_idx],
            gp[block_idx],
        ),
    )
end

function _cpd_join_component_metric_energy(M::ProductManifold, p, g, k::Int, d::Int)
    base = (k - 1) * (d + 1)
    energy = 0.0
    @inbounds for b in eachindex(Base.OneTo(d + 1))
        energy += _cpd_product_block_metric_energy(M, p, g, base + b)
    end
    return energy
end

function _cpd_native_component_metric_energy(M::ProductManifold, p, g, k::Int)
    factors = M.manifolds
    pp = point_parts(p)
    gp = point_parts(g)
    return Float64(ManifoldsBase.inner(factors[k], pp[k], gp[k], gp[k]))
end

function _cpd_canonical_component_metric_energy(M::ProductManifold, p, g, k::Int, d::Int)
    factors = M.manifolds
    pp = point_parts(p)
    gp = point_parts(g)
    energy = abs2(Float64(gp[1][k]))
    @inbounds for m in eachindex(Base.OneTo(d))
        mode_M = factors[m+1]
        energy += Float64(
            ManifoldsBase.inner(mode_M.manifolds[k], pp[m+1][k], gp[m+1][k], gp[m+1][k]),
        )
    end
    return energy
end

function _cpd_component_metric_rgrad_energies(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    p,
    g,
)
    return _cpd_component_metric_rgrad_energies(cpd_model(model), p, g)
end

function _cpd_component_metric_rgrad_energies(model::RankRCPDModel, p, g)
    M = model.M
    r, d = model.r, length(model.dims)
    if M isa ProductManifold
        nf = length(M.manifolds)
        if nf == r * (d + 1)
            return [
                _cpd_join_component_metric_energy(M, p, g, k, d) for
                k in eachindex(Base.OneTo(r))
            ]
        elseif nf == r
            return [
                _cpd_native_component_metric_energy(M, p, g, k) for
                k in eachindex(Base.OneTo(r))
            ]
        elseif nf == d + 1
            return [
                _cpd_canonical_component_metric_energy(M, p, g, k, d) for
                k in eachindex(Base.OneTo(r))
            ]
        end
    end
    return _cpd_component_coordinate_rgrad_energies(model, g)
end

function _cpd_component_metric_rgrad_energies(model::Rank1CPDModel, p, g)
    M = model.M
    if model.nonnegative && M isa ProductManifold
        factors = M.manifolds
        pp = point_parts(p)
        gp = point_parts(g)
        energy = 0.0
        @inbounds for i in eachindex(factors)
            energy += Float64(ManifoldsBase.inner(factors[i], pp[i], gp[i], gp[i]))
        end
        return [energy]
    elseif M isa Manifolds.Segre
        return [Float64(ManifoldsBase.inner(M, p, g, g))]
    end
    return _cpd_component_coordinate_rgrad_energies(model, g)
end

_cpd_component_metric_rgrad_energies(_, _, _) = Float64[]

function _ambient_rankone_tangent_norm2(
    λk::Real,
    u::Vector{<:AbstractVector},
    δλk::Real,
    δu::Vector{<:AbstractVector},
)
    T = promote_type(typeof(λk), eltype(u[1]), typeof(δλk), eltype(δu[1]))
    δT_total = reconstruct_cp_rank1(T(δλk), u)
    @inbounds for m in eachindex(u)
        u_pert = [j == m ? δu[j] : u[j] for j in eachindex(u)]
        δT_total .+= reconstruct_cp_rank1(T(λk), u_pert)
    end
    return Float64(sum(abs2, δT_total))
end

function _decoded_tangent_for_component_k(geometry::Symbol, λ̃, Ũ, gλ, gU, k::Int)
    if geometry == :softplus_metric
        δλk = Float64(_softplus_derivative(λ̃[k]) * gλ[k])
        δu = [
            begin
                ũ = @view Ũ[m][:, k]
                gu = @view gU[m][:, k]
                [_softplus_derivative(ũ[i]) * gu[i] for i in eachindex(ũ)]
            end for m in eachindex(gU)
        ]
        return δλk, δu
    elseif geometry == :squaring_metric
        δλk = Float64(2 * λ̃[k] * gλ[k])
        δu = [@views Float64.(2 .* Ũ[m][:, k] .* gU[m][:, k]) for m in eachindex(gU)]
        return δλk, δu
    else
        δλk = Float64(gλ[k])
        δu = [Vector{Float64}(@view gU[m][:, k]) for m in eachindex(gU)]
        return δλk, δu
    end
end

function _cpd_component_ambient_velocities(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    p,
    g,
)
    return _cpd_component_ambient_velocities(cpd_model(model), p, g)
end

function _cpd_component_ambient_velocities(model::RankRCPDModel, p, g)
    λ̃, Ũ = unpack_point_rankr(p, model.dims, model.r)
    gλ, gU = unpack_point_rankr(g, model.dims, model.r)
    q = cpd_point(model, p)
    λ = lambda(q)
    U = factors(q)
    r = model.r
    energies = Vector{Float64}(undef, r)
    @inbounds for k in eachindex(energies)
        δλk, δu = _decoded_tangent_for_component_k(model.geometry, λ̃, Ũ, gλ, gU, k)
        u_cols = [@view U[m][:, k] for m in eachindex(U)]
        energies[k] = _ambient_rankone_tangent_norm2(λ[k], u_cols, δλk, δu)
    end
    return energies
end

function _decoded_tangent_rank1(geometry::Symbol, λ̃, Ũ, gλ, gU)
    if geometry == :softplus_metric
        δλk = Float64(_softplus_derivative(λ̃) * gλ)
        δu = [
            [_softplus_derivative(Ũ[m][i]) * gU[m][i] for i in eachindex(Ũ[m])] for
            m in eachindex(Ũ)
        ]
        return δλk, δu
    elseif geometry == :squaring_metric
        return Float64(2 * λ̃ * gλ), [Float64.(2 .* Ũ[m] .* gU[m]) for m in eachindex(Ũ)]
    else
        return Float64(gλ), [Vector{Float64}(gU[m]) for m in eachindex(gU)]
    end
end

function _cpd_component_ambient_velocities(model::Rank1CPDModel, p, g)
    λ̃, Ũ = unpack_point_rank1(p, model.dims)
    gλ, gU = unpack_point_rank1(g, model.dims)
    q = cpd_point(model, p)
    geometry =
        model.nonnegative ?
        (_rank1_uses_softplus_metric(model.M) ? :softplus_metric : :squaring_metric) :
        :canonical
    δλk, δu = _decoded_tangent_rank1(geometry, λ̃, Ũ, gλ, gU)
    u_cols = factors(q)
    return [_ambient_rankone_tangent_norm2(lambda(q), u_cols, δλk, δu)]
end

_cpd_component_ambient_velocities(_, _, _) = Float64[]

function _push_component_energy_trace!(
    hist::_CPDComponentTraceHistory,
    energies::AbstractVector{<:Real},
    kind::Symbol,
)
    summary = _component_energy_summary(energies)
    energies_f = Float64.(energies)
    if kind == :coordinate
        push!(hist.coordinate_rgrad_energy_history, energies_f)
        push!(hist.coordinate_rgrad_share_history, summary.shares)
        push!(hist.coordinate_rgrad_top1_share_history, summary.top1)
        push!(hist.coordinate_rgrad_top2_share_history, summary.top2)
        push!(hist.coordinate_rgrad_top3_share_history, summary.top3)
        push!(hist.coordinate_rgrad_effective_components_history, summary.effective)
        push!(hist.coordinate_rgrad_argmax_component_history, summary.argmax_component)
    elseif kind == :metric
        push!(hist.metric_rgrad_energy_history, energies_f)
        push!(hist.metric_rgrad_share_history, summary.shares)
        push!(hist.metric_rgrad_top1_share_history, summary.top1)
        push!(hist.metric_rgrad_top2_share_history, summary.top2)
        push!(hist.metric_rgrad_top3_share_history, summary.top3)
        push!(hist.metric_rgrad_effective_components_history, summary.effective)
        push!(hist.metric_rgrad_argmax_component_history, summary.argmax_component)
    elseif kind == :ambient
        push!(hist.ambient_component_velocity_history, energies_f)
        push!(hist.ambient_velocity_share_history, summary.shares)
        push!(hist.ambient_velocity_top1_share_history, summary.top1)
        push!(hist.ambient_velocity_top2_share_history, summary.top2)
        push!(hist.ambient_velocity_top3_share_history, summary.top3)
        push!(hist.ambient_velocity_effective_components_history, summary.effective)
        push!(hist.ambient_velocity_argmax_component_history, summary.argmax_component)
    end
    return nothing
end

function _safe_cpd_component_gradient_diagnostics(
    rec::_CPDComponentTraceRecorder,
    problem,
    p,
)
    g = try
        Manopt.get_gradient(problem, p)
    catch
        rec.history.rgrad_failed_count += 1
        return (; coordinate = Float64[], metric = Float64[], ambient = Float64[])
    end
    inner = cpd_model(rec.model)
    try
        return (
            coordinate = _cpd_component_coordinate_rgrad_energies(inner, g),
            metric = _cpd_component_metric_rgrad_energies(inner, p, g),
            ambient = _cpd_component_ambient_velocities(inner, p, g),
        )
    catch
        rec.history.rgrad_failed_count += 1
        return (; coordinate = Float64[], metric = Float64[], ambient = Float64[])
    end
end

function _record_cpd_component_trace!(
    rec::_CPDComponentTraceRecorder,
    problem,
    p,
    iter::Int,
)
    q = cpd_point(rec.model, p)
    cost_val = Float64(cost(rec.model, p))
    grad_diags = _safe_cpd_component_gradient_diagnostics(rec, problem, p)
    if rec.previous !== nothing
        hist = rec.history
        deltas = _cpd_component_deltas(rec.previous, q)
        rel_change = abs(rec.previous_cost - cost_val) / max(abs(rec.previous_cost), 1.0)
        push!(hist.iterations, iter)
        push!(hist.cost_history, cost_val)
        push!(hist.cost_rel_change_history, rel_change)
        push!(hist.max_component_delta_history, maximum(deltas))
        push!(hist.component_delta_history, deltas)
        _push_component_energy_trace!(hist, grad_diags.coordinate, :coordinate)
        _push_component_energy_trace!(hist, grad_diags.metric, :metric)
        _push_component_energy_trace!(hist, grad_diags.ambient, :ambient)
    end
    rec.previous = q
    rec.previous_cost = cost_val
    return nothing
end

function _cpd_component_trace_callback(rec::_CPDComponentTraceRecorder)
    return function (problem, state, k)
        p = try
            Manopt.get_iterate(state)
        catch
            return nothing
        end
        _record_cpd_component_trace!(rec, problem, p, Int(k))
        return nothing
    end
end

_trace_history_final(history, default) = isempty(history) ? default : history[end]

function _cpd_component_trace_info(rec::_CPDComponentTraceRecorder)
    hist = rec.history
    return (
        component_trace_iterations = hist.iterations,
        component_trace_cost_history = hist.cost_history,
        component_trace_cost_rel_change_history = hist.cost_rel_change_history,
        component_trace_max_delta_history = hist.max_component_delta_history,
        component_trace_delta_history = hist.component_delta_history,
        component_trace_final_max_delta = _trace_history_final(
            hist.max_component_delta_history,
            NaN,
        ),
        component_trace_coordinate_rgrad_energy_history = hist.coordinate_rgrad_energy_history,
        component_trace_coordinate_rgrad_share_history = hist.coordinate_rgrad_share_history,
        component_trace_coordinate_rgrad_top1_share_history = hist.coordinate_rgrad_top1_share_history,
        component_trace_coordinate_rgrad_top2_share_history = hist.coordinate_rgrad_top2_share_history,
        component_trace_coordinate_rgrad_top3_share_history = hist.coordinate_rgrad_top3_share_history,
        component_trace_coordinate_rgrad_effective_components_history = hist.coordinate_rgrad_effective_components_history,
        component_trace_coordinate_rgrad_argmax_component_history = hist.coordinate_rgrad_argmax_component_history,
        component_trace_metric_rgrad_energy_history = hist.metric_rgrad_energy_history,
        component_trace_metric_rgrad_share_history = hist.metric_rgrad_share_history,
        component_trace_metric_rgrad_top1_share_history = hist.metric_rgrad_top1_share_history,
        component_trace_metric_rgrad_top2_share_history = hist.metric_rgrad_top2_share_history,
        component_trace_metric_rgrad_top3_share_history = hist.metric_rgrad_top3_share_history,
        component_trace_metric_rgrad_effective_components_history = hist.metric_rgrad_effective_components_history,
        component_trace_metric_rgrad_argmax_component_history = hist.metric_rgrad_argmax_component_history,
        component_trace_ambient_component_velocity_history = hist.ambient_component_velocity_history,
        component_trace_ambient_velocity_share_history = hist.ambient_velocity_share_history,
        component_trace_ambient_velocity_top1_share_history = hist.ambient_velocity_top1_share_history,
        component_trace_ambient_velocity_top2_share_history = hist.ambient_velocity_top2_share_history,
        component_trace_ambient_velocity_top3_share_history = hist.ambient_velocity_top3_share_history,
        component_trace_ambient_velocity_effective_components_history = hist.ambient_velocity_effective_components_history,
        component_trace_ambient_velocity_argmax_component_history = hist.ambient_velocity_argmax_component_history,
        component_trace_rgrad_failed_count = hist.rgrad_failed_count,
        component_trace_coordinate_rgrad_top1_share_final = _trace_history_final(
            hist.coordinate_rgrad_top1_share_history,
            NaN,
        ),
        component_trace_coordinate_rgrad_top2_share_final = _trace_history_final(
            hist.coordinate_rgrad_top2_share_history,
            NaN,
        ),
        component_trace_coordinate_rgrad_top3_share_final = _trace_history_final(
            hist.coordinate_rgrad_top3_share_history,
            NaN,
        ),
        component_trace_coordinate_rgrad_effective_components_final = _trace_history_final(
            hist.coordinate_rgrad_effective_components_history,
            NaN,
        ),
        component_trace_coordinate_rgrad_argmax_component_final = _trace_history_final(
            hist.coordinate_rgrad_argmax_component_history,
            0,
        ),
        component_trace_metric_rgrad_argmax_component_final = _trace_history_final(
            hist.metric_rgrad_argmax_component_history,
            0,
        ),
        component_trace_ambient_velocity_argmax_component_final = _trace_history_final(
            hist.ambient_velocity_argmax_component_history,
            0,
        ),
        component_trace_metric_rgrad_top1_share_final = _trace_history_final(
            hist.metric_rgrad_top1_share_history,
            NaN,
        ),
        component_trace_ambient_velocity_top1_share_final = _trace_history_final(
            hist.ambient_velocity_top1_share_history,
            NaN,
        ),
        component_trace_start_rel_error = rec.start_rel_error,
    )
end

_pack_cpd_explicit_p0(_, p0) = p0
_pack_cpd_explicit_p0(model, p0::CPDPoint) = pack_cpd_point(model, p0)
_pack_cpd_explicit_p0(model, p0::CPDResult) = pack_cpd_point(model, cpd_point(p0))

_cpd_model_rank(model::RankRCPDModel) = model.r
_cpd_model_rank(::Rank1CPDModel) = 1

_cpd_auto_init(::ALSSolver) = :tucker
_cpd_auto_init(::AbstractSolver) = :alswarm

function _cpd_effective_init(init, solver::AbstractSolver, warm_steps::Int, warm_init)
    init_resolved = init == :auto ? _cpd_auto_init(solver) : init
    return init_resolved == :alswarm ? ALSWarmStartInit(warm_steps; base_init = warm_init) :
           init_resolved
end

_cpd_solve_normalization(::ALSSolver, nonnegative::Bool) =
    nonnegative ? NoNormalization() : SeparateLambdaNormalization()
_cpd_solve_normalization(::AbstractSolver, nonnegative::Bool) =
    nonnegative ? NonnegativeSeparateLambdaNormalization() : NoNormalization()

function _cpd_normalizations(normalization, solver::AbstractSolver, nonnegative::Bool)
    if normalization != :auto
        policy = _normalization_policy(normalization)
        return policy, policy
    end
    return (
        _cpd_solve_normalization(solver, nonnegative),
        nonnegative ? NoNormalization() : SeparateLambdaNormalization(),
    )
end

function _validate_cpd_solver_supported(solver::AbstractSolver)
    throw(
        ArgumentError(
            "Unsupported CPD solver $(typeof(solver)). Use :als, :rgd, :rgd_fixed, or :rcg.",
        ),
    )
end

_validate_cpd_solver_supported(::Union{ALSSolver,RGDSolver,RGDFixedSolver,RCGSolver}) =
    nothing

function _validate_cpd_solver_options(
    solver::AbstractSolver,
    geometry::Symbol,
    gradient_mode,
    component_trace::Bool,
)
    return nothing
end

function _validate_cpd_solver_options(
    solver::ALSSolver,
    geometry::Symbol,
    gradient_mode,
    component_trace::Bool,
)
    component_trace &&
        throw(ArgumentError("component_trace=true is only supported for manifold solvers."))
    geometry == :canonical || throw(
        ArgumentError(
            "solver=:als does not use manifold geometry. Use geometry=:canonical.",
        ),
    )
    gradient_mode == :riemannian || throw(
        ArgumentError(
            "solver=:als does not use gradient_mode. Use gradient_mode=:riemannian.",
        ),
    )
    return nothing
end

_cpd_nonnegative_init(init) = init == :tucker ? :alswarm : init
_cpd_nonnegative_geometry(::ALSSolver, geometry) = :canonical
_cpd_nonnegative_geometry(::AbstractSolver, geometry) =
    geometry == :canonical ? :softplus_metric : geometry

function _cpd_als_warm_then_pack(
    target::JoinModel{<:AbstractFloat,<:CPDBackend},
    init::ALSWarmStartInit;
    tol::Real,
    normalization,
    verbose::Bool,
    pullback_eps::Real,
    kwargs...,
)
    inner = cpd_model(target)
    A = tensor(target)
    r = _cpd_model_rank(inner)
    T = eltype(A)
    warm_init = init.base_init == :auto ? :tucker : init.base_init
    if r == 1
        warm_out = fit_cp_als(
            A,
            1;
            init = warm_init,
            maxiter = init.nsteps,
            tol = tol,
            normalization = normalization,
            mttkrp_method = get(kwargs, :mttkrp_method, :auto),
            nonnegative = inner.nonnegative,
            verbose = verbose,
            return_stats = true,
            progress_phase = :initialization,
        )
        return pack_cpd_point(target, CPDPoint(warm_out.weights, warm_out.factors))
    end
    warm_model = JoinModel(
        A,
        r;
        geometry = :canonical,
        scale_by_lambda = inner.scale_by_lambda,
        lambda_eps = inner.lambda_eps,
        nonnegative = inner.nonnegative,
        use_pullback_metric = false,
        pullback_eps = pullback_eps,
    )
    warm_result = _solve_model(
        warm_model;
        init = warm_init,
        solver = ALSSolver(),
        maxiter = init.nsteps,
        stepsize = one(T),
        tol = tol,
        gradient_mode = :riemannian,
        normalization = normalization,
        verbose = verbose,
        vector_transport_method = nothing,
        nonnegative = inner.nonnegative,
        progress_phase = :initialization,
        kwargs...,
    )
    warm_cpd = _to_cpd_result(warm_model, warm_result, size(A), r)
    return pack_cpd_point(target, cpd_point(warm_cpd))
end

function initial_point(
    model::RankRCPDModel{T,N},
    init::ALSWarmStartInit;
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    ε = _pullback_eps_value(T, 1e-8)
    target = JoinModel(
        model.A,
        model.r;
        geometry = model.geometry,
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
        nonnegative = model.nonnegative,
        use_pullback_metric = (model.geometry == :squaring_metric),
        pullback_eps = ε,
    )
    norm_als = model.nonnegative ? NoNormalization() : SeparateLambdaNormalization()
    return _cpd_als_warm_then_pack(
        target,
        init;
        tol = T(1e-6),
        normalization = norm_als,
        verbose,
        pullback_eps = ε,
    )
end

function initial_point(
    model::Rank1CPDModel{T,N},
    init::ALSWarmStartInit;
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    ε = _pullback_eps_value(T, 1e-8)
    geom =
        model.nonnegative ?
        (_rank1_uses_softplus_metric(model.M) ? :softplus_metric : :squaring_metric) :
        :native
    target = JoinModel(
        model.A,
        1;
        geometry = geom,
        scale_by_lambda = model.scale_by_lambda,
        lambda_eps = model.lambda_eps,
        nonnegative = model.nonnegative,
        use_pullback_metric = (geom == :squaring_metric),
        pullback_eps = ε,
    )
    norm_als = model.nonnegative ? NoNormalization() : SeparateLambdaNormalization()
    return _cpd_als_warm_then_pack(
        target,
        init;
        tol = T(1e-6),
        normalization = norm_als,
        verbose,
        pullback_eps = ε,
    )
end

function _cpd_manifold_grad_tol(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    solver::AbstractSolver,
    tol::Real,
)
    return nothing
end

function _cpd_manifold_grad_tol(
    model::JoinModel{<:AbstractFloat,<:CPDBackend},
    solver::Union{RGDSolver,RGDFixedSolver,RCGSolver,LBFGSSolver},
    tol::Real,
)
    inner = cpd_model(model)
    inner.nonnegative || return nothing
    return tol
end

function _cpd_point_rel_error(model, p)
    normA2 = sum(abs2, tensor(model))
    cost_val = Float64(cost(model, p))
    return Float64(_relative_error_frob_sq(2 * cost_val, Float64(normA2)))
end

function _cpd_initial_solve_point(model, init_eff, p0, solver::AbstractSolver; kwargs...)
    return _pack_cpd_explicit_p0(model, p0)
end

function _cpd_initial_solve_point(
    model,
    init_eff::ALSWarmStartInit,
    p0::Nothing,
    solver::AbstractSolver;
    tol,
    warm_normalization,
    verbose::Bool,
    pullback_eps,
    kwargs...,
)
    return _cpd_als_warm_then_pack(
        model,
        init_eff;
        tol,
        normalization = warm_normalization,
        verbose,
        pullback_eps,
        kwargs...,
    )
end

function _cpd_initial_solve_point(
    model,
    init_eff::ALSWarmStartInit,
    p0::Nothing,
    solver::ALSSolver;
    kwargs...,
)
    return nothing
end

function _cpd_solver_start_point(
    model,
    p_solve,
    init_eff,
    solver::AbstractSolver;
    verbose::Bool,
)
    return isnothing(p_solve) ? initial_point(model, init_eff; verbose) : p_solve
end

function _cpd_solver_start_point(model, p_solve, init_eff, solver::ALSSolver; verbose::Bool)
    return p_solve
end

function _run_cpd_solver(
    model;
    init_eff,
    p0,
    solver::AbstractSolver,
    maxiter::Int,
    stepsize,
    tol,
    gradient_mode,
    normalization,
    warm_normalization,
    verbose::Bool,
    vector_transport_method,
    pullback_eps,
    component_trace,
    kwargs...,
)
    trace_recorder = component_trace ? _CPDComponentTraceRecorder(model) : nothing
    iteration_callbacks =
        isnothing(trace_recorder) ? () : (_cpd_component_trace_callback(trace_recorder),)
    p_solve = _cpd_initial_solve_point(
        model,
        init_eff,
        p0,
        solver;
        tol,
        warm_normalization,
        verbose,
        pullback_eps,
        kwargs...,
    )
    p_start = _cpd_solver_start_point(model, p_solve, init_eff, solver; verbose)
    if !isnothing(trace_recorder)
        trace_recorder.start_rel_error = _cpd_point_rel_error(model, p_start)
    end

    result = _solve_model(
        model;
        init = init_eff,
        p0 = p_start,
        solver = solver,
        maxiter,
        stepsize,
        tol,
        gradient_mode,
        normalization,
        verbose,
        refinement_verbose = verbose,
        vector_transport_method,
        grad_tol = _cpd_manifold_grad_tol(model, solver, tol),
        iteration_callbacks,
        kwargs...,
    )
    return isnothing(trace_recorder) ? result :
           _merge_res_solver_info(result, _cpd_component_trace_info(trace_recorder))
end

function _cpd_impl(
    A::AbstractArray{T,N},
    r::Int;
    init,
    p0 = nothing,
    warm_steps,
    warm_init,
    solver,
    geometry,
    maxiter,
    stepsize,
    tol,
    gradient_mode,
    normalization,
    scale_by_lambda,
    lambda_eps,
    nonnegative::Bool,
    verbose,
    vector_transport_method,
    pullback_eps = 1e-8,
    component_trace::Bool = false,
    kwargs...,
) where {T<:AbstractFloat,N}
    haskey(kwargs, :softplus_beta) && throw(
        ArgumentError(
            "softplus_beta has been removed. Use pullback_eps to tune softplus pullback regularization.",
        ),
    )

    solver_obj = _solver_object(solver, stepsize; kwargs...)
    init_eff = _cpd_effective_init(init, solver_obj, warm_steps, warm_init)
    geometry_eff = _is_native_rankr_geometry(geometry) ? :native : geometry
    pullback_eps_eff = _pullback_eps_value(T, pullback_eps)
    normalization_eff, warm_normalization_eff =
        _cpd_normalizations(normalization, solver_obj, nonnegative)

    r >= 1 || throw(ArgumentError("rank r must be >= 1, got r=$r"))
    _validate_cpd_solver_supported(solver_obj)
    geometry_eff in (:native, :canonical, :squaring_metric, :softplus_metric) || throw(
        ArgumentError(
            "Unknown geometry=$geometry. Use :native, :canonical, :squaring_metric, or :softplus_metric.",
        ),
    )
    if geometry_eff in (:squaring_metric, :softplus_metric) && !nonnegative
        throw(ArgumentError("geometry=$geometry_eff requires nonnegative=true."))
    end
    _validate_cpd_solver_options(solver_obj, geometry_eff, gradient_mode, component_trace)

    model = JoinModel(
        A,
        r;
        geometry = geometry_eff,
        scale_by_lambda = scale_by_lambda,
        lambda_eps = lambda_eps,
        nonnegative = nonnegative,
        use_pullback_metric = (geometry_eff == :squaring_metric),
        pullback_eps = pullback_eps_eff,
    )

    raw_result = with_phase_progress() do
        _run_cpd_solver(
            model;
            init_eff,
            p0,
            solver = solver_obj,
            maxiter,
            stepsize,
            tol,
            gradient_mode,
            normalization = normalization_eff,
            warm_normalization = warm_normalization_eff,
            verbose,
            vector_transport_method,
            pullback_eps = pullback_eps_eff,
            component_trace,
            nonnegative,
            kwargs...,
        )
    end

    result =
        nonnegative && geometry_eff in (:squaring_metric, :softplus_metric) ?
        _merge_res_solver_info(raw_result, (nncp_pullback_eps = pullback_eps_eff,)) :
        raw_result
    return _to_cpd_result(model, result, size(A), r)
end

#### MAIN CPD ####

function cpd(
    A::AbstractArray{T,N};
    r::Union{Int,Nothing} = nothing,
    kwargs...,
) where {T<:AbstractFloat,N}
    dims = size(A)
    r_eff = r === nothing ? max(1, minimum(dims)) : r
    if r === nothing && get(kwargs, :verbose, true)
        println(
            "Rank not specified. Using heuristic r=$r_eff. Pass r explicitly to control model complexity.",
        )
    end
    return cpd(A, r_eff; kwargs...)
end

"""
    cpd(A, r; kwargs...)

Computes a rank-`r` CP approximation of `A` in two steps: (1) the first step finds an initial point; (2) the second step refines the initial point. Returns a [`CPDResult`](@ref). 
If `r` is omitted, uses the smallest tensor mode as a heuristic rank.

## Main Options 
* `init = :auto`: Sets the algorithm to find the initial point. Possible options are:
    - `:auto`: Uses a default CPD initializer. For `solver = :als`, this uses `TuckerInit`; otherwise, it uses an ALS warm start.
    - `:alswarm`: Runs ALS first and uses the result as the initial point for refinement.
    - customized initial point:
        - `:tucker` (default when `solver = :als`): Uses a default Tucker initializer.
        - `:random`: Uses a random initial point.
        - `:hosvd`: Uses a HOSVD initial point.
* `solver = :rgd`: Sets the algorithm for refinement. Possible options are:
    - `rgd` (default): Riemannian gradient descent
    - `rgd_fixed`: Riemannian gradient descent with fixed step size
    - `rcg`: Riemannian conjugate gradient
    - `als`: Alternating Least Squares

## Extended Options
* `p0 = nothing`: Explicit initial point. If provided, it overrides the default initial point.
* `:alswarm`: ALS warm start option.
    - `warm_init = TuckerInit()`: Before finding the warm start initial point, this sets the good starting point for ALS.
    - `warm_steps = 500`: Once finding the best initial point from warm_init, it runs this many ALS iterations to refine the initial point.
* `maxiter = 500`: Maximum number of Riemannian gradient descent iterations.
* `stepsize = 1.0`: Initial step size for line search in Riemannian gradient descent.
* `tol = 1e-6`: Convergence tolerance.
* `gradient_mode = :riemannian`: Gradient rule for manifold solvers. 
    - If the model has a direct rgrad, it uses that.
    - Otherwise it computes egrad and projects it to the tangent space.
    - This behavior is in src/solvers/abstract.jl (line 289).
* `geometry = :canonical`: Sets the geometry of the manifold. Possible options are:
    - `:canonical`: Standard CPD parameterization with the usual Euclidean factors and canonical Riemannian gradient handling. Best default for general unconstrained CPD.
    - `:squaring_metric`: Nonnegative geometry based on squared latent coordinates. Enforces nonnegativity indirectly, but can become ill-conditioned near zero.
    - `:softplus_metric`: Nonnegative geometry uses a regularized pullback-inspired geometry induced by the softplus chart. Smoother and usually more stable near zero than `:squaring_metric`.
    - `:native`: Native CP manifold geometry using the model’s intrinsic CP/Segre representation not for nonnegative=true. Best for structured join layouts with `Manifolds.Segre` summands.
* `verbose = true`: Enables progress output.
* `nonnegative::Bool = false`: Nonnegative CPD option to be selected by the user. (same as `nncpd`)
* `pullback_eps = 1e-8`: Regularization parameter for pullback-style nonnegative geometries.

## Notes
* `solver = :als` does not use manifold geometry. In that case:
    - `geometry` must be `:canonical`
    - `gradient_mode` is ignored except for validation
* `:squaring_metric` and `:softplus_metric` require `nonnegative = true`.
* When `nonnegative = true`, `cpd(...)` routes to `nncpd(...)`. In that route:
    - if `solver != :als` and `geometry` is left at `:canonical`, the effective geometry becomes `:softplus_metric`
    - if `stepsize` is left at `1.0`, the effective default becomes `0.01`
    - if `init = :tucker`, the effective initializer becomes `:alswarm`
    
## Example 
```julia-repl
julia> A = randn(20, 15, 10); r = 35
julia> res = cpd(A, r)
CPDResult{Float64}
  Order:        3
  Dimensions:   (20, 15, 10)
  Rank:         35
  Rel. error:   0.4359141301703327
```
"""
function cpd(
    A::AbstractArray{T,N},
    r::Int;
    init = :auto,
    p0 = nothing,
    warm_steps = 500,
    warm_init = TuckerInit(),
    solver = :rgd,
    geometry = :canonical,
    maxiter = 500,
    stepsize = 1.0,
    tol = 1e-6,
    gradient_mode = :riemannian,
    normalization = :auto,
    scale_by_lambda = true,
    lambda_eps = 1e-10,
    nonnegative::Bool = false,
    verbose = true,
    vector_transport_method = nothing,
    pullback_eps = 1e-8,
    component_trace::Bool = false,
    kwargs...,
) where {T<:AbstractFloat,N}
    if nonnegative
        solver_obj = _solver_object(solver, stepsize; kwargs...)
        # Align effective defaults with nncpd() on the nonnegative route.
        # Explicitly passed non-default values are preserved.
        init_nn = _cpd_nonnegative_init(init)
        warm_steps_nn = warm_steps
        geometry_nn = _cpd_nonnegative_geometry(solver_obj, geometry)
        stepsize_nn = stepsize == 1.0 ? 0.01 : stepsize
        return nncpd(
            A,
            r;
            init = init_nn,
            p0 = p0,
            warm_steps = warm_steps_nn,
            warm_init = warm_init,
            solver = solver_obj,
            geometry = geometry_nn,
            maxiter = maxiter,
            stepsize = stepsize_nn,
            tol = tol,
            gradient_mode = gradient_mode,
            normalization = normalization,
            scale_by_lambda = scale_by_lambda,
            lambda_eps = lambda_eps,
            pullback_eps = pullback_eps,
            component_trace = component_trace,
            verbose = verbose,
            vector_transport_method = vector_transport_method,
            kwargs...,
        )
    end
    return _cpd_impl(
        A,
        r;
        init = init,
        p0 = p0,
        warm_steps = warm_steps,
        warm_init = warm_init,
        solver = solver,
        geometry = geometry,
        maxiter = maxiter,
        stepsize = stepsize,
        tol = tol,
        gradient_mode = gradient_mode,
        normalization = normalization,
        scale_by_lambda = scale_by_lambda,
        lambda_eps = lambda_eps,
        nonnegative = false,
        pullback_eps = pullback_eps,
        component_trace = component_trace,
        verbose = verbose,
        vector_transport_method = vector_transport_method,
        kwargs...,
    )
end
