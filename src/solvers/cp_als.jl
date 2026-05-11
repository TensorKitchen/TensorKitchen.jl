# solvers/cp_als.jl — CP-ALS
export ALSSolver, fit_cp_als


struct CPALSWorkspace
    Gs::Any # Unpacked G matrices (U[n] * U[n]')
    V::Any # V matrix (U[1] * U[1]') - cached MTTKRP result
    transposed_work::Any
    denom_work::Any # Denominator matrix for non-LS updates
    mttkrp_bufs::Any
    mttkrp_tmp_work::Any
    mttkrp_kr_work::Any # MTTKRP kernel result buffer
    mttkrp_kr_work2::Any
    cross_buf::Any
end

# --- backend-adaptive allocation helpers ------------------------------------
# These wrap `similar(A, ...)` so an ALS run on a `CuArray` / `ROCArray` can
# allocate its workspace on the same device as `A` without any code change.

@inline _cp_als_matrix_workspace_like(
    A::AbstractArray{T},
    n::Int,
    r::Int,
) where {T<:AbstractFloat} = similar(A, T, n, r)

@inline _cp_als_vector_workspace_like(
    A::AbstractArray{T},
    n::Int,
) where {T<:AbstractFloat} = similar(vec(A), T, n)

@inline function _cp_als_random_unit_matrix_like(
    A::AbstractArray{T},
    n::Int,
    r::Int,
) where {T<:AbstractFloat}
    U = _cp_als_matrix_workspace_like(A, n, r)
    rand!(U)
    col_norms = sqrt.(sum(abs2, U; dims = 1))
    col_norms .= max.(col_norms, eps(T))
    U ./= col_norms
    return U
end

@inline function _cp_als_adapt_factors_like(
    A::AbstractArray{T},
    λ::AbstractVector{<:Real},
    U::AbstractVector{<:AbstractMatrix{<:Real}},
) where {T<:AbstractFloat}
    λw = _cp_als_vector_workspace_like(A, length(λ))
    copyto!(λw, T.(λ))
    Uw = [_cp_als_matrix_workspace_like(A, size(Um, 1), size(Um, 2)) for Um in U]
    @inbounds for m in eachindex(U)
        copyto!(Uw[m], T.(U[m]))
    end
    return λw, Uw
end

function CPALSWorkspace(
    A::AbstractArray{T},
    dims::NTuple{N,Int},
    r::Int;
    mttkrp_method::Symbol = :auto,
) where {T<:AbstractFloat,N}
    Gs = [_cp_als_matrix_workspace_like(A, r, r) for _ = 1:N]
    V = _cp_als_matrix_workspace_like(A, r, r)
    transposed_work = [_cp_als_matrix_workspace_like(A, r, dims[n]) for n = 1:N]
    denom_work = [_cp_als_matrix_workspace_like(A, dims[n], r) for n = 1:N]
    mttkrp_bufs = [_cp_als_matrix_workspace_like(A, dims[n], r) for n = 1:N]
    total_dim_prod = prod(dims)
    resolved_mttkrp_methods =
        [_mttkrp_resolve_method(mttkrp_method, dims, r, n) for n = 1:N]
    mttkrp_tmp_work = Any[
        _mttkrp_needs_tmp_workspace(resolved_mttkrp_methods[n]) ?
        _cp_als_matrix_workspace_like(A, dims[n], r) : nothing for n = 1:N
    ]
    mttkrp_kr_work = Any[
        _mttkrp_needs_kr_workspace(resolved_mttkrp_methods[n]) ?
        _cp_als_matrix_workspace_like(A, div(total_dim_prod, dims[n]), r) : nothing for
        n = 1:N
    ]
    mttkrp_kr_work2 = Any[
        _mttkrp_needs_kr_workspace(resolved_mttkrp_methods[n]) ?
        _cp_als_matrix_workspace_like(A, div(total_dim_prod, dims[n]), r) : nothing for
        n = 1:N
    ]
    cross_buf = _cp_als_matrix_workspace_like(A, r, r)
    return CPALSWorkspace(
        Gs,
        V,
        transposed_work,
        denom_work,
        mttkrp_bufs,
        mttkrp_tmp_work,
        mttkrp_kr_work,
        mttkrp_kr_work2,
        cross_buf,
    )
end


@inline function _update_G!(
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
) where {T<:AbstractFloat}
    mul!(G, transpose(U), U)
    return G
end

@inline function _hadamard_G_except!(
    V::AbstractMatrix{T},
    Gs::AbstractVector{<:AbstractMatrix{T}},
    skip::Int,
) where {T<:AbstractFloat}
    fill!(V, one(T))
    @inbounds for m in eachindex(Gs)
        m == skip && continue
        V .*= Gs[m]
    end
    @inbounds for i in axes(V, 1)
        V[i, i] += eps(T)
    end
    return V
end

function _solve_right_spd!(
    Udest::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    workT::AbstractMatrix{T},
) where {T<:AbstractFloat}
    copyto!(workT, transpose(M_mttkrp))
    F = cholesky!(Hermitian(V))
    ldiv!(F, workT)
    copyto!(Udest, transpose(workT))
    return Udest
end

@inline function _als_mode_update!(
    Udest::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    workT::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return _solve_right_spd!(Udest, M_mttkrp, V, workT)
end

"""
    _cp_als_normalize_weights_and_Gs!(λ, U, Gs, normalization_policy, update_policy)

Post-sweep step: reset `λ` to one, apply the normalization policy, clamp to
nonnegative if required, and refresh G matrices. Single pass — see the
module-level design note.
"""
@inline function _cp_als_normalize_weights_and_Gs!(
    λ,
    U,
    Gs,
    normalization_policy,
    update_policy,
)
    T = eltype(λ)
    fill!(λ, one(T))
    normalize_components!(U, λ, normalization_policy)
    update_policy != :ls && _clamp_nonnegative!(λ, U)
    @inbounds for n in eachindex(Gs)
        _update_G!(Gs[n], U[n])
    end
    return nothing
end

function _cp_als_stats(
    A::AbstractArray{T,N},
    normA2::T,
    λ::AbstractVector{T},
    U::AbstractVector{<:AbstractMatrix{T}},
    Gs::AbstractVector{<:AbstractMatrix{T}};
    mttkrp_method::Symbol = :auto,
    mttkrp_buf::Union{Nothing,AbstractMatrix{T}} = nothing,
    mttkrp_work::Union{Nothing,AbstractMatrix{T}} = nothing,
    mttkrp_kr_buf::Union{Nothing,AbstractMatrix{T}} = nothing,
    mttkrp_kr_work::Union{Nothing,AbstractMatrix{T}} = nothing,
    cross_buf::Union{Nothing,AbstractMatrix{T}} = nothing,
) where {T<:AbstractFloat,N}
    if isnothing(cross_buf)
        cross = copy(Gs[1])
    else
        cross = cross_buf
        copyto!(cross, Gs[1])
    end

    @inbounds for m = 2:length(Gs)
        cross .*= Gs[m]
    end
    normX2 = zero(T)
    @inbounds for j in axes(cross, 2)
        cj = zero(T)
        for i in axes(cross, 1)
            cj += cross[i, j] * λ[i]
        end
        normX2 += λ[j] * cj
    end
    M1 = if isnothing(mttkrp_buf)
        mttkrp(A, U, 1; method = mttkrp_method)
    else
        mttkrp!(
            mttkrp_buf,
            A,
            U,
            1;
            method = mttkrp_method,
            work = mttkrp_work,
            kr_buf = mttkrp_kr_buf,
            kr_work = mttkrp_kr_work,
        )
    end
    innerAX = zero(T)
    @inbounds for k in eachindex(λ)
        innerAX += λ[k] * dot(@view(U[1][:, k]), @view(M1[:, k]))
    end
    n2 = normA2 + normX2 - 2 * innerAX
    if _cp_residual_sq_from_G_unreliable(n2, normA2, normX2, innerAX)
        return cp_residual_stats_explicit(A, normA2, λ, U)
    end
    return (n2, T(0.5) * n2, _relative_error_frob_sq(n2, normA2))
end

function fit_cp_als(
    A::AbstractArray{T,N},
    r::Int;
    maxiter::Int = 100,
    tol::Real = 1e-6,
    miniter::Union{Nothing,Int} = nothing,
    projected_grad_tol::Union{Nothing,Real} = nothing,
    nn_update::Symbol = :auto,
    init = :random,
    init_factors = nothing,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = SeparateLambdaNormalization(),
    mttkrp_method::Symbol = :auto,
    nonnegative::Bool = false,
    verbose::Bool = true,
    return_stats::Bool = false,
    progress_phase::Symbol = :refinement,
) where {T<:AbstractFloat,N}
    dims = size(A)
    normA2 = sum(abs2, A)
    update_policy = _cp_update_policy(nonnegative, nn_update)

    if !isnothing(init_factors)
        λ, U = init_factors
    else
        init_sym = _builtin_initializer_symbol(init)
        if init_sym == :random
            U =
                update_policy == :ls ?
                [_cp_als_random_unit_matrix_like(A, d, r) for d in dims] :
                [rand!(_cp_als_matrix_workspace_like(A, d, r)) for d in dims]
            λ = _cp_als_vector_workspace_like(A, r)
            fill!(λ, one(T))
        elseif init_sym in (:hosvd, :tucker, :tucker_diag)
            λ, U = init_cpd_factors(A, r; init = init_sym)
            if update_policy != :ls
                λ .= abs.(λ)
                _clamp_nonnegative!(λ, U)
            end
        else
            throw(ArgumentError("Unknown init=$init"))
        end
    end
    λ, U = _cp_als_adapt_factors_like(A, λ, U)

    fit_old = Inf
    converged = false
    iter_final = maxiter
    miniter_eff = isnothing(miniter) ? (update_policy == :ls ? 0 : 20) : miniter
    pg_tol_eff = if isnothing(projected_grad_tol)
        update_policy != :ls ? max(sqrt(T(tol)), sqrt(eps(T))) : zero(T)
    else
        T(projected_grad_tol)
    end
    pg_norm = update_policy != :ls ? T(Inf) : zero(T)
    normalization_policy = _normalization_policy(normalization)
    nnls_cd_sweeps = 10
    nnls_cd_sweeps_max = 40
    nnls_row_tol = sqrt(eps(T))
    nnls_row_tol_min = nnls_row_tol / 10

    workspace = CPALSWorkspace(A, dims, r; mttkrp_method = mttkrp_method)
    Gs = workspace.Gs
    @inbounds for n = 1:N
        _update_G!(Gs[n], U[n])
    end
    V = workspace.V
    transposed_work = workspace.transposed_work
    denom_work = workspace.denom_work
    mttkrp_bufs = workspace.mttkrp_bufs
    mttkrp_tmp_work = workspace.mttkrp_tmp_work
    mttkrp_kr_work = workspace.mttkrp_kr_work
    mttkrp_kr_work2 = workspace.mttkrp_kr_work2
    cross_buf = workspace.cross_buf

    progress =
        maxiter > 0 ?
        make_als_progress(maxiter; enabled = verbose, phase = progress_phase, dt = 0.2) :
        NoMethodProgress()

    for iter = 1:maxiter
        pg_sq = zero(T)
        u_sq = zero(T)
        for n = 1:N
            _hadamard_G_except!(V, Gs, n)
            M_mttkrp = mttkrp!(
                mttkrp_bufs[n],
                A,
                U,
                n;
                method = mttkrp_method,
                work = mttkrp_tmp_work[n],
                kr_buf = mttkrp_kr_work[n],
                kr_work = mttkrp_kr_work2[n],
            )
            if update_policy != :ls
                _clamp_nonnegative!(M_mttkrp)
                _nncp_mode_update!(
                    U[n],
                    M_mttkrp,
                    V,
                    denom_work[n];
                    nn_update = update_policy,
                    nnls_max_cd_sweeps = nnls_cd_sweeps,
                    nnls_row_tol = nnls_row_tol,
                )
                mul!(denom_work[n], U[n], V)
                pg_sq += _projected_grad_sq_nonnegative(U[n], M_mttkrp, denom_work[n])
                u_sq += sum(abs2, U[n])
            else
                _als_mode_update!(U[n], M_mttkrp, V, transposed_work[n])
            end
            _update_G!(Gs[n], U[n])
        end

        _cp_als_normalize_weights_and_Gs!(λ, U, Gs, normalization_policy, update_policy)

        _, _, rel_error = _cp_als_stats(
            A,
            normA2,
            λ,
            U,
            Gs;
            mttkrp_method,
            mttkrp_buf = mttkrp_bufs[1],
            mttkrp_work = mttkrp_tmp_work[1],
            mttkrp_kr_buf = mttkrp_kr_work[1],
            mttkrp_kr_work = mttkrp_kr_work2[1],
            cross_buf = cross_buf,
        )

        fit = 1.0 - rel_error
        fit_change = abs(fit_old - fit)
        update_policy != :ls && (pg_norm = sqrt(pg_sq / max(u_sq, one(T))))

        if verbose
            showvalues =
                update_policy == :ls ? Any[("Fit", fit), ("Δ Fit", fit_change)] :
                Any[("Fit", fit), ("Δ Fit", fit_change), ("Projected grad", pg_norm)]
            update_progress!(progress, iter; showvalues)
        end

        stop_by_fit = fit_change < tol
        stop_by_kkt = update_policy == :ls || (iter >= miniter_eff && pg_norm <= pg_tol_eff)

        if update_policy == :nnls &&
           iter >= miniter_eff &&
           fit_change < max(T(5) * T(tol), sqrt(eps(T))) &&
           pg_norm > T(5) * pg_tol_eff
            nnls_cd_sweeps = min(nnls_cd_sweeps + 5, nnls_cd_sweeps_max)
            nnls_row_tol = max(nnls_row_tol / 2, nnls_row_tol_min)
        end

        if stop_by_fit && stop_by_kkt
            converged = true
            iter_final = iter
            break
        end
        fit_old = fit
    end

    if verbose
        status = converged ? "Converged" : "Maximum iterations reached"
        finish_progress!(
            progress;
            current = iter_final,
            showvalues = Any[("Status", status), ("Iterations", iter_final)],
        )
    end

    if return_stats
        if update_policy != :ls && !isfinite(pg_norm)
            pg_norm = _projected_grad_norm_nonnegative!(
                A,
                U,
                Gs,
                V,
                denom_work,
                mttkrp_bufs,
                mttkrp_tmp_work,
                mttkrp_kr_work,
                mttkrp_kr_work2;
                mttkrp_method,
            )
        end

        _, cost, rel_error = _cp_als_stats(
            A,
            normA2,
            λ,
            U,
            Gs;
            mttkrp_method,
            mttkrp_buf = mttkrp_bufs[1],
            mttkrp_work = mttkrp_tmp_work[1],
            mttkrp_kr_buf = mttkrp_kr_work[1],
            mttkrp_kr_work = mttkrp_kr_work2[1],
            cross_buf = cross_buf,
        )
        comps = components_from_factors(λ, U)
        return (
            components = comps,
            weights = [c.λ for c in comps],
            factors = factors_from_components(comps),
            point = pack_rankr_canonical_tuple(comps),
            cost = cost,
            rel_error = rel_error,
            grad_norm = update_policy != :ls ? pg_norm : zero(T),
            iterations = iter_final,
            converged = converged,
            solver = :cp_als,
        )
    else
        return λ, U
    end
end

# ========== ALSSolver (AbstractALSSolver) ==========

"""
    ALSSolver

CP-ALS solver; MTTKRP from tensor_ops.jl.
"""
struct ALSSolver{P<:AbstractNormalizationPolicy} <: AbstractALSSolver
    normalization::P
end
ALSSolver(
    policy::Union{AbstractNormalizationPolicy,Symbol,Nothing} = SeparateLambdaNormalization(),
) = ALSSolver{typeof(_normalization_policy(policy))}(_normalization_policy(policy))
solver_symbol(::ALSSolver) = :cp_als

function _cp_init_factors(model::AbstractDecompositionModel{T}, p0) where {T<:AbstractFloat}
    inner = unwrap_model(model)
    inner isa RankRCPDModel || throw(
        ArgumentError(
            "Explicit p0 for ALS/RALS requires RankRCPDModel, got $(typeof(inner))",
        ),
    )
    return _cp_init_factors_from_rankr_point(
        p0,
        inner.dims,
        inner.r;
        nonnegative = inner.nonnegative,
        geometry = getproperty(inner, :geometry),
    )
end

function solve(
    solver::ALSSolver,
    model::AbstractDecompositionModel{T};
    maxiter::Int = 100,
    tol::Real = 1e-6,
    miniter::Union{Nothing,Int} = nothing,
    projected_grad_tol::Union{Nothing,Real} = nothing,
    nn_update::Symbol = :auto,
    init = :random,
    p0 = nothing,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = solver.normalization,
    nonnegative::Bool = false,
    verbose::Bool = true,
    return_stats::Bool = false,
    progress_phase::Symbol = :refinement,
    kwargs...,
) where {T<:AbstractFloat}
    A, r = cp_als_data(model)
    explicit_p0 =
        isnothing(p0) && !_is_builtin_init(init) ? initial_point(model, init; verbose) : p0
    init_factors = isnothing(explicit_p0) ? nothing : _cp_init_factors(model, explicit_p0)
    out = fit_cp_als(
        A,
        r;
        init = init,
        init_factors = init_factors,
        maxiter = maxiter,
        tol = tol,
        miniter = miniter,
        projected_grad_tol = projected_grad_tol,
        nn_update = nn_update,
        normalization = normalization,
        verbose = verbose,
        return_stats = true,
        progress_phase = progress_phase,
        mttkrp_method = get(kwargs, :mttkrp_method, :auto),
        nonnegative,
    )
    return_stats ?
    (
        point = out.point,
        cost = out.cost,
        rel_error = out.rel_error,
        grad_norm = out.grad_norm,
        iterations = out.iterations,
        converged = out.converged,
        solver = :cp_als,
    ) : out.point
end
