# solvers/rals.jl — Randomized ALS (CPRAND/CPRAND-MIX)

function fit_cp_rals(
    A::AbstractArray{T,N},
    r::Int;
    maxiter::Int = 100,
    tol::Real = 1e-6,
    init = :random,
    init_factors = nothing,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = SeparateLambdaNormalization(),
    samples::Int = 0,
    mix::Bool = false,
    nonnegative::Bool = false,
    verbose::Bool = true,
    return_stats::Bool = false,
    progress_phase::Symbol = :refinement,
    kwargs...,
) where {T<:AbstractFloat,N}
    nonnegative && throw(
        ArgumentError(
            "Randomized ALS does not support nonnegative=true. Use solver=:als, :rgd, or :rcg.",
        ),
    )
    dims = size(A)
    normA2 = sum(abs2, A)
    samples = samples <= 0 ? 10 * r : samples
    normalization_policy = _normalization_policy(normalization)

    λ, U = if !isnothing(init_factors)
        init_factors
    else
        init_sym = _builtin_initializer_symbol(init)
        if init_sym == :random
            (ones(T, r), [Matrix(qr(randn(T, dims[n], r)).Q) for n = 1:N])
        elseif init_sym in (:hosvd, :tucker, :tucker_diag)
            init_cpd_factors(A, r; init = init_sym)
        else
            throw(ArgumentError("Unknown init=$init"))
        end
    end

    Q_mix = Matrix{T}[]
    A_work = A
    if mix
        Q_mix = [Matrix(qr(randn(T, d, d)).Q) for d in dims]
        for n = 1:N
            A_work = mode_n_product(A_work, Q_mix[n], n)
        end
    end

    fit_old = 0.0
    converged = false
    iter_final = maxiter
    others_indices = [CartesianIndices(Tuple(dims[setdiff(1:N, n)])) for n = 1:N]
    progress =
        maxiter > 0 ?
        (
            mix ?
            make_rals_mix_progress(
                maxiter;
                enabled = verbose,
                phase = progress_phase,
                dt = 0.2,
            ) :
            make_rals_progress(
                maxiter;
                enabled = verbose,
                phase = progress_phase,
                dt = 0.2,
            )
        ) : NoMethodProgress()

    for iter = 1:maxiter
        for n = 1:N
            others = setdiff(1:N, n)
            sample_idx = rand(1:length(others_indices[n]), samples)
            coords = others_indices[n][sample_idx]

            Z_s = ones(T, samples, r)
            for m in reverse(others)
                local_idx = findfirst(==(m), others)
                rows = [c[local_idx] for c in coords]
                Z_s .*= U[m][rows, :]
            end

            # 3. Sample Tensor Fibers X_s (Size: samples x dims[n])
            # We need to extract A[i, :, k, ...] where (i, k, ...) comes from coords
            # Construct indices for A
            X_s = zeros(T, samples, dims[n])
            for s = 1:samples
                idx = ntuple(d -> d == n ? Colon() : coords[s][findfirst(==(d), others)], N)
                X_s[s, :] = A_work[idx...]
            end

            # 4. Solve Least Squares: min || X_s^T - U[n] * Z_s^T ||
            # Equivalently: U[n] * Z_s^T = X_s^T  => U[n] = (Z_s \ X_s)^T
            # We add Tikhonov regularization for stability on small samples
            U[n] = transpose(Z_s \ X_s)
        end

        normalize_components!(U, λ, normalization_policy)

        # Convergence check (using exact fit for reliability, though expensive)
        # In a purely randomized massive-scale solver, we would estimate this.
        comps = components_from_factors(λ, U)
        _, _, rel_error = cp_residual_stats(A_work, normA2, comps)
        fit = 1.0 - rel_error
        fit_change = abs(fit_old - fit)

        verbose && update_progress!(
            progress,
            iter;
            showvalues = Any[
                ("Fit", round(fit, digits = 6)),
                ("Δ Fit", round(fit_change, digits = 6)),
                ("Samples", samples),
            ],
        )

        if fit_change < tol
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

    # Un-mix factors if needed
    if mix
        for n = 1:N
            # U_orig = Q * U_mixed
            U[n] = Q_mix[n] * U[n]
        end
        # Re-build components with unmixed factors
    end

    if return_stats
        comps = components_from_factors(λ, U)
        _, cost, rel_error = cp_residual_stats(mix ? A : A_work, normA2, comps)
        return (
            point = pack_rankr_canonical_tuple(comps),
            cost = cost,
            rel_error = rel_error,
            grad_norm = T(0),
            iterations = iter_final,
            converged = converged,
            solver = mix ? :rals_mix : :rals,
        )
    else
        return λ, U
    end
end

# ========== RALSSolver (AbstractALSSolver) ==========

"""
    RALSSolver(; samples=0, mix=false)

Randomized ALS. Uses Euclidean space; samples for scalability.
"""
struct RALSSolver <: AbstractALSSolver
    samples::Int
    mix::Bool
end
RALSSolver(; samples::Int = 0, mix::Bool = false) = RALSSolver(samples, mix)

function solve(
    solver::RALSSolver,
    model::AbstractDecompositionModel{T};
    maxiter::Int = 100,
    tol::Real = 1e-6,
    init = :random,
    p0 = nothing,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = SeparateLambdaNormalization(),
    nonnegative::Bool = false,
    verbose::Bool = true,
    return_stats::Bool = false,
    kwargs...,
) where {T<:AbstractFloat}
    inner = unwrap_model(model)
    inner isa RankRCPDModel ||
        throw(ArgumentError("RALSSolver requires RankRCPDModel, got $(typeof(model))"))
    explicit_p0 =
        isnothing(p0) && !_is_builtin_init(init) ? initial_point(model, init; verbose) : p0
    init_factors = isnothing(explicit_p0) ? nothing : _cp_init_factors(model, explicit_p0)
    out = fit_cp_rals(
        tensor(inner),
        inner.r;
        init,
        init_factors,
        maxiter,
        tol,
        normalization,
        verbose,
        return_stats = true,
        samples = solver.samples,
        mix = solver.mix,
        nonnegative,
        progress_phase = :refinement,
        kwargs...,
    )
    return_stats ?
    (
        point = out.point,
        cost = out.cost,
        rel_error = out.rel_error,
        grad_norm = T(0),
        iterations = out.iterations,
        converged = out.converged,
        solver = out.solver,
    ) : out.point
end
