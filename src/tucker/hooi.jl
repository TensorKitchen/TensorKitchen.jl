export hooi

"""
High-order orthogonal iteration (HOOI) for Tucker decomposition.

"""

function hooi(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int};
    maxiter::Int = 50,
    tol::Float64 = 1e-8,
    init = :sthosvd,
    verbose::Bool = false,
) where {T<:AbstractFloat,N}
    dims = size(A)
    d = N
    processing_order = collect(1:d)

    factors0, singular_vals = _hooi_initial_factors(A, ranks, init)

    factors = [copy(factors0[m]) for m in eachindex(factors0)]
    prev_rel_error = T(Inf)
    converged = false
    iter_final = maxiter
    progress =
        maxiter > 0 ?
        make_hooi_progress(maxiter; enabled = verbose, phase = :refinement, dt = 0.2) :
        NoMethodProgress()

    # Iteratively update each factor matrix while holding the others fixed.
    # Reuse the already-projected prefix from updated earlier modes so we do
    # not restart from a full copy of A for every mode update.
    S = copy(A)
    for iter = 1:maxiter
        factors_prev = copy(factors)
        prefix = A
        for k in eachindex(factors)
            Y = prefix
            for j in (k+1):d
                Y = mode_n_product(Y, factors_prev[j]', j)
            end
            Yk = unfold_mode(Y, k)
            F = svd(Yk)
            rk = min(ranks[k], size(F.U, 2))
            factors[k] = Matrix(@view F.U[:, 1:rk])
            prefix = mode_n_product(prefix, factors[k]', k)
        end

        S = prefix
        rel_error = relative_frobenius_error(A, reconstruct_tucker(S, factors))
        rel_change = abs(prev_rel_error - rel_error)

        if verbose
            update_progress!(
                progress,
                iter;
                showvalues = Any[("RelErr", rel_error), ("Δ RelErr", rel_change)],
            )
        end

        if prev_rel_error < T(Inf) && rel_change < T(tol)
            converged = true
            iter_final = iter
            break
        end
        prev_rel_error = rel_error
    end

    if maxiter == 0
        S = copy(A)
        for k in eachindex(factors)
            S = mode_n_product(S, factors[k]', k)
        end
    end

    if verbose
        status = converged ? "Converged" : "Maximum iterations reached"
        finish_progress!(
            progress;
            current = iter_final,
            showvalues = Any[("Status", status), ("Iterations", iter_final)],
        )
    end

    return TuckerResult{T,N}(S, factors, processing_order, singular_vals)
end

function hooi(A::AbstractArray{T,N}, ranks::Vector{Int}; kwargs...) where {T,N}
    @assert length(ranks) == N
    return hooi(A, Tuple(Int.(ranks)); kwargs...)
end

function _hooi_initial_factors(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int},
    init::TuckerResult{T,N},
) where {T<:AbstractFloat,N}
    dims = size(A)
    size(init.core) == ranks || throw(
        DimensionMismatch(
            "hooi: TuckerResult.core has size $(size(init.core)), expected core size $ranks",
        ),
    )
    @inbounds for m in eachindex(ranks)
        size(init.factors[m], 1) == dims[m] || throw(
            DimensionMismatch(
                "hooi: TuckerResult factor $m has $(size(init.factors[m], 1)) rows, expected $(dims[m])",
            ),
        )
        size(init.factors[m], 2) == ranks[m] || throw(
            DimensionMismatch(
                "hooi: TuckerResult factor $m has $(size(init.factors[m], 2)) cols, expected $(ranks[m])",
            ),
        )
    end
    return init.factors, [copy(init.singular_values[m]) for m in eachindex(ranks)]
end

function _hooi_initial_factors(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int},
    init::Symbol,
) where {T<:AbstractFloat,N}
    init == :sthosvd || error("Unknown init: $init. Use :sthosvd or a TuckerResult.")
    td0 = sthosvd(A, ranks)
    return td0.factors, td0.singular_values
end

function _hooi_initial_factors(A::AbstractArray, ranks::Tuple, init)
    error("Unknown init: $init. Use :sthosvd or a TuckerResult.")
end
