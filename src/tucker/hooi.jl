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

    # Initialization
    if init isa TuckerResult
        td0 = init::TuckerResult{T,N}
        size(td0.core) == ranks || throw(
            DimensionMismatch(
                "hooi: TuckerResult.core has size $(size(td0.core)), expected core size $ranks",
            ),
        )
        for m = 1:d
            size(td0.factors[m], 1) == dims[m] || throw(
                DimensionMismatch(
                    "hooi: TuckerResult factor $m has $(size(td0.factors[m],1)) rows, expected $(dims[m])",
                ),
            )
            size(td0.factors[m], 2) == ranks[m] || throw(
                DimensionMismatch(
                    "hooi: TuckerResult factor $m has $(size(td0.factors[m],2)) cols, expected $(ranks[m])",
                ),
            )
        end
        factors0 = td0.factors
        singular_vals = [copy(td0.singular_values[m]) for m = 1:d]
    elseif init == :sthosvd
        td0 = sthosvd(A, ranks)
        factors0 = td0.factors
        singular_vals = td0.singular_values
    else
        error("Unknown init: $init. Use :sthosvd or a TuckerResult.")
    end

    factors = [copy(factors0[m]) for m = 1:d]
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
        for k = 1:d
            Y = prefix
            for j = (k+1):d
                Y = mode_n_product(Y, factors_prev[j]', j)
            end
            Yk = unfold_mode(Y, k)
            F = svd(Yk)
            rk = min(ranks[k], size(F.U, 2))
            factors[k] = F.U[:, 1:rk]
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
        for k = 1:d
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
