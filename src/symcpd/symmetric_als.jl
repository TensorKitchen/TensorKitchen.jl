# Conditional least-squares initialization for symmetric CPD.
# All target access goes through evaluate/contract.

export NormalizedCLSInit

@doc raw"""
    NormalizedCLSInit(; sweeps=10, base_init=:random, damping=1e-10,
        pinv_rtol=sqrt(eps(Float64)), weight_pinv_rtol=pinv_rtol,
        patience=3, tol=1e-6)

Configure normalized conditional least-squares (CLS) initialization for
[`symcpd`](@ref). For unit factor columns
``X=[x_1,\ldots,x_R]`` and tensor order ``D``, one sweep forms

```math
M_{:,r}=\mathcal T(I,x_r,\ldots,x_r),\qquad
H=(X^\top X)^{\circ(D-1)},
```

and solves the coupled linear least-squares surrogate

```math
B=MH^\dagger.
```

The columns of ``B`` are normalized to update the shared directions, after
which the component weights are refitted exactly with
[`refit_symcpd_weights`](@ref). `damping` regularizes the CLS normal system;
`pinv_rtol` truncates its numerical nullspace when damping is zero.
`weight_pinv_rtol` controls only the exact weight refit. `patience` stops an
initializer after repeated failure to improve the strict symmetric objective.

The symmetric PARAFAC motivation follows G. Favier and T. Bouilloc,
"Parametric complexity reduction of Volterra models using tensor
decompositions," *Proc. EUSIPCO* (2009), 2288--2292. The hard/soft symmetric
ALS distinction and its convergence caveats are discussed by P. Comon,
X. Luciani, and A. L. F. de Almeida, *Journal of Chemometrics* 23 (2009),
393--405, doi:10.1002/cem.1236. The explicit third-order conditional LS update
is Algorithm/Table II in G. Favier, A. Y. Kibangou, and T. Bouilloc (2012),
doi:10.1002/acs.1272. TensorKitchen extends its Gram formula to order ``D``,
normalizes columns, refits signed weights, and evaluates the strict symmetric
objective after every sweep; CLS itself is not assumed to be monotone for that
objective. When the result is passed to a Riemannian solver as an initializer,
an exactly zero fitted weight is replaced by the smallest target-scaled
nonzero weight needed by the warped Veronese metric. Standalone
[`SymmetricCLS`](@ref) results retain the exact fitted weights.
"""
struct NormalizedCLSInit{I,T<:AbstractFloat} <: AbstractInitializer
    sweeps::Int
    base_init::I
    damping::T
    pinv_rtol::T
    weight_pinv_rtol::T
    patience::Int
    tol::T
end

function NormalizedCLSInit(;
    sweeps::Int = 10,
    base_init = :random,
    damping::Real = 1.0e-10,
    pinv_rtol::Real = sqrt(eps(Float64)),
    weight_pinv_rtol::Real = pinv_rtol,
    patience::Int = 3,
    tol::Real = 1.0e-6,
)
    sweeps >= 0 || throw(ArgumentError("sweeps must be nonnegative, got $sweeps."))
    damping >= 0 || throw(ArgumentError("damping must be nonnegative, got $damping."))
    pinv_rtol >= 0 || throw(ArgumentError("pinv_rtol must be nonnegative."))
    weight_pinv_rtol >= 0 || throw(ArgumentError("weight_pinv_rtol must be nonnegative."))
    patience >= 1 || throw(ArgumentError("patience must be positive, got $patience."))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol."))
    base_init isa Symbol &&
        base_init in (:cls, :normalized_cls) &&
        throw(ArgumentError("NormalizedCLSInit base_init cannot itself be CLS."))
    T = promote_type(
        typeof(float(damping)),
        typeof(float(pinv_rtol)),
        typeof(float(weight_pinv_rtol)),
        typeof(float(tol)),
    )
    return NormalizedCLSInit{typeof(base_init),T}(
        sweeps,
        base_init,
        T(damping),
        T(pinv_rtol),
        T(weight_pinv_rtol),
        patience,
        T(tol),
    )
end

_symcpd_init_label(::NormalizedCLSInit) = :normalized_cls
_resolve_symcpd_init(
    ::JoinModel{T,B},
    init::NormalizedCLSInit,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend} = init

function _symcpd_normalize_cls_columns!(Xnew, Bfree, Xold)
    floor_norm = sqrt(eps(eltype(Xnew)))
    for r in axes(Xnew, 2)
        column = view(Bfree, :, r)
        column_norm = norm(column)
        if !isfinite(column_norm) || column_norm <= floor_norm
            Xnew[:, r] .= view(Xold, :, r)
        else
            Xnew[:, r] .= column ./ column_norm
        end
        dot(view(Xnew, :, r), view(Xold, :, r)) < 0 && (Xnew[:, r] .*= -1)
    end
    return Xnew
end

@doc raw"""
    _solve_symcpd_cls(model; init=:random, p0=nothing, maxiter=100, ...)

Run normalized conditional least squares as a standalone symmetric CP solver.
Every sweep solves

```math
B=T_{(1)}X^{\odot(D-1)}
  \left((X^\top X)^{\circ(D-1)}\right)^\dagger
```

through target contractions, normalizes the columns of ``B``, and performs an
exact weight refit. Because retying the free factor to every tensor mode is not
a block minimization of the strict symmetric objective, the implementation
retains the lowest-cost strict SymCPD point seen across all sweeps.

The symmetric PARAFAC model follows Favier and Bouilloc (2009), EUSIPCO,
2288--2292. The pseudoinverse symmetric-ALS interpretation and nonconvergence
caveat follow Comon, Luciani, and de Almeida (2009), doi:10.1002/cem.1236. The
conditional update follows Favier, Kibangou, and Bouilloc (2012),
doi:10.1002/acs.1272; column normalization, signed weight refitting, best-point
retention, and the order-``D`` extension are TensorKitchen safeguards.
"""
function _solve_symcpd_cls(
    model::JoinModel{T,B};
    init = :random,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-6,
    damping::Real = 1.0e-10,
    pinv_rtol::Real = sqrt(eps(T)),
    weight_pinv_rtol::Real = pinv_rtol,
    patience::Int = 3,
    verbose::Bool = true,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative."))
    tol > 0 || throw(ArgumentError("tol must be positive."))
    damping >= 0 || throw(ArgumentError("damping must be nonnegative."))
    patience >= 1 || throw(ArgumentError("patience must be positive."))
    init isa Symbol &&
        init in (:cls, :normalized_cls) &&
        throw(ArgumentError("CLS solver requires a non-CLS base initialization."))

    p_start = isnothing(p0) ? initial_point(model, init; verbose) : p0
    X = _symcpd_factor_matrix(model, p_start)
    p_best, initial_weight_info =
        _symcpd_refit_point(model, X; pinv_rtol = weight_pinv_rtol)
    best_cost = cost(model, p_best)
    cost_history = T[best_cost]
    direction_change_history = T[]
    cls_condition_history = T[]
    cls_effective_rank_history = Int[]
    weight_condition_history = T[initial_weight_info.condition_estimate]
    best_iteration = 0
    stagnant = 0
    iterations_done = 0
    converged_flag = false
    termination_reason = :maxiter

    for iteration = 1:maxiter
        gram = transpose(X) * X
        H = gram .^ (model.backend.order - 1)
        M = Matrix{T}(undef, model.backend.n, model.backend.rank)
        for r in axes(X, 2)
            M[:, r] .= contract(model.backend.target, view(X, :, r))
        end
        Btranspose, cls_info = _symcpd_psd_solve(H, transpose(M); damping, rtol = pinv_rtol)
        Bfree = Matrix(transpose(Btranspose))
        Xnew = similar(X)
        _symcpd_normalize_cls_columns!(Xnew, Bfree, X)
        direction_change = norm(Xnew - X) / sqrt(T(model.backend.rank))
        p_candidate, weight_info =
            _symcpd_refit_point(model, Xnew; pinv_rtol = weight_pinv_rtol)
        candidate_cost = cost(model, p_candidate)
        push!(cost_history, candidate_cost)
        push!(direction_change_history, direction_change)
        push!(cls_condition_history, cls_info.condition_estimate)
        push!(cls_effective_rank_history, cls_info.effective_rank)
        push!(weight_condition_history, weight_info.condition_estimate)

        improvement = best_cost - candidate_cost
        improvement_scale = max(one(T), abs(best_cost))
        if improvement > T(tol) * improvement_scale
            p_best = p_candidate
            best_cost = candidate_cost
            best_iteration = iteration
            stagnant = 0
        else
            stagnant += 1
            if candidate_cost < best_cost
                p_best = p_candidate
                best_cost = candidate_cost
                best_iteration = iteration
            end
        end
        X = Xnew
        iterations_done = iteration
        verbose && println(
            "SymCPD CLS iteration $iteration: cost=$candidate_cost, " *
            "direction_change=$direction_change, cond(H)=$(cls_info.condition_estimate)",
        )
        if direction_change <= T(tol)
            converged_flag = true
            termination_reason = :direction_tolerance
            break
        elseif stagnant >= patience
            termination_reason = :plateau
            break
        end
    end

    final_gradient_norm, valid_metric_point = _symcpd_full_gradient_norm(model, p_best)
    norm2 = target_norm2(model.backend.target)
    relative_error =
        norm2 > 0 ? sqrt(max(T(2) * best_cost, zero(T)) / norm2) :
        sqrt(max(T(2) * best_cost, zero(T)))
    return (
        point = p_best,
        cost = best_cost,
        rel_error = relative_error,
        grad_norm = final_gradient_norm,
        iterations = iterations_done,
        converged = converged_flag,
        solver = :cls,
        solver_info = (
            normalized_columns = true,
            exact_weight_refit = true,
            best_iteration = best_iteration,
            cost_history = cost_history,
            direction_change_history = direction_change_history,
            cls_condition_history = cls_condition_history,
            cls_effective_rank_history = cls_effective_rank_history,
            weight_condition_history = weight_condition_history,
            damping = T(damping),
            pinv_rtol = T(pinv_rtol),
            weight_pinv_rtol = T(weight_pinv_rtol),
            valid_metric_point = valid_metric_point,
            termination_reason = termination_reason,
        ),
    )
end

function initial_point(
    model::JoinModel{T,B},
    init::NormalizedCLSInit;
    verbose::Bool = false,
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    cls_solver = SymmetricCLS(
        damping = init.damping,
        pinv_rtol = init.pinv_rtol,
        weight_pinv_rtol = init.weight_pinv_rtol,
        patience = init.patience,
    )
    result = solve(
        cls_solver,
        model;
        init = init.base_init,
        maxiter = init.sweeps,
        tol = init.tol,
        verbose,
        return_stats = true,
    )
    return _symcpd_regularize_zero_weights(model, result.point)
end

_normalized_cls_initial_point(model; kwargs...) =
    initial_point(model, NormalizedCLSInit(); kwargs...)
