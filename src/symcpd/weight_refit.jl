# Exact linear weight refitting shared by symmetric CPD solvers.
# All target access goes through evaluate/contract.

export refit_symcpd_weights

function _symcpd_psd_solve(K, rhs; damping::Real = 0, rtol::Real, atol::Real = 0)
    damping >= 0 || throw(ArgumentError("damping must be nonnegative."))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative."))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative."))
    size(K, 1) == size(K, 2) || throw(DimensionMismatch("K must be square."))
    size(K, 1) == size(rhs, 1) || throw(DimensionMismatch("K and rhs disagree."))
    T = promote_type(float(eltype(K)), float(eltype(rhs)), typeof(float(damping)))
    F = eigen(Symmetric(Matrix{T}(K)))
    values = max.(F.values, zero(T))
    sigma_max = isempty(values) ? zero(T) : maximum(values)
    cutoff = max(T(atol), T(rtol) * sigma_max)
    effective_rank = count(>(cutoff), values)
    inverse_values = if damping > 0
        inv.(values .+ T(damping))
    else
        map(value -> value > cutoff ? inv(value) : zero(T), values)
    end
    projected = transpose(F.vectors) * rhs
    scaled_projected = if projected isa AbstractVector
        inverse_values .* projected
    else
        reshape(inverse_values, :, 1) .* projected
    end
    solution = F.vectors * scaled_projected
    retained = filter(>(cutoff), values)
    condition_estimate = if damping > 0
        smallest = isempty(values) ? zero(T) : minimum(values)
        (sigma_max + T(damping)) / (smallest + T(damping))
    elseif isempty(retained)
        T(Inf)
    else
        sigma_max / minimum(retained)
    end
    residual_norm = norm(Matrix{T}(K) * solution - rhs)
    return solution,
    (
        effective_rank = effective_rank,
        condition_estimate = condition_estimate,
        cutoff = cutoff,
        eigenvalues = values,
        residual_norm = residual_norm,
        damping = T(damping),
    )
end

function _symcpd_weight_system(target::AbstractSymmetricTarget, X::AbstractMatrix)
    n, order = _symmetric_target_size(target)
    size(X, 1) == n ||
        throw(DimensionMismatch("Expected factor dimension $n, got $(size(X, 1))."))
    gram = transpose(X) * X
    K = gram .^ order
    c = [evaluate(target, view(X, :, r)) for r in axes(X, 2)]
    return K, c
end

@doc raw"""
    refit_symcpd_weights(target, factors; pinv_rtol=sqrt(eps(T)), damping=0)
    refit_symcpd_weights(A, factors; compute_type=nothing, ...)

Return the least-squares optimal symmetric CP weights for fixed unit factor
columns. For

```math
\widehat{\mathcal T}(\lambda,X)
=\sum_{r=1}^R\lambda_r x_r^{\otimes D},
```

the weight subproblem is linear. Its normal equations are

```math
K\lambda=c,\qquad
K_{rs}=(x_r^\top x_s)^D,\qquad
c_r=\langle\mathcal T,x_r^{\otimes D}\rangle.
```

The implementation solves this ``R\times R`` positive-semidefinite system by
a truncated eigendecomposition; it never forms a tensor unfolding or a
Khatri--Rao matrix. `damping=0` gives the Moore--Penrose solution on the
numerical range selected by `pinv_rtol`. Positive `damping` solves the
regularized system instead.

This is the linear-variable elimination underlying variable projection; see
G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and
nonlinear least squares problems whose variables separate," *SIAM Journal on
Numerical Analysis* 10(2) (1973), 413--432,
doi:10.1137/0710036.
"""
function refit_symcpd_weights(
    target::AbstractSymmetricTarget{T},
    factors::AbstractMatrix;
    pinv_rtol::Real = sqrt(eps(T)),
    damping::Real = 0,
) where {T<:AbstractFloat}
    K, c = _symcpd_weight_system(target, factors)
    weights, _ = _symcpd_psd_solve(K, c; damping, rtol = pinv_rtol)
    return Vector{promote_type(T, eltype(factors))}(weights)
end

function refit_symcpd_weights(
    A::AbstractArray{<:Real},
    factors::AbstractMatrix;
    compute_type = nothing,
    pinv_rtol = nothing,
    damping::Real = 0,
)
    prepared = prepare_tensor(A; compute_type)
    target = DenseSymmetricTarget(prepared)
    tolerance = isnothing(pinv_rtol) ? sqrt(eps(eltype(target))) : pinv_rtol
    return refit_symcpd_weights(target, factors; pinv_rtol = tolerance, damping)
end

function _symcpd_refit_point(
    model::JoinModel{T,B},
    X::AbstractMatrix;
    pinv_rtol::Real,
    damping::Real = 0,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    K, c = _symcpd_weight_system(model.backend.target, X)
    weights, info = _symcpd_psd_solve(K, c; damping, rtol = pinv_rtol)
    parts = ntuple(model.backend.rank) do r
        _symcpd_point(T(weights[r]), Vector{T}(view(X, :, r)))
    end
    return join_point(manifold(model), parts), info
end

function _symcpd_factor_matrix(
    model::JoinModel{T,B},
    p,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    parts = join_parts(manifold(model), p)
    X = Matrix{T}(undef, model.backend.n, model.backend.rank)
    for r in eachindex(parts)
        X[:, r] .= point_parts(parts[r])[2]
    end
    return X
end

function _symcpd_full_gradient_norm(
    model::JoinModel{T,B},
    p,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    parts = join_parts(manifold(model), p)
    weight_floor = sqrt(eps(T))
    valid_metric_point = all(abs(point_parts(part)[1][1]) > weight_floor for part in parts)
    valid_metric_point || return T(Inf), false
    gradient = rgrad(model, p)
    return T(norm(manifold(model), p, gradient)), true
end

function _symcpd_regularize_zero_weights(
    model::JoinModel{T,B},
    p,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    weight_floor = sqrt(eps(T)) * max(one(T), sqrt(target_norm2(model.backend.target)))
    parts = join_parts(manifold(model), p)
    regularized_parts = ntuple(model.backend.rank) do r
        part = point_parts(parts[r])
        weight = T(part[1][1])
        factor = Vector{T}(part[2])
        reference = iszero(weight) ? T(evaluate(model.backend.target, factor)) : weight
        regularized_weight =
            abs(weight) > weight_floor ? weight : copysign(weight_floor, reference)
        _symcpd_point(regularized_weight, factor)
    end
    return join_point(manifold(model), regularized_parts)
end
