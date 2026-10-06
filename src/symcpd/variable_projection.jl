# Sphere-only variable projection model for symmetric CPD.

export SymCPDVariableProjectionModel

@doc raw"""
    SymCPDVariableProjectionModel(model; pinv_rtol=sqrt(eps(T)))

Reduce a symmetric CP least-squares problem to the product of factor spheres by
eliminating the weights analytically. At directions
``X=(x_1,\ldots,x_R)``, the model solves

```math
\lambda^\star(X)=K(X)^\dagger c(X),\qquad
K_{rs}(X)=(x_r^\top x_s)^D,
```

and optimizes

```math
\widetilde f(X)=f(\lambda^\star(X),X)
```

on ``\mathbb S^{N-1}\times\cdots\times\mathbb S^{N-1}``. On a constant-rank
stratum of ``K``, the envelope theorem removes derivatives of
``\lambda^\star`` from the reduced gradient because
``\partial_\lambda f=0`` at the fitted weights. The tangent gradient is the
sphere projection of

```math
D\lambda_r\left[-\mathcal T(I,x_r,\ldots,x_r)
+\sum_s\lambda_s(x_r^\top x_s)^{D-1}x_s\right].
```

The variable-projection construction follows G. H. Golub and V. Pereyra,
"The differentiation of pseudo-inverses and nonlinear least squares problems
whose variables separate," *SIAM Journal on Numerical Analysis* 10(2)
(1973), 413--432, doi:10.1137/0710036. Rank changes of ``K`` are
nondifferentiable boundaries; `pinv_rtol` defines the numerical
constant-rank stratum used by this implementation.
"""
struct SymCPDVariableProjectionModel{T<:AbstractFloat,P,M} <: AbstractDecompositionModel{T}
    parent::P
    product_manifold::M
    pinv_rtol::T
end

function SymCPDVariableProjectionModel(
    parent::JoinModel{T,B};
    pinv_rtol::Real = sqrt(eps(T)),
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    pinv_rtol >= 0 || throw(ArgumentError("pinv_rtol must be nonnegative."))
    spheres = ntuple(_ -> Sphere(parent.backend.n - 1), parent.backend.rank)
    product = ProductManifold(spheres...)
    return SymCPDVariableProjectionModel{T,typeof(parent),typeof(product)}(
        parent,
        product,
        T(pinv_rtol),
    )
end

manifold(model::SymCPDVariableProjectionModel) = model.product_manifold
tensor(model::SymCPDVariableProjectionModel) = model.parent.backend.target
supports_rgrad(::SymCPDVariableProjectionModel) = true
supports_egrad_project(::SymCPDVariableProjectionModel) = false

function egrad(model::SymCPDVariableProjectionModel, p)
    throw(
        ArgumentError("Variable-projection SymCPD provides a direct Riemannian gradient."),
    )
end

function _symcpd_varpro_factors(model::SymCPDVariableProjectionModel{T}, p) where {T}
    parts = join_parts(manifold(model), p)
    X = Matrix{T}(undef, model.parent.backend.n, model.parent.backend.rank)
    for r in eachindex(parts)
        X[:, r] .= parts[r]
    end
    return X
end

function _symcpd_varpro_point(model::SymCPDVariableProjectionModel, full_point)
    X = _symcpd_factor_matrix(model.parent, full_point)
    parts = ntuple(r -> Vector(view(X, :, r)), model.parent.backend.rank)
    return join_point(manifold(model), parts)
end

function initial_point(
    model::SymCPDVariableProjectionModel,
    init;
    verbose::Bool = false,
    kwargs...,
)
    full_point = initial_point(model.parent, init; verbose, kwargs...)
    return _symcpd_varpro_point(model, full_point)
end

function _symcpd_varpro_fit(model::SymCPDVariableProjectionModel, p)
    X = _symcpd_varpro_factors(model, p)
    K, c = _symcpd_weight_system(model.parent.backend.target, X)
    weights, info = _symcpd_psd_solve(K, c; rtol = model.pinv_rtol)
    return X, K, c, weights, info
end

function cost(model::SymCPDVariableProjectionModel{T}, p) where {T}
    _, K, c, weights, _ = _symcpd_varpro_fit(model, p)
    return T(0.5) * target_norm2(model.parent.backend.target) - dot(weights, c) +
           T(0.5) * dot(weights, K * weights)
end

function rgrad(model::SymCPDVariableProjectionModel{T}, p) where {T}
    X, _, _, weights, _ = _symcpd_varpro_fit(model, p)
    target = model.parent.backend.target
    order = model.parent.backend.order
    rank = model.parent.backend.rank
    gradients = ntuple(rank) do r
        xr = view(X, :, r)
        gr = (-T(order) * weights[r]) .* contract(target, xr)
        for s = 1:rank
            correlation = dot(xr, view(X, :, s))
            gr .+=
                (T(order) * weights[r] * weights[s] * correlation^(order - 1)) .*
                view(X, :, s)
        end
        gr .-= dot(xr, gr) .* xr
        Vector{T}(gr)
    end
    return join_tangent_like(manifold(model), p, gradients)
end

@doc raw"""
    _solve_symcpd_varpro(parent; solver=:rcg, ...)

Solve the reduced symmetric CP problem after eliminating the linear weights.
For a residual Jacobian partitioned into weight and directional blocks,
``J=[J_\lambda\;J_X]``, variable projection uses the projected directional
Jacobian

```math
J_{\mathrm{red}}=(I-J_\lambda J_\lambda^\dagger)J_X.
```

The current implementation uses the corresponding reduced objective and
envelope-theorem gradient with RCG or L-BFGS. It deliberately does not route
to the existing full Gauss--Newton operator: reduced GN would require the
projected Jacobian above, equivalently a Schur complement of the full normal
system.

See G. H. Golub and V. Pereyra, *SIAM Journal on Numerical Analysis* 10(2)
(1973), 413--432, doi:10.1137/0710036.
"""
function _solve_symcpd_varpro(
    parent::JoinModel{T,B};
    init,
    p0,
    solver,
    maxiter::Int,
    stepsize::Real,
    tol::Real,
    verbose::Bool,
    vector_transport_method,
    pinv_rtol::Real,
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    solver in (:rcg, :lbfgs) || throw(
        ArgumentError(
            "variable_projection=true currently supports solver=:rcg or :lbfgs, got $solver.",
        ),
    )
    reduced = SymCPDVariableProjectionModel(parent; pinv_rtol)
    reduced_p0 = isnothing(p0) ? nothing : _symcpd_varpro_point(reduced, p0)
    reduced_result = _solve_model(
        reduced;
        init,
        p0 = reduced_p0,
        solver,
        maxiter,
        stepsize,
        tol,
        gradient_mode = :riemannian,
        normalization = NoNormalization(),
        verbose,
        vector_transport_method,
        observation_norm2_cache = target_norm2(parent.backend.target),
        kwargs...,
    )
    X = _symcpd_varpro_factors(reduced, reduced_result.point)
    full_point, weight_info = _symcpd_refit_point(parent, X; pinv_rtol)
    final_cost = cost(parent, full_point)
    full_gradient_norm, valid_metric_point = _symcpd_full_gradient_norm(parent, full_point)
    reduced_gradient_norm = T(reduced_result.grad_norm)
    norm2 = target_norm2(parent.backend.target)
    relative_error =
        norm2 > 0 ? sqrt(max(T(2) * final_cost, zero(T)) / norm2) :
        sqrt(max(T(2) * final_cost, zero(T)))
    info = merge(
        reduced_result.solver_info,
        (
            variable_projection = true,
            reduced_solver = reduced_result.solver,
            eliminated_weights = parent.backend.rank,
            weight_effective_rank = weight_info.effective_rank,
            weight_condition_estimate = weight_info.condition_estimate,
            weight_pinv_rtol = T(pinv_rtol),
            reduced_grad_norm = reduced_gradient_norm,
            full_grad_norm = full_gradient_norm,
            valid_metric_point = valid_metric_point,
        ),
    )
    return (
        point = full_point,
        cost = final_cost,
        rel_error = relative_error,
        grad_norm = reduced_gradient_norm,
        iterations = reduced_result.iterations,
        converged = reduced_result.converged,
        solver = Symbol("varpro_", reduced_result.solver),
        solver_info = info,
    )
end
