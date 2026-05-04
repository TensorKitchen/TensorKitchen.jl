# core/model.jl — Top-level decomposition model interface
export AbstractDecompositionModel, rgrad, supports_rgrad, tensor, cost, post_step!
"""
    AbstractDecompositionModel{T}

Top-level umbrella type for optimization-based decomposition models in this package.

This includes:
- CPD models
- block-term decomposition models
- generic join/sum-of-manifolds approximation models

Direct Tucker algorithms (`sthosvd`, `thosvd`, `hooi`) are not subtypes of
`AbstractDecompositionModel`; they follow a separate non-optimization path.
"""
abstract type AbstractDecompositionModel{T<:AbstractFloat} end

function cost(model::AbstractDecompositionModel, p)
    error("cost not implemented for $(typeof(model))")
end

function egrad(model::AbstractDecompositionModel, p)
    error("egrad not implemented for $(typeof(model))")
end

"""
    model_cost_function(model)

Return a reusable callable `(M, p) -> cost` for solver loops.
Concrete models can override to avoid rebuilding closures on every evaluation.
"""
model_cost_function(model::AbstractDecompositionModel) = (M, p) -> cost(model, p)
model_cost_egrad_functions(model::AbstractDecompositionModel) =
    (model_cost_function(model), model_egrad_function(model))

"""
    model_egrad_function(model)

Return a reusable callable `(M, p) -> egrad` for solver loops.
Concrete models can override to avoid rebuilding closures on every evaluation.
"""
model_egrad_function(model::AbstractDecompositionModel) = (M, p) -> egrad(model, p)

"""
    model_rgrad_function(model; model_egrad=nothing)

Return a reusable callable `(M, p) -> rgrad` for solver loops.
Concrete models can override to avoid rebuilding helper closures per iteration.
"""
model_rgrad_function(model::AbstractDecompositionModel; model_egrad = nothing) =
    (M, p) -> rgrad(model, p)

"""
    model_exact_native_function(model)

Return a reusable callable `(M, p) -> grad` for `gradient_mode=:exact_native`.
Only models with a dedicated native closed-form/intrinsic gradient should implement this.
"""
function model_exact_native_function(model::AbstractDecompositionModel)
    throw(ArgumentError("exact_native is not implemented for $(typeof(model))."))
end

"""
    unwrap_model(model::AbstractDecompositionModel)

Return the concrete underlying model for routing checks.
Composite front-end wrappers can overload this.
"""
unwrap_model(model::AbstractDecompositionModel) = model

"""Riemannian gradient: rgrad when supported, else egrad or grad(M, p, egrad)."""
function grad(model::AbstractDecompositionModel, p)
    supports_rgrad(model) && return rgrad(model, p)
    eg = egrad(model, p)
    M = manifold(model)
    return isnothing(M) ? eg : grad(M, p, eg)
end

"""
    supports_rgrad(model::AbstractDecompositionModel) -> Bool

Return `true` when `rgrad(model, p)` is implemented as a direct intrinsic
Riemannian gradient for this model.
"""
supports_rgrad(model::AbstractDecompositionModel) = false
supports_egrad_project(model::AbstractDecompositionModel) = true
supports_exact_native(model::AbstractDecompositionModel) = false
supports_exact_join_basis(model::AbstractDecompositionModel) = false

"""
    rgrad(model::AbstractDecompositionModel, p)

Compute a direct intrinsic Riemannian gradient at `p` without routing through
`egrad -> project`.
"""
function rgrad(model::AbstractDecompositionModel, p)
    error(
        "rgrad not implemented for $(typeof(model)); use gradient_mode=:egrad_project or implement rgrad.",
    )
end

"""
    model_exact_join_basis_function(model)

Return a reusable callable `(M, p) -> grad` for `gradient_mode=:exact_join_basis`.
This is a legacy direct-gradient hook for join-style models. Production solver
paths should prefer specialized manifold projections over generic basis sweeps.
"""
function model_exact_join_basis_function(model::AbstractDecompositionModel)
    throw(ArgumentError("exact_join_basis is not implemented for $(typeof(model))."))
end

"""
    cp_als_data(model) -> (A, r)

Return CP tensor/rank pair for ALS/RALS-capable models.
"""
function cp_als_data(model::AbstractDecompositionModel)
    throw(ArgumentError("ALS/RALS are not supported for model $(typeof(model))."))
end

"""
    initial_point(model::AbstractDecompositionModel, init)

Generate an initial point for optimization using the specified initialization strategy.
Returns a point compatible with the model's manifold.
"""
function initial_point(model::AbstractDecompositionModel, init; kwargs...)
    error("initial_point not implemented for $(typeof(model))")
end

initial_point(model::AbstractDecompositionModel, init::PointInit; kwargs...) = init.point
initial_point(model::AbstractDecompositionModel, init::FunctionInit; kwargs...) =
    init.f(model)

supports_normalization_policy(
    model::AbstractDecompositionModel,
    policy::AbstractNormalizationPolicy,
) = policy isa NoNormalization

"""
    cpd_point(model, p)

Convert a model-specific CPD optimization point `p` into a layout-independent
[`CPDPoint`](@ref).

This is a backend adapter used when postprocessing should ignore the current
manifold/packing details. Typical uses are iteration normalization, diagnostics,
and other geometry-agnostic CP utilities.
"""
function cpd_point(model::AbstractDecompositionModel, p)
    throw(
        ArgumentError(
            "Explicit CPD point conversion is not implemented for $(typeof(model)).",
        ),
    )
end

"""
    pack_cpd_point(model, point)

Convert a canonical [`CPDPoint`](@ref) back into the point layout required by
`model`.

This is the inverse backend adapter of [`cpd_point`](@ref). It is used after
normalization or other CP-space postprocessing to continue optimization on the
model's native solver/manifold representation.
"""
function pack_cpd_point(model::AbstractDecompositionModel, point::CPDPoint)
    throw(ArgumentError("CPD point packing is not implemented for $(typeof(model))."))
end

"""
    post_step!(model, p; normalization=..., kwargs...)

Apply backend postprocessing to an optimization iterate.

For CPD models this hook is where iteration-level normalization and similar
layout-agnostic post-step operations live. The default implementation is a
no-op except for normalization-policy validation.
"""
function post_step!(
    model::AbstractDecompositionModel,
    p;
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = nothing,
    kwargs...,
)
    policy = _normalization_policy(normalization)
    supports_normalization_policy(model, policy) || throw(
        ArgumentError(
            "Normalization policy $(typeof(policy)) is not supported for model $(typeof(model)).",
        ),
    )
    return p
end

"""
    manifold(model::AbstractDecompositionModel)

Return the manifold for this model, or `nothing` if the model uses Euclidean space.
"""
function manifold(model::AbstractDecompositionModel)
    error("manifold not implemented for $(typeof(model))")
end

"""
    tensor(model::AbstractDecompositionModel)

Return the target tensor being approximated.
"""
function tensor(model::AbstractDecompositionModel)
    error("tensor not implemented for $(typeof(model))")
end
# ============================================================================
