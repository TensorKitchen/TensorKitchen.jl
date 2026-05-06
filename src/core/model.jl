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

model_cost_function(model::AbstractDecompositionModel) = (M, p) -> cost(model, p)
model_cost_egrad_functions(model::AbstractDecompositionModel) =
    (model_cost_function(model), model_egrad_function(model))

model_egrad_function(model::AbstractDecompositionModel) = (M, p) -> egrad(model, p)

model_rgrad_function(model::AbstractDecompositionModel; model_egrad = nothing) =
    (M, p) -> rgrad(model, p)

function model_exact_native_function(model::AbstractDecompositionModel)
    throw(ArgumentError("exact_native is not implemented for $(typeof(model))."))
end

unwrap_model(model::AbstractDecompositionModel) = model

function grad(model::AbstractDecompositionModel, p)
    supports_rgrad(model) && return rgrad(model, p)
    eg = egrad(model, p)
    M = manifold(model)
    return isnothing(M) ? eg : grad(M, p, eg)
end

supports_rgrad(model::AbstractDecompositionModel) = false
supports_egrad_project(model::AbstractDecompositionModel) = true
supports_exact_native(model::AbstractDecompositionModel) = false
supports_exact_join_basis(model::AbstractDecompositionModel) = false

function rgrad(model::AbstractDecompositionModel, p)
    error(
        "rgrad not implemented for $(typeof(model)); use gradient_mode=:egrad_project or implement rgrad.",
    )
end

function model_exact_join_basis_function(model::AbstractDecompositionModel)
    throw(ArgumentError("exact_join_basis is not implemented for $(typeof(model))."))
end

function cp_als_data(model::AbstractDecompositionModel)
    throw(ArgumentError("ALS/RALS are not supported for model $(typeof(model))."))
end

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

function cpd_point(model::AbstractDecompositionModel, p)
    throw(
        ArgumentError(
            "Explicit CPD point conversion is not implemented for $(typeof(model)).",
        ),
    )
end


function pack_cpd_point(model::AbstractDecompositionModel, point::CPDPoint)
    throw(ArgumentError("CPD point packing is not implemented for $(typeof(model))."))
end

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

function manifold(model::AbstractDecompositionModel)
    error("manifold not implemented for $(typeof(model))")
end

function tensor(model::AbstractDecompositionModel)
    error("tensor not implemented for $(typeof(model))")
end
