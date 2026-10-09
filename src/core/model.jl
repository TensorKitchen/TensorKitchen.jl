# core/model.jl — Top-level decomposition model interface
export AbstractDecompositionModel,
    manifold,
    initial_point,
    egrad,
    rgrad,
    supports_rgrad,
    tensor,
    cost,
    post_step!,
    residual,
    differential_action,
    differential_action!,
    adjoint_action,
    adjoint_action!,
    EuclideanResidualSpace,
    residual_space,
    residual_dimension,
    allocate_residual,
    normal_operator,
    normal_operator!
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

"""Residual coordinates with the ordinary Euclidean inner product.

The coordinates may represent a full tensor, compressed symmetric entries,
or observations only. Compression must preserve the objective norm.
"""
struct EuclideanResidualSpace{T<:AbstractFloat}
    dimension::Int
    function EuclideanResidualSpace{T}(dimension::Int) where {T<:AbstractFloat}
        dimension >= 0 || throw(ArgumentError("Residual dimension must be nonnegative."))
        return new{T}(dimension)
    end
end

"""Return the residual coordinate space, independently of target storage.

The default is a flattened tensor; models with compressed or implicit targets
must specialize this method. A contraction-only model need not implement an
explicit residual evaluation merely to provide a cost or normal operator.
"""
residual_space(model::AbstractDecompositionModel{T}) where {T} =
    EuclideanResidualSpace{T}(length(tensor(model)))
residual_dimension(model::AbstractDecompositionModel) = residual_space(model).dimension
allocate_residual(space::EuclideanResidualSpace{T}) where {T} =
    Vector{T}(undef, space.dimension)
allocate_residual(model::AbstractDecompositionModel, p) =
    allocate_residual(residual_space(model))

function cost(model::AbstractDecompositionModel, p)
    error("cost not implemented for $(typeof(model))")
end

function egrad(model::AbstractDecompositionModel, p)
    error("egrad not implemented for $(typeof(model))")
end

model_cost_function(model::AbstractDecompositionModel) = (M, p) -> cost(model, p)
model_cost_egrad_functions(model::AbstractDecompositionModel) =
    (model_cost_function(model), model_egrad_function(model))
model_cost_egrad_functions(model::AbstractDecompositionModel, ::Real) =
    model_cost_egrad_functions(model)

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

function residual(model::AbstractDecompositionModel, p)
    error("residual not implemented for $(typeof(model))")
end

function differential_action!(out::AbstractVector, model::AbstractDecompositionModel, p, X)
    error("differential_action! not implemented for $(typeof(model))")
end

function differential_action(model::AbstractDecompositionModel, p, X)
    out = allocate_residual(model, p)
    differential_action!(out, model, p, X)
    return out
end

function adjoint_action(
    model::AbstractDecompositionModel,
    p,
    a::AbstractVector;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
)
    M = manifold(model)
    d = manifold_dimension(M)
    length(a) == residual_dimension(model) || throw(
        DimensionMismatch(
            "adjoint_action expected residual vector of length $(residual_dimension(model)) for $(typeof(model)), got $(length(a)).",
        ),
    )
    T = _scalar_eltype(p)
    coeff = zeros(T, d)
    e_j = zeros(T, d)
    col = Vector{T}(undef, length(a))
    @inbounds for j = 1:d
        fill!(e_j, zero(T))
        e_j[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p, e_j, basis)
        differential_action!(col, model, p, Xj)
        coeff[j] = dot(col, a)
    end
    return ManifoldsBase.get_vector(M, p, coeff, basis)
end

"""Write the metric adjoint action into `out`.

Models may specialize this method to reuse tangent workspaces. The default
preserves the allocating `adjoint_action` contract and performs a
manifold-aware copy, so structured tangent representations remain valid.
"""
function adjoint_action!(
    out,
    model::AbstractDecompositionModel,
    p,
    a::AbstractVector;
    kwargs...,
)
    M = manifold(model)
    ManifoldsBase.copyto!(M, out, p, adjoint_action(model, p, a; kwargs...))
    return out
end

"""Apply the Gauss--Newton normal operator `J*J`, using the manifold metric
adjoint and Euclidean residual coordinates. This is not the exact Riemannian
Hessian. Models may specialize this fallback with a fused contraction kernel.
"""
normal_operator(model::AbstractDecompositionModel, p, X) =
    adjoint_action(model, p, differential_action(model, p, X))

function normal_operator!(Y, model::AbstractDecompositionModel, p, X)
    ManifoldsBase.copyto!(manifold(model), Y, p, normal_operator(model, p, X))
    return Y
end

function adjoint_action(model::AbstractDecompositionModel, p, a::AbstractArray; kwargs...)
    return adjoint_action(model, p, vec(a); kwargs...)
end

function adjoint_action!(
    out,
    model::AbstractDecompositionModel,
    p,
    a::AbstractArray;
    kwargs...,
)
    return adjoint_action!(out, model, p, vec(a); kwargs...)
end
