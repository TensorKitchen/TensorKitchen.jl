# join/join_model.jl — Join-front-end model and backend type definitions

export AbstractJoinBackend,
    JoinComponent,
    JoinModel,
    CPDBackend,
    JoinBackend,
    BTDBackend,
    component_manifold,
    component_embedding,
    component_tangent_dimension,
    component_basis_vector,
    component_ambient_embedding!,
    component_ambient_pushforward!
# BTDBackend is defined in `btd/model.jl` (includes contraction workspace).

abstract type AbstractJoinBackend end

struct JoinComponent{M,E}
    manifold::M
    embedding::E
end

struct DefaultJoinEmbedding end

JoinComponent(manifold::M) where {M} =
    JoinComponent{M,DefaultJoinEmbedding}(manifold, DefaultJoinEmbedding())

component_manifold(component::JoinComponent) = component.manifold
component_embedding(component::JoinComponent) = component.embedding
component_tangent_dimension(component::JoinComponent) =
    manifold_dimension(component_manifold(component))
component_tangent_dimension(component::JoinComponent, p) =
    component_tangent_dimension(component)
component_manifold(M::AbstractManifold) = M
component_embedding(::AbstractManifold) = DefaultJoinEmbedding()
component_tangent_dimension(M::AbstractManifold) = manifold_dimension(M)
component_tangent_dimension(M::AbstractManifold, p) = component_tangent_dimension(M)

manifold(component::JoinComponent) = component_manifold(component)

struct JoinModel{T<:AbstractFloat,B<:AbstractJoinBackend} <: AbstractDecompositionModel{T}
    backend::B
end

"""
    _JoinResidualWORO{T}

Write-once-read-once cache for the join residual between gradient and cost evaluations.
"""
mutable struct _JoinResidualWORO{T,V}
    residual::V # Keep the cached residual on the same backend as the join work buffers.
    fresh::Bool
end
_JoinResidualWORO(residual::V) where {V} =
    _JoinResidualWORO{eltype(residual),V}(residual, false)

struct JoinBackend{
    T,
    N,
    CT<:Tuple,
    A<:AbstractArray{T,N},
    V,
    MP<:ProductManifold,
    I,
    W<:_JoinResidualWORO{T},
    C,
} <: AbstractJoinBackend
    components::CT
    r::Int
    target::A
    target_shape::NTuple{N,Int}
    target_flat::V
    M_product::MP
    init_point::I
    work_rec::V
    work_residual::V
    woro::W
    component_bufs::C
end

struct CPDBackend{M<:AbstractDecompositionModel} <: AbstractJoinBackend
    model::M
end

_join_model_error(model::JoinModel, op::AbstractString) =
    error("$op is not implemented for JoinModel backend $(typeof(model.backend)).")

tensor(model::JoinModel) = _join_model_error(model, "tensor")
manifold(model::JoinModel) = _join_model_error(model, "manifold")
initial_point(model::JoinModel, init) = _join_model_error(model, "initial_point")
initial_point(model::JoinModel, init::PointInit; kwargs...) = init.point
initial_point(model::JoinModel, init::FunctionInit; kwargs...) = init.f(model)
cost(model::JoinModel, p) = _join_model_error(model, "cost")
egrad(model::JoinModel, p) = _join_model_error(model, "egrad")
supports_rgrad(::JoinModel) = false
rgrad(model::JoinModel, p) = _join_model_error(model, "rgrad")
supports_egrad_project(::JoinModel) = true
supports_exact_native(::JoinModel) = false
supports_exact_join_basis(::JoinModel) = false
model_exact_native_function(model::JoinModel) = throw(
    ArgumentError(
        "exact_native is not implemented for JoinModel backend $(typeof(model.backend)).",
    ),
)
model_exact_join_basis_function(model::JoinModel) = throw(
    ArgumentError(
        "exact_join_basis is not implemented for JoinModel backend $(typeof(model.backend)).",
    ),
)
cp_als_data(model::JoinModel) = throw(
    ArgumentError(
        "ALS/RALS data is not available for JoinModel backend $(typeof(model.backend)).",
    ),
)
extract_components(model::JoinModel, p) = throw(
    ArgumentError(
        "extract_components is not implemented for JoinModel backend $(typeof(model.backend)).",
    ),
)

unwrap_model(model::JoinModel) = model
