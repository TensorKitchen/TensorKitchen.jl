# core/types.jl — RankOneTensor, CPDResult, ApproxResult, TuckerResult
export RankOneTensor,
    CPDResult,
    ApproxResult,
    BTDResult,
    TuckerResult,
    components,
    blocks,
    weights,
    factors,
    core,
    point,
    cost,
    rel_error,
    grad_norm,
    iterations,
    converged,
    solver,
    solver_info,
    comp_weight,
    tensor,
    kind,
    vectors,
    processing_order,
    singular_values

"""
    RankOneTensor{T}

Representation of a rank-one tensor λ · u₁ ⊗ u₂ ⊗ … ⊗ u_d.
- `λ`: scalar weight
- `vectors`: length-`d` vector of factor vectors (one per mode)
Not a manifold object; used for CP representation and contractions.
"""
struct RankOneTensor{T<:AbstractFloat}
    λ::T
    vectors::Vector{Vector{T}}
end

"""
    CPDResult{T}

Result of a Canonical Polyadic Decomposition.
- `components`: vector of `RankOneTensor`
- `cost`, `rel_error`, `grad_norm`, `iterations`, `converged`, `solver`: metadata
- `solver_info`: lightweight solver diagnostics (for example line-search/evaluation counts)
"""
struct CPDResult{T<:AbstractFloat,C,W,F,S}
    components::C
    weights::W # Store the decoded CP factors once so repeated property access stays allocation-free.
    factors::F
    cost::T
    rel_error::T
    grad_norm::T
    iterations::Int
    converged::Bool
    solver::Symbol
    solver_info::S
end

"""
    DecompositionComponent{T, N}

Concrete component descriptor used by [`ApproxResult`](@ref) and [`BTDResult`](@ref).
It stores only the abstract component payload: the manifold point, the ambient
manifold, and the target tensor shape. Derived structure such as reconstructed
tensors, Tucker cores, or factor matrices is exposed through dispatched
accessors rather than stored eagerly in the container.
"""
struct DecompositionComponent{T<:AbstractFloat,N,P,MT}
    point::P
    manifold::MT
    target_shape::NTuple{N,Int}
end

@inline _component_core(p) =
    throw(ArgumentError("core is not defined for component point type $(typeof(p))."))
@inline _component_factors(p) =
    throw(ArgumentError("factors are not defined for component point type $(typeof(p))."))

@inline _component_core(p::Manifolds.TuckerPoint) = p.hosvd.core
@inline _component_factors(p::Manifolds.TuckerPoint) = collect(p.hosvd.U)

function reconstruct(c::DecompositionComponent)
    p = getfield(c, :point)
    M = getfield(c, :manifold)
    target_shape = getfield(c, :target_shape)
    return _component_tensor(p, M, target_shape)
end

function _component_tensor(p, M, target_shape::Tuple)
    # Generic components reconstruct through the manifold embedding.
    emb = ManifoldsBase.embed(M, p)
    return reshape(vec(emb), target_shape)
end

function _component_tensor(p::Manifolds.TuckerPoint, M, target_shape::Tuple)
    # Tucker points reconstruct directly from their structured core/factor data.
    X = reconstruct_tucker(_component_core(p), p.hosvd.U)
    length(X) == prod(target_shape) || throw(
        DimensionMismatch(
            "Tucker component length $(length(X)) != target length $(prod(target_shape)).",
        ),
    )
    return reshape(vec(X), target_shape)
end

function Base.getproperty(c::DecompositionComponent, name::Symbol)
    # Keep `c.tensor` as a convenience view while avoiding eager storage in the component.
    name === :tensor && return reconstruct(c)
    return getfield(c, name)
end

"""
    ApproxResult{T}

Generic result of `approx(manifolds, target)` (generic join).
- `point`: final point on product manifold
- `components`: extracted component descriptions
- `cost`, `rel_error`, `grad_norm`, `iterations`, `converged`, `solver`: optimization metadata
- `solver_info`: lightweight solver diagnostics
"""
struct ApproxResult{T<:AbstractFloat,P,C,S}
    point::P
    components::C
    cost::T
    rel_error::T
    grad_norm::T
    iterations::Int
    converged::Bool
    solver::Symbol
    solver_info::S
end

"""
    BTDResult{T}

Result of block-term decomposition (`btd`); block
components expose Tucker structure through accessors like
`core(blk)`, `factors(blk)`, and `blk.tensor`.
"""
struct BTDResult{T<:AbstractFloat,P,C,S}
    point::P
    components::C
    cost::T
    rel_error::T
    grad_norm::T
    iterations::Int
    converged::Bool
    solver::Symbol
    solver_info::S
end

function _rank1tensor_column(vectors::Vector{<:AbstractVector{T}}) where {T<:AbstractFloat}
    isempty(vectors) && throw(ArgumentError("_rank1tensor_column: empty vectors"))
    return reduce((a, b) -> kron(a, b), vectors)
end

_cpd_components(weights::Vector{T}, factors::Vector{Matrix{T}}) where {T<:AbstractFloat} = [
    RankOneTensor(weights[k], [factors[m][:, k] for m = 1:length(factors)]) for
    k = 1:length(weights)
]

function CPDResult(
    weights::Vector{T},
    factors::Vector{Matrix{T}},
    cost::T,
    rel_error::T,
    grad_norm::T,
    iterations::Int,
    converged::Bool,
    solver::Symbol,
    solver_info = (;),
) where {T<:AbstractFloat}
    CPDResult(
        _cpd_components(weights, factors),
        weights,
        factors,
        cost,
        rel_error,
        grad_norm,
        iterations,
        converged,
        solver,
        solver_info,
    )
end

λ(c::RankOneTensor) = c.λ
vectors(c::RankOneTensor) = c.vectors
kind(c::DecompositionComponent) = c.kind
point(c::DecompositionComponent) = c.point
tensor(c::DecompositionComponent) = c.tensor
core(c::DecompositionComponent) = c.core
factors(c::DecompositionComponent) = c.factors
point(r::CPDResult) = cpd_point(r)
point(r::ApproxResult) = r.point
point(r::BTDResult) = r.point
cost(r::Union{CPDResult,ApproxResult,BTDResult}) = r.cost
rel_error(r::Union{CPDResult,ApproxResult,BTDResult}) = r.rel_error
grad_norm(r::Union{CPDResult,ApproxResult,BTDResult}) = r.grad_norm
iterations(r::Union{CPDResult,ApproxResult,BTDResult}) = r.iterations
converged(r::Union{CPDResult,ApproxResult,BTDResult}) = r.converged
solver(r::Union{CPDResult,ApproxResult,BTDResult}) = r.solver
solver_info(r::Union{CPDResult,ApproxResult,BTDResult}) = r.solver_info

"""
    components(r) -> AbstractVector

Return the component list of a decomposition result (`CPDResult`, `ApproxResult`,
`BTDResult`, or `LL1Result`). For CP this is a vector of `RankOneTensor`; for
Tucker-block results (BTD, LL1) and generic join (`approx`) it is a vector of
`DecompositionComponent`.
"""
components(r::CPDResult) = r.components
components(r::ApproxResult) = r.components
components(r::BTDResult) = r.components
components(r::NamedTuple) = getproperty(r, :components)

"""
    blocks(r::BTDResult) -> Vector{DecompositionComponent}

Return the Tucker block components of a `BTDResult`. Each block `blk` supports
`core(blk)` and `factors(blk)`.
"""
blocks(r::BTDResult) = r.components

"""
    weights(r::CPDResult) -> Vector{T}

Return the per-component scalar weights `[λ_1, …, λ_r]`.
"""
weights(r::CPDResult) = [λ(c) for c in components(r)]
weights(r::NamedTuple) = getproperty(r, :weights)

"""
    comp_weight(r::CPDResult)

Return component weights from `solver_info(r)` when available, otherwise fall
back to [`weights`](@ref).
"""
function comp_weight(r::CPDResult)
    si = solver_info(r)
    return hasproperty(si, :comp_weight) ? getproperty(si, :comp_weight) : weights(r)
end

"""
    factors(r::CPDResult) -> Vector{Matrix{T}}

Return the mode-wise factor matrices `(A, B, C, …)`; column `k` of mode `m` is
the `m`-th factor vector of the `k`-th rank-one component.
"""
factors(r::CPDResult) = factors_from_components(components(r))
factors(r::NamedTuple) = getproperty(r, :factors)

function Base.show(io::IO, r::CPDResult{T}) where {T}
    comps = components(r)
    dims =
        length(comps) > 0 ?
        Tuple(length(vectors(comps[1])[m]) for m = 1:length(vectors(comps[1]))) : (0,)
    println(io, "CPDResult{$T}")
    println(io, "  Order:        $(length(dims))")
    println(io, "  Dimensions:   $dims")
    println(io, "  Rank:         $(length(comps))")
    print(io, "  Rel. error:   $(rel_error(r))")
end

function Base.show(io::IO, r::ApproxResult{T}) where {T}
    println(io, "ApproxResult{$T}")
    println(io, "  Components:   $(length(components(r)))")
    print(io, "  Rel. error:   $(rel_error(r))")
end

function Base.show(io::IO, r::BTDResult{T}) where {T}
    println(io, "BTDResult{$T}")
    println(io, "  Blocks:       $(length(components(r)))")
    print(io, "  Rel. error:   $(rel_error(r))")
end

"""
    LinearAlgebra.cond(R::CPDResult)

    Compute a conditioning surrogate for a CPD result by assembling tangent-like
    columns per component and returning `1 / σ_min(U)`.

This is an expensive diagnostic routine: it forms dense tangent-basis blocks
using repeated Kronecker products and then computes singular values of the
assembled matrix.
"""
function LinearAlgebra.cond(R::CPDResult{T}) where {T<:AbstractFloat}
    comps = components(R)
    isempty(comps) && throw(ArgumentError("cond(CPDResult): empty components"))

    Us = map(comps) do c
        current_vectors = deepcopy(vectors(c))
        cols = Vector{Vector{T}}()
        push!(cols, _rank1tensor_column(vectors(c)))
        for (j, v) in enumerate(vectors(c))
            P = nullspace(transpose(v))
            for w in eachcol(P)
                current_vectors[j] = Vector{T}(w)
                push!(cols, _rank1tensor_column(current_vectors))
            end
            current_vectors[j] = vectors(c)[j]
        end
        hcat(cols...)
    end
    U = hcat(Us...)
    s = svdvals(U)
    smin = minimum(s)
    return iszero(smin) ? T(Inf) : inv(smin)
end

"""
    TuckerResult{T, N}

Stores a Tucker decomposition: core tensor and factor matrices.
- `core::Array{T,N}` — core tensor
- `factors::Vector{Matrix{T}}` — orthonormal factor matrices
- `processing_order::Vector{Int}` — order modes were processed
- `singular_values::Vector{Vector{T}}` — singular values per truncation
"""
struct TuckerResult{T<:AbstractFloat,N}
    core::Array{T,N}
    factors::Vector{Matrix{T}}
    processing_order::Vector{Int}
    singular_values::Vector{Vector{T}}
end


"""
    core(td::TuckerResult) -> Array{T,N}

Return the core tensor `C` of a Tucker decomposition `Â = (U₁ ⊗ U₂ ⊗ … ⊗ U_N) · C`.
"""
core(td::TuckerResult) = td.core

"""
    factors(td::TuckerResult) -> Vector{Matrix{T}}

Return the factor matrices `(U₁, U₂, …, U_N)` of a Tucker decomposition.
"""
factors(td::TuckerResult) = td.factors
processing_order(td::TuckerResult) = td.processing_order
singular_values(td::TuckerResult) = td.singular_values

"""
    multilinear_rank(td::TuckerResult) -> NTuple{N,Int}

Return the multilinear rank, i.e. the tuple `size(core(td))`.
"""
multilinear_rank(td::TuckerResult) = size(core(td))

"""
    factor_dims(td::TuckerResult) -> NTuple{N,Int}

Return the ambient mode sizes, i.e. `size(U_m, 1)` for each factor.
"""
factor_dims(td::TuckerResult{T,N}) where {T,N} = ntuple(m -> size(factors(td)[m], 1), N)

function Base.show(io::IO, td::TuckerResult{T,N}) where {T,N}
    dims = size(core(td))
    orig_dims = factor_dims(td)
    println(io, "TuckerResult{$T, $N}")
    println(io, "  Original size:    $orig_dims")
    println(io, "  Core size:        $dims")
    println(io, "  Multilinear rank: $dims")
    compression =
        prod(orig_dims) / (prod(dims) + sum(size(U, 1) * size(U, 2) for U in factors(td)))
    print(io, "  Compression:      $(round(compression, digits=2))×")
end
