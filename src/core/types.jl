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


Stores the decoded CP representation together with solver diagnostics:
- `components`: rank-one tensor components
- `weights`: component weights
- `factors`: factor matrices
- `cost`: final objective function value at the returned solution
- `rel_error`: final relative reconstruction error
- `grad_norm`: norm of the final optimization gradient reported by the solver; for manifold solvers this is the Riemannian gradient norm
- `iterations`: number of refinement iterations
- `converged`: whether the solver reported convergence
- `solver`: solver optimization method used to produce the result
- `solver_info`: solver-specific diagnostics/metadata (`NamedTuple`). Typical keys include:
    - `initial_stepsize_eff` (RGD), `memory_size` (LBFGS), `cautious_update` (LBFGS),
    - `initial_scale`, `linesearch`, `has_preconditioner` (LBFGS), and `nncp_pullback_eps` (NNCP).
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
    DecompositionComponent{T,N}

Generic component wrapper used by [`ApproxResult`](@ref) and [`BTDResult`](@ref).

Each component stores:
- `point`: the component point on its manifold
- `manifold`: the component manifold
- `target_shape`: ambient tensor shape used for reconstruction

Use [`tensor`](@ref), [`core`](@ref), and [`factors`](@ref) to inspect the
decoded component structure when supported by the underlying manifold.
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

"""
    reconstruct(c::DecompositionComponent)

Reconstruct a component into its ambient tensor representation using the manifold embedding or directly from the structured core/factor data when the component is a Tucker point.
"""
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

Generic result of `approx(manifolds, target)` (generic join decomposition).
- `point`: final point on the join manifold
- `components`: extracted component descriptions
- `cost`: the value of the optimization objective at the final returned point, that is typically the least-squares objective.
- `rel_error`: relative error of the decomposition which is the ratio of the cost to the target norm. is scale-normalized and easier to compare across problems
- `grad_norm`: norm of the final optimization gradient reported by the solver; for manifold solvers this is typically the Riemannian gradient norm
- `iterations`: number of iterations used
- `converged`: whether the decomposition converged
- `solver`: solver used for the decomposition
- `solver_info`: solver-specific diagnostics/metadata (`NamedTuple`)
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
- `solver_info`: solver-specific diagnostics/metadata (`NamedTuple`). Typical keys include
  BTD-ALS restart diagnostics (`total_iterations`, `stagnation_restarts`,
  `restart_rel_error_history`) and BTD-TSD run settings (`schedule`, `block_repeats`,
  `block_count`, `stepsize`).
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

"""
    CPDResult{T}

Result of a Canonical Polyadic Decomposition.


Stores the decoded CP representation together with solver diagnostics:
- `components`: rank-one tensor components
- `weights`: component weights
- `factors`: factor matrices
- `cost`: final objective function value at the returned solution
- `rel_error`: final relative reconstruction error
- `grad_norm`: norm of the final optimization gradient reported by the solver; for manifold solvers this is the Riemannian gradient norm
- `iterations`: number of refinement iterations
- `converged`: whether the solver reported convergence
- `solver`: solver optimization method used to produce the result
- `solver_info`: solver-specific diagnostics/metadata (`NamedTuple`). Typical keys include:
    - `initial_stepsize_eff` (RGD), `memory_size` (LBFGS), `cautious_update` (LBFGS),
    - `initial_scale`, `linesearch`, `has_preconditioner` (LBFGS), and `nncp_pullback_eps` (NNCP).
"""
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

"""
    vectors(c::RankOneTensor)

Return the factor vectors of a rank-one tensor component.
"""
vectors(c::RankOneTensor) = c.vectors
kind(c::DecompositionComponent) = c.kind

"""
    point(x)

Return the optimization point stored in a result or component.

Supported inputs include [`CPDResult`](@ref), [`ApproxResult`](@ref),
[`BTDResult`](@ref), and [`DecompositionComponent`](@ref).
"""
point(c::DecompositionComponent) = c.point

"""
    tensor(c::DecompositionComponent)

Reconstruct a component into its ambient tensor representation.
"""
tensor(c::DecompositionComponent) = c.tensor

"""
    core(x)

Return the Tucker core stored in a Tucker component.
"""
core(c::DecompositionComponent) = c.core

"""
    factors(x)

Return factor matrices for a CP or Tucker result/component.
"""
factors(c::DecompositionComponent) = c.factors
point(r::CPDResult) = cpd_point(r)
point(r::ApproxResult) = r.point
point(r::BTDResult) = r.point

"""
    cost(r)

Return the final objective function value stored in a decomposition result.
"""
cost(r::Union{CPDResult,ApproxResult,BTDResult}) = r.cost

"""
    rel_error(r)

Return the final relative reconstruction error stored in a decomposition result.
"""
rel_error(r::Union{CPDResult,ApproxResult,BTDResult}) = r.rel_error

"""
    grad_norm(r)

Return the norm of the final optimization gradient reported by the solver; for manifold solvers this is typically the Riemannian gradient norm.
"""
grad_norm(r::Union{CPDResult,ApproxResult,BTDResult}) = r.grad_norm

"""
    iterations(r)

Return the number of refinement iterations used to produce `r`.
"""
iterations(r::Union{CPDResult,ApproxResult,BTDResult}) = r.iterations

"""
    converged(r)

Return whether the solver reported convergence.
"""
converged(r::Union{CPDResult,ApproxResult,BTDResult}) = r.converged

"""
    solver(r)

Return the solver symbol recorded in a decomposition result.
"""
solver(r::Union{CPDResult,ApproxResult,BTDResult}) = r.solver

"""
    solver_info(r)

Return solver-specific diagnostic information stored in a decomposition result.
"""
solver_info(r::Union{CPDResult,ApproxResult,BTDResult}) = r.solver_info

"""
    components(r)

Return the decoded components stored in a decomposition result.
"""
components(r::CPDResult) = r.components
components(r::ApproxResult) = r.components
components(r::BTDResult) = r.components
components(r::NamedTuple) = getproperty(r, :components)

"""
    blocks(r::BTDResult)

Return the Tucker block components of a block-term decomposition result.
"""
blocks(r::BTDResult) = r.components

"""
    weights(r::CPDResult)

Return the CP component weights stored in a CPD result.
"""
weights(r::CPDResult) = [λ(c) for c in components(r)]
weights(r::NamedTuple) = getproperty(r, :weights)

"""
    comp_weight(r::CPDResult)

Return component weights from `solver_info(r)` when available; otherwise fall
back to [`weights`](@ref).
"""
function comp_weight(r::CPDResult)
    si = solver_info(r)
    return hasproperty(si, :comp_weight) ? getproperty(si, :comp_weight) : weights(r)
end

"""
    factors(res::CPDResult)

Return the CP factor matrices of `res` as a vector `[U₁, U₂, ..., U_N]`,
where each `U_m` has size `size(A, m) × rank`.
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

# This forms dense tangent-basis blocksusing repeated Kronecker products and then computes singular values of the assembled matrix.

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
    core(td::TuckerResult)

Return the Tucker core tensor for a Tucker result.
"""
core(td::TuckerResult) = td.core

"""
    factors(td::TuckerResult)

Return the Tucker factor matrices.
"""
factors(td::TuckerResult) = td.factors

processing_order(td::TuckerResult) = td.processing_order

"""
    singular_values(td::TuckerResult)

Return the singular values recorded during Tucker truncation steps.
"""
singular_values(td::TuckerResult) = td.singular_values

"""
    multilinear_rank(td::TuckerResult)

Return the Tucker multilinear rank tuple, i.e. the size of the core tensor.
"""
multilinear_rank(td::TuckerResult) = size(core(td))

"""
    factor_dims(td::TuckerResult)

Return the original mode dimensions represented by the Tucker factor matrices.
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
