struct _NoSimilarArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

Base.size(A::_NoSimilarArray) = size(A.data)
Base.axes(A::_NoSimilarArray) = axes(A.data)
Base.IndexStyle(::Type{<:_NoSimilarArray{T,N,A}}) where {T,N,A} = Base.IndexStyle(A)
Base.getindex(A::_NoSimilarArray, I...) = getindex(A.data, I...)
Base.similar(::_NoSimilarArray, ::Type, ::Dims) =
    error("BTDBackend construction must not allocate target-shaped storage")

# =========================================================================
# tucker/hosvd.jl
# =========================================================================
function _rand_unit_matrix(rng::AbstractRNG, d::Int, r::Int)
    M = randn(rng, d, r)
    for j = 1:r
        nrm = norm(view(M, :, j))
        if nrm > 0
            @views M[:, j] ./= nrm
        else
            M[1, j] = one(eltype(M))
        end
    end
    return M
end
function _rand_orthonormal_matrix(rng::AbstractRNG, d::Int, r::Int)
    Q = Matrix(qr(randn(rng, d, r)).Q)
    return Q[:, 1:r]
end
function _add_relative_noise(
    rng::AbstractRNG,
    A::AbstractArray{T};
    level::Float64 = 1e-2,
) where {T}
    E = randn(rng, size(A))
    α = level * norm(A) / max(norm(E), eps(Float64))
    return A .+ T(α) .* E
end
function _make_cp_tensor(seed::Int; dims = (18, 16, 14), r::Int = 3, noisy::Bool = false)
    rng = MersenneTwister(seed)
    λ = rand(rng, r) .+ 0.5
    U = [_rand_unit_matrix(rng, d, r) for d in dims]
    A = reconstruct_cpd_rankr(λ, U)
    return noisy ? _add_relative_noise(rng, A; level = 1e-2) : A
end
function _make_tucker_tensor(
    seed::Int;
    dims = (20, 16, 12),
    ranks = (4, 3, 2),
    noisy::Bool = false,
)
    rng = MersenneTwister(seed)
    core = randn(rng, ranks...)
    factors = [_rand_orthonormal_matrix(rng, dims[n], ranks[n]) for n in eachindex(dims)]
    A = reconstruct_tucker(core, factors)
    return noisy ? _add_relative_noise(rng, A; level = 1e-2) : A
end

# Deterministic random fixtures. Each call owns its RNG.
_test_rand(seed::Integer, args...) = rand(MersenneTwister(seed), args...)
_test_randn(seed::Integer, args...) = randn(MersenneTwister(seed), args...)
