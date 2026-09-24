# Full-tensor conversion belongs to SymCPD's input/output boundary. The
# compressed coordinate order and weights are owned by Veronese.embed.
@inline _each_veronese_coordinate(f, M::Manifolds.Veronese, ::Type{T}) where {T} =
    Manifolds._veronese_multiindices(f, M, T)

# Visit every distinct index in the permutation orbit of one multi-index.
function _symmetric_orbit(f, α, d)
    counts = copy(α)
    indices = Vector{Int}(undef, d)
    function visit(j)
        if j > d
            f(Tuple(indices))
            return nothing
        end
        for i in eachindex(counts)
            iszero(counts[i]) && continue
            counts[i] -= 1
            indices[j] = i
            visit(j + 1)
            counts[i] += 1
        end
        return nothing
    end
    visit(1)
    return nothing
end

function _compress_symmetric_tensor(M::Manifolds.Veronese, A::AbstractArray{<:Real})
    n, d = Manifolds.get_parameter(M.size)
    size(A) == ntuple(_ -> n, d) ||
        throw(DimensionMismatch("Expected an order-$d tensor with all dimensions $n."))
    Base.require_one_based_indexing(A)
    T = float(eltype(A))
    out = zeros(T, binomial(n + d - 1, d))
    _each_veronese_coordinate(M, T) do k, α, weight
        value = zero(T)
        _symmetric_orbit(α, d) do I
            value += A[I...]
        end
        out[k] = value / weight
    end
    return out
end

function _expand_symmetric_tensor(M::Manifolds.Veronese, a::AbstractVector{<:Real})
    n, d = Manifolds.get_parameter(M.size)
    m = binomial(n + d - 1, d)
    size(a) == (m,) ||
        throw(DimensionMismatch("Expected $m compressed symmetric coordinates."))
    Base.require_one_based_indexing(a)
    T = float(eltype(a))
    out = Array{T}(undef, ntuple(_ -> n, d))
    _each_veronese_coordinate(M, T) do k, α, weight
        value = a[k] / weight
        _symmetric_orbit(α, d) do I
            out[I...] = value
        end
    end
    return out
end
