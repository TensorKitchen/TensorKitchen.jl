# Bombieri--Weyl storage for symmetric tensors. This layer is independent of
# the manifold implementation and fixes TensorKitchen's coordinate convention.

export symmetric_multiindices, multinomial_multiplicity

function _check_symmetric_coordinate_size(n::Int, order::Int)
    n > 0 || throw(ArgumentError("n must be positive, got $n."))
    order > 0 || throw(ArgumentError("order must be positive, got $order."))
    return binomial(n + order - 1, order)
end

"""
    symmetric_multiindices(n, order)

Return all nonnegative multi-indices `alpha` of length `n` satisfying
`sum(alpha) == order`, in descending lexicographic order. This is the fixed
coordinate order used by [`compress_symmetric_tensor`](@ref) and
[`expand_symmetric_tensor`](@ref).
"""
function symmetric_multiindices(n::Int, order::Int)
    count = _check_symmetric_coordinate_size(n, order)
    indices = Vector{Vector{Int}}()
    sizehint!(indices, count)
    _each_symmetric_multiindex(n, order, Float64) do _, alpha, _
        push!(indices, copy(alpha))
    end
    return indices
end

"""
    multinomial_multiplicity(alpha)

Return the number `order! / prod(alpha[i]!)` of full-tensor entries in the
permutation orbit represented by a symmetric multi-index `alpha`.
"""
function multinomial_multiplicity(alpha::AbstractVector{<:Integer})
    all(>=(0), alpha) || throw(ArgumentError("multi-index entries must be nonnegative."))
    remaining = sum(alpha)
    multiplicity = 1
    for exponent in alpha
        multiplicity = Base.Checked.checked_mul(multiplicity, binomial(remaining, exponent))
        remaining -= exponent
    end
    return multiplicity
end

# Callback traversal avoids allocating the complete multi-index table on hot
# contraction paths. The callback must not retain `alpha`, which is reused.
function _each_symmetric_multiindex(f, n::Int, order::Int, ::Type{T}) where {T}
    _check_symmetric_coordinate_size(n, order)
    alpha = zeros(Int, n)
    index = 0
    function visit(i, remaining)
        if i < n
            for exponent = remaining:-1:0
                alpha[i] = exponent
                visit(i + 1, remaining - exponent)
            end
            return nothing
        end
        alpha[n] = remaining
        weight = one(T)
        degree = order
        for exponent in alpha, k = 1:exponent
            weight *= sqrt(T(degree) / T(k))
            degree -= 1
        end
        index += 1
        f(index, alpha, weight)
        return nothing
    end
    visit(1, order)
    return nothing
end

# Evaluate normalized monomials and, optionally, their gradients without
# division by factor entries, including when a factor coordinate is zero.
function _symmetric_coordinates(f, n::Int, order::Int, x; differential::Bool = false)
    length(x) == n || throw(DimensionMismatch("Expected a factor of length $n."))
    T = float(eltype(x))
    prefix = ones(T, n + 1)
    suffix = differential ? ones(T, n + 1) : nothing
    gradient = differential ? zeros(T, n) : nothing
    _each_symmetric_multiindex(n, order, T) do index, alpha, weight
        for j = 1:n
            prefix[j+1] = prefix[j] * x[j]^alpha[j]
        end
        if differential
            for j = n:-1:1
                suffix[j] = suffix[j+1] * x[j]^alpha[j]
            end
            for j = 1:n
                gradient[j] =
                    iszero(alpha[j]) ? zero(T) :
                    weight * alpha[j] * x[j]^(alpha[j] - 1) * prefix[j] * suffix[j+1]
            end
        end
        f(index, weight * prefix[n+1], gradient)
    end
    return nothing
end

function _symcpd_embed_coordinates!(out::AbstractVector, M, p)
    n, order = _symcpd_manifold_size(M)
    expected = _check_symmetric_coordinate_size(n, order)
    length(out) == expected ||
        throw(DimensionMismatch("Expected an output vector of length $expected."))
    _symmetric_coordinates(n, order, p[2]) do k, monomial, _
        out[k] = p[1][1] * monomial
    end
    return out
end

function _symcpd_embed_coordinates!(out::AbstractVector, M, p, X)
    n, order = _symcpd_manifold_size(M)
    expected = _check_symmetric_coordinate_size(n, order)
    length(out) == expected ||
        throw(DimensionMismatch("Expected an output vector of length $expected."))
    _symmetric_coordinates(n, order, p[2]; differential = true) do k, monomial, gradient
        out[k] = X[1][1] * monomial + p[1][1] * dot(gradient, X[2])
    end
    return out
end

function _symcpd_embed_coordinates(M, p)
    n, order = _symcpd_manifold_size(M)
    T = promote_type(eltype(p[1]), eltype(p[2]))
    out = zeros(T, _check_symmetric_coordinate_size(n, order))
    return _symcpd_embed_coordinates!(out, M, p)
end

function _symcpd_pullback_coordinates(M, p, ambient::AbstractVector)
    n, order = _symcpd_manifold_size(M)
    expected = _check_symmetric_coordinate_size(n, order)
    length(ambient) == expected ||
        throw(DimensionMismatch("Expected an ambient cotangent of length $expected."))
    T = promote_type(eltype(p[1]), eltype(p[2]), eltype(ambient))
    factor_covector = zeros(T, n)
    radial = zero(T)
    _symmetric_coordinates(n, order, p[2]; differential = true) do k, monomial, gradient
        radial += ambient[k] * monomial
        factor_covector .+= ambient[k] .* gradient
    end
    factor_gradient = (factor_covector .- order .* radial .* p[2]) ./ (order * p[1][1])
    factor_gradient .-= dot(p[2], factor_gradient) .* p[2]
    return _symcpd_tangent(T(radial), factor_gradient)
end

# Visit every distinct index in the permutation orbit of one multi-index.
function _symmetric_orbit(f, alpha, order)
    counts = copy(alpha)
    indices = Vector{Int}(undef, order)
    function visit(j)
        if j > order
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

function _compress_symmetric_tensor(M, A::AbstractArray{<:Real})
    n, order = _symcpd_manifold_size(M)
    size(A) == ntuple(_ -> n, order) ||
        throw(DimensionMismatch("Expected an order-$order tensor with all dimensions $n."))
    Base.require_one_based_indexing(A)
    T = float(eltype(A))
    out = zeros(T, _check_symmetric_coordinate_size(n, order))
    _each_symmetric_multiindex(n, order, T) do k, alpha, weight
        value = zero(T)
        _symmetric_orbit(alpha, order) do I
            value += A[I...]
        end
        out[k] = value / weight
    end
    return out
end

function _expand_symmetric_tensor(M, a::AbstractVector{<:Real})
    n, order = _symcpd_manifold_size(M)
    expected = _check_symmetric_coordinate_size(n, order)
    length(a) == expected ||
        throw(DimensionMismatch("Expected $expected compressed symmetric coordinates."))
    Base.require_one_based_indexing(a)
    T = float(eltype(a))
    out = Array{T}(undef, ntuple(_ -> n, order))
    _each_symmetric_multiindex(n, order, T) do k, alpha, weight
        value = a[k] / weight
        _symmetric_orbit(alpha, order) do I
            out[I...] = value
        end
    end
    return out
end
