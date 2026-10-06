# results/reconstruct.jl — result reconstruction helpers
export reconstruct

"""
    reconstruct(res::CPDResult)

Reconstruct the dense tensor represented by a CP decomposition result.

For a rank-`R` CPD result, this returns
`sum(weights(res)[k] * u_1k ⊗ ... ⊗ u_Nk for k = 1:R)`.
"""
reconstruct(res::CPDResult) = reconstruct_cpd_rankr(components(res))

function reconstruct(c::SymCPDComponent)
    return expand_symmetric_tensor(
        _symcpd_embed_coordinates(_symcpd_manifold(length(c.factor), c.order), c.point),
        length(c.factor),
        c.order,
    )
end

tensor(c::SymCPDComponent) = reconstruct(c)

function reconstruct(res::SymCPDResult)
    n = size(res.factors, 1)
    return expand_symmetric_tensor(compressed_coordinates(res), n, res.order)
end

"""
    reconstruct(res::ApproxResult)

Reconstruct the dense ambient object represented by a generic join
approximation result by summing its component tensors.
"""
function reconstruct(res::ApproxResult)
    comps = components(res)
    isempty(comps) && throw(ArgumentError("ApproxResult has no components to reconstruct."))
    X0 = tensor(comps[1])
    X = zero.(X0)
    for c in comps
        X .+= tensor(c)
    end
    return X
end

"""
    reconstruct(res::BTDResult)

Reconstruct the dense tensor represented by a block-term decomposition result
by summing the reconstructed Tucker blocks.
"""
function reconstruct(res::BTDResult)
    comps = components(res)
    isempty(comps) && throw(ArgumentError("BTDResult has no components to reconstruct."))
    X0 = tensor(comps[1])
    X = zero.(X0)
    for c in comps
        X .+= tensor(c)
    end
    return X
end
