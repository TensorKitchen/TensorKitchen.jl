# results/reconstruct.jl — result reconstruction helpers
export reconstruct

"""
    reconstruct(res::CPDResult)

Reconstruct the dense tensor represented by a CP decomposition result.

For a rank-`R` CPD result, this returns
`sum(weights(res)[k] * u_1k ⊗ ... ⊗ u_Nk for k = 1:R)`.
"""
reconstruct(res::CPDResult) = reconstruct_cpd_rankr(components(res))

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
