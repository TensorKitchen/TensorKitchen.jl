# results/reconstruct.jl — result reconstruction helpers
export reconstruct

reconstruct(res::CPDResult) = reconstruct_cpd_rankr(components(res))

"""
    reconstruct(res::ApproxResult)
    reconstruct(res::BTDResult)
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
