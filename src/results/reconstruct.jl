# results/reconstruct.jl — result reconstruction helpers
export reconstruct

"""
    reconstruct(res::CPDResult)

Reconstruct by summing `component.tensor` for all components: If the decomposition in `res` is `A = λ_1 A_1 + … + λ_r A_r`, return `A`.
"""
reconstruct(r::CPDResult) = reconstruct_cpd_rankr(components(r))

"""
    reconstruct(res::ApproxResult)
    reconstruct(res::BTDResult)

Reconstruct by summing `component.tensor` for all components.
"""
function reconstruct(r::ApproxResult)
    comps = components(r)
    isempty(comps) && throw(ArgumentError("ApproxResult has no components to reconstruct."))
    X0 = tensor(comps[1])
    X = zero.(X0)
    for c in comps
        X .+= tensor(c)
    end
    return X
end

function reconstruct(r::BTDResult)
    comps = components(r)
    isempty(comps) && throw(ArgumentError("BTDResult has no components to reconstruct."))
    X0 = tensor(comps[1])
    X = zero.(X0)
    for c in comps
        X .+= tensor(c)
    end
    return X
end
