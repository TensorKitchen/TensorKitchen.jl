# results/reconstruct.jl — result reconstruction helpers
export reconstruct

"""
    reconstruct(r::CPDResult)

Reconstruct the tensor: ∑_k λ_k · (U1[:,k] ⊗ U2[:,k] ⊗ …).
"""
reconstruct(r::CPDResult) = reconstruct_cpd_rankr(components(r))

"""
    reconstruct(r::ApproxResult)
    reconstruct(r::BTDResult)

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
