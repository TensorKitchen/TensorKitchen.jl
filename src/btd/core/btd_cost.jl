# btd/core/btd_cost.jl — BTD cost without ambient Tucker reconstruction

function _check_tucker_block(pb, b)
    pb isa Manifolds.TuckerPoint && return nothing

    throw(ArgumentError(
        "BTD cost expects TuckerPoint blocks, got $(typeof(pb)) at block $b.",
    ))
end

function _btd_cost(backend::BTDBackend, p)
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "BTD cost")

    s = backend.target_normsq

    # Target interaction:
    #
    #     -2⟨A, Σ_b X_b⟩
    #
    for (b, pb) in enumerate(parts)
        _check_tucker_block(pb, b)
        s -= 2 * _target_tucker_inner(backend, backend.target, pb)
    end

    # Model self-interaction:
    #
    #     ⟨Σ_b X_b, Σ_c X_c⟩
    #   = Σ_b ⟨X_b, X_b⟩ + 2Σ_{b<c} ⟨X_b, X_c⟩
    #
    for (b, pb) in enumerate(parts)
        s += _tucker_tucker_inner(backend, pb, pb)

        for pc in Iterators.drop(parts, b)
            s += 2 * _tucker_tucker_inner(backend, pb, pc)
        end
    end

    return s / 2
end

cost(model::JoinModel{<:AbstractFloat,<:BTDBackend}, p) =
    _btd_cost(model.backend, p)