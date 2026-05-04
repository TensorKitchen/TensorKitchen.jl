# btd/core/btd_cost.jl — BTD cost without ambient Tucker reconstruction

function _btd_cost(backend::BTDBackend, p)
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "BTD cost")

    # Previous ambient-reconstruction path kept for comparison:
    #
    # residual = _join_residual!(backend, p)
    # return 0.5 * sum(abs2, residual)
    #
    # which expands to:
    #
    # _join_reconstruct!(backend.work_rec, backend.manifolds, backend.r, p)
    # backend.work_residual .= backend.work_rec
    # backend.work_residual .-= backend.target_flat
    # return 0.5 * sum(abs2, backend.work_residual)

    s = backend.target_normsq

    @inbounds for b = 1:backend.r
        pb = parts[b]
        pb isa Manifolds.TuckerPoint || throw(
            ArgumentError(
                "BTD cost expects TuckerPoint blocks, got $(typeof(pb)) at block $b.",
            ),
        )
        s -= 2 * _target_tucker_inner(backend, backend.target, pb)
    end

    @inbounds for b = 1:backend.r
        s += _tucker_tucker_inner(backend, parts[b], parts[b])
        for c = (b+1):backend.r
            s += 2 * _tucker_tucker_inner(backend, parts[b], parts[c])
        end
    end

    return 0.5 * s
end

cost(model::JoinModel{<:AbstractFloat,<:BTDBackend}, p) = _btd_cost(model.backend, p)
