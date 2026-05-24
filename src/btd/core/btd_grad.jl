# btd/core/btd_grad.jl — BTD gradient assembly without ambient Tucker reconstruction

function _btd_core_grad_block(backend::BTDBackend, parts, b::Int)
    pb = parts[b]
    Gb, _ = _tucker_data(pb)

    grad_core = copy(Gb)
    grad_core .-= _tucker_project_target(backend, pb, backend.target)
    @inbounds for c in eachindex(parts)
        c == b && continue
        grad_core .+= _tucker_cross_core(backend, pb, parts[c])
    end
    return grad_core
end

function _btd_factor_grad_block(backend::BTDBackend, parts, b::Int)
    pb = parts[b]
    Gb, Ub = _tucker_data(pb)
    T = eltype(Gb)
    N = length(Ub)
    grad_factors = Vector{Matrix{T}}(undef, N)
    core_unfolds = [unfold_mode(Gb, m) for m in eachindex(Ub)]
    core_gram_factors =
        [core_unfolds[m] * transpose(core_unfolds[m]) for m in eachindex(Ub)]
    oneT = one(T)
    zeroT = zero(T)

    @inbounds for m in eachindex(Ub)
        Gbm = core_unfolds[m]
        GbmT = transpose(Gbm)
        nmode = size(Ub[m], 1)
        rmode = size(Ub[m], 2)

        target_proj = _tucker_project_target_except_mode(backend, pb, backend.target, m)
        target_unfold = unfold_mode(target_proj, m)
        grad_m = Matrix{T}(undef, nmode, rmode)
        mul!(grad_m, target_unfold, GbmT, -oneT, zeroT)

        mul!(grad_m, Ub[m], core_gram_factors[m], oneT, oneT)

        for c in eachindex(parts)
            c == b && continue
            _, Uc = _tucker_data(parts[c])
            cross_unfold =
                unfold_mode(_tucker_cross_except_mode(backend, pb, parts[c], m), m)
            cross_tmp = Matrix{T}(undef, size(cross_unfold, 1), rmode)
            mul!(cross_tmp, cross_unfold, GbmT)
            mul!(grad_m, Uc[m], cross_tmp, oneT, oneT)
        end
        grad_factors[m] = grad_m
    end

    return grad_factors
end

function _btd_block_egrad(backend::BTDBackend, parts, b::Int)
    pb = parts[b]
    pb isa Manifolds.TuckerPoint || throw(
        ArgumentError(
            "BTD gradient expects TuckerPoint blocks, got $(typeof(pb)) at block $b.",
        ),
    )

    Gb, Ub = _tucker_data(pb)
    N = length(Ub)

    # Previous ambient-residual path kept for comparison:
    #
    # residual = _join_residual!(backend, wrap_like_point(parts, parts))
    # return _tucker_egrad(backend.manifolds[b], pb, residual)

    grad_core = _btd_core_grad_block(backend, parts, b)
    grad_factors = _btd_factor_grad_block(backend, parts, b)
    return Manifolds.TuckerTangentVector(grad_core, Tuple(grad_factors))
end

function _btd_egrad(backend::BTDBackend, p)
    parts = point_parts(p)
    _check_parts_len(parts, backend.r, "BTD egrad")
    vals = ntuple(k -> _btd_block_egrad(backend, parts, k), backend.r)
    return wrap_like_point(p, vals)
end
