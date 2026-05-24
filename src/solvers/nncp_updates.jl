# solvers/nncp_updates.jl — nonnegative CP update


@inline _cp_update_policy(nonnegative::Bool, nn_update) =
    _cp_update_policy(nonnegative, nn_update_policy(nn_update))

@inline _cp_update_policy(nonnegative::Bool, ::AutomaticNNUpdate) =
    nonnegative ? :nnls : :ls

@inline function _cp_update_policy(nonnegative::Bool, ::LeastSquaresNNUpdate)
    nonnegative && throw(
        ArgumentError(
            "nn_update=:ls requires nonnegative=false. Use :mu, :hals, or :nnls when nonnegative=true.",
        ),
    )
    return :ls
end

@inline function _cp_update_policy(
    nonnegative::Bool,
    policy::Union{MultiplicativeNNUpdate,HALSNNUpdate,NNLSUpdate},
)
    nonnegative || throw(
        ArgumentError(
            "nn_update=$(nn_update_symbol(policy)) requires nonnegative=true. Use nn_update=:ls or omit the option.",
        ),
    )
    return nn_update_symbol(policy)
end

@inline function _clamp_nonnegative!(A::AbstractArray{T}) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    A .= max.(A, floor)
    return A
end

@inline function _clamp_nonnegative!(
    λ::AbstractVector{T},
    U::AbstractVector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    _clamp_nonnegative!(λ)
    @inbounds for m in eachindex(U)
        _clamp_nonnegative!(U[m])
    end
    return λ, U
end

function _nncp_mu_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    denom::AbstractMatrix{T},
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    U .= max.(U .* M_mttkrp ./ (denom .+ floor), floor)
    return U
end

function _nncp_hals_mode_update!(
    U::StridedMatrix{T},
    M_mttkrp::StridedMatrix{T},
    V::StridedMatrix{T},
    work::StridedMatrix{T},
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    mul!(work, U, V)
    @inbounds for k in axes(U, 2)
        vkk = max(V[k, k], floor)
        for i in axes(U, 1)
            old = U[i, k]
            new = max((M_mttkrp[i, k] - work[i, k] + old * vkk) / vkk, floor)
            Δ = new - old
            U[i, k] = new
            if !iszero(Δ)
                for j in axes(U, 2)
                    work[i, j] += Δ * V[k, j]
                end
            end
        end
    end
    return U
end

function _nncp_hals_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    work::AbstractMatrix{T},
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    d = max.(diag(V), floor)
    d_row = reshape(d, 1, :)
    mul!(work, U, V)
    U .= max.((M_mttkrp .- work .+ U .* d_row) ./ d_row, floor)
    return U
end

function _nncp_nnls_row_update!(
    x::StridedVector{T},
    g::StridedVector{T},
    V::StridedMatrix{T},
    work::StridedVector{T};
    max_cd_sweeps::Int = 10,
    row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    mul!(work, V, x)
    @inbounds for _ in eachindex(Base.OneTo(max_cd_sweeps))
        max_delta = zero(T)
        max_x = maximum(x)
        for k in eachindex(x)
            vkk = max(V[k, k], floor)
            old = x[k]
            new = max((g[k] - work[k] + old * vkk) / vkk, floor)
            Δ = new - old
            x[k] = new
            max_delta = max(max_delta, abs(Δ))
            if !iszero(Δ)
                for j in eachindex(work)
                    work[j] += Δ * V[j, k]
                end
            end
        end
        max_delta <= row_tol * max(max_x, one(T)) && break
    end
    return x
end

function _nncp_nnls_row_update!(
    x::AbstractVector{T},
    g::AbstractVector{T},
    V::AbstractMatrix{T},
    work::AbstractVector{T};
    max_cd_sweeps::Int = 10,
    row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    d = max.(diag(V), floor)
    @inbounds for _ in eachindex(Base.OneTo(max_cd_sweeps))
        mul!(work, V, x)
        x_new = max.(x .- (work .- g) ./ d, floor)
        max_delta = maximum(abs.(x_new .- x))
        copyto!(x, x_new)
        max_delta <= row_tol * max(maximum(x), one(T)) && break
    end
    return x
end

function _nncp_nnls_mode_update!(
    U::StridedMatrix{T},
    M_mttkrp::StridedMatrix{T},
    V::StridedMatrix{T},
    work::StridedMatrix{T};
    max_cd_sweeps::Int = 10,
    row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    @inbounds for i in axes(U, 1)
        _nncp_nnls_row_update!(
            view(U, i, :),
            view(M_mttkrp, i, :),
            V,
            view(work, i, :);
            max_cd_sweeps,
            row_tol,
        )
    end
    return U
end

function _nncp_nnls_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    work::AbstractMatrix{T};
    max_cd_sweeps::Int = 10,
    row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    d = max.(diag(V), floor)
    d_row = reshape(d, 1, :)
    @inbounds for _ in eachindex(Base.OneTo(max_cd_sweeps))
        mul!(work, U, V)
        U_new = max.(U .- (work .- M_mttkrp) ./ d_row, floor)
        max_delta = maximum(abs.(U_new .- U))
        copyto!(U, U_new)
        max_delta <= row_tol * max(maximum(U), one(T)) && break
    end
    return U
end

function _nncp_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    denom::AbstractMatrix{T};
    nn_update,
    nnls_max_cd_sweeps::Int = 10,
    nnls_row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    return _nncp_mode_update!(
        U,
        M_mttkrp,
        V,
        denom,
        nn_update_policy(nn_update);
        nnls_max_cd_sweeps,
        nnls_row_tol,
    )
end

@inline function _nncp_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    denom::AbstractMatrix{T},
    ::MultiplicativeNNUpdate;
    nnls_max_cd_sweeps::Int = 10,
    nnls_row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    mul!(denom, U, V)
    return _nncp_mu_mode_update!(U, M_mttkrp, denom)
end


@inline function _nncp_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    denom::AbstractMatrix{T},
    ::HALSNNUpdate;
    nnls_max_cd_sweeps::Int = 10,
    nnls_row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    return _nncp_hals_mode_update!(U, M_mttkrp, V, denom)
end

@inline function _nncp_mode_update!(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    V::AbstractMatrix{T},
    denom::AbstractMatrix{T},
    ::NNLSUpdate;
    nnls_max_cd_sweeps::Int = 10,
    nnls_row_tol::T = sqrt(eps(T)),
) where {T<:AbstractFloat}
    return _nncp_nnls_mode_update!(
        U,
        M_mttkrp,
        V,
        denom;
        max_cd_sweeps = nnls_max_cd_sweeps,
        row_tol = nnls_row_tol,
    )
end

@inline function _projected_grad_sq_nonnegative(
    U::AbstractMatrix{T},
    M_mttkrp::AbstractMatrix{T},
    denom::AbstractMatrix{T},
) where {T<:AbstractFloat}
    floor = sqrt(eps(T))
    active_floor = 10 * floor
    grad = denom .- M_mttkrp
    pg = ifelse.(U .<= active_floor, min.(grad, zero(T)), grad)
    return sum(abs2, pg)
end

function _projected_grad_norm_nonnegative!(
    A::AbstractArray{T,N},
    U::AbstractVector{<:AbstractMatrix{T}},
    grams::AbstractVector{<:AbstractMatrix{T}},
    V::AbstractMatrix{T},
    denom_work::AbstractVector{<:AbstractMatrix{T}},
    mttkrp_bufs::AbstractVector{<:AbstractMatrix{T}},
    mttkrp_tmp_work::AbstractVector,
    mttkrp_kr_work::AbstractVector,
    mttkrp_kr_work2::AbstractVector;
    mttkrp_method::Symbol = :auto,
) where {T<:AbstractFloat,N}
    sq = zero(T)
    u_sq = zero(T)
    for n in eachindex(U)
        _hadamard_G_except!(V, grams, n)
        M_mttkrp = mttkrp!(
            mttkrp_bufs[n],
            A,
            U,
            n;
            method = mttkrp_method,
            work = mttkrp_tmp_work[n],
            kr_buf = mttkrp_kr_work[n],
            kr_work = mttkrp_kr_work2[n],
        )
        _clamp_nonnegative!(M_mttkrp)
        mul!(denom_work[n], U[n], V)
        sq += _projected_grad_sq_nonnegative(U[n], M_mttkrp, denom_work[n])
        u_sq += sum(abs2, U[n])
    end
    return sqrt(sq / max(u_sq, one(T)))
end
