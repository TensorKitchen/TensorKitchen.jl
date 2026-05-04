

function _rand_unit_matrix(rng::AbstractRNG, d::Int, r::Int)
    M = randn(rng, d, r)
    for j = 1:r
        nrm = norm(view(M, :, j))
        if nrm > 0
            @views M[:, j] ./= nrm
        else
            M[1, j] = one(eltype(M))
        end
    end
    return M
end

function _rand_orthonormal_matrix(rng::AbstractRNG, d::Int, r::Int)
    Q = Matrix(qr(randn(rng, d, r)).Q)
    return Q[:, 1:r]
end

function _add_relative_noise(
    rng::AbstractRNG,
    A::AbstractArray{T};
    level::Float64 = 1e-2,
) where {T}
    E = randn(rng, size(A))
    α = level * norm(A) / max(norm(E), eps(Float64))
    return A .+ T(α) .* E
end

function _make_cp_tensor(seed::Int; dims = (18, 16, 14), r::Int = 3, noisy::Bool = false)
    rng = MersenneTwister(seed)
    λ = rand(rng, r) .+ 0.5
    U = [_rand_unit_matrix(rng, d, r) for d in dims]
    A = reconstruct_cpd_rankr(λ, U)
    return noisy ? _add_relative_noise(rng, A; level = 1e-2) : A
end

function _make_tucker_tensor(
    seed::Int;
    dims = (20, 16, 12),
    ranks = (4, 3, 2),
    noisy::Bool = false,
)
    rng = MersenneTwister(seed)
    core = randn(rng, ranks...)
    factors = [_rand_orthonormal_matrix(rng, dims[n], ranks[n]) for n in eachindex(dims)]
    A = reconstruct_tucker(core, factors)
    return noisy ? _add_relative_noise(rng, A; level = 1e-2) : A
end


@testset "cpd exact synthetic regression" begin
    rels = Float64[]
    for seed = 1:3
        A = _make_cp_tensor(seed; noisy = false)
        Random.seed!(10_000 + seed)
        res = cpd(
            A,
            3;
            solver = :als,
            init = :tucker,
            maxiter = 80,
            tol = 1e-8,
            verbose = false,
        )
        push!(rels, Float64(res.rel_error))
        @test res.converged
        @test res.rel_error < 1e-7
        @test res.iterations <= 80
        @test isfinite(res.grad_norm)
    end
    @test median(rels) < 1e-8
end

@testset "cpd noisy synthetic regression" begin
    rels = Float64[]
    for seed = 1:3
        A = _make_cp_tensor(seed; noisy = true)
        Random.seed!(20_000 + seed)
        res = cpd(
            A,
            3;
            solver = :als,
            init = :tucker,
            maxiter = 80,
            tol = 1e-8,
            verbose = false,
        )
        push!(rels, Float64(res.rel_error))
        @test res.converged
        @test res.rel_error < 0.02
        @test res.iterations <= 80
        @test isfinite(res.grad_norm)
    end
    @test median(rels) < 0.01
end

@testset "tucker exact synthetic regression" begin
    methods = (:hosvd, :sthosvd, :thosvd, :hooi)
    for method in methods
        for seed = 1:3
            A = _make_tucker_tensor(seed; noisy = false)
            kwargs = method == :hooi ? (; maxiter = 20, tol = 1e-8, verbose = false) : (;)
            td = tucker(A, (4, 3, 2); method = method, kwargs...)
            @test rel_error(A, td) < 1e-12
        end
    end
end

@testset "tucker noisy synthetic regression" begin
    for seed = 1:3
        A = _make_tucker_tensor(seed; noisy = true)
        td_hosvd = tucker(A, (4, 3, 2); method = :hosvd)
        td_hooi =
            tucker(A, (4, 3, 2); method = :hooi, maxiter = 20, tol = 1e-8, verbose = false)
        err_hosvd = rel_error(A, td_hosvd)
        err_hooi = rel_error(A, td_hooi)
        @test err_hosvd < 0.02
        @test err_hooi < 0.02
        @test err_hooi <= err_hosvd + 1e-8
    end
end
