# Shifted symmetric higher-order power iteration on symmetric target operators.

export SSHOPM,
    TensorEigenpairResult,
    tensor_eigenpair,
    best_symmetric_rank1,
    eigenvalue,
    eigenvector,
    residual_norm

@doc raw"""
    SSHOPM(; shift=nothing)

Configure the shifted symmetric higher-order power method. With a unit vector
``x_k``, target contraction ``b_k=A(I,x_k,\ldots,x_k)``, direction sign
``\chi\in\{-1,1\}``, and signed shift ``\alpha``, one iteration is

```math
x_{k+1}
=\chi\frac{b_k+\alpha x_k}{\|b_k+\alpha x_k\|_2}.
```

`shift=nothing` selects a value slightly above the conservative magnitude
``(D-1)\|A\|_F``. A nonnegative numeric `shift` uses that magnitude directly;
[`tensor_eigenpair`](@ref) chooses its sign from `which`.

This is Algorithm 2 of T. G. Kolda and J. R. Mayo, "Shifted Power Method for
Computing Tensor Eigenpairs," *SIAM Journal on Matrix Analysis and
Applications* 32(4) (2011), 1095--1124,
doi:10.1137/100801482. The automatic value uses the Frobenius-norm upper bound
on the paper's curvature threshold; it is safe but may be conservative.
"""
struct SSHOPM{S}
    shift::S
end

function SSHOPM(; shift = nothing)
    isnothing(shift) ||
        (shift isa Real && isfinite(shift) && shift >= 0) ||
        throw(ArgumentError("shift must be nothing or a finite nonnegative real value."))
    return SSHOPM{typeof(shift)}(shift)
end

"""
    TensorEigenpairResult

Result returned by [`tensor_eigenpair`](@ref). `value` and `vector` satisfy
`contract(target, vector) ≈ value * vector`; `residual_norm` measures the norm
of this equation's residual.
"""
struct TensorEigenpairResult{T<:AbstractFloat,V<:AbstractVector{T}}
    value::T
    vector::V
    residual_norm::T
    iterations::Int
    converged::Bool
    shift::T
    direction::Symbol
end

"""Return the tensor eigenvalue stored in a [`TensorEigenpairResult`](@ref)."""
eigenvalue(result::TensorEigenpairResult) = result.value

"""Return the unit eigenvector stored in a [`TensorEigenpairResult`](@ref)."""
eigenvector(result::TensorEigenpairResult) = result.vector

@doc raw"""Return ``\|A(I,x,\ldots,x)-\lambda x\|_2`` for a computed eigenpair."""
residual_norm(result::TensorEigenpairResult) = result.residual_norm
iterations(result::TensorEigenpairResult) = result.iterations
converged(result::TensorEigenpairResult) = result.converged
solver(::TensorEigenpairResult) = :sshopm

function _sshopm_shift(target::AbstractSymmetricTarget{T}, method::SSHOPM) where {T}
    _, order = _symmetric_target_size(target)
    magnitude = if isnothing(method.shift)
        T(order - 1) * sqrt(target_norm2(target))
    else
        T(method.shift)
    end
    isnothing(method.shift) || return max(magnitude, eps(T))
    return magnitude + sqrt(eps(T)) * max(one(T), magnitude)
end

function _sshopm_start(target::AbstractSymmetricTarget{T}, x0, rng::AbstractRNG) where {T}
    n, _ = _symmetric_target_size(target)
    x = isnothing(x0) ? randn(rng, T, n) : T.(x0)
    length(x) == n || throw(DimensionMismatch("Expected an initial vector of length $n."))
    all(isfinite, x) || throw(ArgumentError("The initial vector must be finite."))
    xnorm = norm(x)
    xnorm > zero(T) || throw(ArgumentError("The initial vector must be nonzero."))
    return x ./ xnorm
end

@doc raw"""
    tensor_eigenpair(target; method=SSHOPM(), x0=nothing,
                     which=:largest, maxiter=500, tol=1e-10, rng=Random.default_rng())

Compute a real unit-norm symmetric tensor eigenpair

```math
A(I,x,\ldots,x)=\lambda x,
\qquad \|x\|_2=1,
```

using only [`contract`](@ref) and [`evaluate`](@ref). `which=:largest` uses a
positive shifted iteration and `which=:smallest` uses its negative counterpart.
These names identify the two SS-HOPM stability directions; one start does not
guarantee a globally extremal eigenvalue. Use [`best_symmetric_rank1`](@ref)
for a multistart search by absolute eigenvalue.

The update and convergence target follow Kolda and Mayo, "Shifted Power Method
for Computing Tensor Eigenpairs," *SIAM J. Matrix Anal. Appl.* 32(4) (2011),
1095--1124, Algorithm 2, doi:10.1137/100801482.
"""
function tensor_eigenpair(
    target::AbstractSymmetricTarget{T};
    method::SSHOPM = SSHOPM(),
    x0 = nothing,
    which::Symbol = :largest,
    maxiter::Int = 500,
    tol::Real = 1.0e-10,
    rng::AbstractRNG = Random.default_rng(),
) where {T}
    _, order = _symmetric_target_size(target)
    order >= 2 || throw(ArgumentError("SS-HOPM requires a tensor order of at least two."))
    which in (:largest, :smallest) ||
        throw(ArgumentError("which must be :largest or :smallest, got $which."))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive, got $maxiter."))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol."))

    x = _sshopm_start(target, x0, rng)
    shift_magnitude = _sshopm_shift(target, method)
    direction_sign = which == :largest ? one(T) : -one(T)
    signed_shift = direction_sign * shift_magnitude
    tolerance = T(tol)
    value = T(evaluate(target, x))
    residual = T(Inf)

    for iteration = 1:maxiter
        b = T.(contract(target, x))
        y = b .+ signed_shift .* x
        ynorm = norm(y)
        ynorm > eps(T) * max(one(T), norm(b)) || throw(
            ArgumentError(
                "SS-HOPM encountered a zero shifted update; choose a different shift or start.",
            ),
        )
        x = direction_sign .* (y ./ ynorm)
        value = T(evaluate(target, x))
        eigen_residual = T.(contract(target, x)) .- value .* x
        residual = norm(eigen_residual)
        scale = max(one(T), abs(value), norm(eigen_residual .+ value .* x))
        if residual <= tolerance * scale
            return TensorEigenpairResult(
                value,
                Vector{T}(x),
                T(residual),
                iteration,
                true,
                signed_shift,
                which,
            )
        end
    end

    return TensorEigenpairResult(
        value,
        Vector{T}(x),
        T(residual),
        maxiter,
        false,
        signed_shift,
        which,
    )
end

function _sshopm_candidates(
    target::AbstractSymmetricTarget;
    method::SSHOPM,
    x0,
    starts::Int,
    maxiter::Int,
    tol::Real,
    rng::AbstractRNG,
)
    starts >= 1 || throw(ArgumentError("starts must be positive, got $starts."))
    candidates = TensorEigenpairResult[]
    n, _ = _symmetric_target_size(target)
    T = eltype(target)
    for start = 1:starts
        start_vector = start == 1 && !isnothing(x0) ? x0 : normalize(randn(rng, T, n))
        for which in (:largest, :smallest)
            push!(
                candidates,
                tensor_eigenpair(
                    target;
                    method,
                    x0 = start_vector,
                    which,
                    maxiter,
                    tol,
                    rng,
                ),
            )
        end
    end
    return candidates
end

@doc raw"""
    best_symmetric_rank1(target; method=SSHOPM(), x0=nothing, starts=8, ...)

Search for a symmetric rank-one approximation

```math
\min_{\lambda,\,\|x\|_2=1}
\frac12\|A-\lambda x^{\otimes D}\|_F^2.
```

For a fixed unit vector, the optimal weight is
``\lambda=\langle A,x^{\otimes D}\rangle``. The routine therefore runs both
SS-HOPM stability directions from multiple starts and returns the converged
candidate with largest ``|\lambda|`` as a rank-one [`SymCPDResult`](@ref).
SS-HOPM is a local method, so `starts` controls search breadth rather than
providing a global-optimality certificate.

The equivalence between symmetric tensor eigenpairs and stationary symmetric
rank-one approximation is described in T. G. Kolda and J. R. Mayo, *SIAM J.
Matrix Anal. Appl.* 32(4) (2011), 1095--1124, Sections 2--3,
doi:10.1137/100801482.
"""
function best_symmetric_rank1(
    target::AbstractSymmetricTarget{T};
    method::SSHOPM = SSHOPM(),
    x0 = nothing,
    starts::Int = 8,
    maxiter::Int = 500,
    tol::Real = 1.0e-10,
    rng::AbstractRNG = Random.default_rng(),
) where {T}
    candidates = _sshopm_candidates(target; method, x0, starts, maxiter, tol, rng)
    converged_candidates = filter(converged, candidates)
    pool = isempty(converged_candidates) ? candidates : converged_candidates
    selected = pool[argmax(abs(eigenvalue(candidate)) for candidate in pool)]

    model = SymCPDModel(target, 1)
    component_point = _symcpd_point(eigenvalue(selected), copy(eigenvector(selected)))
    p = join_point(manifold(model), (component_point,))
    final_cost = cost(model, p)
    gradient_norm = norm(manifold(model), p, rgrad(model, p))
    norm2 = target_norm2(target)
    relative_error =
        norm2 > zero(T) ? sqrt(max(T(2) * final_cost, zero(T)) / norm2) :
        sqrt(max(T(2) * final_cost, zero(T)))
    result = (
        point = p,
        cost = final_cost,
        rel_error = relative_error,
        grad_norm = gradient_norm,
        iterations = sum(iterations, candidates),
        converged = converged(selected),
        solver = :sshopm,
        solver_info = (
            selected_eigenpair = selected,
            candidates = candidates,
            starts = starts,
            converged_candidates = count(converged, candidates),
        ),
    )
    n, order = _symmetric_target_size(target)
    return _symcpd_result(model, result, n, order)
end

function _sshopm_initial_point(
    model::JoinModel{T,B};
    starts::Int = max(8, 4 * model.backend.rank),
    maxiter::Int = 500,
    tol::Real = 1.0e-8,
    max_correlation::Real = 0.98,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    candidates = _sshopm_candidates(
        backend.target;
        method = SSHOPM(),
        x0 = nothing,
        starts,
        maxiter,
        tol,
        rng,
    )
    sort!(
        candidates;
        by = candidate -> (converged(candidate), abs(eigenvalue(candidate))),
        rev = true,
    )
    selected = TensorEigenpairResult[]
    for candidate in candidates
        all(
            previous ->
                abs(dot(eigenvector(previous), eigenvector(candidate))) <= max_correlation,
            selected,
        ) || continue
        push!(selected, candidate)
        length(selected) == backend.rank && break
    end

    while length(selected) < backend.rank
        x = normalize(randn(rng, T, backend.n))
        value = T(evaluate(backend.target, x))
        push!(
            selected,
            TensorEigenpairResult(value, x, T(Inf), 0, false, zero(T), :random_fill),
        )
    end

    factors = [copy(eigenvector(candidate)) for candidate in selected]
    gram = Matrix{T}(undef, backend.rank, backend.rank)
    rhs = Vector{T}(undef, backend.rank)
    for r = 1:backend.rank
        rhs[r] = T(evaluate(backend.target, factors[r]))
        for s = 1:backend.rank
            gram[r, s] = dot(factors[r], factors[s])^backend.order
        end
    end
    weights = try
        gram \ rhs
    catch error
        error isa LinearAlgebra.SingularException || rethrow()
        pinv(gram) * rhs
    end
    weight_floor = sqrt(eps(T)) * max(one(T), sqrt(target_norm2(backend.target)))
    parts = ntuple(backend.rank) do r
        weight =
            abs(weights[r]) > weight_floor ? weights[r] : copysign(weight_floor, rhs[r])
        _symcpd_point(T(weight), factors[r])
    end
    return join_point(backend.product_manifold, parts)
end
