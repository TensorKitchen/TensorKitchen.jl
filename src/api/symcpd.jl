export symcpd, compressed_coordinates, compress_symmetric_tensor, expand_symmetric_tensor

function _symmetric_tensor_manifold(A::AbstractArray)
    d = ndims(A)
    d >= 1 || throw(ArgumentError("symcpd requires an array of order at least one."))
    n = size(A, 1)
    all(==(n), size(A)) ||
        throw(DimensionMismatch("symcpd requires equal mode sizes, got $(size(A))."))
    return _symcpd_manifold(n, d), n, d
end

function _check_symmetric_input(A; atol, rtol)
    for I in CartesianIndices(A)
        canonical = Tuple(sort(collect(Tuple(I)); rev = true))
        isapprox(A[I], A[canonical...]; atol, rtol) || throw(
            ArgumentError(
                "symcpd input is not symmetric; entries $I and $canonical differ.",
            ),
        )
    end
    return nothing
end

@doc raw"""
    compress_symmetric_tensor(A)

Convert a dense tensor into the orthonormal symmetric coordinates used by
TensorKitchen's Bombieri--Weyl basis. An arbitrary nonsymmetric input is first
orthogonally projected onto the symmetric tensor subspace. This is an
input-boundary operation; it does not allocate a second full tensor.

The Bombieri--Weyl normalization is the polynomial inner product used by
Khouja, Khalil, and Mourrain (2022), doi:10.1016/j.laa.2021.12.008.
"""
function compress_symmetric_tensor(A::AbstractArray{<:Real})
    M, _, _ = _symmetric_tensor_manifold(A)
    return _compress_symmetric_tensor(M, A)
end

@doc raw"""
    expand_symmetric_tensor(a, n, d)

Expand orthonormal symmetric coordinates into a dense order-`d` tensor of
shape `(n, ..., n)`. This allocates `n^d` entries at the output boundary.
The coordinate normalization follows Khouja, Khalil, and Mourrain (2022),
doi:10.1016/j.laa.2021.12.008.
"""
function expand_symmetric_tensor(a::AbstractVector{<:Real}, n::Int, d::Int)
    return _expand_symmetric_tensor(_symcpd_manifold(n, d), a)
end

@doc raw"""
    symcpd(A, r; solver=:rgd, target_backend=:dense, init=:random, ...)
    symcpd(target::AbstractSymmetricTarget, r; solver=:rgd, init=:random, ...)

Approximate a dense symmetric order-`D` tensor by

```math
\widehat A=\sum_{k=1}^r\lambda_kx_k^{\otimes D},
\qquad \|x_k\|_2=1,
```

using `JoinModel(SymmetricRankOne(N, D), r, target)`. Thus symmetric CPD uses
the same join-of-rank-one-components architecture as ordinary CPD, with
Veronese geometry replacing Segre geometry. `cost`, `rgrad`, and the
Gauss--Newton normal action never construct `Ahat`, an ambient residual, or
compressed model-coordinate vectors.

Passing an [`AbstractSymmetricTarget`](@ref) directly skips dense input
preparation and permits operator-defined targets such as
[`FunctionalSymmetricTarget`](@ref).

# Backends

- `target_backend=:dense` retains `A` and evaluates target contractions
  directly.
- `target_backend=:compressed` converts `A` once to the Bombieri--Weyl
  orthonormal symmetric basis. Model-model terms remain kernelized.
- Passing a `FunctionalSymmetricTarget` evaluates a user-supplied polynomial
  and contraction operator without storing the tensor.

# Initialization

- `init=:random` samples product-manifold components.
- `init=:sshopm` uses multistart shifted power iterations, removes nearly
  collinear candidates, and solves a small kernel least-squares problem for
  the initial weights.

# Solvers

- `:rgd`, `:rcg`, and `:lbfgs` use TensorKitchen's first-order Riemannian path.
- `:gn_cg` uses damped Riemannian Gauss--Newton with the analytic matrix-free
  `J'J` operator and tangent CG.
- `:gn_dense` builds the intrinsic `r*N` square normal matrix from operator
  columns and solves it directly. It is intended for validation and small
  problems.

The product-of-Veronese formulation and GN block equations follow R. Khouja,
H. Khalil, and B. Mourrain, "Riemannian Newton optimization methods for the
symmetric tensor approximation problem," *Linear Algebra and its Applications*
637 (2022), 175--211, doi:10.1016/j.laa.2021.12.008. The operator-CG execution
follows the matrix-free tensor GN pattern of Sorber, Van Barel, and De
Lathauwer (2013), doi:10.1137/120868323, and Singh, Ma, Yang, and Solomonik
(2021), doi:10.1137/20M1344561.
"""
function symcpd(
    A::AbstractArray{<:Real},
    r::Int;
    init = :random,
    p0 = nothing,
    solver = :rgd,
    target_backend::Symbol = :dense,
    maxiter::Int = 500,
    stepsize::Real = 1.0,
    tol::Real = 1.0e-6,
    gradient_mode = :riemannian,
    verbose::Bool = true,
    compute_type = nothing,
    check_symmetric::Bool = true,
    symmetry_atol::Real = 0,
    symmetry_rtol::Real = sqrt(eps(float(eltype(A)))),
    vector_transport_method = nothing,
    damping::Real = 1.0e-6,
    damping_increase::Real = 10,
    damping_decrease::Real = 0.3,
    cg_tol::Real = 1.0e-2,
    adaptive_cg::Bool = true,
    cg_min_tol::Real = 1.0e-10,
    cg_forcing_scale::Real = 1,
    cg_forcing_power::Real = 0.5,
    cg_maxiter = nothing,
    max_damping_trials::Int = 8,
    acceptance_ratio::Real = 1.0e-4,
    poor_step_ratio::Real = 0.25,
    good_step_ratio::Real = 0.75,
    kwargs...,
)
    r >= 1 || throw(ArgumentError("symcpd requires r >= 1, got $r."))
    prepared = prepare_tensor(A; compute_type)
    M, n, d = _symmetric_tensor_manifold(prepared)
    check_symmetric &&
        _check_symmetric_input(prepared; atol = symmetry_atol, rtol = symmetry_rtol)
    target = if target_backend == :dense
        DenseSymmetricTarget(prepared)
    elseif target_backend == :compressed
        CompressedSymmetricTarget(_compress_symmetric_tensor(M, prepared), n, d)
    else
        throw(
            ArgumentError(
                "target_backend must be :dense or :compressed, got $target_backend.",
            ),
        )
    end
    return symcpd(
        target,
        r;
        init,
        p0,
        solver,
        maxiter,
        stepsize,
        tol,
        gradient_mode,
        verbose,
        vector_transport_method,
        damping,
        damping_increase,
        damping_decrease,
        cg_tol,
        adaptive_cg,
        cg_min_tol,
        cg_forcing_scale,
        cg_forcing_power,
        cg_maxiter,
        max_damping_trials,
        acceptance_ratio,
        poor_step_ratio,
        good_step_ratio,
        kwargs...,
    )
end

function symcpd(
    target::AbstractSymmetricTarget,
    r::Int;
    init = :random,
    p0 = nothing,
    solver = :rgd,
    maxiter::Int = 500,
    stepsize::Real = 1.0,
    tol::Real = 1.0e-6,
    gradient_mode = :riemannian,
    verbose::Bool = true,
    vector_transport_method = nothing,
    damping::Real = 1.0e-6,
    damping_increase::Real = 10,
    damping_decrease::Real = 0.3,
    cg_tol::Real = 1.0e-2,
    adaptive_cg::Bool = true,
    cg_min_tol::Real = 1.0e-10,
    cg_forcing_scale::Real = 1,
    cg_forcing_power::Real = 0.5,
    cg_maxiter = nothing,
    max_damping_trials::Int = 8,
    acceptance_ratio::Real = 1.0e-4,
    poor_step_ratio::Real = 0.25,
    good_step_ratio::Real = 0.75,
    kwargs...,
)
    r >= 1 || throw(ArgumentError("symcpd requires r >= 1, got $r."))
    n, d = _symmetric_target_size(target)
    component = SymmetricRankOne(n, d)
    model = JoinModel(component, r, target)
    result = if solver in (:gn_cg, :gn_dense)
        cg_iterations =
            isnothing(cg_maxiter) ? max(20 * manifold_dimension(manifold(model)), 200) :
            Int(cg_maxiter)
        _solve_symcpd_gn(
            model;
            init,
            p0,
            maxiter,
            tol,
            linear_solver = solver == :gn_cg ? :cg : :dense,
            damping,
            damping_increase,
            damping_decrease,
            cg_tol,
            adaptive_cg,
            cg_min_tol,
            cg_forcing_scale,
            cg_forcing_power,
            cg_maxiter = cg_iterations,
            max_damping_trials,
            acceptance_ratio,
            poor_step_ratio,
            good_step_ratio,
            verbose,
        )
    else
        _solve_model(
            model;
            init,
            p0,
            solver,
            maxiter,
            stepsize,
            tol,
            gradient_mode,
            normalization = NoNormalization(),
            verbose,
            vector_transport_method,
            observation_norm2_cache = target_norm2(target),
            kwargs...,
        )
    end
    return _symcpd_result(model, result, n, d)
end
