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
    symcpd(A, r; solver=:gn_cg, target_backend=:dense, init=:auto, ...)
    symcpd(target::AbstractSymmetricTarget, r; solver=:gn_cg, init=:auto, ...)

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

- `init=:auto` uses SS-HOPM for rank one and random product-manifold
  components for rank greater than one. An explicit `p0` takes precedence.
- `init=:random` samples product-manifold components.
- `init=:sshopm` uses multistart shifted power iterations, removes nearly
  collinear candidates, and solves a small kernel least-squares problem for
  the initial weights.
- `init=:cls` (or `:normalized_cls`) runs normalized conditional least
  squares and then refits all weights by solving the exact linear subproblem
  ``K\lambda=c``. Use [`NormalizedCLSInit`](@ref) to configure its sweep and
  regularization parameters.

# Solvers

- `:rgd`, `:rcg`, and `:lbfgs` use TensorKitchen's first-order Riemannian path.
- `:gn_cg` or `GaussNewtonSolver(linear_solver=:cg)` uses damped Riemannian
  Gauss--Newton with the analytic matrix-free `J'J` operator and tangent CG.
- `:gn_dense` or `GaussNewtonSolver(linear_solver=:dense)` builds the intrinsic
  `r*N` square normal matrix from operator columns and solves it directly. It
  is intended for validation and small problems.
- `:cls` or [`SymmetricCLS`](@ref) runs normalized conditional least squares
  as a standalone solver.
  Every sweep uses tensor contractions instead of an unfolding, normalizes
  the updated columns, and exactly refits the component weights.

Set `variable_projection=true` with `solver=:rcg` or `:lbfgs` to optimize only
the factor directions on a product of spheres. At every objective and gradient
evaluation, the weights are eliminated analytically as

```math
\lambda^\star(X)=K(X)^\dagger c(X),\qquad
K_{rs}(X)=(x_r^\top x_s)^D.
```

This removes the `r` radial variables and their scale conditioning from the
nonlinear optimization. `weight_pinv_rtol` selects the numerical range of the
small positive-semidefinite weight system. Reduced Gauss--Newton is not used:
its projected Jacobian requires a separate Schur-complement implementation.

The product-of-Veronese formulation and GN block equations follow R. Khouja,
H. Khalil, and B. Mourrain, "Riemannian Newton optimization methods for the
symmetric tensor approximation problem," *Linear Algebra and its Applications*
637 (2022), 175--211, doi:10.1016/j.laa.2021.12.008. The operator-CG execution
follows the matrix-free tensor GN pattern of Sorber, Van Barel, and De
Lathauwer (2013), doi:10.1137/120868323, and Singh, Ma, Yang, and Solomonik
(2021), doi:10.1137/20M1344561. Weight elimination follows G. H. Golub and
V. Pereyra (1973), doi:10.1137/0710036. The conditional least-squares update
follows G. Favier, A. Y. Kibangou, and T. Bouilloc (2012),
doi:10.1002/acs.1272.
"""
function symcpd(
    A::AbstractArray{<:Real},
    r::Int;
    init = :auto,
    p0 = nothing,
    solver = :gn_cg,
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
    damping::Union{Nothing,Real} = nothing,
    damping_increase::Union{Nothing,Real} = nothing,
    damping_decrease::Union{Nothing,Real} = nothing,
    inner::Union{Nothing,InnerSolveOptions} = nothing,
    max_damping_trials::Union{Nothing,Int} = nothing,
    acceptance_ratio::Union{Nothing,Real} = nothing,
    poor_step_ratio::Union{Nothing,Real} = nothing,
    good_step_ratio::Union{Nothing,Real} = nothing,
    variable_projection::Bool = false,
    weight_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_damping::Real = 1.0e-10,
    cls_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_weight_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_patience::Int = 3,
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
        inner,
        max_damping_trials,
        acceptance_ratio,
        poor_step_ratio,
        good_step_ratio,
        variable_projection,
        weight_pinv_rtol,
        cls_damping,
        cls_pinv_rtol,
        cls_weight_pinv_rtol,
        cls_patience,
        kwargs...,
    )
end

function symcpd(
    target::AbstractSymmetricTarget,
    r::Int;
    init = :auto,
    p0 = nothing,
    solver = :gn_cg,
    maxiter::Int = 500,
    stepsize::Real = 1.0,
    tol::Real = 1.0e-6,
    gradient_mode = :riemannian,
    verbose::Bool = true,
    vector_transport_method = nothing,
    damping::Union{Nothing,Real} = nothing,
    damping_increase::Union{Nothing,Real} = nothing,
    damping_decrease::Union{Nothing,Real} = nothing,
    inner::Union{Nothing,InnerSolveOptions} = nothing,
    max_damping_trials::Union{Nothing,Int} = nothing,
    acceptance_ratio::Union{Nothing,Real} = nothing,
    poor_step_ratio::Union{Nothing,Real} = nothing,
    good_step_ratio::Union{Nothing,Real} = nothing,
    variable_projection::Bool = false,
    weight_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_damping::Real = 1.0e-10,
    cls_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_weight_pinv_rtol::Union{Nothing,Real} = nothing,
    cls_patience::Int = 3,
    kwargs...,
)
    r >= 1 || throw(ArgumentError("symcpd requires r >= 1, got $r."))
    n, d = _symmetric_target_size(target)
    component = SymmetricRankOne(n, d)
    model = JoinModel(component, r, target)
    target_type = eltype(target)
    default_pinv_rtol = sqrt(eps(target_type))
    weight_pinv_rtol_eff =
        isnothing(weight_pinv_rtol) ? default_pinv_rtol : weight_pinv_rtol
    cls_pinv_rtol_eff = isnothing(cls_pinv_rtol) ? default_pinv_rtol : cls_pinv_rtol
    cls_weight_pinv_rtol_eff =
        isnothing(cls_weight_pinv_rtol) ? cls_pinv_rtol_eff : cls_weight_pinv_rtol
    result = if solver == :cls || solver isa SymmetricCLS
        variable_projection && throw(
            ArgumentError(
                "solver=:cls and variable_projection=true are distinct execution paths; " *
                "use solver=:rcg or :lbfgs for variable projection.",
            ),
        )
        cls_solver = if solver isa SymmetricCLS
            solver
        else
            SymmetricCLS(
                damping = cls_damping,
                pinv_rtol = cls_pinv_rtol_eff,
                weight_pinv_rtol = cls_weight_pinv_rtol_eff,
                patience = cls_patience,
            )
        end
        solve(cls_solver, model; init, p0, maxiter, tol, verbose, return_stats = true)
    elseif variable_projection
        _solve_symcpd_varpro(
            model;
            init,
            p0,
            solver,
            maxiter,
            stepsize,
            tol,
            verbose,
            vector_transport_method,
            pinv_rtol = weight_pinv_rtol_eff,
            kwargs...,
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
            damping,
            damping_increase,
            damping_decrease,
            inner = solver == :lm && isnothing(inner) ?
                    _default_symcpd_inner_options() : inner,
            max_damping_trials,
            acceptance_ratio,
            poor_step_ratio,
            good_step_ratio,
            kwargs...,
        )
    end
    return _symcpd_result(model, result, n, d)
end
