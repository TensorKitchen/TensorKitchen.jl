# Standalone normalized conditional least-squares solver for symmetric CPD.

export SymmetricCLS

@doc raw"""
    SymmetricCLS(; damping=1e-10, pinv_rtol=nothing,
        weight_pinv_rtol=nothing, patience=3)

Standalone normalized conditional least-squares solver for a symmetric CP
model

```math
\widehat{\mathcal T}=\sum_{r=1}^R\lambda_r x_r^{\otimes D},
\qquad \|x_r\|_2=1.
```

For ``X=[x_1,\ldots,x_R]``, one CLS sweep computes

```math
M_{:,r}=\mathcal T(I,x_r,\ldots,x_r),\qquad
H=(X^\top X)^{\circ(D-1)},\qquad
B=MH^\dagger.
```

TensorKitchen evaluates the columns of ``M`` by target contractions, so the
solver does not materialize ``\mathcal T_{(1)}`` or
``X^{\odot(D-1)}``. It normalizes the columns of ``B`` and then exactly refits
the signed weights from

```math
K\lambda=c,\qquad
K_{rs}=(x_r^\top x_s)^D,
\qquad c_r=\langle\mathcal T,x_r^{\otimes D}\rangle.
```

`damping` regularizes the CLS system ``H``. `pinv_rtol=nothing` chooses
``\sqrt{\epsilon(T)}`` for the target element type; `weight_pinv_rtol`
independently controls the exact weight solve and defaults to the resolved CLS
tolerance. `patience` terminates after consecutive sweeps fail to improve the
strict symmetric objective. Because tying the free updated factor back across
all modes is not a block minimization of that strict objective, the solver
returns the best iterate observed rather than assuming monotonicity.

The symmetric PARAFAC setting and its Volterra-kernel use follow G. Favier and
T. Bouilloc, "Parametric complexity reduction of Volterra models using tensor
decompositions," *Proc. EUSIPCO* (2009), 2288--2292. The pseudoinverse ALS
updates, hard/soft symmetric variants, and their convergence limitations are
described by P. Comon, X. Luciani, and A. L. F. de Almeida, "Tensor
decompositions, alternating least squares and other tales," *Journal of
Chemometrics* 23 (2009), 393--405, doi:10.1002/cem.1236. The explicit
conditional-LS formulation for third-order symmetric PARAFAC is given by
G. Favier, A. Y. Kibangou, and T. Bouilloc (2012), doi:10.1002/acs.1272;
TensorKitchen uses its order-``D`` Gram generalization.
"""
struct SymmetricCLS{T<:AbstractFloat,R1,R2} <: AbstractALSSolver
    damping::T
    pinv_rtol::R1
    weight_pinv_rtol::R2
    patience::Int
end

function SymmetricCLS(;
    damping::Real = 1.0e-10,
    pinv_rtol::Union{Nothing,Real} = nothing,
    weight_pinv_rtol::Union{Nothing,Real} = nothing,
    patience::Int = 3,
)
    damping >= 0 || throw(ArgumentError("damping must be nonnegative."))
    isnothing(pinv_rtol) ||
        pinv_rtol >= 0 ||
        throw(ArgumentError("pinv_rtol must be nonnegative."))
    isnothing(weight_pinv_rtol) ||
        weight_pinv_rtol >= 0 ||
        throw(ArgumentError("weight_pinv_rtol must be nonnegative."))
    patience >= 1 || throw(ArgumentError("patience must be positive, got $patience."))
    numeric_types = Type[typeof(float(damping))]
    isnothing(pinv_rtol) || push!(numeric_types, typeof(float(pinv_rtol)))
    isnothing(weight_pinv_rtol) || push!(numeric_types, typeof(float(weight_pinv_rtol)))
    T = promote_type(numeric_types...)
    cls_rtol = isnothing(pinv_rtol) ? nothing : T(pinv_rtol)
    weight_rtol = isnothing(weight_pinv_rtol) ? nothing : T(weight_pinv_rtol)
    return SymmetricCLS{T,typeof(cls_rtol),typeof(weight_rtol)}(
        T(damping),
        cls_rtol,
        weight_rtol,
        patience,
    )
end

solver_symbol(::SymmetricCLS) = :cls

function solve(
    solver::SymmetricCLS,
    model::JoinModel{T,B};
    init = :random,
    p0 = nothing,
    maxiter::Int = 100,
    tol::Real = 1.0e-6,
    normalization::Union{AbstractNormalizationPolicy,Symbol,Nothing} = NoNormalization(),
    verbose::Bool = true,
    return_stats::Bool = false,
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    policy = _normalization_policy(normalization)
    policy isa NoNormalization || throw(
        ArgumentError(
            "SymmetricCLS uses normalized shared factors and explicit weights; " *
            "normalization must be NoNormalization() or :none.",
        ),
    )
    cls_rtol = isnothing(solver.pinv_rtol) ? sqrt(eps(T)) : T(solver.pinv_rtol)
    weight_rtol = isnothing(solver.weight_pinv_rtol) ? cls_rtol : T(solver.weight_pinv_rtol)
    result = _solve_symcpd_cls(
        model;
        init,
        p0,
        maxiter,
        tol,
        damping = T(solver.damping),
        pinv_rtol = cls_rtol,
        weight_pinv_rtol = weight_rtol,
        patience = solver.patience,
        verbose,
    )
    return return_stats ? result : result.point
end
