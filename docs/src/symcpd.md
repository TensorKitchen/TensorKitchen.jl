# Symmetric CP decomposition

Symmetric CP decomposition represents a symmetric tensor as a sum of repeated
vector patterns:

```math
\widehat A
=\sum_{r=1}^{R}\lambda_r x_r^{\otimes D},
\qquad \|x_r\|_2=1.
```

Here ``D`` is the tensor order, ``R`` is the number of components,
``\lambda_r`` is a signed weight, and ``x_r`` is a unit vector. The notation
``x_r^{\otimes D}`` means that the same vector appears in all ``D`` modes.

The input must have equal mode sizes and be symmetric. For example, an
order-three input has shape ``(N,N,N)`` and satisfies
``A[i,j,k]=A[k,i,j]`` for every permutation of the indices.

## Basic use

```julia
using TensorKitchen

result = symcpd(A, 3)

lambda = weights(result)       # three component weights
X = factors(result)            # column r is x_r
error = rel_error(A, result)
```

Call `reconstruct(result)` only when the full fitted tensor is needed:

```julia
A_fit = reconstruct(result)
```

The factors are compact, but `A_fit` has the same ``N^D`` entries as the
input.

The default solver is matrix-free Gauss--Newton (`solver=:gn_cg`). With
`init=:auto`, rank one uses SS-HOPM and higher rank uses a random starting
point. See [Symmetric CPD methods](advanced/symcpd.md) for the other solver,
initialization, and storage choices.

## Relation to ordinary CPD

Ordinary CPD allows a different vector in each mode. Symmetric CPD uses one
shared vector:

```math
\begin{array}{c|c}
\text{ordinary CPD} & \text{symmetric CPD} \\
\hline
\lambda_r u_r^{(1)}\otimes\cdots\otimes u_r^{(D)}
& \lambda_r x_r^{\otimes D}
\end{array}
```

TensorKitchen represents one symmetric component with
`SymmetricRankOne(N,D)`. Its geometry is a Veronese manifold. A rank-``R``
model is a sum, or Join, of ``R`` such components:

```julia
component = SymmetricRankOne(N, D)
target = DenseSymmetricTarget(A)
model = JoinModel(component, R, target)
```

`symcpd(A,R)` constructs this model for you.

## Matrix-free objective

The least-squares objective is

```math
f
=\frac12\left\|A-\sum_r\lambda_r x_r^{\otimes D}\right\|_F^2.
```

It can be evaluated as

```math
f
=\frac12\|A\|_F^2
-\sum_r\lambda_r\langle A,x_r^{\otimes D}\rangle
+\frac12\sum_{r,s}\lambda_r\lambda_s(x_r^\top x_s)^D.
```

This form uses contractions with the target and inner products between factor
vectors. It does not require a full fitted tensor or residual tensor.

## Target storage

The same model can use three target representations.

```julia
# Store the full tensor. This is the default for an array input.
dense_result = symcpd(A, R; target_backend=:dense)

# Store only unique symmetric coordinates.
compressed_result = symcpd(A, R; target_backend=:compressed)
```

A functional target stores contraction functions instead of tensor entries:

```julia
target = FunctionalSymmetricTarget(
    N,
    D,
    normA2;
    evaluate = x -> target_polynomial(x),
    contract = x -> target_contraction(x),
)

result = symcpd(target, R)
```

The three supplied values must describe the same tensor:

- `normA2` is ``\|A\|_F^2``;
- `evaluate(x)` is ``\langle A,x^{\otimes D}\rangle``;
- `contract(x)` is ``A(x,\ldots,x,\mathord\cdot)``.

## Rank-one eigenpairs

For ``R=1``, a unit tensor eigenvector satisfies

```math
A(I,x,\ldots,x)=\lambda x,
\qquad \|x\|_2=1.
```

Use SS-HOPM through the rank-one functions:

```julia
target = DenseSymmetricTarget(A)

pair = tensor_eigenpair(target; method=SSHOPM())
rank_one = best_symmetric_rank1(target; starts=8)
```

These functions find a local solution from one or more starting points. They
do not certify the globally best rank-one approximation.

## Equivalent signs

For odd ``D``, ``(\lambda,x)`` and ``(-\lambda,-x)`` represent the same
component. For even ``D``, ``x`` and ``-x`` represent the same component with
the same weight. Account for these equivalent signs when comparing factors.

## API reference

```@docs
symcpd
AbstractJoinComponent
AbstractSymmetricTarget
SymmetricRankOne
JoinModel
SymCPDModel
SymmetricCPDBackend
DenseSymmetricTarget
CompressedSymmetricTarget
FunctionalSymmetricTarget
target_norm2
evaluate
contract
SSHOPM
TensorEigenpairResult
tensor_eigenpair
best_symmetric_rank1
NormalizedCLSInit
SymmetricCLS
refit_symcpd_weights
SymCPDVariableProjectionModel
eigenvalue
eigenvector
residual_norm
component_inner
data_inner
pushforward!
pullback
normal_operator!
normal_operator
dense_normal_matrix
SymCPDResult
SymCPDComponent
compressed_coordinates
compress_symmetric_tensor
expand_symmetric_tensor
symmetric_multiindices
multinomial_multiplicity
```
