# Residual and differential operator contracts

A decomposition model's target storage and residual coordinate space are
independent. `residual_space(model)` describes Euclidean residual coordinates;
`residual_dimension(model)` reports their dimension and
`allocate_residual(model, p)` allocates a buffer. The default flattens tensor
storage. Models with implicit targets, observed entries, or compressed
coordinates specialize `residual_space` without providing dense storage.

For an explicitly evaluable least-squares objective, residuals use the sign
`r(p) = F(p) - A`, and `cost(model, p) = dot(r(p), r(p)) / 2`.
`differential_action(model, p, X)` is `D F(p)[X]`. The metric adjoint satisfies
`dot(differential_action(model, p, X), z) = inner(M, p, X, adjoint_action(model, p, z))`.
Consequently the intrinsic gradient is `adjoint_action(model, p, residual(model, p))`.
Compressed coordinates must preserve the tensor norm; an arbitrary unweighted
list of symmetric entries does not satisfy this contract.

`normal_operator(model, p, X)` applies the Gauss–Newton operator `J*J`, not
the exact Riemannian Hessian. Its default composes differential and adjoint
actions. `normal_operator!` copies the result using manifold-aware tangent
storage. Structured models override these defaults with fused contraction
kernels. A contraction-only model can supply cost, intrinsic gradient, and
normal action without materializing residual coordinates.

The generic adjoint is a small-problem reference: it evaluates differential
actions over an orthonormal tangent basis. Production models should implement
an analytic adjoint. The allocating differential uses the model's residual
allocator, while `differential_action!` must overwrite its entire output.

Manopt's LM adapter scales residual, differential, and adjoint consistently
when normalizing the objective. Residual coordinate dimensions come from this
interface, not the target array length. Functional targets may still reject
explicit residual evaluation; describing the coordinate space does not promise
that every operation is supported.

Weighted objectives require either whitening into Euclidean residual
coordinates or a separate metric-aware residual-space implementation. The
current `EuclideanResidualSpace` does not represent a weighted inner product.
