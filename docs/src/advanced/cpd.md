# Advanced CPD Methods

For an input tensor ``\mathcal A \in \mathbb R^{n_1\times\cdots\times n_d}``,
a rank-``R`` CP approximation is

```math
\hat{\mathcal A}
= \sum_{r=1}^{R} \lambda_r
  u_r^{(1)} \otimes u_r^{(2)} \otimes \cdots \otimes u_r^{(d)}.
```

The fitted parameters minimize the least-squares objective

```math
f(\lambda,U^{(1)},\ldots,U^{(d)})
= \frac12\left\|\mathcal A-\hat{\mathcal A}\right\|_F^2.
```

CP parameters are not unique as raw coordinates: components may be permuted,
and nonzero scaling can be moved between factors as long as the product of the
mode-wise scalings is unchanged. Compare reconstructed components or the full
tensor, rather than comparing factor columns without first aligning their order
and scale.

## Initialization and refinement

The CP objective is nonconvex. A structured or ALS-based warm start can place a
manifold solver in a better basin than a random point, but no initializer
guarantees the globally best rank-``R`` approximation. Compare multiple seeds or
initializers when the fitted model will support scientific conclusions.

ALS updates one factor matrix at a time. Riemannian gradient, conjugate-gradient,
L-BFGS, and Levenberg--Marquardt methods instead refine the full parameter point
on the selected geometry. See [Optimization methods](optimization.md) for solver
constructors and diagnostics.

## Nonnegative CPD

Nonnegative CPD constrains the fitted weights and factors to be nonnegative. It
is appropriate only when negative components would be physically or
semantically invalid. Nonnegativity changes the feasible model; it is not merely
a numerical stabilization of unconstrained CPD.

The canonical, squaring, and softplus geometries use different coordinates near
zero. Their current availability and defaults are maintained in the `nncpd`
docstring below.

## API, methods, and options

These source docstrings are the complete reference for supported initialization,
solver, geometry, normalization, stopping, and diagnostic options:

```@docs
cpd
nncpd
fit_cp_als
```
