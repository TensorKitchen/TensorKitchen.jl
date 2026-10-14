# LM inner solves and diagnostics

`LMSolver` retains Manopt's matrix-free Levenberg–Marquardt algorithm.
TensorKitchen supplies residual/JVP/VJP callbacks and configures a tangent
conjugate-residual solve. No dense Jacobian is constructed on this path.

## Configuration

```julia
solver = LMSolver(
    inner = LMInnerOptions(
        tolerance = AdaptiveResidualTolerance(
            maximum = 1e-2, minimum = 1e-8, scale = 1, power = 0.5),
        maxiter = 200),
    diagnostics = true)
result = solve(solver, model; p0, maxiter = 100, verbose = false)
info = solver_info(result)
```

Every inner solve starts from zero, including after the tangent base point
changes. All CR buffers are supplied explicitly so constructor defaults do
not consume randomness or evaluate operators before initialization.

The default forcing tolerance is
`clamp(‖grad f‖^0.5, 1e-8, 1e-2)` in unnormalized objective units, with bounds
floored at `eps(T)`. CR stops when its relative residual falls below that
value. `RelativeResidualTolerance(1e-2)` selects a fixed relative threshold.
`AbsoluteResidualTolerance(1e-16)` is available for comparison with the previous
accuracy requirement; initialization remains deterministic in all modes.
A zero right-hand side stops before the first CR step. The default iteration
cap is `max(20 * dim(M), 200)` and can be overridden independently of outer
stopping criteria.

`solver_info(result).inner` groups the inner solver name, policy, iteration
counts, tolerances, residual histories, convergence flags, and termination
reasons.
With `diagnostics=true`, `operator_calls` and `operator_seconds` report
`residual`, `differential`, and `adjoint` evaluations, and `inner.seconds`
records each CR solve. These count only LM callbacks, not separate cost or
gradient evaluations in outer reporting. Operator time overlaps inner time;
do not add these values. Timing is disabled by default.

## Architecture

- `core/model.jl` defines residual coordinates independently of target storage
  and defines metric-adjoint and normal-operator contracts.
- `solvers/inner_tolerance.jl` defines tolerance policies without Manopt types.
- `solvers/lm_inner.jl` owns the Manopt CR lifecycle and termination history.
- `solvers/lm_diagnostics.jl` instruments operator callbacks when requested.
- `solvers/lm.jl` connects model operations to the outer LM solver. JVP callbacks
  write directly into the supplied residual buffer; adjoint results use
  manifold-aware copying for structured tangent representations.
- `compat/manopt.jl` confines structured-storage compatibility to a private
  manifold decorator, including tangent-space copying and coordinate scratch
  allocation. The upstream scratch allocation itself is still present.

SymCPD LM uses orthonormal compressed symmetric residual coordinates for dense
or compressed symmetric targets. The fixed target coordinates are cached in
its residual callback. Functional targets remain contraction-only and should
use GN-CG. Explicit SymCPD LM operators are correctness/reference paths; GN-CG
uses fused contractions without materializing residual coordinates.

## Controlled comparisons

For a controlled CPD or SymCPD comparison, give each policy a copy of the same
seeded initial point and warm each run before measurement. Record final relative
error, outer and inner iterations, operator counts and times, and allocations
for the full solve and isolated callbacks. Single samples are not statistical
performance claims. Large cases can be expensive for explicit SymCPD operators.

The comparison isolates absolute, fixed-relative, and adaptive inner accuracy
with zero initialization throughout; it is not a reproduction of the old
random-warm-start implementation. GN-CG is a contextual SymCPD reference.
Manopt multiplies its damping scale by the current cost; GN-CG adds its damping
directly. Their globalization and outer stopping criteria differ.

The default damping policy is preserved. To test decreasing damping explicitly,
set `damping_reduction_threshold` to a value in `[η, 1]`; `nothing` uses the
existing `expect_zero_residual` policy, and `Inf` disables reduction. Compare
final accuracy as well as runtime before choosing a policy or replacing LM.
