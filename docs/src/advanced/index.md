# Advanced Guide

This section explains the mathematics and algorithm options behind each
decomposition. The main decomposition pages focus on inputs, outputs, and basic
examples.

## Topics

- [CPD methods](cpd.md): CP indeterminacies, initialization, nonnegative CPD,
  and complete API options.
- [Symmetric CPD methods](symcpd.md): solver, initialization, target backend,
  scaling, and conditioning choices.
- [Tucker methods](tucker.md): ST-HOSVD, implicit randomized sketching, HOOI,
  and error estimates.
- [BTD methods](btd.md): initialization, alternating updates, and refinement.
- [Optimization methods](optimization.md): objective functions, solver choices,
  stopping criteria, and diagnostics.
- [Join models](join.md): supported `approx` forms and automatic routing to
  specialized decompositions.

The public API reference on each decomposition page remains the source of truth
for supported keyword arguments and their current defaults.
