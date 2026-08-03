# Advanced Guide

This section collects algorithm and solver details that are useful for method
selection, reproducibility, and performance tuning. The main decomposition
pages remain focused on inputs, outputs, and basic examples.

## Topics

- [CPD methods](cpd.md): CP indeterminacies, initialization, nonnegative CPD,
  and complete API options.
- [Tucker methods](tucker.md): ST-HOSVD, implicit randomized sketching, HOOI,
  and error estimates.
- [BTD methods](btd.md): initialization, alternating updates, refinement, and
  robustness controls.
- [Optimization methods](optimization.md): objective functions, solver choices,
  stopping criteria, and diagnostics.
- [Join models](join.md): supported `approx` forms and automatic routing to
  specialized decompositions.

The public API reference on each decomposition page remains the source of truth
for supported keyword arguments and their current defaults.
