# Benchmarks

## CPD Stagnation Trace

Run:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl
```

Quick smoke run:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --quick
```

The benchmark writes:

- `benchmark/results/cpd_stagnation_trace.tsv`: one row per scenario, seed, and solver.
- `benchmark/results/cpd_stagnation_trace.md`: grouped summary table.

Implementation layout:

- `benchmark/cpd_stagnation_trace.jl`: thin module entry point.
- `benchmark/cpd_stagnation_trace/`: reusable scenario, CLI, generation, diagnostics, reporting, concentration, and runner code.

The key diagnostic columns are:

- `cost_rel_change_tail_median`: median of the last 10 relative cost changes.
- `component_delta_tail_median`: median of the last 10 maximum rank-one term movements.
- `component_delta_final`: final maximum rank-one term movement.
- `rgrad_top1_share_tail_median`: median of the last 10 internal solver rgrad block-coordinate energy shares in the largest CP component.
- `rgrad_top3_share_tail_median`: median of the last 10 internal solver rgrad block-coordinate energy shares in the largest three CP components.
- `rgrad_effective_components_tail_median`: effective number of CP components carrying internal solver rgrad block-coordinate energy.
- `accepted_stepsize_tail_median`: median accepted line-search step size.
- `line_search_trials_tail_median`: median number of line-search trials.
- `classification`: coarse diagnosis of the run.

For nonnegative pullback-style geometries, the rgrad columns diagnose internal solver coordinates. They should not be interpreted as exact intrinsic component contributions under the pullback metric.

Interpretation:

- `converged_terms_stuck`: the cost is flat, the rank-one terms are also barely moving, and the final gradient is small.
- `stagnated_terms_stuck_grad_large`: the cost and rank-one terms are barely moving, but the final gradient is still large.
- `cost_flat_terms_moving`: the cost is flat but the rank-one terms are still moving, which indicates flat directions, gauge effects, or cancellation.
- `not_stagnated_or_still_descending`: the cost is not yet flat and the gradient is not small.

Useful focused commands:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --scenarios=exact_cp,ill_conditioned_cp --solvers=rgd,rcg --seeds=1,2,3 --maxiter=1000
```

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --scenarios=overranked_abs_randn --solvers=rgd --seeds=1 --maxiter=1000
```

Border-rank W-state stress test:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --scenarios=border_rank_w --solvers=rgd --seeds=1,2,3 --maxiter=5000 --tol=1e-12 --init=random --rank=10
```

Override ranks globally:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --rank=10 --overranked-rank=50 ...
```

- `--rank` applies to `exact_cp`, `ill_conditioned_cp`, `exact_nncp`, and `border_rank_w`.
- `--overranked-rank` applies only to `overranked_abs_randn`.

Run RGD without ALS warm start:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl --scenarios=exact_cp,ill_conditioned_cp,exact_nncp --solvers=rgd --seeds=1,2,3 --maxiter=1000 --init=random
```

RGD + ALS warm start concentration sweep (W-state and exact NNCP):

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=border_rank_w,exact_nncp \
  --solvers=rgd \
  --init=alswarm \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_concentration_rgd_alswarm.tsv \
  --summary=benchmark/results/exp_concentration_rgd_alswarm.md \
  --concentration-summary=benchmark/results/exp_concentration_rgd_alswarm_concentration.md
```

The concentration summary groups by `(scenario, rank, nonnegative, noise_level, collinearity_noise, solver, init)`, labels rank ≤ 2 as `rank_degenerate`, and reports independent counts plus outcome groups (`concentrated_converged`, `concentrated_cost_flat_grad_large`, `concentrated_grad_large`, `concentrated_maxiter`, `nonconcentrated_converged`, `nonconcentrated_grad_large`, `other`) with per-group medians. Benign vs pathological concentrated counts use `--rel-error-tol`.

For border-rank W-state with tighter tolerance, add `--maxiter=5000 --tol=1e-12`.

NNCP init comparison (ALS warm start vs random, same seeds):

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=exact_nncp \
  --solvers=rgd \
  --inits=alswarm,random \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_concentration_rgd_nncp_init_compare.tsv \
  --summary=benchmark/results/exp_concentration_rgd_nncp_init_compare.md \
  --concentration-summary=benchmark/results/exp_concentration_rgd_nncp_init_compare_concentration.md
```

Random init only:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=exact_nncp --solvers=rgd --init=random --replicates=50 \
  --concentration-summary=benchmark/results/exp_concentration_rgd_random_nncp_concentration.md \
  --out=benchmark/results/exp_concentration_rgd_random_nncp.tsv \
  --summary=benchmark/results/exp_concentration_rgd_random_nncp.md
```

Refinement diagnostics in TSV / concentration summary:

- `start_rel_error`: rel_error at RGD start (after init / ALS warm start)
- `rgrad_argmax_tail_persistence`: same rgrad argmax component across tail iterations
- `dominant_component_delta_tail_median`: movement of the rgrad-dominant component
- `dominant_movement_pattern`: blocked vs compensating movement heuristic
- `noise_level`: additive noise scale for `noisy_exact_cp` (`NaN` otherwise)
- `collinearity_noise`: factor perturbation scale for `ill_conditioned_cp` (`NaN` otherwise)
- `rgrad_share_tail_median`: semicolon-separated tail median rgrad share per component (`s1;s2;…;sr`)
- `rgrad_share_entropy_tail_median`: tail median Shannon entropy of the rgrad share vector
- `rgrad_share_normalized_entropy_tail_median`: tail median entropy / log(r)
- `component_deltas_tail_median`: semicolon-separated tail median movement per component

True noisy exact CP (additive noise on an exact rank-`r` tensor). Include `0` in `--noise-levels` for the σ=0 exact control (same clean tensor as `exact_cp`, labeled `noise_level=0`).

Rank 5 init comparison with noise sweep, rank-aware concentration thresholds, and per-component rgrad shares:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=noisy_exact_cp \
  --rank=5 \
  --solvers=rgd \
  --inits=alswarm,random \
  --noise-sweep-default \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_noisy_exact_cp_r5_init_compare.tsv \
  --summary=benchmark/results/exp_noisy_exact_cp_r5_init_compare.md \
  --concentration-summary=benchmark/results/exp_noisy_exact_cp_r5_init_compare_concentration.md
```

Equivalent explicit sweep (`0,1e-4,1e-3,1e-2,1e-1`). With `--rank=5`, row-specific concentration defaults become top3≈0.75 and effective≤2.5; fixed top3=0.9 counts are also reported in the concentration summary.

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=noisy_exact_cp \
  --rank=5 \
  --solvers=rgd \
  --inits=alswarm,random \
  --noise-levels=0,1e-4,1e-3,1e-2,1e-1 \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_noisy_exact_cp_r5_noise_sweep.tsv \
  --summary=benchmark/results/exp_noisy_exact_cp_r5_noise_sweep.md \
  --concentration-summary=benchmark/results/exp_noisy_exact_cp_r5_noise_sweep_concentration.md
```

General noisy exact CP sweep:

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=noisy_exact_cp \
  --solvers=rgd \
  --inits=alswarm,random \
  --noise-sweep-default \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_noisy_exact_cp_noise_sweep.tsv \
  --summary=benchmark/results/exp_noisy_exact_cp_noise_sweep.md \
  --concentration-summary=benchmark/results/exp_noisy_exact_cp_noise_sweep_concentration.md
```

Notes:
- `exact_cp` and σ=0 noisy runs use `noise_level=0` in the TSV for the same σ axis.
- Per-component columns: `rgrad_share_tail_median`, `component_deltas_tail_median`, `rgrad_share_entropy_tail_median`.
- `--concentration-rank-aware=true` (default) adjusts top3/effective from each row's rank, so mixed-rank runs do not reuse one global threshold; fixed top3=0.9 counts appear in the concentration summary.

Collinearity noise sweep (`ill_conditioned_cp`):

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=ill_conditioned_cp \
  --solvers=rgd \
  --inits=alswarm,random \
  --collinearity-noises=1e-1,1e-2,1e-3,1e-4,1e-5 \
  --replicates=50 \
  --maxiter=1000 \
  --out=benchmark/results/exp_ill_conditioned_collinearity_sweep.tsv \
  --summary=benchmark/results/exp_ill_conditioned_collinearity_sweep.md \
  --concentration-summary=benchmark/results/exp_ill_conditioned_collinearity_sweep_concentration.md
```

Use `--noise-level=1e-3` or `--collinearity-noise=1e-4` for a single override without expanding the sweep list.

## Experiment suites

Use `--out` and `--summary` to keep runs organized under `benchmark/results/`. Suggested neutral names:

| Suite | Purpose | Suggested outputs |
|-------|---------|-------------------|
| **A** | Border-rank positive control | `exp_A_border_rank_r10.{tsv,md}` |
| **B** | Gradient-concentration / overrank stress | `exp_B_gradient_concentration_r10_r50.{tsv,md}` |
| **C** | Healthy exact-CP control | `exp_C_exact_cp_r10.{tsv,md}` |

**A — border-rank pathology**

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=border_rank_w \
  --solvers=rgd,rcg,rgd_fixed \
  --inits=random \
  --seeds=1,2,3 \
  --maxiter=5000 \
  --tol=1e-12 \
  --rank=10 \
  --out=benchmark/results/exp_A_border_rank_r10.tsv \
  --summary=benchmark/results/exp_A_border_rank_r10.md
```

**B — gradient concentration (hypothesis test)**

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=overranked_abs_randn,exact_nncp,ill_conditioned_cp \
  --solvers=rgd,rcg \
  --inits=random,alswarm \
  --seeds=1,2,3 \
  --maxiter=1000 \
  --rank=10 \
  --overranked-rank=50 \
  --out=benchmark/results/exp_B_gradient_concentration_r10_r50.tsv \
  --summary=benchmark/results/exp_B_gradient_concentration_r10_r50.md
```

**C — healthy finite convergence**

```bash
julia --project=. benchmark/cpd_stagnation_trace.jl \
  --scenarios=exact_cp \
  --solvers=rgd,rcg \
  --inits=alswarm,random \
  --seeds=1,2,3 \
  --maxiter=1000 \
  --rank=10 \
  --out=benchmark/results/exp_C_exact_cp_r10.tsv \
  --summary=benchmark/results/exp_C_exact_cp_r10.md
```

Use `--rank` for exact/border/ill-conditioned/NNCP scenarios (defaults 2–3) and `--overranked-rank` for `overranked_abs_randn` (default 35).

All files under `benchmark/results/` are local outputs (gitignored).
