const DEFAULT_SCENARIOS =
    (:exact_cp, :ill_conditioned_cp, :exact_nncp, :border_rank_w, :overranked_abs_randn)
const KNOWN_SCENARIOS = (DEFAULT_SCENARIOS..., :noisy_exact_cp)
const DEFAULT_SOLVERS = (:rgd, :rcg)
const SUPPORTED_SOLVERS = (:rgd, :rgd_fixed, :rcg)
const TRACE_TAIL_LENGTH = 10
const DEFAULT_NOISE_SWEEP_LEVELS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
const DEFAULT_CONCENTRATION_TOP1 = 0.5
const FIXED_CONCENTRATION_TOP3 = 0.9
const FIXED_CONCENTRATION_EFFECTIVE_MAX = 2.0
