# ---------- Tucker ----------
include("tucker/hosvd.jl")
include("tucker/sthosvd.jl")
include("tucker/hooi.jl")

# ---------- CPD ----------
include("cpd/core/mttkrp.jl")
include("cpd/core/cpd_init.jl")
include("cpd/core/cp_cost.jl")
include("cpd/core/reconstruct.jl")
include("cpd/model/parameterizations.jl")
include("cpd/model/rank1.jl")
include("cpd/model/rankr.jl")

# ---------- BTD + Join ----------
include("join/join_init.jl")
include("join/join_model.jl")
include("btd/model.jl")
include("join/cpd_backend.jl")
include("join/join_backend.jl")
include("btd/core/inner_prod.jl")
include("btd/core/btd_cost.jl")
include("btd/core/btd_grad.jl")
