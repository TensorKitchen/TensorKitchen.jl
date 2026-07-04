module TensorKitchen

using LinearAlgebra
import LinearAlgebra: norm
using Random
using ManifoldsBase
using Manifolds
using Manopt
using PrecompileTools
import ProgressMeter
using RecursiveArrayTools
using TensorOperations
import Base: show
const PM = ProgressMeter

const AbstractRealManifold = AbstractManifold{ManifoldsBase.RealNumbers}

include("core.jl")
include("manifolds.jl")
include("decompositions.jl")
include("backend.jl")
include("api.jl")
include("precompile.jl")

function __init__()
    return nothing
end

end
