#!/usr/bin/env julia

module CPDStagnationTraceBenchmark

using LinearAlgebra
using Printf
using Random
using TensorKitchen

include("cpd_stagnation_trace/constants.jl")
include("cpd_stagnation_trace/scenarios.jl")
include("cpd_stagnation_trace/options.jl")
include("cpd_stagnation_trace/cli.jl")
include("cpd_stagnation_trace/generation.jl")
include("cpd_stagnation_trace/diagnostics.jl")
include("cpd_stagnation_trace/reporting.jl")
include("cpd_stagnation_trace/concentration.jl")
include("cpd_stagnation_trace/runner.jl")

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CPDStagnationTraceBenchmark.main()
end
