
using Test
using Random
using LinearAlgebra
using Statistics
using Manifolds
using ManifoldsBase
using RecursiveArrayTools
using TensorKitchen

include("test_helpers.jl")
include("tucker_tests.jl")
include("cpd_tests.jl")
include("nncpd_tests.jl")
include("btd_tests.jl")
include("join_tests.jl")
include("symcpd_tests.jl")
include("solver_tests.jl")
include("core_tests.jl")
include("integration_tests.jl")
