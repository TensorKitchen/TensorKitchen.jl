
using Test
using Random
using LinearAlgebra
using Statistics
using Manifolds
using ManifoldsBase
using RecursiveArrayTools
using TensorKitchen

Random.seed!(42)

include("data_tests.jl")
include("basic_tests.jl")
include("convergence_tests.jl")
