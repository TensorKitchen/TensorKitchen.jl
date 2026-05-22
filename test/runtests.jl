
using Test
using Random
using LinearAlgebra
using Statistics
using Manifolds
using ManifoldsBase
using RecursiveArrayTools
using TensorKitchen

Random.seed!(42)

include("basic_tests.jl")
include("latent_egrad_fd.jl")
include("convergence_tests.jl")
