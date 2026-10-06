# Deterministic random fixtures for tests whose structural coverage benefits
# from varied dense values. Each call owns its RNG, so changing test order or
# adding an unrelated random draw cannot change an existing fixture.
_test_rand(seed::Integer, args...) = rand(MersenneTwister(seed), args...)
_test_randn(seed::Integer, args...) = randn(MersenneTwister(seed), args...)
