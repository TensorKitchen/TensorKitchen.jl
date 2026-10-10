# An implicit target intentionally has no tensor storage or length method.
struct _ObservedLinearModel{T<:AbstractFloat} <: AbstractDecompositionModel{T}
    J::Matrix{T}
end
TensorKitchen.manifold(::_ObservedLinearModel) = Euclidean(2)
TensorKitchen.residual_space(model::_ObservedLinearModel{T}) where {T} =
    EuclideanResidualSpace{T}(size(model.J, 1))
TensorKitchen.differential_action!(out::AbstractVector, model::_ObservedLinearModel, p, X) =
    mul!(out, model.J, X)

@testset "Residual space and metric operator contract" begin
    for T in (Float32, Float64)
        model = _ObservedLinearModel(T[1 2; 3 4; 5 6])
        M = manifold(model)
        p = zeros(T, 2)
        X = T[0.3, -0.2]
        z = T[0.2, 0.5, -0.4]
        @test residual_dimension(model) == 3
        @test eltype(allocate_residual(model, p)) === T
        @test differential_action(model, p, X) ≈ model.J * X
        adjoint = adjoint_action(model, p, z)
        @test adjoint ≈ model.J' * z
        adjoint_buffer = zero_vector(M, p)
        @test adjoint_action!(adjoint_buffer, model, p, z) === adjoint_buffer
        @test adjoint_buffer ≈ adjoint
        @test dot(differential_action(model, p, X), z) ≈ inner(M, p, X, adjoint)
        @test_throws DimensionMismatch adjoint_action(model, p, zeros(T, 2))
        @test normal_operator(model, p, X) ≈ model.J' * model.J * X
        Y = zeros(T, 2)
        @test normal_operator!(Y, model, p, X) === Y
        @test Y ≈ model.J' * model.J * X
        @test TensorKitchen._lm_raw_jacobian_matrix(model, M, p) ≈ model.J
    end
    @test_throws ArgumentError EuclideanResidualSpace{Float64}(-1)

    component = SymmetricRankOne(3, 3)
    for target in (
        DenseSymmetricTarget(zeros(3, 3, 3)),
        CompressedSymmetricTarget(zeros(10), 3, 3),
        FunctionalSymmetricTarget(3, 3, 0.0; evaluate = x -> 0.0, contract = x -> zeros(3)),
    )
        model = JoinModel(component, 1, target)
        p = (([1.0], [1.0, 0.0, 0.0]),)
        @test residual_dimension(model) == 10
        @test length(allocate_residual(model, p)) == 10
    end
end
