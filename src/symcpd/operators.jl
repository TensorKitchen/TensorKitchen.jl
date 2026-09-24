# Structured differential algebra for the intrinsic Veronese join.

export pushforward!, pullback, normal_operator!, normal_operator, dense_normal_matrix

@doc raw"""
    pushforward!(out, component::SymmetricRankOne, p, X)
    pushforward!(out, model, p, X)

Reference Jacobian-vector product for the join map

```math
\sigma(p_1,\ldots,p_R)=\sum_r\lambda_r x_r^{\otimes D}.
```

For `X_r = (nu_r, u_r)`, it evaluates

```math
D\sigma_p[X]
=\sum_r\left[
\nu_r x_r^{\otimes D}
+\lambda_r D\,\operatorname{sym}
(u_r\otimes x_r^{\otimes(D-1)})
\right]
```

in compressed orthonormal symmetric coordinates. This routine intentionally
materializes one ambient vector and is provided as a correctness oracle; the
production [`normal_operator!`](@ref) does not call it.

The differential is the Veronese tangent map used in Khouja, Khalil, and
Mourrain (2022), doi:10.1016/j.laa.2021.12.008, Sections 4.2--4.3. The
separation into a JVP API follows the matrix-free nonlinear least-squares
practice of Sorber, Van Barel, and De Lathauwer (2013),
doi:10.1137/120868323.
"""
function pushforward!(out::AbstractVector, component::SymmetricRankOne, p, X)
    expected = ambient_length(component)
    length(out) == expected ||
        throw(DimensionMismatch("Expected an output vector of length $expected."))
    ManifoldsBase.embed!(component.manifold, out, p, X)
    return out
end

function pushforward!(
    out::AbstractVector,
    model::JoinModel{T,B},
    p,
    X,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    expected = ambient_length(backend.component)
    length(out) == expected ||
        throw(DimensionMismatch("Expected an output vector of length $expected."))
    M = backend.product_manifold
    parts = join_parts(M, p)
    xparts = join_parts(M, X)
    length(parts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) point components."))
    length(xparts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) tangent components."))
    fill!(out, zero(eltype(out)))
    work = similar(out)
    @inbounds for r = 1:backend.rank
        pushforward!(work, backend.component, parts[r], xparts[r])
        out .+= work
    end
    return out
end

function differential_action!(
    out::AbstractVector,
    model::JoinModel{T,B},
    p,
    X,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    return pushforward!(out, model, p, X)
end

function differential_action(
    model::JoinModel{T,B},
    p,
    X,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    out = zeros(T, ambient_length(model.backend.component))
    return pushforward!(out, model, p, X)
end

@doc raw"""
    pullback(component::SymmetricRankOne, p, a)
    pullback(model, p, a)

Apply the metric adjoint `D sigma_p^*` to a cotangent `a` represented in
orthonormal symmetric coordinates. Component `r` is the orthogonal Veronese
tangent projection of `a` at `p_r`. This is the analytic VJP corresponding to
the `J'F` construction in Khouja, Khalil, and Mourrain (2022), Proposition 4.9,
doi:10.1016/j.laa.2021.12.008.

This materialized-ambient method is a reference path. Target-specific
gradients use contractions directly, and [`normal_operator!`](@ref) evaluates
`J'J*X` from kernel derivatives without constructing `a`.
"""
function pullback(component::SymmetricRankOne, p, a::AbstractVector)
    expected = ambient_length(component)
    length(a) == expected ||
        throw(DimensionMismatch("Expected an ambient cotangent of length $expected."))
    return ManifoldsBase.project(component.manifold, p, a)
end

function pullback(
    model::JoinModel{T,B},
    p,
    a::AbstractVector,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    expected = ambient_length(backend.component)
    length(a) == expected ||
        throw(DimensionMismatch("Expected an ambient cotangent of length $expected."))
    M = backend.product_manifold
    parts = join_parts(M, p)
    values = ntuple(r -> pullback(backend.component, parts[r], a), backend.rank)
    return join_tangent_like(M, p, values)
end

function adjoint_action(
    model::JoinModel{T,B},
    p,
    a::AbstractVector;
    kwargs...,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    return pullback(model, p, a)
end

@doc raw"""
    normal_operator!(Y, model, p, X)

Apply the Riemannian Gauss--Newton normal operator `J(p)^*J(p)` without
forming a Jacobian, a normal matrix, an ambient tensor, or a compressed
Veronese vector.

For components `p_r=(lambda_r,x_r)`, `p_s=(lambda_s,x_s)`, tangents
`X_r=(nu_r,u_r)`, `Z_s=(xi_s,v_s)`, and `c=x_r' * x_s`, the cross-component
bilinear form is

```math
\begin{aligned}
B_{rs}(X_r,Z_s)={}&
\nu_r\xi_s c^D
+\nu_r\lambda_sD c^{D-1}x_r^\top v_s\\
&+\lambda_r\xi_sD c^{D-1}u_r^\top x_s
+\lambda_r\lambda_sD c^{D-1}u_r^\top v_s\\
&+\lambda_r\lambda_sD(D-1)c^{D-2}
(u_r^\top x_s)(x_r^\top v_s).
\end{aligned}
```

The implementation first accumulates the scalar and factor covectors of this
form, then applies the inverse Veronese metric
`diag(1, (D*lambda_s^2)^(-1) P_{x_s})`. For `r=s`, tangent orthogonality makes
`B_rr` exactly the induced metric, a useful implementation invariant.

The kernel-derived block formula is the real, explicit-weight counterpart of
Khouja, Khalil, and Mourrain (2022), Proposition 4.9,
doi:10.1016/j.laa.2021.12.008. Unlike their TensorDec implementation, this
routine applies the blocks as operators and does not assemble a dense `J'J`.
Matrix-free normal products inside CG follow the established GN-CG pattern in
N. Singh, L. Ma, H. Yang, and E. Solomonik, *SIAM J. Sci. Comput.* 43(4)
(2021), C290--C311, doi:10.1137/20M1344561.
"""
function normal_operator!(
    Y,
    model::JoinModel{T,B},
    p,
    X,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    M = backend.product_manifold
    parts = join_parts(M, p)
    xparts = join_parts(M, X)
    yparts = join_parts(M, Y)
    length(parts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) point components."))
    length(xparts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) input tangent components."))
    length(yparts) == backend.rank ||
        throw(DimensionMismatch("Expected $(backend.rank) output tangent components."))
    d = backend.order
    dT = T(d)
    @inbounds for s = 1:backend.rank
        ps = point_parts(parts[s])
        lambda_s = ps[1][1]
        x_s = ps[2]
        ys = point_parts(yparts[s])
        radial = zero(T)
        fill!(ys[2], zero(T))
        for r = 1:backend.rank
            pr = point_parts(parts[r])
            xr = point_parts(xparts[r])
            lambda_r = pr[1][1]
            x_r = pr[2]
            nu_r = xr[1][1]
            u_r = xr[2]
            c = dot(x_r, x_s)
            c_dm1 = c^(d - 1)
            uxs = dot(u_r, x_s)
            radial += nu_r * c^d + lambda_r * dT * c_dm1 * uxs
            x_coefficient = nu_r * lambda_s * dT * c_dm1
            if d > 1
                x_coefficient += lambda_r * lambda_s * dT * T(d - 1) * c^(d - 2) * uxs
            end
            ys[2] .+= x_coefficient .* x_r
            ys[2] .+= (lambda_r * lambda_s * dT * c_dm1) .* u_r
        end
        ys[1][1] = radial
        ys[2] .-= dot(x_s, ys[2]) .* x_s
        ys[2] ./= dT * lambda_s^2
        ys[2] .-= dot(x_s, ys[2]) .* x_s
    end
    return Y
end

"""Allocate and return `J(p)^*J(p)X`; see [`normal_operator!`](@ref)."""
function normal_operator(
    model::JoinModel{T,B},
    p,
    X,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    Y = ManifoldsBase.zero_vector(model.backend.product_manifold, p)
    return normal_operator!(Y, model, p, X)
end

@doc raw"""
    dense_normal_matrix(model, p; reference=false)

Construct a dense coordinate matrix for `J(p)^*J(p)` in the default
orthonormal tangent basis. With `reference=false`, columns are generated by
[`normal_operator!`](@ref). With `reference=true`, this function explicitly
forms the compressed-coordinate Jacobian `J` using [`pushforward!`](@ref) and
returns `J'J`.

The `reference=true` path mirrors the dense normal-matrix validation strategy
of Khouja, Khalil, and Mourrain (2022), Proposition 4.9,
doi:10.1016/j.laa.2021.12.008. It is intended only for small tests: production
GN-CG should use the operator directly.
"""
function dense_normal_matrix(
    model::JoinModel{T,B},
    p;
    reference::Bool = false,
) where {T<:AbstractFloat,B<:SymmetricCPDBackend}
    backend = model.backend
    M = backend.product_manifold
    basis = ManifoldsBase.DefaultOrthonormalBasis()
    tangent_dim = manifold_dimension(M)
    coefficients = zeros(T, tangent_dim)
    if reference
        ambient_dim = ambient_length(backend.component)
        J = Matrix{T}(undef, ambient_dim, tangent_dim)
        @inbounds for j = 1:tangent_dim
            fill!(coefficients, zero(T))
            coefficients[j] = one(T)
            Xj = ManifoldsBase.get_vector(M, p, coefficients, basis)
            pushforward!(view(J, :, j), model, p, Xj)
        end
        return transpose(J) * J
    end
    H = Matrix{T}(undef, tangent_dim, tangent_dim)
    @inbounds for j = 1:tangent_dim
        fill!(coefficients, zero(T))
        coefficients[j] = one(T)
        Xj = ManifoldsBase.get_vector(M, p, coefficients, basis)
        Yj = normal_operator(model, p, Xj)
        H[:, j] .= ManifoldsBase.get_coordinates(M, p, Yj, basis)
    end
    return H
end
