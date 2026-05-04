# core/initialization.jl — Initializer objects and basic init helpers.
export AbstractInitializer,
    RandomInit,
    HOSVDInit,
    TuckerInit,
    TuckerDiagInit,
    ALSWarmStartInit,
    BTDALSWarmStartInit,
    BTDHOSVDMultistartInit,
    PointInit,
    FunctionInit,
    random_unit_matrix

abstract type AbstractInitializer end
abstract type BuiltinInitializer <: AbstractInitializer end

struct RandomInit <: BuiltinInitializer end
struct HOSVDInit <: BuiltinInitializer end
struct TuckerInit <: BuiltinInitializer end
struct TuckerDiagInit <: BuiltinInitializer end

struct ALSWarmStartInit{I} <: AbstractInitializer
    nsteps::Int
    base_init::I
end

struct BTDALSWarmStartInit{I} <: AbstractInitializer
    nsteps::Int
    base_init::I
    block_method::Symbol
    block_maxiter::Int
end

struct BTDHOSVDMultistartInit <: AbstractInitializer
    candidates::Int
    screening_steps::Int
    include_sequential::Bool
    block_method::Symbol
    block_maxiter::Int
    seed::Union{Nothing,Int}
end

function ALSWarmStartInit(nsteps::Int = 10; base_init = TuckerInit())
    nsteps >= 0 ||
        throw(ArgumentError("ALSWarmStartInit requires nsteps >= 0, got $nsteps"))
    return ALSWarmStartInit{typeof(base_init)}(nsteps, base_init)
end

function BTDALSWarmStartInit(
    nsteps::Int = 20;
    base_init = TuckerInit(),
    block_method::Symbol = :hooi,
    block_maxiter::Int = 5,
)
    nsteps >= 0 ||
        throw(ArgumentError("BTDALSWarmStartInit requires nsteps >= 0, got $nsteps"))
    block_maxiter >= 0 || throw(
        ArgumentError(
            "BTDALSWarmStartInit requires block_maxiter >= 0, got $block_maxiter",
        ),
    )
    return BTDALSWarmStartInit{typeof(base_init)}(
        nsteps,
        base_init,
        block_method,
        block_maxiter,
    )
end

function BTDHOSVDMultistartInit(
    candidates::Int = 24;
    screening_steps::Int = 5,
    include_sequential::Bool = true,
    block_method::Symbol = :hooi,
    block_maxiter::Int = 10,
    seed::Union{Nothing,Integer} = nothing,
)
    candidates >= 1 || throw(
        ArgumentError("BTDHOSVDMultistartInit requires candidates >= 1, got $candidates"),
    )
    screening_steps >= 0 || throw(
        ArgumentError(
            "BTDHOSVDMultistartInit requires screening_steps >= 0, got $screening_steps",
        ),
    )
    block_maxiter >= 0 || throw(
        ArgumentError(
            "BTDHOSVDMultistartInit requires block_maxiter >= 0, got $block_maxiter",
        ),
    )
    seed_eff = isnothing(seed) ? nothing : Int(seed)
    return BTDHOSVDMultistartInit(
        candidates,
        screening_steps,
        include_sequential,
        block_method,
        block_maxiter,
        seed_eff,
    )
end

struct PointInit{P} <: AbstractInitializer
    point::P
end

struct FunctionInit{F} <: AbstractInitializer
    f::F
end

_builtin_initializer_symbol(init::Symbol) = init
_builtin_initializer_symbol(::RandomInit) = :random
_builtin_initializer_symbol(::HOSVDInit) = :hosvd
_builtin_initializer_symbol(::TuckerInit) = :tucker
_builtin_initializer_symbol(::TuckerDiagInit) = :tucker_diag
function _builtin_initializer_symbol(init)
    throw(
        ArgumentError(
            "Initializer $(typeof(init)) does not map to a built-in symbol. " *
            "Use a built-in initializer or pass an explicit point via `PointInit`/`p0`.",
        ),
    )
end

_is_builtin_init(init) = init isa Union{Symbol,BuiltinInitializer}

function _cp_init_factors_from_rankr_point(
    p0,
    dims::NTuple{N,Int},
    r::Int;
    nonnegative::Bool = false,
    geometry::Symbol = :squaring_metric,
) where {N}
    from_explicit = p0 isa Union{CPDResult,CPDPoint}
    λ, U = if p0 isa CPDPoint
        (copy(lambda(p0)), [Matrix(F) for F in factors(p0)])
    elseif p0 isa CPDResult
        (copy(weights(p0)), [Matrix(F) for F in factors(p0)])
    else
        unpack_point_rankr(p0, dims, r)
    end
    if nonnegative && !from_explicit
        if geometry == :softplus_metric
            λ = _softplus_value.(λ)
            U = [_softplus_value.(Um) for Um in U]
        else
            λ = λ .^ 2
            U = [Um .^ 2 for Um in U]
        end
    end
    return λ, U
end

function _cp_init_factors_from_rank1_point(
    p0,
    dims::NTuple{N,Int};
    nonnegative::Bool = false,
    geometry::Symbol = :squaring_metric,
) where {N}
    from_explicit = p0 isa Union{CPDResult,CPDPoint}
    if p0 isa CPDPoint
        length(lambda(p0)) == 1 || throw(
            ArgumentError(
                "Rank-1 warm start requires exactly one CP component, got $(length(lambda(p0)))",
            ),
        )
        λ = [lambda(p0)[1]]
        U = [reshape(Vector(F[:, 1]), :, 1) for F in factors(p0)]
    elseif p0 isa CPDResult
        λ = weights(p0)
        length(λ) == 1 || throw(
            ArgumentError(
                "Rank-1 warm start requires exactly one CP component, got $(length(λ))",
            ),
        )
        λ = [λ[1]]
        U = [reshape(Vector(F[:, 1]), :, 1) for F in factors(p0)]
    else
        λ0, U0 = unpack_point_rank1(p0, dims)
        λ = [λ0]
        U = [reshape(u, :, 1) for u in U0]
    end
    if nonnegative && !from_explicit
        if geometry == :softplus_metric
            λ = _softplus_value.(λ)
            U = [_softplus_value.(Um) for Um in U]
        else
            λ = λ .^ 2
            U = [Um .^ 2 for Um in U]
        end
    end
    return λ, U
end

function random_unit_vector(n::Int, ::Type{T} = Float64) where {T<:AbstractFloat}
    v = randn(T, n)
    return v / norm(v)
end

function random_unit_matrix(n::Int, r::Int, ::Type{T} = Float64) where {T<:AbstractFloat}
    U = zeros(T, n, r)
    for k = 1:r
        U[:, k] = random_unit_vector(n, T)
    end
    return U
end

function _tucker_diag(core::AbstractArray{T}, r::Int) where {T<:AbstractFloat}
    N = ndims(core)
    out = zeros(T, r)
    maxr = minimum(size(core))
    rr = min(r, maxr)
    for k = 1:rr
        idx = ntuple(_ -> k, N)
        out[k] = core[idx...]
    end
    return out
end
