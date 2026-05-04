# core/abstract.jl — Typed strategy objects for internal dispatch.

abstract type AbstractGradientMode end
struct RiemannianGradientMode <: AbstractGradientMode end
struct ProjectedEGradientMode <: AbstractGradientMode end
struct ExactNativeGradientMode <: AbstractGradientMode end
struct ExactJoinGradientMode <: AbstractGradientMode end
struct ExactJoinBasisGradientMode <: AbstractGradientMode end

gradient_mode_policy(mode::AbstractGradientMode) = mode

const _exact_native_mode_warned = Ref(false)
const _exact_join_mode_warned = Ref(false)
const _exact_join_basis_mode_warned = Ref(false)

@inline function _warn_exact_gradient_mode_deprecated(
    mode::Symbol,
    warned::Base.RefValue{Bool},
)
    warned[] && return nothing
    warned[] = true
    Base.depwarn(
        "gradient_mode=$mode is advanced/experimental and is planned to be removed from public frontend APIs in a future release. " *
        "Prefer :riemannian (default) or :egrad_project for frontend usage. " *
        "If you need $mode, use advanced/internal solver entry points.",
        :gradient_mode,
    )
    return nothing
end

function gradient_mode_policy(mode::Symbol)
    mode === :riemannian && return RiemannianGradientMode()
    mode === :egrad_project && return ProjectedEGradientMode()
    if mode === :exact_native
        _warn_exact_gradient_mode_deprecated(mode, _exact_native_mode_warned)
        return ExactNativeGradientMode()
    end
    if mode === :exact_join
        _warn_exact_gradient_mode_deprecated(mode, _exact_join_mode_warned)
        return ExactJoinGradientMode()
    end
    if mode === :exact_join_basis
        _warn_exact_gradient_mode_deprecated(mode, _exact_join_basis_mode_warned)
        return ExactJoinBasisGradientMode()
    end
    throw(
        ArgumentError(
            "Unknown gradient_mode=$mode. Use :riemannian or :egrad_project (frontend defaults). " *
            "Advanced/experimental modes: :exact_native, :exact_join, :exact_join_basis.",
        ),
    )
end
gradient_mode_symbol(::RiemannianGradientMode) = :riemannian
gradient_mode_symbol(::ProjectedEGradientMode) = :egrad_project
gradient_mode_symbol(::ExactNativeGradientMode) = :exact_native
gradient_mode_symbol(::ExactJoinGradientMode) = :exact_join
gradient_mode_symbol(::ExactJoinBasisGradientMode) = :exact_join_basis

abstract type AbstractApproxDispatch end
struct AutoApproxDispatch <: AbstractApproxDispatch end
struct GenericApproxDispatch <: AbstractApproxDispatch end
struct CPDApproxDispatch <: AbstractApproxDispatch end
struct BTDApproxDispatch <: AbstractApproxDispatch end

approx_dispatch(dispatch::AbstractApproxDispatch) = dispatch
function approx_dispatch(dispatch::Symbol)
    dispatch === :auto && return AutoApproxDispatch()
    dispatch === :generic && return GenericApproxDispatch()
    dispatch === :cpd && return CPDApproxDispatch()
    dispatch === :btd && return BTDApproxDispatch()
    throw(
        ArgumentError(
            "Unknown approx dispatch=$dispatch. Use :auto, :generic, :cpd, or :btd.",
        ),
    )
end
approx_dispatch_symbol(::AutoApproxDispatch) = :auto
approx_dispatch_symbol(::GenericApproxDispatch) = :generic
approx_dispatch_symbol(::CPDApproxDispatch) = :cpd
approx_dispatch_symbol(::BTDApproxDispatch) = :btd

abstract type AbstractNNUpdatePolicy end
struct AutomaticNNUpdate <: AbstractNNUpdatePolicy end
struct LeastSquaresNNUpdate <: AbstractNNUpdatePolicy end
struct MultiplicativeNNUpdate <: AbstractNNUpdatePolicy end
struct HALSNNUpdate <: AbstractNNUpdatePolicy end
struct NNLSUpdate <: AbstractNNUpdatePolicy end

nn_update_policy(policy::AbstractNNUpdatePolicy) = policy
function nn_update_policy(policy::Symbol)
    policy === :auto && return AutomaticNNUpdate()
    policy === :ls && return LeastSquaresNNUpdate()
    policy === :mu && return MultiplicativeNNUpdate()
    policy === :hals && return HALSNNUpdate()
    policy === :nnls && return NNLSUpdate()
    throw(ArgumentError("Unknown nn_update=$policy. Use :auto, :ls, :mu, :hals, or :nnls."))
end
nn_update_symbol(::AutomaticNNUpdate) = :auto
nn_update_symbol(::LeastSquaresNNUpdate) = :ls
nn_update_symbol(::MultiplicativeNNUpdate) = :mu
nn_update_symbol(::HALSNNUpdate) = :hals
nn_update_symbol(::NNLSUpdate) = :nnls

abstract type AbstractBTDBlockMethod end
struct HOOIBlockMethod <: AbstractBTDBlockMethod end
struct STHOSVDBlockMethod <: AbstractBTDBlockMethod end
struct THOSVDBlockMethod <: AbstractBTDBlockMethod end
struct HOSVDBlockMethod <: AbstractBTDBlockMethod end

btd_block_method(method::AbstractBTDBlockMethod) = method
function btd_block_method(method::Symbol)
    method === :hooi && return HOOIBlockMethod()
    method === :sthosvd && return STHOSVDBlockMethod()
    method === :thosvd && return THOSVDBlockMethod()
    method === :hosvd && return HOSVDBlockMethod()
    throw(
        ArgumentError(
            "Unknown block_method=$method. Use :hooi, :sthosvd, :thosvd, or :hosvd.",
        ),
    )
end
btd_block_method_symbol(::HOOIBlockMethod) = :hooi
btd_block_method_symbol(::STHOSVDBlockMethod) = :sthosvd
btd_block_method_symbol(::THOSVDBlockMethod) = :thosvd
btd_block_method_symbol(::HOSVDBlockMethod) = :hosvd
