# dispatch/approx_routing.jl — manifold-family routing for approx()

# If every summand is a Manifolds.Segre with the same factor_dims, the call is a
# plain rank-r CPD. Route those directly into the cpd() tree so they get the
# CPDBackend, CPDResult, and all CPD-specific kwargs (geometry, nonnegative, ...).
function _all_segre_uniform(manifolds)
    isempty(manifolds) && return false
    all(m -> m isa Manifolds.Segre, manifolds) || return false
    d0 = factor_dims(first(manifolds))
    return all(m -> factor_dims(m) == d0, manifolds)
end

function _all_tucker_uniform(manifolds, target_shape::Tuple)
    isempty(manifolds) && return false
    all(m -> m isa Manifolds.Tucker, manifolds) || return false
    dims0 = factor_dims(first(manifolds))
    ranks0 = multilinear_rank(first(manifolds))
    dims0 == target_shape || return false
    return all(m -> factor_dims(m) == dims0 && multilinear_rank(m) == ranks0, manifolds)
end

@inline function _normalize_approx_dispatch(dispatch::Symbol)
    dispatch in (:auto, :generic, :cpd, :btd) || throw(
        ArgumentError(
            "Unknown approx dispatch=$dispatch. Use :auto, :generic, :cpd, or :btd.",
        ),
    )
    return dispatch
end
