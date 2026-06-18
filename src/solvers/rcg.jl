# solvers/rcg.jl — Riemannian Conjugate Gradient
export RCGSolver

struct SegreProjectionTransport <: ManifoldsBase.AbstractVectorTransportMethod end

function _uses_segre_projection_transport(M)
    return _uses_segre_projection_transport_unwrapped(_unwrap_solver_manifold(M))
end

_uses_segre_projection_transport_unwrapped(::Manifolds.Segre) = true
_uses_segre_projection_transport_unwrapped(M::ProductManifold) =
    all(_uses_segre_projection_transport, M.manifolds)
_uses_segre_projection_transport_unwrapped(M) =
    hasproperty(M, :native) ? _uses_segre_projection_transport(getproperty(M, :native)) :
    false

function ManifoldsBase.vector_transport_to(
    M::Manifolds.Segre,
    p,
    X,
    q,
    ::SegreProjectionTransport,
)
    xparts = point_parts(X)
    qparts = point_parts(q)
    length(xparts) == length(qparts) || throw(
        DimensionMismatch(
            "Segre tangent/point part count mismatch: $(length(xparts)) vs $(length(qparts)).",
        ),
    )
    T = promote_type(eltype(_unwrap_part(xparts[1])), eltype(_unwrap_part(qparts[1])))
    ν = T(_unwrap_part(xparts[1])[1])
    Udot = Vector{Vector{T}}(undef, length(xparts) - 1)
    @inbounds for m in eachindex(Udot)
        xm = Vector{T}(_unwrap_part(xparts[m+1]))
        qm = _unwrap_part(qparts[m+1])
        length(xm) == length(qm) ||
            throw(DimensionMismatch("Segre mode $m transport length mismatch."))
        xm .-= dot(qm, xm) .* qm
        Udot[m] = xm
    end
    return pack_tangent_rank1_segre(ν, Udot)
end

function ManifoldsBase.vector_transport_to!(
    M::Manifolds.Segre,
    Y,
    p,
    X,
    q,
    m::SegreProjectionTransport,
)
    Ynew = vector_transport_to(M, p, X, q, m)
    yparts = point_parts(Y)
    newparts = point_parts(Ynew)
    length(yparts) == length(newparts) || throw(
        DimensionMismatch(
            "Segre transport destination part count mismatch: $(length(yparts)) vs $(length(newparts)).",
        ),
    )
    @inbounds for k in eachindex(newparts)
        yparts[k] = newparts[k]
    end
    return Y
end

"""
    solve_rcg(model_cost, model_egrad, M, p0; maxiter, tol, verbose, return_stats, model_grad, vector_transport_method)

Riemannian conjugate gradient. Uses a custom projection-style transport for
`Manifolds.Segre` / `ProductManifold(Manifolds.Segre(...), ...)`, and otherwise
prefers `ProjectionTransport()` when the current manifold/point layout
supports it, falling back to `SchildsLadderTransport()` as needed. Callers can
override this with `vector_transport_method=...` when they want an explicit
transport choice.
"""
function _default_vector_transport_method(M, p, retraction_method)
    if _uses_segre_projection_transport(M)
        return SegreProjectionTransport()
    end
    vt = ManifoldsBase.ProjectionTransport()
    try
        X = zero_vector(M, p)
        q = retract(M, p, X, retraction_method)
        Y = vector_transport_to(M, p, X, q, vt)
        return isnothing(check_vector(M, q, Y)) ? vt :
               ManifoldsBase.SchildsLadderTransport()
    catch
        return ManifoldsBase.SchildsLadderTransport()
    end
end

function solve_rcg(
    model_cost,
    model_egrad,
    M,
    p0;
    maxiter::Int = 2000,
    tol::Real = 1e-8,
    verbose::Bool = true,
    return_stats::Bool = false,
    normA2 = nothing,
    model_grad = nothing,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback = nothing,
    diagnostics_recorder = nothing,
    iteration_callbacks = (),
    grad_tol = nothing,
    normalized_objective::Bool = true,
)
    setup = _prepare_manopt_solver_functions(
        model_cost,
        model_egrad,
        M,
        p0;
        normA2,
        model_grad,
        tol,
        grad_tol,
        normalized_objective,
    )
    p0_local = setup.p0
    T = setup.T
    retraction_method = _solver_retraction_method(M, p0_local)
    transport =
        isnothing(vector_transport_method) ?
        _default_vector_transport_method(M, p0_local, retraction_method) :
        vector_transport_method
    tol_g = setup.dual_grad_tol
    dual_stop = StopWhenCostRelChangeAndGradientLess(T(tol), tol_g)

    stopping = _manopt_stopping(maxiter, setup.grad_stop_tol, dual_stop)
    callbacks = _manopt_callbacks(
        n -> make_rcg_progress(n; enabled = verbose, phase = :refinement, dt = 0.2),
        maxiter,
        verbose,
        setup.solver_cost,
        setup.solver_grad,
        M;
        diagnostics_recorder,
        post_step_callback,
        iteration_callbacks,
    )

    state = conjugate_gradient_descent(
        M,
        setup.solver_cost,
        setup.solver_grad,
        p0_local;
        retraction_method = retraction_method,
        vector_transport_method = transport,
        stopping_criterion = stopping,
        debug = callbacks.debug_actions,
        count = [:Cost, :Gradient],
        return_state = true,
    )

    return _manopt_finish_result(
        get_solver_result(state),
        state,
        callbacks.progress,
        diagnostics_recorder,
        setup.solver_cost,
        setup.solver_grad,
        M,
        normA2;
        tol_T = T(tol),
        maxiter,
        solver = :rcg,
        tiny_grad_tol = tol_g,
        return_stats,
        verbose,
        normalized_objective = setup.uses_relative_objective,
    )
end

# ========== RCGSolver (AbstractFirstOrderROSolver) ==========

"""
    RCGSolver

Riemannian conjugate gradient. Call via
`solve(RCGSolver(), model; init=:random, gradient_mode=:riemannian, vector_transport_method=nothing)`.
"""
struct RCGSolver <: AbstractFirstOrderROSolver end

solver_symbol(::RCGSolver) = :rcg
first_order_diagnostics_recorder(::RCGSolver) =
    _SolverDiagnosticsRecorder(line_search_enabled = true)

function run_first_order_solver(
    solver::RCGSolver,
    setup;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
    return_stats::Bool,
    vector_transport_method::Union{ManifoldsBase.AbstractVectorTransportMethod,Nothing} = nothing,
    post_step_callback,
    diagnostics_recorder,
    iteration_callbacks,
    grad_tol = nothing,
    normalized_objective::Bool = true,
)
    return solve_rcg(
        setup.model_cost,
        setup.model_egrad,
        setup.M,
        setup.p0;
        maxiter,
        tol,
        verbose,
        return_stats,
        normA2 = setup.normA2,
        model_grad = setup.model_grad,
        vector_transport_method,
        post_step_callback,
        diagnostics_recorder,
        iteration_callbacks,
        grad_tol,
        normalized_objective,
    )
end
