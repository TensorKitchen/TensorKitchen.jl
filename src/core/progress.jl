# core/progress.jl — phase-aware progress infrastructure

abstract type AbstractMethodProgress end
abstract type AbstractPhaseProgress end
abstract type AbstractProgressFamily end

struct NoMethodProgress <: AbstractMethodProgress end

struct ALSFamily <: AbstractProgressFamily end
struct ManoptFamily <: AbstractProgressFamily end
struct TuckerFamily <: AbstractProgressFamily end

mutable struct FamilyProgress{F<:AbstractProgressFamily} <: AbstractMethodProgress
    meter::Union{PM.Progress,Nothing}
    phase::Symbol
    method::String
    was_rendered::Bool
end

mutable struct PhaseProgress <: AbstractPhaseProgress
    phase::Symbol
    initialization::AbstractMethodProgress
    refinement::AbstractMethodProgress
end

PhaseProgress(; phase::Symbol = :initialization) =
    PhaseProgress(phase, NoMethodProgress(), NoMethodProgress())

const _PHASE_PROGRESS_TLS_KEY = :tensor_kitchen_phase_progress

@inline function _current_phase_tracker()
    return get(task_local_storage(), _PHASE_PROGRESS_TLS_KEY, nothing)
end

function with_phase_progress(f::Function; tracker::PhaseProgress = PhaseProgress())
    tls = task_local_storage()
    old = get(tls, _PHASE_PROGRESS_TLS_KEY, nothing)
    tls[_PHASE_PROGRESS_TLS_KEY] = tracker
    try
        return f()
    finally
        if isnothing(old)
            pop!(tls, _PHASE_PROGRESS_TLS_KEY, nothing)
        else
            tls[_PHASE_PROGRESS_TLS_KEY] = old
        end
    end
end

@inline function phase_desc(phase::Symbol)
    phase == :initialization && return "Find initialization"
    return "Compute decomposition"
end

@inline _method_name(::NoMethodProgress) = "Unknown"
@inline _method_name(p::FamilyProgress) = p.method

@inline _meter(::NoMethodProgress) = nothing
@inline _meter(p::AbstractMethodProgress) = p.meter

@inline function _progress_barlen(desc::AbstractString, io::IO)
    return min(PM.tty_width(desc, io, false), 40)
end

function _make_meter(
    n::Integer;
    enabled::Bool,
    phase::Symbol,
    dt::Real = 0.2,
    delay::Real = 0.0,
    output::IO = stdout,
)
    enabled || return nothing
    desc = phase_desc(phase)
    progress = PM.Progress(
        n;
        dt = Float64(dt),
        desc,
        barlen = _progress_barlen(desc, output),
        output,
    )
    progress.tlast += Float64(delay)
    return progress
end

function _make_family_progress(
    ::Type{F},
    n::Integer;
    enabled::Bool,
    phase::Symbol = :refinement,
    method::AbstractString,
    kwargs...,
) where {F<:AbstractProgressFamily}
    progress = FamilyProgress{F}(
        _make_meter(n; enabled, phase, kwargs...),
        phase,
        String(method),
        false,
    )
    tracker = _current_phase_tracker()
    if tracker isa PhaseProgress
        attach_progress!(tracker, progress)
    end
    return progress
end

make_als_family_progress(
    n::Integer;
    enabled::Bool,
    phase::Symbol = :refinement,
    method::AbstractString = "ALS",
    kwargs...,
) = _make_family_progress(ALSFamily, n; enabled, phase, method, kwargs...)
make_manopt_family_progress(
    n::Integer;
    enabled::Bool,
    phase::Symbol = :refinement,
    method::AbstractString,
    kwargs...,
) = _make_family_progress(ManoptFamily, n; enabled, phase, method, kwargs...)
make_tucker_family_progress(
    n::Integer;
    enabled::Bool,
    phase::Symbol = :refinement,
    method::AbstractString,
    kwargs...,
) = _make_family_progress(TuckerFamily, n; enabled, phase, method, kwargs...)

# Backward-compatible wrappers used by solver call sites
make_als_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_als_family_progress(n; enabled, phase, method = "ALS", kwargs...)
make_rals_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_als_family_progress(n; enabled, phase, method = "CPRAND", kwargs...)
make_rals_mix_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_als_family_progress(n; enabled, phase, method = "CPRAND-MIX", kwargs...)
make_rgd_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_manopt_family_progress(n; enabled, phase, method = "RGD", kwargs...)
make_rgd_fixed_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_manopt_family_progress(n; enabled, phase, method = "RGD-fixed", kwargs...)
make_rcg_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_manopt_family_progress(n; enabled, phase, method = "RCG", kwargs...)
make_btd_tsd_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_manopt_family_progress(n; enabled, phase, method = "BTD-TSD", kwargs...)
make_hooi_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_tucker_family_progress(n; enabled, phase, method = "HOOI", kwargs...)
make_sthosvd_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_tucker_family_progress(n; enabled, phase, method = "ST-HOSVD", kwargs...)
make_thosvd_progress(n::Integer; enabled::Bool, phase::Symbol = :refinement, kwargs...) =
    make_tucker_family_progress(n; enabled, phase, method = "T-HOSVD", kwargs...)

@inline function set_phase!(tracker::PhaseProgress, phase::Symbol)
    tracker.phase = phase
    return tracker
end

function attach_progress!(tracker::PhaseProgress, progress::AbstractMethodProgress)
    if progress isa NoMethodProgress
        return tracker
    end
    if progress.phase == :initialization
        tracker.initialization = progress
    else
        tracker.refinement = progress
    end
    return tracker
end

@inline function active_progress(tracker::PhaseProgress)
    return tracker.phase == :initialization ? tracker.initialization : tracker.refinement
end

@inline _was_rendered(::NoMethodProgress) = false
@inline _was_rendered(p::FamilyProgress) = p.was_rendered
@inline _mark_rendered!(::NoMethodProgress) = nothing
@inline function _mark_rendered!(p::FamilyProgress)
    p.was_rendered = true
    return nothing
end

@inline function _force_visible_phase_finish(tracker::PhaseProgress, progress)
    return tracker.phase == :refinement &&
           _was_rendered(tracker.initialization) &&
           !_was_rendered(progress)
end

function _render_unrendered_completion!(meter, showvalues)
    # ProgressMeter does not render a meter that reaches 100% before its first
    # visible update. Give it one display-only step before completion.
    PM.update!(
        meter,
        meter.n;
        showvalues,
        force = true,
        max_steps = meter.n + 1,
    )
    return nothing
end

function _force_visible_unrendered_progress!(progress, meter, showvalues)
    _was_rendered(progress) && return nothing
    _render_unrendered_completion!(meter, showvalues)
    _mark_rendered!(progress)
    return nothing
end

update_progress!(::NoMethodProgress, args...; kwargs...) = nothing

function update_progress!(
    progress::AbstractMethodProgress,
    current::Integer;
    showvalues = nothing,
    force::Bool = false,
)
    meter = _meter(progress)
    isnothing(meter) && return nothing
    tracker = _current_phase_tracker()
    if tracker isa PhaseProgress
        set_phase!(tracker, progress.phase)
    end
    t = time()
    renders_by_time = t > meter.tlast + meter.dt
    if force || current >= meter.n || renders_by_time
        showvalues_with_method = if isnothing(showvalues)
            Any[("Method", _method_name(progress))]
        else
            Any[("Method", _method_name(progress)); showvalues]
        end
        PM.update!(meter, current; showvalues = showvalues_with_method, force)
        if force || current < meter.n || _was_rendered(progress)
            _mark_rendered!(progress)
        end
    end
    return nothing
end

finish_progress!(::NoMethodProgress; kwargs...) = nothing

function finish_progress!(
    tracker::PhaseProgress;
    current::Union{Nothing,Integer} = nothing,
    showvalues = nothing,
)
    progress = active_progress(tracker)
    progress isa NoMethodProgress && return nothing
    meter = _meter(progress)
    isnothing(meter) && return nothing

    showvalues_with_method = if isnothing(showvalues)
        Any[("Method", _method_name(progress))]
    else
        Any[("Method", _method_name(progress)); showvalues]
    end

    if _force_visible_phase_finish(tracker, progress)
        _force_visible_unrendered_progress!(progress, meter, showvalues_with_method)
        PM.finish!(meter; showvalues = showvalues_with_method)
        return nothing
    end

    _was_rendered(progress) || return nothing

    PM.finish!(meter; showvalues = showvalues_with_method)
    _mark_rendered!(progress)
    return nothing
end

function finish_progress!(
    progress::AbstractMethodProgress;
    current::Union{Nothing,Integer} = nothing,
    showvalues = nothing,
)
    meter = _meter(progress)
    isnothing(meter) && return nothing
    tracker = _current_phase_tracker()
    if tracker isa PhaseProgress
        set_phase!(tracker, progress.phase)
        if active_progress(tracker) === progress
            return finish_progress!(tracker; current, showvalues)
        end
        if _force_visible_phase_finish(tracker, progress)
            showvalues_with_method = if isnothing(showvalues)
                Any[("Method", _method_name(progress))]
            else
                Any[("Method", _method_name(progress)); showvalues]
            end
            _force_visible_unrendered_progress!(progress, meter, showvalues_with_method)
            PM.finish!(meter; showvalues = showvalues_with_method)
            return nothing
        end
    end
    showvalues_with_method = if isnothing(showvalues)
        Any[("Method", _method_name(progress))]
    else
        Any[("Method", _method_name(progress)); showvalues]
    end
    _was_rendered(progress) || return nothing
    PM.finish!(meter; showvalues = showvalues_with_method)
    _mark_rendered!(progress)
    return nothing
end
# ============================================================================
