# Optional callback instrumentation. No timing or counting wrapper is installed
# when diagnostics=false. Allocation measurement belongs in benchmarks.
mutable struct _LMOperatorStats
    calls::Vector{Int}
    nanoseconds::Vector{UInt64}
end
_LMOperatorStats() = _LMOperatorStats(zeros(Int, 3), zeros(UInt64, 3))

struct _MeasuredLMOperator{F}
    f::F
    stats::_LMOperatorStats
    index::Int
end
function (op::_MeasuredLMOperator)(args...)
    start = time_ns()
    try
        return op.f(args...)
    finally
        op.stats.calls[op.index] += 1
        op.stats.nanoseconds[op.index] += time_ns() - start
    end
end
_measure_lm(f, ::Nothing, index) = f
_measure_lm(f, stats::_LMOperatorStats, index) = _MeasuredLMOperator(f, stats, index)
_lm_operator_info(::Nothing) = (operator_calls = nothing, operator_seconds = nothing)
function _lm_operator_info(stats::_LMOperatorStats)
    labels = (:residual, :differential, :adjoint)
    return (
        operator_calls = NamedTuple{labels}(Tuple(stats.calls)),
        operator_seconds = NamedTuple{labels}(Tuple(stats.nanoseconds ./ 1e9)),
    )
end
